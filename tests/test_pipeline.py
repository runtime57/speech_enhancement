import copy
import importlib.util
import logging
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch

from src.loss.Loss import DfAlphaLoss, SpectralLoss
from src.model.modules import DfOp
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.config_utils import setup_config
from src.utils.df_utils import (audio_lengths_to_frame_lengths, exp_unit_norm,
                                istft_with_df_config, stft_with_df_config)
from src.utils.erb_utils import compute_erb_feats_from_stft

ROOT = Path(__file__).resolve().parents[1]


class NullWriter:
    def __init__(self):
        self.scalars = []
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_step(self, step, mode="train"):
        self.step, self.mode = step, mode

    def add_scalar(self, name, value):
        self.scalars.append((name, value, self.step, self.epoch, self.mode))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/configs")):
            cls.config = compose(config_name="dfnet")

    def setUp(self):
        setup_config(self.config)
        torch.manual_seed(42)

    def test_alpha_matches_squared_paper_equation_and_excludes_padding(self):
        alpha = torch.tensor([[[0.2], [0.3], [0.4], [0.9]]], requires_grad=True)
        snr = torch.tensor([[-12., 0., -7., -20.]])
        result = DfAlphaLoss(0.05)(alpha, snr, lengths=torch.tensor([3]))
        torch.testing.assert_close(result, torch.tensor(0.05 * (0.2**2 + 0.7**2) / 3))
        result.backward()
        torch.testing.assert_close(alpha.grad, torch.tensor([[[0.05 * 0.4 / 3],
                                                            [-0.05 * 1.4 / 3], [0.], [0.]]]))

    def test_spectral_loss_matches_complex_norm_in_equation_five(self):
        target = torch.tensor([[[[1+2j, 2-1j], [3+1j, 1-1j]]]])
        enhanced = torch.tensor([[[[2+1j, 1-2j], [2+2j, 2-1j]]]])
        cm = enhanced.abs().pow(0.6)
        tm = target.abs().pow(0.6)
        cc = torch.polar(cm, enhanced.angle())
        tc = torch.polar(tm, target.angle())
        expected = ((cm-tm).square() + (cc-tc).abs().square()).sum(-1).mean()
        actual = SpectralLoss(gamma=0.6, frequency_reduction="sum")(enhanced, target)
        torch.testing.assert_close(actual, expected)

    def test_paper_loss_and_gradients_ignore_added_padding(self):
        criterion = instantiate(self.config.loss_function)
        clean = torch.randn(1, 1, 6, 161, 2) * 0.01
        noisy = clean + torch.randn_like(clean) * 0.004
        enh = clean * 0.8
        alpha = torch.full((1, 6, 1), 0.5)
        expected = criterion(clean, noisy, enh=enh, df_alpha=alpha)["loss"]
        padded = lambda x, value: torch.cat((x, torch.full_like(x, value)), dim=2)
        padded_enh = padded(enh, 123).requires_grad_()
        padded_alpha = torch.cat((alpha, torch.full_like(alpha, 456)), dim=1).requires_grad_()
        result = criterion(padded(clean, 42), padded(noisy, 84), enh=padded_enh,
                           df_alpha=padded_alpha, frame_lengths=torch.tensor([6]))
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        self.assertEqual(padded_enh.grad[:, :, 6:].count_nonzero().item(), 0)
        self.assertEqual(padded_alpha.grad[:, 6:].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(padded_enh.grad).all())

    def test_stft_and_synthesis_are_independent_of_other_batch_lengths(self):
        lengths = torch.tensor([2201, 4001])
        audio = torch.randn(2, 4001) * 0.1
        audio[0, 2201:] = 0
        spectrum = stft_with_df_config(audio, lengths)
        for i, size in enumerate(lengths.tolist()):
            single = stft_with_df_config(audio[i, :size])
            torch.testing.assert_close(spectrum[i, :, :single.shape[1]], single)
        reconstructed = istft_with_df_config(spectrum, lengths=lengths)
        torch.testing.assert_close(reconstructed, audio, atol=1e-6, rtol=1e-5)

    def test_features_preserve_real_prefix_and_zero_invalid_frames(self):
        lengths = torch.tensor([2201, 4001])
        audio = torch.randn(2, 4001) * 0.1
        audio[0, 2201:] = 0
        spectrum = stft_with_df_config(audio, lengths)
        frames = audio_lengths_to_frame_lengths(lengths, 320, 160)
        erb = compute_erb_feats_from_stft(spectrum, lengths_frames=frames)
        packed = spectrum.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()
        df, _ = exp_unit_norm(packed[:, :, :, :96], frames)
        n = frames[0].item()
        single = spectrum[0:1, :, :n]
        torch.testing.assert_close(erb[0:1, :, :n], compute_erb_feats_from_stft(single))
        self.assertEqual(erb[0, :, n:].count_nonzero().item(), 0)
        self.assertEqual(df[0, :, n:].count_nonzero().item(), 0)

    def test_deep_filter_matches_direct_complex_multiply_and_gate(self):
        spec = torch.randn(2, 1, 9, 7, 2)
        coefs = torch.randn(2, 9, 5, 3, 2) * 0.1
        alpha = torch.rand(2, 9, 1)
        x, c = torch.view_as_complex(spec), torch.view_as_complex(coefs)
        expected = x.clone()
        for t in range(9):
            filtered = torch.zeros(2, 3, dtype=torch.complex64)
            for tap in range(5):
                source_t = t + tap - 3
                if 0 <= source_t < 9:
                    filtered += x[:, 0, source_t, :3] * c[:, t, tap]
            expected[:, 0, t, :3] = alpha[:, t] * filtered + (1-alpha[:, t]) * x[:, 0, t, :3]
        for method in ("real_loop", "real_unfold", "real_strided", "complex_strided"):
            with self.subTest(method=method):
                actual = DfOp(3, 5, 1, method=method)(spec, coefs, alpha)
                torch.testing.assert_close(actual, torch.view_as_real(expected), atol=1e-6, rtol=1e-5)

    def test_model_evaluation_matches_single_utterance_inside_padded_batch(self):
        model = instantiate(self.config.model).eval()
        lengths = torch.tensor([2201, 4001])
        audio = torch.randn(2, 4001) * 0.1
        audio[0, 2201:] = 0
        def enhance(waveforms, sizes):
            spectrum = stft_with_df_config(waveforms, sizes)
            frames = audio_lengths_to_frame_lengths(sizes, 320, 160)
            packed = spectrum.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()
            df, _ = exp_unit_norm(packed[:, :, :, :96], frames)
            erb = compute_erb_feats_from_stft(spectrum, lengths_frames=frames)
            enhanced = model(packed, erb, df)[0]
            return istft_with_df_config(enhanced.squeeze(1).permute(0, 2, 1, 3), lengths=sizes)
        with torch.no_grad():
            batch = enhance(audio, lengths)
            single = enhance(audio[:1, :2201], lengths[:1])
        torch.testing.assert_close(batch[:1, :2201], single, atol=2e-5, rtol=2e-4)

    def test_optional_losses_ignore_padding_when_lengths_are_provided(self):
        config = copy.deepcopy(self.config)
        config.loss_function.mask_factor = 1.0
        config.loss_function.mr_factor = 0.5
        config.loss_function.mr_factor_complex = 0.5
        config.loss_function.sdr_factor = 0.1
        config.loss_function.lsnr_factor = 0.2
        criterion = instantiate(config.loss_function)
        clean_wave = torch.randn(1, 3201) * 0.1
        noisy_wave = clean_wave + torch.randn_like(clean_wave) * 0.02
        packed = lambda x: stft_with_df_config(x).unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()
        clean, noisy = packed(clean_wave), packed(noisy_wave)
        t = clean.shape[2]
        enh = clean * 0.8
        mask = torch.full((1, 1, t, 24), 0.8)
        alpha = torch.full((1, t, 1), 0.5)
        lsnr = torch.zeros_like(alpha)
        arguments = dict(frame_lengths=torch.tensor([t]), lengths=torch.tensor([3201]))
        expected = criterion(clean, noisy, enh, mask, lsnr, alpha, **arguments)
        pad = lambda x: torch.cat((x, torch.zeros_like(x)), dim=2)
        pad_alpha = lambda x: torch.cat((x, torch.zeros_like(x)), dim=1)
        actual = criterion(pad(clean), pad(noisy), pad(enh), pad(mask), pad_alpha(lsnr),
                           pad_alpha(alpha), **arguments)
        for name in expected:
            with self.subTest(loss=name):
                torch.testing.assert_close(actual[name], expected[name])

    def test_full_epoch_average_scheduler_and_validation_selection(self):
        class ToyTrainer(BaseTrainer):
            def process_batch(self, batch, metrics, step):
                self.optimizer.zero_grad()
                (self.model.weight.sum()-1).square().backward()
                self.optimizer.step()
                metrics.update("loss", batch["value"], n=batch["weight"])
                return {"loss": torch.tensor(batch["value"])}

            def _evaluation_epoch(self, epoch, part, loader):
                self.evaluated.append(part)
                return {"loss": 1.0}

            def _log_batch(self, *args):
                pass

            def _save_checkpoint(self, *args, **kwargs):
                pass

        config = copy.deepcopy(self.config)
        config.trainer.n_epochs = 4
        config.trainer.log_step = 2
        config.writer = {"loss_names": ["loss"], "run_name": "test_pipeline"}
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
        loader = [{"value": 1., "weight": 1}, {"value": 3., "weight": 2}, {"value": 7., "weight": 1}]
        writer = NullWriter()
        trainer = ToyTrainer(model, None, {"train": [], "inference": []}, optimizer, scheduler,
                             config, "cpu", {"train": loader, "val": [None], "test": [None]},
                             logging.getLogger("pipeline_test"), writer, batch_transforms={})
        trainer.evaluated = []
        self.assertEqual(trainer._train_epoch(1)["loss"], 3.5)
        self.assertIn(("epoch/loss", 3.5, 3, 1, "train"), writer.scalars)
        self.assertEqual(trainer.evaluated, [])
        trainer.evaluated.clear()
        trainer._train_process()
        self.assertEqual(trainer.evaluated, ["val"] * 4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0009)

    def test_speech_inference_saves_unique_files_for_short_final_batch(self):
        import torchaudio
        from omegaconf import OmegaConf
        from src.datasets.collate import collate_fn
        from src.trainer.inferencer import Inferencer
        class IdentityEnhancer(torch.nn.Module):
            def forward(self, spec, **kwargs):
                return spec, None, None, None
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            OmegaConf.set_struct(config, False)
            config.inferencer = {"device_tensors": ["noisy", "clean"], "amp": False,
                                 "save_predictions": True}
            audios = [torch.randn(size) * 0.1 for size in (2201, 4001, 3201)]
            items = [{"clean": audio, "noisy": audio} for audio in audios]
            loader = [collate_fn(items[:2]), collate_fn(items[2:])]
            inferencer = Inferencer(IdentityEnhancer(), config, "cpu", {"test": loader},
                                    Path(directory), metrics=None, skip_model_load=True)
            self.assertEqual(inferencer.run_inference(), {"test": {}})
            files = sorted((Path(directory) / "test").glob("*.wav"))
            self.assertEqual([p.name for p in files], ["00000000.wav", "00000001.wav", "00000002.wav"])
            for path, original in zip(files, audios):
                waveform, sr = torchaudio.load(str(path))
                self.assertEqual(sr, 16000)
                torch.testing.assert_close(waveform[0], original, atol=4e-5, rtol=1e-4)

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "TensorBoard dependency required")
    def test_tensorboard_events_are_readable(self):
        from src.logger.tensorboard import TensorBoardWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as directory:
            writer = TensorBoardWriter(logging.getLogger("tb_test"), {"trainer": {"save_dir": directory}},
                                       "speech_enhancement", "test", log_dir=directory)
            writer.set_step(337, "val")
            writer.add_scalar("loss", 0.25)
            writer.close()
            events = EventAccumulator(directory).Reload()
            scalar = events.Scalars("loss_val")[0]
            self.assertEqual((scalar.step, scalar.value), (337, 0.25))


if __name__ == "__main__":
    unittest.main()
