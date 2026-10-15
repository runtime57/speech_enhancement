"""Exercise AMP through the real trainer, and half precision waveform inference."""

import logging
from pathlib import Path
import unittest

import torch
from omegaconf import OmegaConf

from src.model.dfnet_grnn import DfNetGRNN
from src.model.modules import GroupedGRNNLayer, GroupedGRU, GroupedGRULayer
from src.utils.config_utils import setup_config
from src.utils.precision import amp_settings, set_inference_precision


ROOT = Path(__file__).resolve().parents[1]
CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
LOCAL_AUDIO = (ROOT/'data/vctk/onebatch/00000000.pt').exists() and (ROOT/'models/grnn.pth').exists()


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.config = OmegaConf.load(ROOT/'src/configs/grnn.yaml')
        setup_config(self.config)

    def test_cpu_amp_is_disabled_and_options_are_validated(self):
        enabled, dtype = amp_settings({'amp': True}, 'cpu')
        self.assertFalse(enabled)
        self.assertEqual(dtype, torch.float16)
        with self.assertRaises(ValueError):
            amp_settings({'amp_dtype': 'invalid'}, 'cuda')
        with self.assertRaises(ValueError):
            set_inference_precision(torch.nn.Linear(2, 2), 'invalid')

    @unittest.skipUnless(CUDA, 'CUDA SM80+ required')
    def test_grouped_gru_half_state_and_residual_preserve_dtype(self):
        for model in (GroupedGRULayer(48, 64, 4),
                      GroupedGRU(48, 64, groups=4, num_layers=2, add_outputs=True)):
            with self.subTest(model=type(model).__name__):
                model = model.cuda().half()
                x = torch.randn(3, 23, 48, device='cuda', dtype=torch.float16)
                with torch.no_grad():
                    output, state = model(x)
                self.assertEqual(output.dtype, torch.float16)
                self.assertEqual(state.dtype, torch.float16)
                self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(CUDA, 'CUDA SM80+ required')
    def test_amp_grnn_outputs_and_gradients_are_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = GroupedGRNNLayer(48, 64, 4, backend='triton').cuda()
                x = torch.randn(3, 23, 48, device='cuda', requires_grad=True)
                with torch.autocast('cuda', dtype=dtype):
                    output, state = model(x)
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(state.dtype, dtype)
                (output.float().square().mean()+state.float().square().mean()).backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter in model.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertEqual(parameter.grad.dtype, torch.float32)

    @unittest.skipUnless(CUDA and LOCAL_AUDIO, 'Local audio, checkpoint and CUDA SM80+ required')
    def test_fp16_and_amp_denoise_preserve_waveform_shape(self):
        from denoise import _denoise_waveform
        from safetensors.torch import load_file

        model = DfNetGRNN(mask_pf=True).cuda().eval()
        checkpoint = torch.load(ROOT/'models/grnn.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        noisy = load_file(ROOT/'data/vctk/onebatch/00000000.pt')['noisy'].cuda().unsqueeze(0)
        expected = _denoise_waveform(model, noisy)
        for precision in ('amp', 'fp16'):
            with self.subTest(precision=precision):
                set_inference_precision(model, precision)
                actual = _denoise_waveform(model, noisy, precision=precision)
                self.assertEqual(actual.shape, noisy.shape)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all())
                relative_error = torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)
                self.assertLess(relative_error.item(), 0.01)
                self.assertEqual(next(model.parameters()).dtype,
                                 torch.float16 if precision == 'fp16' else torch.float32)

    @unittest.skipUnless(CUDA and LOCAL_AUDIO, 'Local audio, checkpoint and CUDA SM80+ required')
    def test_trainer_amp_step_clipping_and_validation(self):
        from hydra.utils import instantiate
        from safetensors.torch import load_file
        from src.metrics.tracker import MetricTracker
        from src.trainer.trainer import Trainer

        config = self.config
        config.writer = {'loss_names': ['loss'], 'run_name': 'amp_test'}
        config.trainer.max_grad_norm = 1.0
        model = DfNetGRNN(mask_pf=True).cuda()
        checkpoint = torch.load(ROOT/'models/grnn.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        trainer = Trainer(model, instantiate(config.loss_function).cuda(),
                          {'train': [], 'inference': []}, optimizer, scheduler, config, 'cuda',
                          {'train': [None]}, logging.getLogger('amp_test'), writer=None,
                          batch_transforms={})
        source = load_file(ROOT/'data/vctk/onebatch/00000000.pt')
        batch = dict(noisy=source['noisy'].unsqueeze(0), clean=source['clean'].unsqueeze(0),
                     lengths=torch.tensor([source['noisy'].numel()]))
        initial = next(model.parameters()).detach().clone()
        for attempt in range(16):
            result = trainer.process_batch(dict(batch), MetricTracker('loss'), step=attempt)
            self.assertEqual(result['loss'].dtype, torch.float32)
            self.assertTrue(torch.isfinite(result['loss']))
            if not torch.equal(initial, next(model.parameters())):
                break
        else:
            self.fail('GradScaler did not produce an optimizer update in 16 attempts.')
        gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
        self.assertLessEqual(norm.item(), 1.0001)
        scaler_state = trainer.grad_scaler.state_dict()
        self.assertGreater(scaler_state['scale'], 0)
        clone = torch.amp.GradScaler('cuda')
        clone.load_state_dict(scaler_state)
        self.assertEqual(clone.state_dict(), scaler_state)
        trainer.is_train = False
        model.eval()
        before = next(model.parameters()).detach().clone()
        with torch.no_grad():
            result = trainer.process_batch(dict(batch), MetricTracker('loss'), step=None)
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertEqual(result['enh_wav'].shape, batch['noisy'].shape)
        self.assertEqual(result['enh_wav'].dtype, torch.float32)
        self.assertTrue(torch.equal(before, next(model.parameters())))


if __name__ == '__main__':
    unittest.main()
