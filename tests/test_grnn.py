"""Differential tests against the original per-timestep FastGRNN equations.

Run with: python -m unittest discover -s tests -v
"""

import copy
from pathlib import Path
import types
import unittest

import torch
from omegaconf import OmegaConf

from src.model.dfnet_grnn import DfNetGRNN
from src.model.modules import GroupedGRNN, GroupedGRNNLayer
from src.utils.config_utils import setup_config


ROOT = Path(__file__).resolve().parents[1]
CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def original_direction(self, inputs, h0, params, reverse=False):
    b, t, _, _ = inputs.shape
    outputs = inputs.new_empty(b, t, self.groups, self.hidden_size)
    h = h0
    for idx in range(t-1, -1, -1) if reverse else range(t):
        h = self._grouped_step(inputs[:, idx], h, params)
        outputs[:, idx] = h
    return outputs, h


def reference_model(model):
    model = copy.deepcopy(model)
    for layer in model.modules():
        if isinstance(layer, GroupedGRNNLayer):
            layer._run_direction = types.MethodType(original_direction, layer)
    return model


class GRNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def compare_layer(self, device, dtype, backend, b=3, t=7, h=64, g=4,
                      bidirectional=False, batch_first=False, w_rank=None, u_rank=None,
                      gate='sigmoid', update='tanh', gamma=0.7):
        model = GroupedGRNNLayer(12*g, h, g, batch_first=batch_first,
                                bidirectional=bidirectional, w_rank=w_rank, u_rank=u_rank,
                                gate_non_linearity=gate, update_non_linearity=update,
                                backend=backend).to(device=device, dtype=dtype)
        model.gamma.data.fill_(gamma)
        model.lambd.data.fill_(0.2)
        ref = reference_model(model)
        # Start from noncontiguous sequence tensors to exercise layout conversion.
        x = torch.randn(b, t, 12*g, device=device, dtype=dtype).transpose(0, 1)
        if batch_first:
            x = x.transpose(0, 1)
        x = x.detach().requires_grad_()
        xr = x.detach().clone().requires_grad_()
        h0 = torch.randn(g*(2 if bidirectional else 1), b, h//g,
                         device=device, dtype=dtype, requires_grad=True)
        y, hn = model(x, h0)
        yr, hnr = ref(xr, h0)
        atol, rtol = (1e-12, 1e-10) if dtype == torch.float64 else (1e-5, 2e-5)
        torch.testing.assert_close(y, yr, atol=atol, rtol=rtol)
        torch.testing.assert_close(hn, hnr, atol=atol, rtol=rtol)
        dy = torch.randn_like(y)/max(y.numel(), 1)
        dh = torch.randn_like(hn)/max(hn.numel(), 1)
        torch.autograd.backward((y, hn), (dy, dh))
        torch.autograd.backward((yr, hnr), (dy, dh))
        if dtype == torch.float32:
            atol, rtol = 2e-6, 2e-4
        for name, grad, expected in [('input', x.grad, xr.grad)] + [
            (name, p.grad, q.grad) for (name, p), q in zip(model.named_parameters(), ref.parameters())
        ]:
            torch.testing.assert_close(grad, expected, atol=atol, rtol=rtol,
                                       msg=lambda message: name+' '+message)
        self.assertIsNone(h0.grad)  # Preserve the layer's deliberate state detachment.

    def test_torch_cpu_outputs_and_gradients(self):
        for bidirectional in (False, True):
            for batch_first in (False, True):
                for wr, ur in ((None, None), (3, None), (None, 3), (3, 3)):
                    with self.subTest(bidirectional=bidirectional, batch_first=batch_first, wr=wr, ur=ur):
                        self.compare_layer('cpu', torch.float64, 'torch',
                                           bidirectional=bidirectional, batch_first=batch_first,
                                           w_rank=wr, u_rank=ur)

    def test_alternative_activations_use_torch_fallback(self):
        for gate, update in [('quantSigm4', 'quantTanh'), ('relu', 'relu'),
                             ('tanh', 'sigmoid'), (torch.sigmoid, torch.tanh)]:
            with self.subTest(gate=gate, update=update):
                self.compare_layer('cpu', torch.float64, 'auto', gate=gate, update=update)

    def test_multilayer_default_state_preserves_dtype(self):
        model = GroupedGRNN(48, 64, num_layers=3, groups=4, add_outputs=True,
                            batch_first=False, backend='torch').double()
        ref = reference_model(model)
        x = torch.randn(7, 2, 48, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        y, h = model(x)
        yr, hr = ref(xr)
        torch.testing.assert_close(y, yr, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(h, hr, atol=1e-12, rtol=1e-10)
        y.square().mean().backward()
        yr.square().mean().backward()
        torch.testing.assert_close(x.grad, xr.grad, atol=1e-12, rtol=1e-10)

    def test_backend_validation(self):
        with self.assertRaises(ValueError):
            GroupedGRNNLayer(48, 64, 4, backend='invalid')
        with self.assertRaises(RuntimeError):
            GroupedGRNNLayer(48, 64, 4, backend='triton')(torch.randn(2, 7, 48))

    @unittest.skipUnless(CUDA, 'CUDA SM80+ required')
    def test_cuda_outputs_and_all_gradients(self):
        # Include masked batches, padded hidden dimensions, reverse direction,
        # low-rank factors and gamma clamp boundaries, with changing lengths.
        cases = [dict(b=1, t=1, h=8, g=8), dict(b=2, t=9),
                 dict(b=32, t=30, h=512, g=8), dict(b=17, t=16, h=384, g=8),
                 dict(b=3, t=21, h=128, g=8, w_rank=3, u_rank=3, bidirectional=True),
                 dict(b=3, t=17, h=704, g=8, batch_first=True),
                 dict(b=17, t=200, h=640, g=8),
                 dict(b=2, t=17, h=1024, g=8, w_rank=3, u_rank=3, bidirectional=True),
                 dict(b=1, t=200, h=512, g=8), dict(gamma=0.0), dict(gamma=1.0),
                 dict(gamma=-0.2), dict(gamma=1.2)]
        for case in cases:
            with self.subTest(**case):
                self.compare_layer('cuda', torch.float32, 'triton', **case)

    @unittest.skipUnless(CUDA, 'CUDA SM80+ required')
    def test_full_model_outputs_and_parameter_gradients(self):
        setup_config(OmegaConf.load(ROOT/'src/configs/grnn.yaml'))
        model = DfNetGRNN(mask_pf=True, grnn_backend='triton').cuda().train()
        ref = reference_model(model)
        inputs = dict(spec=torch.randn(2, 13, 161, 2, device='cuda').unsqueeze(1),
                      feat_erb=torch.randn(2, 1, 13, 24, device='cuda'),
                      feat_spec=torch.randn(2, 1, 13, 96, 2, device='cuda'))
        out, expected = model(**inputs), ref(**inputs)
        for y, yr in zip(out, expected):
            # Training-mode BatchNorm amplifies float32 reduction-order changes.
            # Both paths have larger error versus a float64 reference than this.
            torch.testing.assert_close(y, yr, atol=2e-4, rtol=3e-5)
        sum(y.square().mean() for y in out).backward()
        sum(y.square().mean() for y in expected).backward()
        for (name, p), q in zip(model.named_parameters(), ref.parameters()):
            # BatchNorm also amplifies rounding in gradients. Compare tensor
            # norms so individual near-zero components do not dominate error.
            error = torch.linalg.vector_norm(p.grad-q.grad)
            limit = 5e-4*torch.linalg.vector_norm(q.grad)+2e-6*q.numel()**0.5
            self.assertLessEqual(error.item(), limit.item(), name)
        self.assertEqual(list(model.state_dict()), list(ref.state_dict()))

    @unittest.skipUnless(CUDA and (ROOT/'models/grnn.pth').exists()
                         and (ROOT/'data/vctk/onebatch/00000000.pt').exists(),
                         'Local checkpoint, onebatch sample and CUDA SM80+ required')
    def test_pretrained_real_audio_loss_and_optimizer_step(self):
        from hydra.utils import instantiate
        from safetensors.torch import load_file
        from src.utils.df_utils import (audio_lengths_to_frame_lengths, exp_unit_norm,
                                        stft_with_df_config)
        from src.utils.erb_utils import compute_erb_feats_from_stft

        config = OmegaConf.load(ROOT/'src/configs/grnn.yaml')
        setup_config(config)
        model = DfNetGRNN(mask_pf=True, grnn_backend='triton').cuda().train()
        checkpoint = torch.load(ROOT/'models/grnn.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        ref = reference_model(model)
        data = load_file(ROOT/'data/vctk/onebatch/00000000.pt')
        noisy, clean = data['noisy'].cuda().unsqueeze(0), data['clean'].cuda().unsqueeze(0)
        ns, cs = stft_with_df_config(noisy), stft_with_df_config(clean)
        lengths = audio_lengths_to_frame_lengths(torch.tensor([noisy.shape[-1]], device='cuda'),
                                                config.stft.fft_size, config.stft.hop_length)
        noisy_spec = ns.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()
        clean_spec = cs.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()
        feat_spec, _ = exp_unit_norm(noisy_spec[:, :, :, :config.stft.nb_df, :], lengths)
        inputs = dict(spec=noisy_spec, feat_spec=feat_spec,
                      feat_erb=compute_erb_feats_from_stft(ns, lengths_frames=lengths))
        criterion = instantiate(config.loss_function).cuda()
        optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (model, ref)]
        losses = []
        for m in (model, ref):
            enhanced, mask, lsnr, alpha = m(**inputs)
            loss = criterion(clean=clean_spec, noisy=noisy_spec, enh=enhanced, m=mask,
                             lsnr=lsnr, df_alpha=alpha)['loss']
            self.assertTrue(torch.isfinite(loss))
            losses.append(loss.detach())
            loss.backward()
        torch.testing.assert_close(losses[0], losses[1], atol=1e-4, rtol=2e-5)
        for (name, p), q in zip(model.named_parameters(), ref.parameters()):
            if p.grad is None or q.grad is None:
                self.assertIs(p.grad, q.grad)
            else:
                # Fractional spectral powers amplify rounding near quiet bins;
                # test the full gradient, rather than relative error near zero.
                error = torch.linalg.vector_norm(p.grad-q.grad)
                limit = 2e-3*torch.linalg.vector_norm(q.grad)+2e-6*q.numel()**0.5
                self.assertLessEqual(error.item(), limit.item(), name)
        for optimizer in optimizers:
            optimizer.step()
        for (name, p), q in zip(model.named_parameters(), ref.parameters()):
            error = torch.linalg.vector_norm(p-q)
            limit = 2e-3*torch.linalg.vector_norm(q)+2e-6*q.numel()**0.5
            self.assertLessEqual(error.item(), limit.item(), name)


if __name__ == '__main__':
    unittest.main()
