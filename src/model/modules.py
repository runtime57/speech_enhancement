# code from Official DeepFilterNet repository: https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/modules.py

import math
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from typing_extensions import Final

from src.model.comfifastgrnn import gen_non_linearity
from src.utils.config_utils import stft_config

from src.utils.df_utils import as_complex, as_real, get_norm_alpha
from src.utils.erb_utils import erb_widths

class Conv2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Iterable[int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
    ):
        """Causal Conv2d by delaying the signal for any lookahead.

        Expected input format: [B, C, T, F]
        """
        lookahead = 0  # This needs to be handled on the input feature side
        # Padding on time axis
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        if fpad:
            fpad_ = kernel_size[1] // 2 + dilation - 1
        else:
            fpad_ = 0
        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False
        layers.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(0, fpad_),
                stride=(1, fstride),  # Stride over time is always 1
                dilation=(1, dilation),  # Same for dilation
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
    ):
        """Causal ConvTranspose2d.

        Expected input format: [B, C, T, F]
        """
        # Padding on time axis, with lookahead = 0
        lookahead = 0  # This needs to be handled on the input feature side
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if fpad:
            fpad_ = kernel_size[1] // 2
        else:
            fpad_ = 0
        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(kernel_size[0] - 1, fpad_ + dilation - 1),
                output_padding=(0, fpad_),
                stride=(1, fstride),  # Stride over time is always 1
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


def convkxf(
    in_ch: int,
    out_ch: Optional[int] = None,
    k: int = 1,
    f: int = 3,
    fstride: int = 2,
    lookahead: int = 0,
    batch_norm: bool = False,
    act: nn.Module = nn.ReLU(inplace=True),
    mode="normal",  # must be "normal", "transposed" or "upsample"
    depthwise: bool = True,
    complex_in: bool = False,
):
    bias = batch_norm is False
    assert f % 2 == 1
    stride = 1 if f == 1 else (1, fstride)
    if out_ch is None:
        out_ch = in_ch * 2 if mode == "normal" else in_ch // 2
    fpad = (f - 1) // 2
    convpad = (0, fpad)
    modules = []
    # Manually pad for time axis kernel to not introduce delay
    pad = (0, 0, k - 1 - lookahead, lookahead)
    if any(p > 0 for p in pad):
        modules.append(("pad", nn.ConstantPad2d(pad, 0.0)))
    if depthwise:
        groups = min(in_ch, out_ch)
    else:
        groups = 1
    if in_ch % groups != 0 or out_ch % groups != 0:
        groups = 1
    if complex_in and groups % 2 == 0:
        groups //= 2
    convkwargs = {
        "in_channels": in_ch,
        "out_channels": out_ch,
        "kernel_size": (k, f),
        "stride": stride,
        "groups": groups,
        "bias": bias,
    }
    if mode == "normal":
        modules.append(("sconv", nn.Conv2d(padding=convpad, **convkwargs)))
    elif mode == "transposed":
        # Since pytorch's transposed conv padding does not correspond to the actual padding but
        # rather the padding that was used in the encoder conv, we need to set time axis padding
        # according to k. E.g., this disables the padding for k=2:
        #     dilation - (k - 1) - padding
        #   = 1        - (2 - 1) - 1 = 0; => padding = fpad (=1 for k=2)
        padding = (k - 1, fpad)
        modules.append(
            ("sconvt", nn.ConvTranspose2d(padding=padding, output_padding=convpad, **convkwargs))
        )
    elif mode == "upsample":
        modules.append(("upsample", FreqUpsample(fstride)))
        convkwargs["stride"] = 1
        modules.append(("sconv", nn.Conv2d(padding=convpad, **convkwargs)))
    else:
        raise NotImplementedError()
    if groups > 1:
        modules.append(("1x1conv", nn.Conv2d(out_ch, out_ch, 1, bias=False)))
    if batch_norm:
        modules.append(("norm", nn.BatchNorm2d(out_ch)))
    modules.append(("act", act))
    return nn.Sequential(OrderedDict(modules))


class FreqUpsample(nn.Module):
    def __init__(self, factor: int, mode="nearest"):
        super().__init__()
        self.f = float(factor)
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=[1.0, self.f], mode=self.mode)


def _erb_fb(widths: np.ndarray, sr: int, normalized: bool = True, inverse: bool = False) -> Tensor:
    n_freqs = int(np.sum(widths))
    all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]

    b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]

    fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
    for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
        fb[b : b + w, i] = 1
    # Normalize to constant energy per resulting band
    if inverse:
        fb = fb.t()
        if not normalized:
            fb /= fb.sum(dim=1, keepdim=True)
    else:
        if normalized:
            fb /= fb.sum(dim=0)
    return fb

def erb_filterbank(
    normalized: bool = True,
    inverse: bool = False,
) -> Tensor:
    """
    Build a rectangular ERB filter bank.

    Returns a matrix of shape:
      - [F, B] when inverse=False
      - [B, F] when inverse=True
    where F = fft_size // 2 + 1 and B = nb_erb.
    """
    p = stft_config()
    widths = erb_widths(p.sr, p.fft_size, p.nb_erb, min_nb_freqs=p.min_nb_freqs)
    return _erb_fb(widths, p.sr, normalized=normalized, inverse=inverse)




class Mask(nn.Module):
    def __init__(self, erb_inv_fb: Tensor, post_filter: bool = False, eps: float = 1e-12):
        super().__init__()
        self.erb_inv_fb: Tensor
        self.register_buffer("erb_inv_fb", erb_inv_fb)
        self.clamp_tensor = torch.__version__ > "1.9.0" or torch.__version__ == "1.9.0"
        self.post_filter = post_filter
        self.eps = eps

    def pf(self, mask: Tensor, beta: float = 0.02) -> Tensor:
        """Post-Filter proposed by Valin et al. [1].

        Args:
            mask (Tensor): Real valued mask, typically of shape [B, C, T, F].
            beta: Global gain factor.
        Refs:
            [1]: Valin et al.: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech.
        """
        mask_sin = mask * torch.sin(np.pi * mask / 2)
        mask_pf = (1 + beta) * mask / (1 + beta * mask.div(mask_sin.clamp_min(self.eps)).pow(2))
        return mask_pf

    def forward(self, spec: Tensor, mask: Tensor, atten_lim: Optional[Tensor] = None) -> Tensor:
        # spec (real) [B, 1, T, F, 2], F: freq_bins
        # mask (real): [B, 1, T, Fe], Fe: erb_bins
        # atten_lim: [B]
        if not self.training and self.post_filter:
            mask = self.pf(mask)
        if atten_lim is not None:
            # dB to amplitude
            atten_lim = 10 ** (-atten_lim / 20)
            # Greater equal (__ge__) not implemented for TorchVersion.
            if self.clamp_tensor:
                # Supported by torch >= 1.9
                mask = mask.clamp(min=atten_lim.view(-1, 1, 1, 1))
            else:
                m_out = []
                for i in range(atten_lim.shape[0]):
                    m_out.append(mask[i].clamp_min(atten_lim[i].item()))
                mask = torch.stack(m_out, dim=0)
        mask = mask.matmul(self.erb_inv_fb)  # [B, 1, T, F]
        if not spec.is_complex():
            mask = mask.unsqueeze(4)
        return spec * mask


class ExponentialUnitNorm(nn.Module):
    """Unit norm for a complex spectrogram.

    This should match the rust code:
    ```rust
        for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
            *s = x.norm() * (1. - alpha) + *s * alpha;
            *x /= s.sqrt();
        }
    ```
    """

    alpha: Final[float]
    eps: Final[float]

    def __init__(self, alpha: float, num_freq_bins: int, eps: float = 1e-14):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.init_state: Tensor
        s = torch.from_numpy(unit_norm_init(num_freq_bins)).view(1, 1, num_freq_bins, 1)
        self.register_buffer("init_state", s)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, F, 2]
        b, c, t, f, _ = x.shape
        x_abs = x.square().sum(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        state = self.init_state.clone().expand(b, c, f, 1)
        out_states: List[Tensor] = []
        for t in range(t):
            state = x_abs[:, :, t] * (1 - self.alpha) + state * self.alpha
            out_states.append(state)
        return x / torch.stack(out_states, 2).sqrt()


class DfOp(nn.Module):
    df_order: Final[int]
    df_bins: Final[int]
    df_lookahead: Final[int]
    freq_bins: Final[int]

    def __init__(
        self,
        df_bins: int,
        df_order: int = 5,
        df_lookahead: int = 0,
        method: str = "complex_strided",
        freq_bins: int = 0,
    ):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins
        self.df_lookahead = df_lookahead
        self.freq_bins = freq_bins
        self.set_forward(method)

    def set_forward(self, method: str):
        # All forward methods should be mathematically similar.
        # DeepFilterNet results are obtained with 'real_unfold'.
        forward_methods = {
            "real_loop": self.forward_real_loop,
            "real_strided": self.forward_real_strided,
            "real_unfold": self.forward_real_unfold,
            "complex_strided": self.forward_complex_strided,
            "real_one_step": self.forward_real_no_pad_one_step,
            "real_hidden_state_loop": self.forward_real_hidden_state_loop,
        }
        if method not in forward_methods.keys():
            raise NotImplementedError(
                f"`method` must be one of {forward_methods.keys()}, but got '{method}'"
            )
        if method == "real_hidden_state_loop":
            assert self.freq_bins >= self.df_bins
            self.spec_buf: Tensor
            # Currently only designed for batch size of 1
            self.register_buffer(
                "spec_buf", torch.zeros(1, 1, self.df_order, self.freq_bins, 2), persistent=False
            )
        self.forward = forward_methods[method]

    def forward_real_loop(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version 0: Manual loop over df_order, maybe best for onnx export?
        b, _, t, _, _ = spec.shape
        f = self.df_bins
        padded = spec_pad(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )

        spec_f = torch.zeros((b, t, f, 2), device=spec.device)
        for i in range(self.df_order):
            spec_f[..., 0] += padded[:, i : i + t, ..., 0] * coefs[:, :, i, :, 0]
            spec_f[..., 0] -= padded[:, i : i + t, ..., 1] * coefs[:, :, i, :, 1]
            spec_f[..., 1] += padded[:, i : i + t, ..., 1] * coefs[:, :, i, :, 0]
            spec_f[..., 1] += padded[:, i : i + t, ..., 0] * coefs[:, :, i, :, 1]
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_strided(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version1: Use as_strided instead of unfold
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = as_strided(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        # Complex numbers are not supported by onnx
        re = padded[..., 0] * coefs[..., 0]
        re -= padded[..., 1] * coefs[..., 1]
        im = padded[..., 1] * coefs[..., 0]
        im += padded[..., 0] * coefs[..., 1]
        spec_f = torch.stack((re, im), -1).sum(2)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_unfold(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version2: Unfold
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = spec_pad(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        padded = padded.unfold(dimension=1, size=self.df_order, step=1)  # [B, T, F, 2, O]
        padded = padded.permute(0, 1, 4, 2, 3)
        spec_f = torch.empty_like(padded)
        spec_f[..., 0] = padded[..., 0] * coefs[..., 0]  # re1
        spec_f[..., 0] -= padded[..., 1] * coefs[..., 1]  # re2
        spec_f[..., 1] = padded[..., 1] * coefs[..., 0]  # im1
        spec_f[..., 1] += padded[..., 0] * coefs[..., 1]  # im2
        spec_f = spec_f.sum(dim=2)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_complex_strided(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version3: Complex strided; definatly nicest, no permute, no indexing, but complex gradient
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = as_strided(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        spec_f = torch.sum(torch.view_as_complex(padded) * torch.view_as_complex(coefs), dim=2)
        spec_f = torch.view_as_real(spec_f)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_no_pad_one_step(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version4: Only viable for onnx handling. `spec` needs external (ring-)buffer handling.
        # Thus, time steps `t` must be equal to `df_order`.

        # spec (real) [B, 1, O, F', 2]
        # coefs (real) [B, 1, O, F, 2]
        assert (
            spec.shape[2] == self.df_order
        ), "This forward method needs spectrogram buffer with `df_order` time steps as input"
        assert coefs.shape[1] == 1, "This forward method is only valid for 1 time step"
        sre, sim = spec[..., : self.df_bins, :].split(1, -1)
        cre, cim = coefs.split(1, -1)
        outr = torch.sum(sre * cre - sim * cim, dim=2).squeeze(-1)
        outi = torch.sum(sre * cim + sim * cre, dim=2).squeeze(-1)
        spec_f = torch.stack((outr, outi), dim=-1)
        return assign_df(
            spec[:, :, self.df_order - self.df_lookahead - 1],
            spec_f.unsqueeze(1),
            self.df_bins,
            alpha,
        )

    def forward_real_hidden_state_loop(self, spec: Tensor, coefs: Tensor, alpha: Tensor) -> Tensor:
        # Version5: Designed for onnx export. `spec` buffer handling is done via a torch buffer.

        # spec (real) [B, 1, T, F', 2]
        # coefs (real) [B, T, O, F, 2]
        b, _, t, _, _ = spec.shape
        spec_out = torch.empty((b, 1, t, self.freq_bins, 2), device=spec.device)
        for t in range(spec.shape[2]):
            self.spec_buf = self.spec_buf.roll(-1, dims=2)
            self.spec_buf[:, :, -1] = spec[:, :, t]
            sre, sim = self.spec_buf[..., : self.df_bins, :].split(1, -1)
            cre, cim = coefs[:, t : t + 1].split(1, -1)
            outr = torch.sum(sre * cre - sim * cim, dim=2).squeeze(-1)
            outi = torch.sum(sre * cim + sim * cre, dim=2).squeeze(-1)
            spec_f = torch.stack((outr, outi), dim=-1)
            spec_out[:, :, t] = assign_df(
                self.spec_buf[:, :, self.df_order - self.df_lookahead - 1].unsqueeze(2),
                spec_f.unsqueeze(1),
                self.df_bins,
                alpha[:, t],
            ).squeeze(2)
        return spec_out


def assign_df(spec: Tensor, spec_f: Tensor, df_bins: int, alpha: Optional[Tensor]):
    spec_out = spec.clone()
    if alpha is not None:
        b = spec.shape[0]
        alpha = alpha.view(b, 1, -1, 1, 1)
        spec_out[..., :df_bins, :] = spec_f * alpha + spec[..., :df_bins, :] * (1 - alpha)
    else:
        spec_out[..., :df_bins, :] = spec_f
    return spec_out


def spec_pad(x: Tensor, window_size: int, lookahead: int, dim: int = 0) -> Tensor:
    pad = [0] * x.dim() * 2
    if dim >= 0:
        pad[(x.dim() - dim - 1) * 2] = window_size - lookahead - 1
        pad[(x.dim() - dim - 1) * 2 + 1] = lookahead
    else:
        pad[(-dim - 1) * 2] = window_size - lookahead - 1
        pad[(-dim - 1) * 2 + 1] = lookahead
    return F.pad(x, pad)


def as_strided(x: Tensor, window_size: int, lookahead: int, step: int = 1, dim: int = 0) -> Tensor:
    shape = list(x.shape)
    shape.insert(dim + 1, window_size)
    x = spec_pad(x, window_size, lookahead, dim=dim)
    # torch.fx workaround
    step = 1
    stride = [x.stride(0), x.stride(1), x.stride(2), x.stride(3)]
    stride.insert(dim, stride[dim] * step)
    return torch.as_strided(x, shape, stride)


class GroupedGRULayer(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    out_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    groups: Final[int]
    batch_first: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        kwargs = {
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.groups = groups
        self.batch_first = batch_first
        assert (self.hidden_size % groups) == 0, "Hidden size must be divisible by groups"
        self.layers = nn.ModuleList(
            (nn.GRU(self.input_size, self.hidden_size, **kwargs) for _ in range(groups))
        )

    def flatten_parameters(self):
        for layer in self.layers:
            layer.flatten_parameters()

    def get_h0(self, batch_size: int = 1, device: torch.device = torch.device("cpu")):
        return torch.zeros(
            self.groups * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def forward(self, input: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # input shape: [B, T, I] if batch_first else [T, B, I], B: batch_size, I: input_size
        # state shape: [G*D, B, H], where G: groups, D: num_directions, H: hidden_size

        if h0 is None:
            dim0, dim1 = input.shape[:2]
            bs = dim0 if self.batch_first else dim1
            h0 = self.get_h0(bs, device=input.device)
        outputs: List[Tensor] = []
        outstates: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            o, s = layer(
                input[..., i * self.input_size : (i + 1) * self.input_size],
                h0[i * self.num_directions : (i + 1) * self.num_directions].detach(),
            )
            outputs.append(o)
            outstates.append(s)
        output = torch.cat(outputs, dim=-1)
        h = torch.cat(outstates, dim=0)
        return output, h


class GroupedGRU(nn.Module):
    groups: Final[int]
    num_layers: Final[int]
    batch_first: Final[bool]
    hidden_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    shuffle: Final[bool]
    add_outputs: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        groups: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()
        kwargs = {
            "groups": groups,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        assert num_layers > 0
        self.input_size = input_size
        self.groups = groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size // groups
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        if groups == 1:
            shuffle = False  # Fully connected, no need to shuffle
        self.shuffle = shuffle
        self.add_outputs = add_outputs
        self.grus: List[GroupedGRULayer] = nn.ModuleList()  # type: ignore
        self.grus.append(GroupedGRULayer(input_size, hidden_size, **kwargs))
        for _ in range(1, num_layers):
            self.grus.append(GroupedGRULayer(hidden_size, hidden_size, **kwargs))
        self.flatten_parameters()

    def flatten_parameters(self):
        for gru in self.grus:
            gru.flatten_parameters()

    def get_h0(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tensor:
        return torch.zeros(
            (self.num_layers * self.groups * self.num_directions, batch_size, self.hidden_size),
            device=device,
        )

    def forward(self, input: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        dim0, dim1, _ = input.shape
        b = dim0 if self.batch_first else dim1
        if state is None:
            state = self.get_h0(b, input.device)
        output = torch.zeros(
            dim0, dim1, self.hidden_size * self.num_directions * self.groups, device=input.device
        )
        outstates = []
        h = self.groups * self.num_directions
        for i, gru in enumerate(self.grus):
            input, s = gru(input, state[i * h : (i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                input = (
                    input.view(dim0, dim1, -1, self.groups).transpose(2, 3).reshape(dim0, dim1, -1)
                )
            if self.add_outputs:
                output += input
            else:
                output = input
        outstate = torch.cat(outstates, dim=0)
        return output, outstate


class GroupedGRNNLayer(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    out_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    groups: Final[int]
    batch_first: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        gate_non_linearity: str = "sigmoid",
        update_non_linearity: str = "tanh",
        w_rank: Optional[int] = None,
        u_rank: Optional[int] = None,
        zeta_init: float = 1.0,
        nu_init: float = -4.0,
        lambda_init: float = 0.0,
        gamma_init: float = 0.999,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        if not bias:
            raise NotImplementedError("FastGRNN-based grouped layers require internal bias terms.")
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.groups = groups
        self.batch_first = batch_first
        self.gate_non_linearity = gate_non_linearity
        self.update_non_linearity = update_non_linearity
        self.w_rank = w_rank
        self.u_rank = u_rank
        self._init_grouped_fastgrnn_params(
            zeta_init=zeta_init,
            nu_init=nu_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
        )

    def _init_grouped_fastgrnn_params(
        self,
        zeta_init: float,
        nu_init: float,
        lambda_init: float,
        gamma_init: float,
    ) -> None:
        shape_gh = (self.num_directions, self.groups, self.hidden_size)
        shape_g1 = (self.num_directions, self.groups, 1)
        if self.w_rank is None:
            self.w = nn.Parameter(
                torch.randn(
                    self.num_directions, self.groups, self.input_size, self.hidden_size
                )
                * 0.1
            )
        else:
            self.w1 = nn.Parameter(
                torch.randn(self.num_directions, self.groups, self.input_size, self.w_rank) * 0.1
            )
            self.w2 = nn.Parameter(
                torch.randn(self.num_directions, self.groups, self.w_rank, self.hidden_size) * 0.1
            )

        if self.u_rank is None:
            self.u = nn.Parameter(
                torch.randn(
                    self.num_directions, self.groups, self.hidden_size, self.hidden_size
                )
                * 0.1
            )
        else:
            self.u1 = nn.Parameter(
                torch.randn(self.num_directions, self.groups, self.hidden_size, self.u_rank) * 0.1
            )
            self.u2 = nn.Parameter(
                torch.randn(self.num_directions, self.groups, self.u_rank, self.hidden_size) * 0.1
            )

        self.bias_gate = nn.Parameter(torch.ones(shape_gh))
        self.bias_update = nn.Parameter(torch.ones(shape_gh))
        self.zeta = nn.Parameter(torch.full(shape_g1, zeta_init, dtype=torch.float32))
        self.nu = nn.Parameter(torch.full(shape_g1, nu_init, dtype=torch.float32))
        self.lambd = nn.Parameter(torch.full(shape_g1, lambda_init, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full(shape_g1, gamma_init, dtype=torch.float32))

    def flatten_parameters(self):
        # FastGRNN is implemented from cells, so there are no packed RNN parameters to flatten.
        return

    def get_h0(self, batch_size: int = 1, device: torch.device = torch.device("cpu")):
        return torch.zeros(
            self.groups * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def _direction_index(self, forward: bool) -> int:
        if forward:
            return 0
        if not self.bidirectional:
            raise ValueError("Backward direction requested for unidirectional GroupedGRNNLayer.")
        return 1

    def _select_direction_params(self, forward: bool = True) -> dict[str, Tensor]:
        d = self._direction_index(forward)
        params = {
            "bias_gate": self.bias_gate[d].unsqueeze(0),
            "bias_update": self.bias_update[d].unsqueeze(0),
            "zeta": torch.sigmoid(self.zeta[d]).unsqueeze(0),
            "nu": torch.sigmoid(self.nu[d]).unsqueeze(0),
            "lambd": self.lambd[d].unsqueeze(0),
            "gamma": torch.clamp(self.gamma[d], 0.0, 1.0).unsqueeze(0),
        }
        if self.w_rank is None:
            params["w"] = self.w[d]
        else:
            params["w1"] = self.w1[d]
            params["w2"] = self.w2[d]
        if self.u_rank is None:
            params["u"] = self.u[d]
        else:
            params["u1"] = self.u1[d]
            params["u2"] = self.u2[d]
        return params

    def _project_input(self, x_t: Tensor, params: dict[str, Tensor]) -> Tensor:
        if self.w_rank is None:
            return torch.bmm(x_t.transpose(0, 1), params["w"]).transpose(0, 1)
        w_proj = torch.bmm(x_t.transpose(0, 1), params["w1"]).transpose(0, 1)
        return torch.bmm(w_proj.transpose(0, 1), params["w2"]).transpose(0, 1)

    def _project_hidden(self, h_prev: Tensor, params: dict[str, Tensor]) -> Tensor:
        if self.u_rank is None:
            return torch.bmm(h_prev.transpose(0, 1), params["u"]).transpose(0, 1)
        u_proj = torch.bmm(h_prev.transpose(0, 1), params["u1"]).transpose(0, 1)
        return torch.bmm(u_proj.transpose(0, 1), params["u2"]).transpose(0, 1)

    def _grouped_step(self, x_t: Tensor, h_prev: Tensor, params: dict[str, Tensor]) -> Tensor:
        # x_t: [B, G, I/G], h_prev: [B, G, H/G]
        preact = self._project_input(x_t, params) + self._project_hidden(h_prev, params)
        z = gen_non_linearity(preact + params["bias_gate"], self.gate_non_linearity)
        h_hat = gen_non_linearity(preact + params["bias_update"], self.update_non_linearity)

        zeta = params["zeta"]
        nu = params["nu"]
        h = z * h_prev + (zeta * (1 - z) + nu) * h_hat

        gamma = params["gamma"]
        lambd = params["lambd"]
        return gamma * h + (1 - gamma) * lambd

    def _run_direction(
        self,
        inputs: Tensor,
        h0: Tensor,
        params: dict[str, Tensor],
        reverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        # inputs: [B, T, G, I/G], h0: [B, G, H/G]
        b, t, _, _ = inputs.shape
        outputs = inputs.new_empty(b, t, self.groups, self.hidden_size)
        h = h0
        time_indices = range(t - 1, -1, -1) if reverse else range(t)
        for idx in time_indices:
            h = self._grouped_step(inputs[:, idx], h, params)
            outputs[:, idx] = h
        return outputs, h

    def forward(self, input: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # input shape: [B, T, I] if batch_first else [T, B, I], B: batch_size, I: input_size
        # state shape: [G*D, B, H], where G: groups, D: num_directions, H: hidden_size
        transposed = False
        if not self.batch_first:
            input = input.transpose(0, 1)
            transposed = True

        if h0 is None:
            bs = input.shape[0]
            h0 = input.new_zeros(self.groups * self.num_directions, bs, self.hidden_size)

        batch_size, seq_len, _ = input.shape
        grouped_input = input.reshape(batch_size, seq_len, self.groups, self.input_size)
        grouped_state = (
            h0.reshape(self.groups, self.num_directions, batch_size, self.hidden_size)
            .permute(1, 2, 0, 3)
            .contiguous()
        )

        fw_params = self._select_direction_params(forward=True)
        fw_out, h_fw = self._run_direction(grouped_input, grouped_state[0].detach(), fw_params)

        outputs = [fw_out]
        outstates = [h_fw]
        if self.bidirectional:
            bw_params = self._select_direction_params(forward=False)
            bw_out, h_bw = self._run_direction(
                grouped_input, grouped_state[1].detach(), bw_params, reverse=True
            )
            outputs.append(bw_out)
            outstates.append(h_bw)

        output = torch.cat(outputs, dim=-1).reshape(batch_size, seq_len, -1)
        h = torch.stack(outstates, dim=0).permute(2, 0, 1, 3).reshape(
            self.groups * self.num_directions, batch_size, self.hidden_size
        )
        if transposed:
            output = output.transpose(0, 1)
        return output, h

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_probe = f"{prefix}layers.0.cells_fwd.0.bias_gate"
        if legacy_probe in state_dict:
            self._upgrade_legacy_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _upgrade_legacy_state_dict(self, state_dict, prefix: str) -> None:
        def cell_prefix(group: int, direction: str) -> str:
            base = f"{prefix}layers.{group}."
            if direction == "fwd":
                return base + "cells_fwd.0."
            return base + "cells_bwd.0."

        def collect(name: str, direction: str) -> list[Tensor]:
            return [state_dict.pop(cell_prefix(g, direction) + name) for g in range(self.groups)]

        directions = ["fwd"] + (["bwd"] if self.bidirectional else [])
        bias_gate = self.bias_gate.detach().clone()
        bias_update = self.bias_update.detach().clone()
        zeta = self.zeta.detach().clone()
        nu = self.nu.detach().clone()
        lambd = self.lambd.detach().clone()
        gamma = self.gamma.detach().clone()

        if self.w_rank is None:
            w = self.w.detach().clone()
        else:
            w1 = self.w1.detach().clone()
            w2 = self.w2.detach().clone()
        if self.u_rank is None:
            u = self.u.detach().clone()
        else:
            u1 = self.u1.detach().clone()
            u2 = self.u2.detach().clone()

        for dir_idx, direction in enumerate(directions):
            bias_gate[dir_idx] = torch.stack(collect("bias_gate", direction), dim=0).squeeze(1)
            bias_update[dir_idx] = torch.stack(collect("bias_update", direction), dim=0).squeeze(1)
            zeta[dir_idx] = torch.stack(collect("zeta", direction), dim=0).view(self.groups, 1)
            nu[dir_idx] = torch.stack(collect("nu", direction), dim=0).view(self.groups, 1)
            lambd[dir_idx] = torch.stack(collect("lambd", direction), dim=0).view(self.groups, 1)
            gamma[dir_idx] = torch.stack(collect("gamma", direction), dim=0).view(self.groups, 1)
            if self.w_rank is None:
                w[dir_idx] = torch.stack(collect("w_matrix", direction), dim=0)
            else:
                w1[dir_idx] = torch.stack(collect("w_matrix_1", direction), dim=0)
                w2[dir_idx] = torch.stack(collect("w_matrix_2", direction), dim=0)
            if self.u_rank is None:
                u[dir_idx] = torch.stack(collect("u_matrix", direction), dim=0)
            else:
                u1[dir_idx] = torch.stack(collect("u_matrix_1", direction), dim=0)
                u2[dir_idx] = torch.stack(collect("u_matrix_2", direction), dim=0)

        state_dict[f"{prefix}bias_gate"] = bias_gate
        state_dict[f"{prefix}bias_update"] = bias_update
        state_dict[f"{prefix}zeta"] = zeta
        state_dict[f"{prefix}nu"] = nu
        state_dict[f"{prefix}lambd"] = lambd
        state_dict[f"{prefix}gamma"] = gamma
        if self.w_rank is None:
            state_dict[f"{prefix}w"] = w
        else:
            state_dict[f"{prefix}w1"] = w1
            state_dict[f"{prefix}w2"] = w2
        if self.u_rank is None:
            state_dict[f"{prefix}u"] = u
        else:
            state_dict[f"{prefix}u1"] = u1
            state_dict[f"{prefix}u2"] = u2


class GroupedGRNN(nn.Module):
    groups: Final[int]
    num_layers: Final[int]
    batch_first: Final[bool]
    hidden_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    shuffle: Final[bool]
    add_outputs: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        groups: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
        gate_non_linearity: str = "sigmoid",
        update_non_linearity: str = "tanh",
        w_rank: Optional[int] = None,
        u_rank: Optional[int] = None,
        zeta_init: float = 1.0,
        nu_init: float = -4.0,
        lambda_init: float = 0.0,
        gamma_init: float = 0.999,
    ):
        super().__init__()
        kwargs = {
            "groups": groups,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "gate_non_linearity": gate_non_linearity,
            "update_non_linearity": update_non_linearity,
            "w_rank": w_rank,
            "u_rank": u_rank,
            "zeta_init": zeta_init,
            "nu_init": nu_init,
            "lambda_init": lambda_init,
            "gamma_init": gamma_init,
        }
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        assert num_layers > 0
        self.input_size = input_size
        self.groups = groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size // groups
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.add_outputs = add_outputs
        self.grnns: List[GroupedGRNNLayer] = nn.ModuleList()  # type: ignore
        self.grnns.append(GroupedGRNNLayer(input_size, hidden_size, **kwargs))
        for _ in range(1, num_layers):
            self.grnns.append(GroupedGRNNLayer(hidden_size, hidden_size, **kwargs))
        self.flatten_parameters()

    def flatten_parameters(self):
        for grnn in self.grnns:
            grnn.flatten_parameters()

    def get_h0(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tensor:
        return torch.zeros(
            (self.num_layers * self.groups * self.num_directions, batch_size, self.hidden_size),
            device=device,
        )

    def forward(self, input: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        dim0, dim1, _ = input.shape
        b = dim0 if self.batch_first else dim1
        if state is None:
            state = self.get_h0(b, input.device)
        output = torch.zeros(
            dim0,
            dim1,
            self.hidden_size * self.num_directions * self.groups,
            device=input.device,
            dtype=input.dtype,
        )
        outstates = []
        h = self.groups * self.num_directions
        for i, grnn in enumerate(self.grnns):
            input, s = grnn(input, state[i * h : (i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                input = (
                    input.view(dim0, dim1, -1, self.groups).transpose(2, 3).reshape(dim0, dim1, -1)
                )
            if self.add_outputs:
                output += input
            else:
                output = input
        outstate = torch.cat(outstates, dim=0)
        return output, outstate


class SqueezedGRU(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
        linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups), linear_act_layer()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups), linear_act_layer()
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        input = self.linear_in(input)
        x, h = self.gru(input, h)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        x = self.linear_out(x)
        return x, h


class SqueezedGRU_S(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
        linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups), linear_act_layer()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups), linear_act_layer()
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, h = self.gru(x, h)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h


class GroupedLinearEinsum(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        # self.weight: Tensor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert input_size % groups == 0, f"Input size {input_size} not divisible by {groups}"
        assert hidden_size % groups == 0, f"Hidden size {hidden_size} not divisible by {groups}"
        self.ws = input_size // groups
        self.register_parameter(
            "weight",
            Parameter(
                torch.zeros(groups, input_size // groups, hidden_size // groups), requires_grad=True
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., I]
        b, t, _ = x.shape
        # new_shape = list(x.shape)[:-1] + [self.groups, self.ws]
        new_shape = (b, t, self.groups, self.ws)
        x = x.view(new_shape)
        # The better way, but not supported by torchscript
        # x = x.unflatten(-1, (self.groups, self.ws))  # [..., G, I/G]
        x = torch.einsum("btgi,gih->btgh", x, self.weight)  # [..., G, H/G]
        x = x.flatten(2, 3)  # [B, T, H]
        return x

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(input_size: {self.input_size}, hidden_size: {self.hidden_size}, groups: {self.groups})"


class GroupedLinear(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]
    shuffle: Final[bool]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1, shuffle: bool = True):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups)
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(x[..., i * self.input_size : (i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)
        if self.shuffle:
            orig_shape = output.shape
            output = (
                output.view(-1, self.hidden_size, self.groups).transpose(-1, -2).reshape(orig_shape)
            )
        return output


class LocalSnrTarget(nn.Module):
    def __init__(
        self, ws: int = 20, db: bool = True, ws_ns: Optional[int] = None, target_snr_range=None
    ):
        super().__init__()
        self.ws = self.calc_ws(ws)
        self.ws_ns = self.ws * 2 if ws_ns is None else self.calc_ws(ws_ns)
        self.db = db
        self.range = target_snr_range

    def calc_ws(self, ws_ms: int) -> int:
        # Calculates windows size in stft domain given a window size in ms
        p = stft_config()
        ws = ws_ms - p.fft_size / p.sr * 1000  # length ms of an fft_window
        ws = 1 + ws / (p.hop_length / p.sr * 1000)  # consider hop_length
        return max(int(round(ws)), 1)

    def forward(self, clean: Tensor, noise: Tensor, max_bin: Optional[int] = None) -> Tensor:
        # clean: [B, 1, T, F] (complex) or [B, 1, T, F, 2] (packed real/imag)
        # out: [B, T']
        if max_bin is not None:
            if clean.dim() == 5:
                clean = as_complex(clean[..., :max_bin, :])
                noise = as_complex(noise[..., :max_bin, :])
            else:
                clean = as_complex(clean[..., :max_bin])
                noise = as_complex(noise[..., :max_bin])
        return (
            local_snr(clean, noise, window_size=self.ws, db=self.db, window_size_ns=self.ws_ns)[0]
            .clamp(self.range[0], self.range[1])
            .squeeze(1)
        )


def _local_energy(x: Tensor, ws: int, device: torch.device) -> Tensor:
    if (ws % 2) == 0:
        ws += 1
    ws_half = ws // 2
    x = F.pad(x.pow(2).sum(-1).sum(-1), (ws_half, ws_half, 0, 0))
    w = torch.hann_window(ws, device=device, dtype=x.dtype)
    x = x.unfold(-1, size=ws, step=1) * w
    return torch.sum(x, dim=-1).div(ws)


def local_snr(
    clean: Tensor,
    noise: Tensor,
    window_size: int,
    db: bool = False,
    window_size_ns: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    # clean shape: [B, C, T, F]
    clean = as_real(clean)
    noise = as_real(noise)
    assert clean.dim() == 5

    E_speech = _local_energy(clean, window_size, clean.device)
    window_size_ns = window_size if window_size_ns is None else window_size_ns
    E_noise = _local_energy(noise, window_size_ns, clean.device)

    snr = E_speech / E_noise.clamp_min(eps)
    if db:
        snr = snr.clamp_min(eps).log10().mul(10)
    return snr, E_speech, E_noise
