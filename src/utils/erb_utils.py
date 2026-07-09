# erb_utils.py
from __future__ import annotations

from typing import Optional, Tuple, Union

import math
import numpy as np

import torch

from src.utils.config_utils import stft_config

def _freq2erb(freq_hz: np.float32) -> np.float32:
    """Mirrors `freq2erb()` in `libDF/src/lib.rs` (uses f32 math in Rust)."""
    return np.float32(9.265) * np.log1p(freq_hz / (np.float32(24.7) * np.float32(9.265)))


def _erb2freq(n_erb: np.float32) -> np.float32:
    """Mirrors `erb2freq()` in `libDF/src/lib.rs` (uses f32 math in Rust)."""
    return (
        np.float32(24.7)
        * np.float32(9.265)
        * (np.exp(n_erb / np.float32(9.265)) - np.float32(1.0))
    )


def _round_half_away_from_zero_pos_f32(x: np.float32) -> int:
    """Round-half-away-from-zero for x>=0 (Rust f32::round semantics for positive numbers).

    Python/NumPy round ties-to-even, which does not match Rust's `round()` for halfway cases.
    """
    return int(np.floor(x + np.float32(0.5)))


def erb_widths(
    sr: int,
    fft_size: int,
    nb_bands: int = 32,
    min_nb_freqs: int = 1,
    *,
    dtype: Optional[np.dtype] = np.uint64,
) -> np.ndarray:
    """Compute ERB band widths (number of FFT bins per ERB band).

    This implements the same algorithm as `erb_fb()` in `libDF/src/lib.rs` and is the source
    of what `df_state.erb_widths()` returns in Python.

    Args:
        sr: Sample rate in Hz.
        fft_size: FFT size in samples. The number of frequency bins is `fft_size // 2 + 1`.
        nb_bands: Number of ERB bands.
        min_nb_freqs: Minimum number of FFT bins per ERB band (enforced).
        dtype: Optional numpy dtype for the returned array. Defaults to `np.uint64` (like usize).

    Returns:
        widths: 1D numpy array of length `nb_bands`. Sum equals `fft_size // 2 + 1`.
    """
    if sr <= 0:
        raise ValueError(f"`sr` must be > 0, got {sr}")
    if fft_size <= 0:
        raise ValueError(f"`fft_size` must be > 0, got {fft_size}")
    if nb_bands <= 0:
        raise ValueError(f"`nb_bands` must be > 0, got {nb_bands}")
    if min_nb_freqs <= 0:
        raise ValueError(f"`min_nb_freqs` must be > 0, got {min_nb_freqs}")

    nyq_freq = sr // 2  # matches Rust integer division
    freq_width = np.float32(sr) / np.float32(fft_size)
    erb_low = _freq2erb(np.float32(0.0))
    erb_high = _freq2erb(np.float32(nyq_freq))
    step = (erb_high - erb_low) / np.float32(nb_bands)

    widths = np.zeros(nb_bands, dtype=np.int64)
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = _erb2freq(erb_low + np.float32(i) * step)
        fb = _round_half_away_from_zero_pos_f32(f / freq_width)
        nb_freqs = int(fb) - int(prev_freq) - int(freq_over)
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        widths[i - 1] = nb_freqs
        prev_freq = fb

    # since we have `fft_size/2 + 1` frequency bins (incl. Nyquist)
    widths[-1] += 1

    expected = fft_size // 2 + 1
    too_large = int(widths.sum()) - int(expected)
    if too_large > 0:
        widths[-1] -= too_large

    if int(widths.sum()) != int(expected):
        raise RuntimeError(
            f"erb_widths() invariant violated: sum(widths)={int(widths.sum())} != {expected}"
        )

    if dtype is None:
        return widths
    return widths.astype(dtype, copy=False)


# -----------------------------
# Rectangular ERB filterbank
# -----------------------------

def erb_fb_matrix_np(
    sr: int,
    fft_size: int,
    nb_bands: int,
    min_nb_freqs: int = 1,
    *,
    normalize: str = "mean",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build rectangular ERB filterbank matrix.

    Returns matrix FB with shape [nb_bands, F_full], where F_full = fft_size//2 + 1.

    normalize:
        - "mean": each band averages its bins (weights sum to 1 inside band)
        - "sum":  each band sums its bins (weights are 1 inside band)
    """
    widths = erb_widths(sr, fft_size, nb_bands=nb_bands, min_nb_freqs=min_nb_freqs, dtype=None).astype(int)
    F_full = fft_size // 2 + 1
    if int(widths.sum()) != int(F_full):
        raise RuntimeError("erb_widths returned wrong total width")

    starts = np.concatenate([[0], np.cumsum(widths[:-1])])
    ends = starts + widths

    fb = np.zeros((nb_bands, F_full), dtype=dtype)
    for b, (s, e) in enumerate(zip(starts, ends)):
        if normalize == "mean":
            fb[b, s:e] = np.float32(1.0) / np.float32(e - s)
        elif normalize == "sum":
            fb[b, s:e] = np.float32(1.0)
        else:
            raise ValueError(f"normalize must be 'mean' or 'sum', got {normalize!r}")
    return fb


def erb_fb_apply_np(
    x: np.ndarray,
    fb: np.ndarray,
) -> np.ndarray:
    """Apply ERB FB to a log-power spectrogram.

    x shape: [T, F_full] or [B, T, F_full]
    fb shape: [Fe, F_full]
    returns: [T, Fe] or [B, T, Fe]
    """
    if x.ndim == 2:
        # [T,F] @ [F,Fe] -> [T,Fe]
        return x @ fb.T
    if x.ndim == 3:
        # [B,T,F] @ [F,Fe] -> [B,T,Fe]
        return x @ fb.T
    raise ValueError(f"Expected x as [T,F] or [B,T,F], got {x.shape}")


# -----------------------------
# Exponential mean normalization (decay in seconds)
# -----------------------------

def exp_mean_norm_np(
    x: np.ndarray,        # [T,F] or [B,T,F]
    hop_length: int,
    sr: int,
    decay_sec: float = 1.0,
) -> np.ndarray:
    """Exponential mean normalization: y_t = x_t - EMA(x_t).

    EMA update: mu_t = alpha*mu_{t-1} + (1-alpha)*x_t
    alpha = exp(-(hop/sr)/decay_sec)
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected x as [T,F] or [B,T,F], got {x.shape}")

    dt = hop_length / float(sr)
    alpha = float(math.exp(-dt / float(decay_sec)))

    if x.ndim == 2:
        T, F = x.shape
        mu = np.zeros((F,), dtype=x.dtype)
        y = np.empty_like(x)
        for t in range(T):
            mu = alpha * mu + (1.0 - alpha) * x[t]
            y[t] = x[t] - mu
        return y

    # [B,T,F]
    B, T, F = x.shape
    mu = np.zeros((B, F), dtype=x.dtype)
    y = np.empty_like(x)
    for t in range(T):
        mu = alpha * mu + (1.0 - alpha) * x[:, t, :]
        y[:, t, :] = x[:, t, :] - mu
    return y

def make_erb_fb_matrix_torch(
    sr: int,
    fft_size: int,
    nb_bands: int,
    min_nb_freqs: int = 1,
    *,
    normalize: str = "mean",
    device=None,
    dtype=None,
):
    """Torch version of rectangular ERB FB matrix: [Fe, F_full]."""
    fb_np = erb_fb_matrix_np(sr, fft_size, nb_bands, min_nb_freqs, normalize=normalize, dtype=np.float32)
    fb = torch.tensor(fb_np, device=device, dtype=dtype if dtype is not None else torch.float32)
    return fb


def exp_mean_norm_torch(
    x,  # torch.Tensor [B,T,F]
    hop_length: int,
    sr: int,
    decay_sec: float = 1.0,
    lengths: Optional["torch.Tensor"] = None,  # [B] in frames
):
    """Torch exponential mean normalization: y = x - EMA(x)."""
    if x.dim() != 3:
        raise ValueError(f"Expected x as [B,T,F], got {tuple(x.shape)}")

    B, T, F = x.shape
    dt = hop_length / float(sr)
    alpha = float(math.exp(-dt / float(decay_sec)))

    mu = torch.zeros((B, F), device=x.device, dtype=x.dtype)
    y = torch.empty_like(x)

    if lengths is not None:
        valid_t = (torch.arange(T, device=x.device)[None, :] < lengths[:, None])  # [B,T]
    else:
        valid_t = None

    for t in range(T):
        xt = x[:, t, :]  # [B,F]
        if valid_t is None:
            mu = alpha * mu + (1.0 - alpha) * xt
        else:
            v = valid_t[:, t].unsqueeze(1)  # [B,1]
            mu_new = alpha * mu + (1.0 - alpha) * xt
            mu = torch.where(v, mu_new, mu)
        y[:, t, :] = xt - mu
    return y


def compute_erb_feats_from_stft(
    noisy_stft,
    *,
    decay_sec: float = 1.0,
    fb_normalize: str = "mean",         # "mean" or "sum"
    eps: float = 1e-10,
    lengths_frames: Optional["torch.Tensor"] = None,  # [B] in frames (optional)
):
    """Compute ERB features as in the paper: log-power -> exp mean norm (decay=1s) -> rectangular ERB FB.

    Returns:
        feat_erb: [B, 1, T, nb_erb]
    """
    p = stft_config()
    sr = p.sr
    fft_size = p.fft_size
    hop_length = p.hop_length
    nb_erb = p.nb_erb
    min_nb_freqs = p.min_nb_freqs
    if noisy_stft.dim() != 4 or noisy_stft.size(-1) != 2:
        raise ValueError(f"Expected noisy_stft as [B,F,T,2], got {tuple(noisy_stft.shape)}")

    # power: [B,F,T]
    re = noisy_stft[..., 0]
    im = noisy_stft[..., 1]
    power = re * re + im * im

    # log-power: [B,F,T]
    logp = torch.log(power + eps)
    logp = logp.permute(0, 2, 1).contiguous()

    # exp mean norm (decay=1s by default)
    logp_norm = exp_mean_norm_torch(
        logp, hop_length=hop_length, sr=sr, decay_sec=decay_sec, lengths=lengths_frames
    )

    # ERB FB
    fb = make_erb_fb_matrix_torch(
        sr, fft_size, nb_erb, min_nb_freqs,
        normalize=fb_normalize,
        device=logp_norm.device,
        dtype=logp_norm.dtype,
    )  # [Fe,F]

    # apply: [B,T,F] @ [F,Fe] = [B,T,Fe]
    erb_feats = torch.matmul(logp_norm, fb.t())

    # encoder expects [B,1,T,Fe]
    return erb_feats.unsqueeze(1)
