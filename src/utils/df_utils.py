import collections
import math
import os
import random
import subprocess
from socket import gethostname
from typing import Any, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torch.types import Number

from src.utils.config_utils import stft_config


def as_complex(x: Tensor):
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dimension need to be of length 2 (re + im), but got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


def as_real(x: Tensor):
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


class angle_re_im(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, re: Tensor, im: Tensor):
        ctx.save_for_backward(re, im)
        return torch.atan2(im, re)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tuple[Tensor, Tensor]:
        re, im = ctx.saved_tensors
        grad_inv = grad / (re.square() + im.square()).clamp_min_(1e-12)
        return -im * grad_inv, re * grad_inv


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-12)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))


def check_finite_module(obj, name="Module", _raise=True) -> Set[str]:
    out: Set[str] = set()
    if isinstance(obj, torch.nn.Module):
        for name, child in obj.named_children():
            out = out | check_finite_module(child, name)
        for name, param in obj.named_parameters():
            out = out | check_finite_module(param, name)
        for name, buf in obj.named_buffers():
            out = out | check_finite_module(buf, name)
    if _raise and len(out) > 0:
        raise ValueError(f"{name} not finite during checkpoint writing including: {out}")
    return out


def make_np(x: Union[Tensor, np.ndarray, Number]) -> np.ndarray:
    """Transforms Tensor to numpy.
    Args:
      x: An instance of torch tensor or caffe blob name

    Returns:
        numpy.array: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    raise NotImplementedError(
        "Got {}, but numpy array, scalar, or torch tensor are expected.".format(type(x))
    )


def get_norm_alpha(log: bool = True) -> float:
    p = stft_config()
    a_ = _calculate_norm_alpha(sr=p.sr, hop_length=p.hop_length, tau=p.norm_tau)
    precision = 3
    a = 1.0
    while a >= 1.0:
        a = round(a_, precision)
        precision += 1
    return a


def exp_unit_norm(
    x: Tensor,
    lengths: Optional[Tensor] = None,  # [B] в фреймах, если есть паддинг
) -> Tuple[Tensor, Tensor]:
    assert x.dim() == 5 and x.size(-1) == 2
    B, C, T, F, _ = x.shape
    device, dtype = x.device, x.dtype

    alpha = get_norm_alpha()
    eps = 1e-8
    state = torch.ones((B, C, F), device=device, dtype=dtype)
    
    y = x.clone()

    if lengths is not None:
        valid_t = (torch.arange(T, device=device)[None, :] < lengths[:, None])  # [B,T]
    else:
        valid_t = None

    for t in range(T):
        re = y[:, :, t, :, 0]                # [B,C,F]
        im = y[:, :, t, :, 1]
        mag2 = re * re + im * im             # |X|^2

        if valid_t is None:
            state = alpha * state + (1 - alpha) * mag2
        else:
            v = valid_t[:, t].view(B, 1, 1)  # [B,1,1]
            state = torch.where(v, alpha * state + (1 - alpha) * mag2, state)

        inv = torch.rsqrt(state + eps)
        y[:, :, t, :, 0] = re * inv
        y[:, :, t, :, 1] = im * inv

    return y, state


def _calculate_norm_alpha(sr: int, hop_length: int, tau: float):
    """Exponential decay factor alpha for a given tau (decay window size [s])."""
    dt = hop_length / sr
    return math.exp(-dt / tau)


def check_manual_seed(seed: Optional[int] = None):
    """If manual seed is not specified, choose a random one and communicate it to the user."""
    seed = seed or random.randint(1, 10000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_git_root():
    """Returns the top level git directory or None if not called within the git repository."""
    try:
        git_local_dir = os.path.dirname(os.path.abspath(__file__))
        args = ["git", "-C", git_local_dir, "rev-parse", "--show-toplevel"]
        return subprocess.check_output(args).strip().decode()
    except subprocess.CalledProcessError:
        return None


def get_commit_hash():
    """Returns the current git commit."""
    try:
        git_dir = get_git_root()
        if git_dir is None:
            return None
        args = ["git", "-C", git_dir, "rev-parse", "--short", "--verify", "HEAD"]
        return subprocess.check_output(args).strip().decode()
    except subprocess.CalledProcessError:
        # probably not in git repo
        return None


def get_host() -> str:
    return gethostname()


def get_branch_name():
    try:
        git_dir = os.path.dirname(os.path.abspath(__file__))
        args = ["git", "-C", git_dir, "rev-parse", "--abbrev-ref", "HEAD"]
        branch = subprocess.check_output(args).strip().decode()
    except subprocess.CalledProcessError:
        # probably not in git repo
        branch = None
    return branch


# from pytorch/ignite:
def apply_to_tensor(input_, func):
    """Apply a function on a tensor or mapping, or sequence of tensors."""
    if isinstance(input_, torch.nn.Module):
        return [apply_to_tensor(c, func) for c in input_.children()]
    elif isinstance(input_, torch.nn.Parameter):
        return func(input_.data)
    elif isinstance(input_, Tensor):
        return func(input_)
    elif isinstance(input_, (str, bytes)):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: apply_to_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Iterable):
        return [apply_to_tensor(sample, func) for sample in input_]
    elif input_ is None:
        return input_
    else:
        return input_


def detach_hidden(hidden: Any) -> Any:
    """Cut backpropagation graph.
    Auxillary function to cut the backpropagation graph by detaching the hidden
    vector.
    """
    return apply_to_tensor(hidden, Tensor.detach)


def download_file(url: str, download_dir: str, extract: bool = False):
    import shutil
    import zipfile

    import requests

    local_filename = url.split("/")[-1]
    local_filename = os.path.join(download_dir, local_filename)
    with requests.get(url, stream=True) as r:
        if r.status_code >= 400:
            print(f"Error downloading file {url} ({r.status_code}): {r.reason}")
            exit(1)
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    if extract:
        if os.path.splitext(local_filename)[1] != ".zip":
            print("File not supported. Cannot extract.")
            exit(1)

        with zipfile.ZipFile(local_filename) as zf:
            zf.extractall(download_dir)
        os.remove(local_filename)

    return local_filename


def get_cache_dir():
    try:
        from appdirs import user_cache_dir

        return user_cache_dir("DeepFilterNet")
    except ImportError:
        import sys

        if sys.platform == "linux":
            return os.path.expanduser("~/.cache/DeepFilterNet/")
        else:
            raise ValueError(
                "Could not get cache dir. Please install `appdirs` via `pip install appdirs`"
            )
