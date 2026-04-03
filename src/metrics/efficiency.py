import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class KeywordModelWrapper(nn.Module):
    """
    Wrap a model that expects **kwargs into a model that accepts
    positional input (one dict) for fvcore FLOP counting.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs_dict):
        return self.model(**inputs_dict)


@dataclass
class ModelMetrics:
    n_parameters: int
    model_size_bytes: Optional[int] = None
    model_size_mb: Optional[float] = None
    flops: Optional[float] = None
    giga_flops: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    avg_inference_time_sec: Optional[float] = None
    device: Optional[str] = None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_size(path: Optional[str]) -> Optional[int]:
    if path is None or not os.path.exists(path):
        return None
    size_bytes = os.path.getsize(path)
    return size_bytes


def measure_flops(
    model: nn.Module,
    inputs: dict,
) -> Optional[float]:
    """
    Measure FLOPs using fvcore
    Returns: flops (float)
    """
    model.eval()
    try:
        from fvcore.nn import FlopCountAnalysis

        model_wrapped = KeywordModelWrapper(model)
        flops = FlopCountAnalysis(model_wrapped, (inputs,))
        return flops.total()
    except Exception as e:
        print(f"exception during measure_flops: {type(e).__name__}: {e}")
        pass

    return None

def measure_macs(
    model: nn.Module,
    inputs: dict,
) -> Optional[float]:
    """
    Measure MACs using thop.
    Returns: macs (float)
    """
    try:
        from copy import deepcopy
        from thop import profile

        model_copy = deepcopy(model)
        model_copy.eval()

        model_wrapped = KeywordModelWrapper(model_copy)
        macs, _ = profile(
            model_wrapped,
            inputs=(inputs,),
            verbose=False,
        )

        del model_copy
        return float(macs)
    except Exception as e:
        print(f"exception during measure_macs: {type(e).__name__}: {e}")
        pass
    return None

def measure_inference_time(
    model: nn.Module,
    inputs: dict,
    device: str = "cuda",
    warmup: int = 5,
    iters: int = 20,
) -> float | None:
    """
    Measure average inference time (seconds) over `iters` runs,
    after a warmup phase.
    Supports only "cuda" device
    """
    cuda_flg = device.startswith("cuda")

    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
            if cuda_flg:
                torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(**inputs)
            if cuda_flg:
                torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters


def measure_peak_memory(
    model: nn.Module,
    inputs: dict,
    device: str = "cuda",
) -> Optional[int]:
    """
    Measure peak CUDA memory for one forward pass in bytes.
    CPU path is a no-op (returns None).
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None

    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        _ = model(**inputs)
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes