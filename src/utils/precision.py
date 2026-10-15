"""Precision choices shared by training and waveform inference."""

import torch


def amp_settings(config, device):
    name = str(config.get("amp_dtype", "float16"))
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtypes:
        raise ValueError(f"Unsupported AMP dtype: {name}. Use float16 or bfloat16.")
    enabled = bool(config.get("amp", False)) and torch.device(device).type == "cuda"
    return enabled, dtypes[name]


def model_autocast(device, enabled=False, dtype=torch.float16):
    device_type = torch.device(device).type
    return torch.autocast(
        device_type=device_type, dtype=dtype, enabled=enabled and device_type == "cuda"
    )


def set_inference_precision(model, precision):
    if precision not in ("fp32", "fp16", "amp"):
        raise ValueError(f"Unsupported inference precision: {precision}. Use fp32, fp16 or amp.")
    return model.half() if precision == "fp16" else model.float()
