"""Sequence execution for grouped FastGRNN, with an optional CUDA backend."""

from functools import lru_cache

import torch

from src.model.comfifastgrnn import gen_non_linearity


@lru_cache(maxsize=None)
def _cuda_runner(device_index):
    if torch.version.hip is not None or torch.cuda.get_device_capability(device_index)[0] < 8:
        return None
    try:
        from src.model.fastgrnn_cuda import persistent_grnn
    except ImportError:
        return None
    return persistent_grnn


def run_grouped_fastgrnn(
    inputs,
    h0,
    params,
    gate_non_linearity,
    update_non_linearity,
    reverse=False,
    backend="auto",
):
    """Run one direction, preserving [B, T, G, H] output and [B, G, H] state.

    Project all inputs together. unbind/stack avoids a separate full-sequence
    gradient allocation at each timestep, as happens with indexed reads/writes.
    CUDA float32/float16/bfloat16 with default activations and at most 128 hidden
    units per group uses a persistent kernel. Other cases use PyTorch.
    """
    b, t, g, i = inputs.shape
    h = h0.shape[-1]
    if t == 0:
        return inputs.new_empty(b, 0, g, h), h0
    x = inputs.permute(2, 0, 1, 3).reshape(g, b * t, i)
    if "w" in params:
        projected = torch.bmm(x, params["w"])
    else:
        projected = torch.bmm(torch.bmm(x, params["w1"]), params["w2"])
    projected = projected.reshape(g, b, t, h).permute(2, 0, 1, 3).contiguous()
    state = h0.transpose(0, 1).contiguous()

    runner = None
    if (
        backend != "torch"
        and projected.is_cuda
        and projected.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and gate_non_linearity == "sigmoid"
        and update_non_linearity == "tanh"
        and 0 < h <= 128
    ):
        runner = _cuda_runner(projected.device.index)
    if runner is not None:
        # Keep the recurrence and its gradients in float32 under AMP/half models.
        # Input/output projections can still use Tensor Cores in the lower dtype.
        output_dtype = projected.dtype
        float_params = {name: value.float() for name, value in params.items()}
        with torch.autocast(device_type="cuda", enabled=False):
            output = runner(projected.float(), state.float(), float_params, reverse)
        output = output.to(output_dtype)
        return output.permute(2, 0, 1, 3), output[0 if reverse else -1].transpose(0, 1)
    if backend == "triton":
        raise RuntimeError(
            "Triton GRNN requires CUDA float32/float16/bfloat16 on NVIDIA SM80+, sigmoid/tanh, "
            "at most 128 hidden units per group, and an installed Triton package."
        )

    steps = projected.unbind(0)
    if reverse:
        steps = steps[::-1]
    bias_gate = params["bias_gate"].transpose(0, 1)
    bias_update = params["bias_update"].transpose(0, 1)
    zeta = params["zeta"].transpose(0, 1)
    nu = params["nu"].transpose(0, 1)
    gamma = params["gamma"].transpose(0, 1)
    drift = (1 - gamma) * params["lambd"].transpose(0, 1)
    outputs = []
    for wx in steps:
        if "u" in params:
            recurrent = torch.bmm(state, params["u"])
        else:
            recurrent = torch.bmm(torch.bmm(state, params["u1"]), params["u2"])
        preact = wx + recurrent
        z = gen_non_linearity(preact + bias_gate, gate_non_linearity)
        candidate = gen_non_linearity(preact + bias_update, update_non_linearity)
        state = gamma * (z * state + (zeta * (1 - z) + nu) * candidate) + drift
        outputs.append(state)
    if reverse:
        outputs.reverse()
    return torch.stack(outputs).permute(2, 0, 1, 3), state.transpose(0, 1)
