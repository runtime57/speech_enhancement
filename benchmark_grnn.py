"""Compare full DFNet models with warmed-up CPU or synchronized GPU timings.

python benchmark_grnn.py --output benchmarks/grnn_cuda.json
"""

import argparse
import json
import os
import platform
import statistics
import time
import types
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.model.modules import GroupedGRNNLayer
from src.utils.config_utils import setup_config
from src.utils.precision import model_autocast, set_inference_precision


ROOT = Path(__file__).resolve().parent


def original_direction(self, inputs, h0, params, reverse=False):
    """The previous implementation, retained only as a benchmark reference."""
    b, t, _, _ = inputs.shape
    outputs = inputs.new_empty(b, t, self.groups, self.hidden_size)
    h = h0
    for idx in range(t - 1, -1, -1) if reverse else range(t):
        h = self._grouped_step(inputs[:, idx], h, params)
        outputs[:, idx] = h
    return outputs, h


def make_model(name, device):
    original = name.endswith("_original")
    config_name = name.removesuffix("_original")
    torch.manual_seed(42)
    config = OmegaConf.load(ROOT / "src/configs/model" / f"{config_name}.yaml")
    model = instantiate(config).to(device)
    if original:
        for layer in model.modules():
            if isinstance(layer, GroupedGRNNLayer):
                layer._run_direction = types.MethodType(original_direction, layer)
    return model


def measure(model, inputs, training, warmup, iterations, device, precision):
    model.train(training)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step():
        if training:
            model.zero_grad(set_to_none=True)
            with model_autocast(device, enabled=precision == "amp"):
                output = model(**inputs)
            output = tuple(y.float() for y in output)
            sum(y.square().mean() for y in output).backward()
        else:
            with torch.no_grad():
                with model_autocast(device, enabled=precision == "amp"):
                    model(**inputs)

    for _ in range(warmup):
        step()
    samples = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        step()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(
        median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[200, 400])
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "deepfilternet", "deepfilternetgrnn_original", "deepfilternetgrnn", "grnn_emb_128"
        ],
    )
    parser.add_argument("--precision", choices=["fp32", "amp", "fp16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_sizes + args.seq_lengths + [args.warmup, args.iters]) <= 0:
        parser.error("Batch sizes, sequence lengths, warmup and iterations must be positive.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    config = OmegaConf.load(ROOT / "src/configs/grnn.yaml")
    setup_config(config)
    p = config.stft
    hardware = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                hardware = line.split(":", 1)[1].strip()
                break
    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(device)
    report = dict(
        torch=torch.__version__,
        device=str(device),
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        cpu_affinity=(
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        hardware=hardware,
        mkldnn_enabled=torch.backends.mkldnn.enabled,
        precision=args.precision,
        warmup=args.warmup,
        iterations=args.iters,
        scope="Full model only. Synthetic loss; no optimizer, audio preprocessing or metrics.",
        results=[],
    )
    for batch in args.batch_sizes:
        for frames in args.seq_lengths:
            torch.manual_seed(1)
            inputs = dict(
                spec=torch.randn(batch, 1, frames, p.fft_size // 2 + 1, 2, device=device),
                feat_erb=torch.randn(batch, 1, frames, p.nb_erb, device=device),
                feat_spec=torch.randn(batch, 1, frames, p.nb_df, 2, device=device),
            )
            for name in args.models:
                model = make_model(name, device)
                set_inference_precision(model, args.precision)
                model_inputs = dict(inputs)
                if args.precision == "fp16":
                    for key in ("feat_erb", "feat_spec"):
                        model_inputs[key] = model_inputs[key].half()
                forward = measure(
                    model, model_inputs, False, args.warmup, args.iters, device, args.precision
                )
                training = None
                if args.precision != "fp16":
                    training = measure(
                        model, model_inputs, True, args.warmup, args.iters, device, args.precision
                    )
                result = dict(
                    model=name,
                    batch_size=batch,
                    frames=frames,
                    parameters=sum(p.numel() for p in model.parameters()),
                    forward=forward,
                    forward_backward=training,
                )
                report["results"].append(result)
                print(json.dumps(result), flush=True)
                del model
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
