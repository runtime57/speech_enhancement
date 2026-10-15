"""Reproducible CPU waveform-to-waveform and network-only speech enhancement timings."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

from src.model.modules import GroupedGRNNLayer, GroupedLinearEinsum
from src.utils.config_utils import setup_config
from src.utils.enhancement import prepare_enhancement_inputs, synthesize_enhancement

ROOT = Path(__file__).resolve().parent
MODELS = [
    ("dfnet", "DFNet (grouped GRU)", "deepfilternet", "dfnet_full_20261014"),
    ("grnn", "FastGRNN-512", "deepfilternetgrnn", "grnn_full_20261014"),
    ("grnn_emb_256", "FastGRNN-256", "grnn_emb_256", "grnn_emb_256_full_20261014"),
    ("grnn_emb_128", "FastGRNN-128", "grnn_emb_128", "grnn_emb_128_full_50e_eval5_20261015"),
    ("grnn_rebalanced", "Rebalanced-1", "rebalancedgrnn", "grnn_rebalanced_full_50e_eval5_20261015"),
    ("grnn_rebalanced2", "Rebalanced-2", "rebalanced2grnn", "grnn_rebalanced2_full_50e_eval5_20261015"),
    ("grnn_rebalanced3", "Rebalanced-3", "rebalanced3grnn", "grnn_rebalanced3_full_50e_eval5_20261015"),
    *[(name, name.upper() if name != "conv" else "Conv", config, f"{name}_full_50e_eval5_20261015")
      for name, config in [("dws", "dws"), ("mb", "mm"), ("conv", "conv"),
                           ("xdws", "xdws"), ("xmb", "xmm"), ("xconv", "xconv")]],
]


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checkpoint_for(entry):
    path = ROOT / "saved" / entry[3] / "checkpoint-epoch50.pth"
    return path if path.exists() else None


def stats(samples_ms, duration):
    values = np.asarray(samples_ms, dtype=np.float64)
    rng = np.random.default_rng(20261015)
    medians = np.median(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    median = float(np.median(values))
    return {"median_ms": median, "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(values.mean()), "std_ms": float(values.std(ddof=1)),
            "median_ci95_ms": np.percentile(medians, [2.5, 97.5]).tolist(),
            "rtf_median": median / (duration * 1000),
            "rtf_p95": float(np.percentile(values, 95)) / (duration * 1000),
            "audio_seconds_per_second": duration * 1000 / median,
            "samples_ms": values.tolist()}


def count_neural_macs(model, inputs):
    """Dense Conv/Linear/recurrent MACs; excludes FFT, elementwise ops and DF filtering."""
    total, hooks = [0], []

    def count(module, args, output):
        x = args[0]
        if isinstance(module, torch.nn.Conv2d):
            total[0] += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, torch.nn.ConvTranspose2d):
            total[0] += x.numel() * (module.out_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, torch.nn.Linear):
            total[0] += output.numel() * module.in_features
        elif isinstance(module, GroupedLinearEinsum):
            total[0] += (x.numel() // module.input_size) * module.input_size * module.hidden_size // module.groups
        elif isinstance(module, torch.nn.GRU):
            batch, steps = (x.shape[:2] if module.batch_first else (x.shape[1], x.shape[0]))
            directions = 2 if module.bidirectional else 1
            for layer in range(module.num_layers):
                features = module.input_size if layer == 0 else module.hidden_size * directions
                total[0] += batch * steps * directions * 3 * module.hidden_size * (features + module.hidden_size)
        elif isinstance(module, GroupedGRNNLayer):
            batch, steps = (x.shape[:2] if module.batch_first else (x.shape[1], x.shape[0]))
            i, h = module.input_size, module.hidden_size
            w = i * h if module.w_rank is None else module.w_rank * (i + h)
            u = h * h if module.u_rank is None else 2 * h * module.u_rank
            total[0] += batch * steps * module.groups * module.num_directions * (w + u)

    counted = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear,
               torch.nn.GRU, GroupedGRNNLayer, GroupedLinearEinsum)
    for module in model.modules():
        if isinstance(module, counted):
            hooks.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return total[0]


def clips(duration, count, sample_rate):
    index = json.loads((ROOT / "data/vctk/test/index.json").read_text())
    length = round(duration * sample_rate)
    selected, sources = [], []
    for entry in index:
        path = Path(entry["element_path"])
        with safe_open(path, framework="pt") as source:
            size = source.get_slice("noisy").get_shape()[0]
        if size < length:
            continue
        start = (size - length) // 2
        noisy = load_file(path)["noisy"][start:start + length].clone().unsqueeze(0)
        selected.append(noisy)
        sources.append({"recording": Path(entry["original_noisy"]).name,
                        "sample_offset": start, "samples": length,
                        "waveform_sha256": hashlib.sha256(noisy.numpy().tobytes()).hexdigest()})
        if len(selected) == min(count, 15):
            break
    if not selected:
        raise ValueError(f"No official test clips are long enough for {duration}s")
    # Two passes over 15 fixed recordings give 30 timings without padding short recordings.
    return ([selected[i % len(selected)] for i in range(count)],
            [sources[i % len(sources)] for i in range(count)])


def worker(args):
    torch.set_num_threads(args.threads[0])
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(True)
    torch.manual_seed(1)
    entry = next(m for m in MODELS if m[0] == args.worker)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/configs")):
        config = compose(config_name="report_queue", overrides=[f"model={entry[2]}", "++model.mask_pf=true"])
    if config.model._target_ != "src.model.DfNet":
        OmegaConf.update(config, "model.grnn_backend", "torch", force_add=True)
    setup_config(config)
    model = instantiate(config.model).cpu().float().eval()
    checkpoint = checkpoint_for(entry)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state.get("epoch_complete") and state["epoch"] == 50
        model.load_state_dict(state["state_dict"], strict=True)
        del state
    state_bytes = sum(p.numel() * p.element_size() for p in model.state_dict().values())
    report = {"model": entry[0], "label": entry[1], "model_config": entry[2],
              "training_run": entry[3], "parameters": sum(p.numel() for p in model.parameters()),
              "fp32_state_mib": state_bytes / 1024 ** 2, "threads": torch.get_num_threads(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "weights_trained_50_epochs": bool(checkpoint),
              "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
              "weights_source": str(checkpoint.relative_to(ROOT)) if checkpoint else "seed=1 initialization; architecture timing only",
              "durations": []}

    def features(wave):
        lengths = torch.full((1,), wave.shape[-1], dtype=torch.long)
        inputs, _ = prepare_enhancement_inputs(wave, lengths)
        return inputs, lengths

    def enhance(wave):
        inputs, lengths = features(wave)
        enhanced = model(**inputs)[0]
        return synthesize_enhancement(enhanced, lengths, output_length=wave.shape[-1])

    with torch.inference_mode():
        for duration in args.durations:
            waves, sources = clips(duration, args.iterations, config.stft.sr)
            for i in range(args.warmup):
                enhance(waves[i % len(waves)])
            model_times, end_times = [], []
            for wave in waves:
                inputs, _ = features(wave)
                start = time.perf_counter_ns()
                model(**inputs)
                model_times.append((time.perf_counter_ns() - start) / 1e6)
                start = time.perf_counter_ns()
                enhanced = enhance(wave)
                end_times.append((time.perf_counter_ns() - start) / 1e6)
                assert enhanced.shape == wave.shape and torch.isfinite(enhanced).all()
            inputs, _ = features(waves[0])
            report["durations"].append({"audio_seconds": duration,
                                        "stft_frames": inputs["spec"].shape[2],
                                        "distinct_clips": len({s["waveform_sha256"] for s in sources}),
                                        "neural_macs": count_neural_macs(model, inputs),
                                        "model_only": stats(model_times, duration),
                                        "waveform_to_waveform": stats(end_times, duration),
                                        "clips": sources})
            print(f"{entry[0]} {args.threads[0]}T {duration:g}s: {statistics.median(end_times):.2f} ms end-to-end", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[m[0] for m in MODELS], default=[m[0] for m in MODELS])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--durations", nargs="+", type=float, default=[1, 5])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/cpu_speech_enhancement_20261015.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/cpu_benchmark_20261015")
    parser.add_argument("--worker", choices=[m[0] for m in MODELS])
    args = parser.parse_args()
    if min([*args.threads, *args.durations, args.iterations, args.warmup]) <= 0 or args.iterations < 2:
        parser.error("Use positive arguments and at least two measurements")
    if args.worker:
        worker(args)
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": file_hash(__file__), "torch": torch.__version__,
              "python": platform.python_version(), "platform": platform.platform(),
              "hardware": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
              "precision": "float32", "batch_size": 1, "sample_rate": 16000,
              "fft_size": 320, "hop_length": 160, "interop_threads": 1,
              "flush_denormal": True, "mkldnn_enabled": torch.backends.mkldnn.enabled,
              "warmup": args.warmup, "iterations": args.iterations,
              "scope": "Fixed-duration middle crops of official test waveforms; preloaded. Network-only and STFT/features/model/ISTFT separately. No disk I/O, resampling, batching, loss or quality metrics in timed regions. Offline whole-utterance inference, not streaming latency.",
              "mac_scope": "Dense Conv2d, ConvTranspose2d (input lattice), Linear, grouped linear, GRU and FastGRNN matrix products. Excludes activation functions, biases, normalization, FFT and parameter-free DF filtering.",
              "results": []}
    for name in args.models:
        for threads in args.threads:
            path = args.work_dir / f"{name}_t{threads}.json"
            checkpoint = checkpoint_for(next(m for m in MODELS if m[0] == name))
            signature = {"source": report["source_sha256"], "checkpoint": file_hash(checkpoint) if checkpoint else None,
                         "warmup": args.warmup, "iterations": args.iterations, "durations": args.durations}
            cached = json.loads(path.read_text()) if path.exists() else {}
            if cached.get("signature") != signature:
                cores = [0] if threads == 1 else [0, 2, 4, 6][:threads]
                if threads > len(cores):
                    raise ValueError("This protocol reserves four physical cores for CPU benchmarking")
                command = ["taskset", "-c", ",".join(map(str, cores)), sys.executable, "-u", str(Path(__file__).resolve()),
                           "--worker", name, "--threads", str(threads), "--durations", *map(str, args.durations),
                           "--iterations", str(args.iterations), "--warmup", str(args.warmup), "--output", str(path)]
                env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1")
                subprocess.run(command, cwd=ROOT, env=env, check=True)
                cached = json.loads(path.read_text())
                cached["signature"] = signature
                path.write_text(json.dumps(cached, indent=2) + "\n")
            report["results"].append(cached)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            (args.work_dir / "status.json").write_text(json.dumps({"model": name, "threads": threads,
                                                                 "completed": len(report["results"]),
                                                                 "total": len(args.models) * len(args.threads)}, indent=2) + "\n")
    (args.work_dir / "completed").write_text(datetime.now(timezone.utc).isoformat() + "\n")


if __name__ == "__main__":
    main()
