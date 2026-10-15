"""Render paper tables and figures from measured CPU timings and committed epoch-50 results."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from benchmark_cpu import MODELS, ROOT
MODELS = [(name, "XConv" if label == "XCONV" else label, config, run) for name, label, config, run in MODELS]

BENCHMARK = ROOT / "benchmarks/cpu_speech_enhancement_20261015.json"
NOISY = ROOT / "benchmarks/noisy_test_baseline_20261015.json"
REPORT = ROOT / "paper/results/cpu_benchmark_20261015.md"
CURVES = ROOT / "benchmarks/training_curves_20261015.json"


def noisy_baseline():
    from src.metrics import PESQ, STOI, SI_SDR
    torch.set_num_threads(1)
    metrics = [PESQ(device="cpu"), STOI(device="cpu"), SI_SDR(device="cpu")]
    records = []
    for entry in json.loads((ROOT / "data/vctk/test/index.json").read_text()):
        audio = load_file(entry["element_path"])
        scores = {m.name: m(separated_audio=audio["noisy"].unsqueeze(0),
                            audio_target=audio["clean"].unsqueeze(0),
                            lengths=torch.tensor([audio["noisy"].numel()])) for m in metrics}
        assert all(np.isfinite(list(scores.values())))
        records.append({"recording": Path(entry["original_noisy"]).name, **scores})
        if len(records) % 100 == 0:
            print(f"Noisy baseline: {len(records)}/824", flush=True)
    assert len(records) == 824
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "samples": len(records),
              "sample_rate": 16000, "scope": "Unprocessed official test recordings, full lengths. Same metric implementations as training.",
              "mean": {m.name: float(np.mean([r[m.name] for r in records])) for m in metrics}, "recordings": records}
    NOISY.parent.mkdir(exist_ok=True)
    NOISY.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mean"]), flush=True)


def histories():
    result = {}
    cached = json.loads(CURVES.read_text()) if CURVES.exists() else {}
    for name, label, config, run in MODELS:
        path = ROOT / "saved" / run / "epoch_metrics.jsonl"
        if path.exists():
            rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            result[name] = {r["epoch"]: r for r in rows if r["epoch"] <= 50}
        elif name in cached:
            result[name] = {int(epoch): row for epoch, row in cached[name]["epochs"].items()}
    CURVES.write_text(json.dumps({name: {"run_name": next(m[3] for m in MODELS if m[0] == name), "epochs": rows}
                                 for name, rows in result.items()}, indent=2) + "\n")
    return result


def duration_result(result, seconds):
    return next(d for d in result["durations"] if d["audio_seconds"] == seconds)


def figures(rows, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "savefig.bbox": "tight",
                         "svg.hashsalt": "speech-enhancement-20261015"})
    colors = plt.get_cmap("tab20")
    directory = ROOT / "paper/graphics"
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*", "8"]
    for i, (name, label, _, _) in enumerate(MODELS):
        quality = history.get(name, {}).get(50)
        if not quality or (name, 1) not in rows:
            continue
        row = rows[name, 1]
        latency = duration_result(row, 1)["waveform_to_waveform"]["median_ms"]
        for ax, x in zip(axes, [row["parameters"] / 1e6, latency]):
            ax.scatter(x, quality["test_PESQ"], color=colors(i), marker=markers[i], s=45, label=label)
    for ax, xlab in zip(axes, ["Parameters (millions)", "CPU end-to-end median, 1 s input (ms), 1 thread"]):
        ax.set(xlabel=xlab, ylabel="Test WB-PESQ, fixed epoch 50")
        ax.grid(alpha=.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
    for ext in ("svg", "png"):
        path = directory / f"cpu_quality_tradeoff_20261015.{ext}"
        fig.savefig(path, dpi=220, metadata={"Date": None} if ext == "svg" else {})
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for i, (name, label, _, _) in enumerate(MODELS):
        h = history.get(name, {})
        selected = [h[e] for e in sorted(h) if e % 5 == 0 and "val_loss" in h[e]]
        if not selected:
            continue
        for ax, metric in zip(axes, ["val_loss", "test_PESQ"]):
            ax.plot([r["epoch"] for r in selected], [r[metric] for r in selected],
                    marker=".", label=label, linewidth=1.2, color=colors(i))
    axes[0].set(xlabel="Training epoch", ylabel="Validation spectral + alpha loss")
    axes[1].set(xlabel="Training epoch", ylabel="Test WB-PESQ")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
    for ext in ("svg", "png"):
        path = directory / f"learning_curves_20261015.{ext}"
        fig.savefig(path, dpi=220, metadata={"Date": None} if ext == "svg" else {})
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def render():
    benchmark = json.loads(BENCHMARK.read_text())
    rows = {(r["model"], r["threads"]): r for r in benchmark["results"]}
    assert len(rows) == 2 * len(MODELS), "CPU benchmark has not finished all models/threads"
    history = histories()
    quality_count = sum(50 in history.get(m[0], {}) for m in MODELS)
    figures(rows, history)
    lines = ["# CPU performance and fixed-budget speech enhancement results", "",
             f"Study updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
             f"CPU measurements: {len(MODELS)} model architectures; completed epoch-50 quality results: {quality_count}/{len(MODELS)}.", "",
             "## CPU protocol", "",
             f"Hardware: **{benchmark['hardware']}**, WSL2 on Windows (shared desktop host). "
             f"Software: Python {benchmark['python']}, PyTorch {benchmark['torch']}, FP32, evaluation mode, inference mode, oneDNN enabled. "
             "Batch size is 1. One-thread runs are pinned to logical CPU 0; four-thread runs use CPUs 0, 2, 4, 6 (four distinct physical cores). "
             "PyTorch inter-op threads = 1, denormals are flushed. Concurrent GPU training is pinned to CPUs 8–15. "
             "These are timings of this repository's Python/PyTorch implementations at 16 kHz, not the upstream Rust/C++ DeepFilterNet runtime.", "",
             "Inputs are fixed **1 s and 5 s middle crops** of 15 eligible official test recordings per duration, "
             "with two passes (30 timed observations) after 5 warm-up calls per duration. Waveforms are loaded before timing. "
             "The same clips, crop offsets and audio hashes are recorded for every model. "
             "The network-only measurement excludes feature extraction. End-to-end timing includes STFT, ERB/DF feature extraction and normalization, "
             "the model, and ISTFT; it excludes disk I/O, resampling, losses and quality metrics. "
             "Network-only and end-to-end calls are interleaved per clip. Outputs are checked for correct length and finite samples.", "",
             "**RTF = processing seconds / audio seconds** (lower is better); throughput is `1 / RTF` audio seconds per wall-clock second. "
             "Latency below is the processing time of a whole buffered utterance. It does not measure streaming packet deadlines, first-sample latency "
             "or algorithmic look-ahead. The STFT is centered, with FFT/window = 320 samples (20 ms) and hop = 160 samples (10 ms).", "",
             "## Main performance table: 1 s inputs", "",
             "| Model | Params (M) | FP32 state (MiB) | Neural MMAC/s | Network p50 (ms) | End-to-end p50 / p95 (ms) | RTF, 1 thread | RTF, 4 threads | Throughput, 1 thread (× real time) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label, _, _ in MODELS:
        row = rows[name, 1]
        a, b = duration_result(row, 1), duration_result(rows[name, 4], 1)
        full = a["waveform_to_waveform"]
        mark = "†" if not row["weights_trained_50_epochs"] else ""
        lines.append(f"| {label}{mark} | {row['parameters']/1e6:.3f} | {row['fp32_state_mib']:.2f} | {a['neural_macs']/1e6:.2f} | "
                     f"{a['model_only']['median_ms']:.2f} | {full['median_ms']:.2f} / {full['p95_ms']:.2f} | "
                     f"{full['rtf_median']:.4f} | {b['waveform_to_waveform']['rtf_median']:.4f} | {full['audio_seconds_per_second']:.1f} |")
    lines += ["", "FP32 state is the total byte size of model parameters and buffers, excluding optimizer state, "
              "serialization overhead, Python/PyTorch runtime and activation memory. Neural MACs count Conv2d, ConvTranspose2d (input-lattice convention), "
              "Linear, grouped linear and recurrent matrix products; one multiply–accumulate counts as one MAC. "
              "FFT, bias additions, activations, normalization and parameter-free deep filtering are excluded. "
              "For 1 s inputs, centered STFT produces 101 frames; these boundary frames are included.", "",
              "† CPU timing currently uses seed-1 initialization because a complete 50-epoch checkpoint was not available at measurement time. "
              "This measures the dense architecture's execution cost; it does not constitute a quality result. "
              "Unmarked rows load this study's exact epoch-50 checkpoint. Raw reports contain checkpoint SHA-256 hashes. "
              "After the remaining checkpoints become available, rerunning the benchmark updates only affected rows.", "",
              "## Longer inputs and timing uncertainty", "",
              "| Model | 5 s end-to-end p50 / p95 (ms), 1 thread | 5 s RTF, 1 thread | 5 s RTF, 4 threads | 1 s p50 bootstrap 95% interval (ms), 1 thread |",
              "|---|---:|---:|---:|---:|"]
    for name, label, _, _ in MODELS:
        a, b = duration_result(rows[name, 1], 5), duration_result(rows[name, 4], 5)
        timing = a["waveform_to_waveform"]
        ci = duration_result(rows[name, 1], 1)["waveform_to_waveform"]["median_ci95_ms"]
        lines.append(f"| {label} | {timing['median_ms']:.2f} / {timing['p95_ms']:.2f} | {timing['rtf_median']:.4f} | "
                     f"{b['waveform_to_waveform']['rtf_median']:.4f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
    lines += ["", "Timing intervals are percentile bootstrap intervals of the median over the 30 observed runtimes "
              "(2,000 resamples; fixed seed 20261015), conditional on the selected clips and this shared host. "
              "They are not confidence intervals for enhancement quality or cross-device generalization.", "",
              "On this host, the full FastGRNN-512 model processes a 1 s waveform "
              f"{duration_result(rows['dfnet', 1], 1)['waveform_to_waveform']['median_ms']/duration_result(rows['grnn', 1], 1)['waveform_to_waveform']['median_ms']:.2f}× "
              "faster than the grouped-GRU DFNet, with "
              f"{100*(1-rows['grnn',1]['parameters']/rows['dfnet',1]['parameters']):.1f}% fewer parameters. "
              "Four threads do not uniformly improve short-utterance latency. The X-variants also illustrate why "
              "MAC counts should be accompanied by measured runtime: elementwise work and framework dispatch are outside the neural-MAC estimate.", "",
              "## Quality at an equal training budget", "",
              "Every enhanced result below is the committed **last epoch 50**, with 10,802 training recordings, "
              "770 speaker-held-out validation recordings and all 824 official test recordings. "
              "The test set is not used to select a checkpoint. No best-over-70/94-epoch result is substituted. "
              "Baseline DFNet/FastGRNN-512/FastGRNN-256 originally ran longer, but only their epoch-50 results are compared here. "
              "Runs use the same full-recording loss/optimizer recipe and effective batch 32, seed 1. "
              "Published values are the training evaluator's sample-averaged WB-PESQ, STOI and SI-SDR (AMP inference); "
              "the CPU speed benchmark uses FP32. A single seed is used, so no claim of statistical superiority is made.", "",
              "| Model | Epoch | Test recordings | WB-PESQ ↑ | STOI ↑ | SI-SDR (dB) ↑ |",
              "|---|---:|---:|---:|---:|---:|"]
    if NOISY.exists():
        noise = json.loads(NOISY.read_text())
        v = noise["mean"]
        lines.append(f"| Noisy (unprocessed) | — | {noise['samples']} | {v['PESQ']:.3f} | {v['STOI']:.4f} | {v['SI_SDR']:.2f} |")
    for name, label, _, _ in MODELS:
        row = history.get(name, {}).get(50)
        if row:
            assert row["test_samples"] == 824 and row["samples"] == 10802
            lines.append(f"| {label} | 50 | 824 | {row['test_PESQ']:.3f} | {row['test_STOI']:.4f} | {row['test_SI_SDR']:.2f} |")
        else:
            lines.append(f"| {label} (training pending/in progress) | — | — | — | — | — |")
    lines += ["", "## Figures for the article", "",
              "![Model size and CPU processing time versus epoch-50 quality](../graphics/cpu_quality_tradeoff_20261015.svg)", "",
              "![Validation loss and test quality over the first 50 epochs](../graphics/learning_curves_20261015.svg)", "",
              "Both figures have vector SVG and 220-dpi PNG versions in `paper/graphics`. "
              "They use measured timings and completed epoch results only. Curves are sampled every five epochs for a common display cadence.", "",
              "## Reproduction and raw evidence", "", "```bash",
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmark_cpu.py",
              "taskset -c 0 python paper_results.py --noisy-baseline",
              "python paper_results.py", "```", "",
              "- [Raw CPU observations, clip identities and checkpoint hashes](../../benchmarks/cpu_speech_enhancement_20261015.json)",
              "- [Raw training/evaluation histories used for quality and figures](../../benchmarks/training_curves_20261015.json)",
              "- [Noisy-baseline per-recording scores](../../benchmarks/noisy_test_baseline_20261015.json)",
              "- [CPU measurement code](../../benchmark_cpu.py)",
              "- [Table/figure and Noisy-baseline code](../../paper_results.py)",
              "- [Comet project](https://www.comet.com/runtime57/speech-enhancement)", "",
              "CPU measurements depend on this desktop/WSL2 host and these PyTorch implementations. "
              "They support controlled comparisons within the table, rather than absolute latency claims for ARM, microcontrollers or a streaming implementation.", ""]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(REPORT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy-baseline", action="store_true")
    args = parser.parse_args()
    noisy_baseline() if args.noisy_baseline else render()
