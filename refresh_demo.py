"""Regenerate the listening demo from complete, fixed epoch-50 inference weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.config_utils import setup_config
from src.utils.enhancement import prepare_enhancement_inputs, synthesize_enhancement

ROOT = Path(__file__).resolve().parent
MODELS = [
    ("dfnet", "DFNet (grouped GRU)", "dfnet", "dfnet.pth"),
    ("grnn", "FastGRNN · emb 512", "grnn", "grnn.pth"),
    ("grnn_emb_256", "FastGRNN · emb 256", "grnn_emb_256", "grnn_emb_256.pth"),
    ("grnn_emb_128", "FastGRNN · emb 128", "grnn_emb_128", "grnn_emb_128.pth"),
    ("rebalanced", "Rebalanced-1", "grnn_rebalanced", "rebalanced_1.pth"),
    ("rebalanced2", "Rebalanced-2", "grnn_rebalanced2", "rebalanced_2.pth"),
    ("rebalanced3", "Rebalanced-3", "grnn_rebalanced3", "rebalanced_3.pth"),
]
SAMPLE_RATE = 16000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_audio(path):
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE and waveform.shape[1] == 1, f"Expected mono 16 kHz: {path}"
    assert np.isfinite(waveform).all() and waveform.shape[0] > 0
    return torch.from_numpy(waveform[:, 0].copy()).unsqueeze(0)


def magnitude(waveform):
    return torch.stft(waveform.squeeze(0), n_fft=512, hop_length=128,
                      window=torch.hann_window(512), return_complex=True).abs().numpy()


def spectrogram(spec, reference, duration, path):
    """Use one reference and fixed color limits for all variants of a recording."""
    db = 20 * np.log10(np.maximum(spec, reference * 1e-4) / reference)
    plt.rcParams.update({"font.size": 10, "text.color": "#e3e9f5", "axes.labelcolor": "#e3e9f5",
                         "xtick.color": "#b6c0d4", "ytick.color": "#b6c0d4"})
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor="#0b1020", layout="constrained")
    half_step = 128 / SAMPLE_RATE / 2
    image = ax.imshow(db, origin="lower", aspect="auto", cmap="inferno", vmin=-80, vmax=0,
                      extent=(-half_step, (spec.shape[1] - .5) * 128 / SAMPLE_RATE,
                              -SAMPLE_RATE / 512 / 2 / 1000, (SAMPLE_RATE / 2 + SAMPLE_RATE / 512 / 2) / 1000))
    ax.set(xlim=(0, duration), ylim=(0, 8), xlabel="Time (s)", ylabel="Frequency (kHz)")
    bar = fig.colorbar(image, ax=ax, pad=.02, ticks=[-80, -60, -40, -20, 0])
    bar.set_label("dB re. shared peak")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, help="Optional outputs/paper_exports from the training checkout")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(True)
    torch.manual_seed(1)
    manifest = {"schema_version": 1, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_epoch": 50, "sample_rate": SAMPLE_RATE, "precision": "FP32, CPU, batch 1",
                "spectrogram": {"n_fft": 512, "hop_length": 128, "window": "Hann", "range_db": [-80, 0],
                                "reference": "One peak magnitude shared by all nine variants of each sample"},
                "models": [], "samples": []}
    samples = {}
    for noisy_path in sorted((ROOT / "audio/noisy").glob("sample*.wav")):
        sample = noisy_path.stem
        noisy = read_audio(noisy_path)
        clean = read_audio(ROOT / "audio/clean" / noisy_path.name)
        assert noisy.shape == clean.shape, f"Unaligned reference for {sample}"
        samples[sample] = {"noisy": noisy, "clean": clean}
    assert len(samples) == 3, "Expected the three existing listening examples"
    with torch.inference_mode():
        for model_id, label, export_name, filename in MODELS:
            path = ROOT / "models" / filename
            if args.weights_root:
                path.parent.mkdir(exist_ok=True)
                shutil.copy2(args.weights_root / export_name / "model_epoch50.pth", path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            assert checkpoint["epoch"] == 50 and checkpoint["epoch_complete"]
            assert checkpoint["checkpoint_kind"] == "inference_only"
            config = OmegaConf.create(checkpoint["config"])
            if config.model._target_ != "src.model.DfNet":
                OmegaConf.update(config, "model.grnn_backend", "torch", force_add=True)
            setup_config(config)
            model = instantiate(config.model).cpu().float().eval()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            manifest["models"].append({"id": model_id, "label": label, "epoch": 50,
                                       "parameters": sum(p.numel() for p in model.parameters()),
                                       "run_name": checkpoint["run_name"], "weights_url": f"models/{filename}",
                                       "weights_sha256": sha256(path),
                                       "source_checkpoint_sha256": checkpoint["source_checkpoint_sha256"],
                                       "inference_config": OmegaConf.to_container(config)})
            for sample, waves in samples.items():
                noisy = waves["noisy"]
                lengths = torch.tensor([noisy.shape[-1]])
                inputs, _ = prepare_enhancement_inputs(noisy, lengths)
                enhanced = synthesize_enhancement(model(**inputs)[0], lengths, output_length=noisy.shape[-1])
                assert enhanced.shape == noisy.shape and torch.isfinite(enhanced).all()
                assert enhanced.abs().max() <= 1, f"Clipping would be required: {model_id}/{sample}"
                output = ROOT / "audio/enhanced" / model_id / f"{sample}.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output, enhanced.squeeze(0).numpy(), SAMPLE_RATE, subtype="PCM_16")
                waves[model_id] = read_audio(output)
                print(f"Enhanced {sample}: {label}", flush=True)
            del model, checkpoint
    for sample, waves in samples.items():
        spectra = {variant: magnitude(wave) for variant, wave in waves.items()}
        reference = max(float(spec.max()) for spec in spectra.values())
        assert reference > 0
        item = {"id": sample, "label": f"Sample {sample.removeprefix('sample')}",
                "samples": waves["noisy"].shape[-1], "duration_seconds": waves["noisy"].shape[-1] / SAMPLE_RATE,
                "spectrogram_reference_magnitude": reference, "variants": {}}
        for variant, wave in waves.items():
            subdirectory = variant if variant in ("noisy", "clean") else f"enhanced/{variant}"
            audio_path, image_path = f"audio/{subdirectory}/{sample}.wav", f"images/{subdirectory}/{sample}.png"
            spectrogram(spectra[variant], reference, item["duration_seconds"], ROOT / image_path)
            item["variants"][variant] = {"audio": audio_path, "image": image_path,
                                          "audio_sha256": sha256(ROOT / audio_path),
                                          "image_sha256": sha256(ROOT / image_path)}
        manifest["samples"].append(item)
        print(f"Rendered {len(waves)} spectrograms: {sample}", flush=True)
    (ROOT / "demo_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("Updated 7 models, 21 enhanced clips and 27 spectrograms", flush=True)


if __name__ == "__main__":
    main()
