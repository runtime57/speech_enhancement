"""Publish epoch-50 quality, lean inference weights and measured CPU results to Comet."""

import hashlib
import fcntl
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from benchmark_cpu import MODELS, ROOT, checkpoint_for, file_hash
from sync_comet import CometRestClient, save_state


def upload(client, key, name, path, state, state_path):
    data = path.read_bytes()
    if client.api_key.encode() in data:
        raise ValueError("API key found in an export; upload stopped")
    digest = hashlib.sha256(data).hexdigest()
    if state.setdefault("assets", {}).get(name, {}).get("sha256") != digest:
        state["assets"][name] = client.upload_asset(key, name, data)
        save_state(state_path, state)
        print("Verified", name, len(data), "bytes", flush=True)


def main():
    torch.set_num_threads(1)
    root = ROOT / "outputs/paper_exports"
    root.mkdir(exist_ok=True)
    lock = (root / "artifacts.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    benchmark = json.loads((ROOT / "benchmarks/cpu_speech_enhancement_20261015.json").read_text())
    client = CometRestClient((ROOT / ".comet_key").read_text().strip(), "windows", timeout=20)
    try:
        for entry in MODELS:
            checkpoint = checkpoint_for(entry)
            if checkpoint is None:
                continue
            run = checkpoint.parent
            comet_state = run / "comet_sync.json"
            if not comet_state.exists():
                continue
            experiment = json.loads(comet_state.read_text())
            key = experiment["experiment_key"]
            directory = root / entry[0]
            directory.mkdir(exist_ok=True)
            state_path = directory / "export.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            checksum = file_hash(checkpoint)
            weights_path = directory / "model_epoch50.pth"
            if state.get("export_format_version") != 2 or state.get("source_checkpoint_sha256") != checksum or not weights_path.exists():
                loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert loaded["epoch"] == 50 and loaded["epoch_complete"]
                assert all(torch.isfinite(t).all() for t in loaded["state_dict"].values())
                config = OmegaConf.load(run / "config.yaml")
                torch.save({"state_dict": loaded["state_dict"], "epoch": 50, "epoch_complete": True,
                            "checkpoint_kind": "inference_only", "source_checkpoint_sha256": checksum, "run_name": run.name,
                            "config": {k: OmegaConf.to_container(config[k]) for k in ("model", "stft")}}, weights_path)
                state["source_checkpoint_sha256"] = checksum
                state["export_format_version"] = 2
                del loaded
            quality = next(json.loads(s) for s in (run / "epoch_metrics.jsonl").read_text().splitlines()
                           if s.strip() and json.loads(s)["epoch"] == 50)
            assert quality["test_samples"] == 824 and quality["samples"] == 10802
            quality_path = directory / "paper_metrics_epoch50.json"
            quality_path.write_text(json.dumps({"model": entry[0], "run_name": run.name,
                                               "checkpoint_epoch": 50, "quality": quality,
                                               "selection": "Fixed epoch 50, not selected on test performance"}, indent=2) + "\n")
            print(entry[0], experiment["url"], flush=True)
            upload(client, key, "model_epoch50.pth", weights_path, state, state_path)
            upload(client, key, "paper_metrics_epoch50.json", quality_path, state, state_path)
            cpu = [r for r in benchmark["results"] if r["model"] == entry[0] and r["checkpoint_sha256"] == checksum]
            metrics = {f"paper50/{metric}": quality[f"test_{metric}"] for metric in ("PESQ", "STOI", "SI_SDR")}
            if cpu:
                path = directory / "cpu_benchmark.json"
                path.write_text(json.dumps({"protocol": {k:v for k,v in benchmark.items() if k != "results"},
                                            "results": cpu}, indent=2) + "\n")
                upload(client, key, "cpu_benchmark.json", path, state, state_path)
                metrics.update(parameters=cpu[0]["parameters"], model_state_mib=cpu[0]["fp32_state_mib"])
                for row in cpu:
                    for d in row["durations"]:
                        prefix = f"cpu/{row['threads']}t/{d['audio_seconds']:g}s/"
                        metrics.update({prefix+k:d["waveform_to_waveform"][k] for k in
                                        ("median_ms", "p95_ms", "rtf_median", "audio_seconds_per_second")})
            signature = hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()
            if state.get("metrics_sha256") != signature:
                client.log_metrics(key, metrics, step=50*338, epoch=50,
                                   timestamp=int(checkpoint.stat().st_mtime*1000))
                state["metrics_sha256"] = signature
                save_state(state_path, state)
            if entry[0] == "dfnet":
                for path in [ROOT / "paper/results/cpu_benchmark_20261015.md",
                             ROOT / "benchmarks/cpu_speech_enhancement_20261015.json",
                             ROOT / "benchmarks/noisy_test_baseline_20261015.json",
                             ROOT / "benchmarks/training_curves_20261015.json",
                             *sorted((ROOT / "paper/graphics").glob("*20261015.svg"))]:
                    if path.exists():
                        upload(client, key, path.name, path, state, state_path)
            print("Published paper results:", entry[0], flush=True)
    finally:
        client.close()
        lock.close()


if __name__ == "__main__":
    main()
