"""Persistent, sequential local training queue; Comet is handled by sync_queue.py."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from src.utils.config_utils import setup_config
from src.utils.training_suite import MODELS, REPORT_MODELS
from sync_comet import save_state
from train_suite import environment, snapshot, wait_for_game

ROOT = Path(__file__).resolve().parent


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid, expected):
    try:
        path = Path(f"/proc/{pid}")
        stat = (path / "stat").read_text().split()
        if stat[2] in ("Z", "X") or expected.encode() not in (path / "cmdline").read_bytes():
            return None
        return {"pid": int(pid), "start_ticks": stat[21],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, ValueError):
        return None


def is_live(identity, expected):
    return bool(identity) and process_identity(identity["pid"], expected) == identity


def wait_for_running_runs(manifest, path):
    """Adopt existing trainers before starting any newly queued earlier entry."""
    adopted = []
    for entry in manifest["runs"]:
        if not is_live(entry.get("identity"), entry["run_name"]):
            continue
        adopted.append(entry)
        print(f"Adopting {entry['name']} (PID {entry['identity']['pid']}); waiting for completion", flush=True)
        while is_live(entry.get("identity"), entry["run_name"]):
            manifest.update(status="running", active_model=entry["name"], updated_at_utc=timestamp())
            save_state(path, manifest)
            time.sleep(15)
    return adopted


def run_environment(entry):
    env = environment()
    if entry.get("cuda_launch_blocking"):
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    return env


def run_preflight(command, log_path, attempts=3, env=None):
    """Retry a temporary unavailable CUDA device, preserving every failed attempt."""
    for attempt in range(1, attempts + 1):
        start = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open("a") as log:
            result = subprocess.run(command, cwd=ROOT, env=env or environment(), stdout=log,
                                    stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return
        with log_path.open("rb") as log:
            log.seek(start)
            failure = log.read().decode(errors="replace")
        if "CUDA driver error: device not ready" not in failure or attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"CUDA device not ready; retrying preflight ({attempt + 1}/{attempts})", flush=True)
        time.sleep(15 * attempt)


def make_manifest(after=None, prepend_models=None):
    extra = [(name, dict(MODELS + REPORT_MODELS)[name]) for name in (prepend_models or [])]
    models = extra + [(name, config) for name, config in REPORT_MODELS if name not in dict(extra)]
    return {"created_at_utc": timestamp(), "status": "queued", "after": after,
            "protocol": {"epochs": 50, "evaluation_period": 5, "batch_size": 32,
                         "full_recordings": True, "train_samples": 10802,
                         "val_samples": 770, "test_samples": 824, "train_batches": 338,
                         "drop_last": False},
            "runs": [{"name": name, "model_config": config, "status": "pending",
                      "run_name": f"{name}_full_50e_eval5_20261015"}
                     for name, config in models]}


def records(run_dir):
    path = run_dir / "epoch_metrics.jsonl"
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                result[row["epoch"]] = row
    return result


def training_command(entry, resume=None):
    command = [sys.executable, "-u", "train.py", "-cn=report_queue",
               f"model={entry['model_config']}", f"writer.run_name={entry['run_name']}",
               "++model.mask_pf=true", "++model.grnn_backend=triton"]
    if resume:
        command.append(f"trainer.resume_from={resume.name}")
    return command


def resume_checkpoint(run_dir):
    journal = records(run_dir)
    for epoch in sorted(journal, reverse=True):
        path = run_dir / f"checkpoint-epoch{epoch}.pth"
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("epoch_complete"):
                candidates = [r for e, r in journal.items() if e <= epoch and "val_loss" in r]
                if candidates:
                    best = min(candidates, key=lambda row: row["val_loss"])
                    shutil.copyfile(run_dir / f"checkpoint-epoch{best['epoch']}.pth", run_dir / "model_best.pth")
                return path
    return None


def audit_local(run_dir):
    config = OmegaConf.load(run_dir / "config.yaml")
    result = json.loads((run_dir / "training_result.json").read_text())
    assert config.trainer.n_epochs == config.trainer.early_stop == 50
    assert config.trainer.evaluation_period == 5
    assert config.datasets.train.segment_seconds is None
    assert config.trainer.epoch_len is None and not config.trainer.drop_last
    assert config.dataloader.batch_size == 32
    assert list(config.trainer.evaluation_partitions) == ["val", "test"]
    assert not config.metrics.train
    assert result["epochs_completed"] == result["epochs_requested"] == 50
    assert result["epoch_len"] == 338
    assert result["dataset_sizes"] == {"train": 10802, "val": 770, "test": 824}
    history = records(run_dir)
    assert set(history) == set(range(1, 51))
    evaluations = set()
    for epoch, row in history.items():
        assert (row["samples"], row["batches"]) == (10802, 338)
        assert all(math.isfinite(value) for value in row.values())
        if epoch % 5 == 0:
            assert (row["val_samples"], row["test_samples"]) == (770, 824)
            for part in ("val", "test"):
                assert all(f"{part}_{metric}" in row for metric in ("loss", "PESQ", "STOI", "SI_SDR"))
            evaluations.add(epoch)
        else:
            assert not any(name.startswith(("val_", "test_")) for name in row)
    assert evaluations == set(range(5, 51, 5))
    setup_config(config)
    model = instantiate(config.model)
    for name in ("checkpoint-epoch50.pth", "model_best.pth"):
        checkpoint = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        assert checkpoint["epoch_complete"]
        if name == "checkpoint-epoch50.pth":
            assert checkpoint["epoch"] == 50
        assert all(torch.isfinite(t).all() for t in checkpoint["state_dict"].values())
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        for state in checkpoint["optimizer"]["state"].values():
            assert all(torch.isfinite(t).all() for t in state.values() if isinstance(t, torch.Tensor))
    assert checkpoint["epoch"] in evaluations
    minimum = min(history[e]["val_loss"] for e in evaluations)
    assert math.isclose(checkpoint["monitor_best"], minimum, abs_tol=1e-9)
    return {"epochs": 50, "evaluation_epochs": sorted(evaluations), "best_epoch": checkpoint["epoch"],
            "best_val_loss": minimum, "local_passed": True,
            "parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=ROOT / "outputs/report_queue_20261015")
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--after-run", default="grnn_emb_256_full_20261014")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepend-models", nargs="*", default=[],
                        choices=sorted(dict(MODELS + REPORT_MODELS)))
    args = parser.parse_args()
    directory = args.queue_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = directory / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        expected = make_manifest(prepend_models=args.prepend_models)
        expected_names = {r["name"]: r["model_config"] for r in expected["runs"]}
        assert len(expected_names) == len(expected["runs"]), "Duplicate selected model"
        assert all(expected_names.get(r["name"]) == r["model_config"] for r in manifest["runs"])
        existing = {r["name"]: r for r in manifest["runs"]}
        manifest["runs"] = [existing.get(r["name"], r) for r in expected["runs"]]
        assert manifest["protocol"] == make_manifest()["protocol"]
        save_state(path, manifest)
    else:
        prior = process_identity(args.after_pid, args.after_run) if args.after_pid else None
        after = {"identity": prior, "run_name": args.after_run} if prior else None
        manifest = make_manifest(after, args.prepend_models)
        save_state(path, manifest)
    if args.prepare:
        print(f"Prepared {len(manifest['runs'])} jobs: 50 epochs, evaluation every 5", flush=True)
        return
    (directory / "controller.pid").write_text(str(os.getpid()) + "\n")
    after = manifest.get("after")
    while after and is_live(after["identity"], after["run_name"]):
        manifest.update(status="waiting_current", updated_at_utc=timestamp())
        save_state(path, manifest)
        time.sleep(15)
    manifest.update(status="running", updated_at_utc=timestamp())
    manifest.pop("finished_at_utc", None)
    save_state(path, manifest)
    for entry in wait_for_running_runs(manifest, path):
        output = directory / entry["name"]
        run_dir = ROOT / "saved" / entry["run_name"]
        if (run_dir / "training_result.json").exists():
            try:
                verification = audit_local(run_dir)
                save_state(output / "verification.json", verification)
                entry.update(status="trained", finished_at_utc=timestamp(), **verification)
                print(f"Trained and verified adopted {entry['name']}", flush=True)
            except Exception as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
        else:
            entry["status"] = "interrupted"
        save_state(path, manifest)
    for entry in manifest["runs"]:
        if entry["status"] in ("trained", "failed"):
            continue
        output = directory / entry["name"]
        output.mkdir(exist_ok=True)
        run_dir = ROOT / "saved" / entry["run_name"]
        try:
            # Reattach after controller restarts; never run another model beside it.
            while is_live(entry.get("identity"), entry["run_name"]):
                manifest.update(active_model=entry["name"], updated_at_utc=timestamp())
                save_state(path, manifest)
                time.sleep(15)
            if not (run_dir / "training_result.json").exists():
                entry["status"] = "waiting_gpu"
                manifest["active_model"] = entry["name"]
                save_state(path, manifest)
                wait_for_game(directory, {entry["name"]})
                if not (output / "preflight.json").exists():
                    entry["status"] = "preflight"
                    save_state(path, manifest)
                    command = [sys.executable, "-u", "preflight_suite.py", "--config", "report_queue",
                               "--model", entry["model_config"], "--output", str(output / "preflight.json"),
                               "--override", "++model.mask_pf=true", "--override", "++model.grnn_backend=triton"]
                    run_preflight(command, output / "preflight.log", env=run_environment(entry))
                resume = resume_checkpoint(run_dir) if run_dir.exists() else None
                if run_dir.exists() and resume is None:
                    archive = run_dir.with_name(run_dir.name + "_interrupted_init_" + str(int(time.time())))
                    run_dir.rename(archive)
                command = training_command(entry, resume)
                entry.update(status="training", command=command, started_at_utc=timestamp())
                with (output / "train.log").open("a") as log:
                    train = subprocess.Popen(command, cwd=ROOT, env=run_environment(entry), stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                identity = None
                for _ in range(100):
                    identity = process_identity(train.pid, entry["run_name"])
                    if identity or train.poll() is not None:
                        break
                    time.sleep(.05)
                if not identity:
                    train.terminate()
                    train.wait()
                    raise RuntimeError("Could not establish training process identity")
                entry.update(pid=train.pid, identity=identity)
                (output / "train.pid").write_text(str(train.pid) + "\n")
                manifest.update(active_model=entry["name"])
                save_state(path, manifest)
                print(f"Training {entry['name']} (PID {train.pid})", flush=True)
                while not (run_dir / "config.yaml").exists() and train.poll() is None:
                    time.sleep(1)
                if train.poll() is not None:
                    raise RuntimeError(f"Training failed during initialization: {train.returncode}")
                snapshot(run_dir)
                with (output / "monitor.log").open("a") as log:
                    monitor = subprocess.Popen([sys.executable, "-u", "monitor_training.py", "--pid-file",
                                                str(output / "train.pid"), "--run-dir", str(run_dir),
                                                "--output-dir", str(output), "--status-file",
                                                str(ROOT / "outputs/dfnet_launch/status.json")],
                                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                code = train.wait()
                (output / "train.exit").write_text(str(code) + "\n")
                entry["exit_code"] = code
                try:
                    monitor.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    monitor.terminate()
                if code:
                    raise RuntimeError(f"Training exited with {code}")
            entry["status"] = "verifying_local"
            save_state(path, manifest)
            verification = audit_local(run_dir)
            save_state(output / "verification.json", verification)
            entry.update(status="trained", finished_at_utc=timestamp(), **verification)
            print(f"Trained and verified {entry['name']}; Comet uploads independently", flush=True)
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(f"Job failed: {entry['name']}: {entry['error']}", flush=True)
        manifest["updated_at_utc"] = timestamp()
        save_state(path, manifest)
    manifest.update(status="trained" if all(r["status"] == "trained" for r in manifest["runs"]) else "finished_with_errors",
                    finished_at_utc=timestamp(), active_model=None)
    save_state(path, manifest)


if __name__ == "__main__":
    main()
