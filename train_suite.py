"""Train all requested full-recording models sequentially and verify Comet history."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from src.utils.config_utils import setup_config
from src.utils.training_suite import MODELS
from sync_comet import CometRestClient, comet_metric_names, save_state


ROOT = Path(__file__).resolve().parent


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def environment():
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for name in list(env):
        if "proxy" in name.lower():
            env.pop(name)
    return env


def wait_for_game(directory, selected):
    """Delay the selected run until ARC Raiders exits and GPU memory is free."""
    probe = ('if (Get-Process -Name PioneerGame -ErrorAction SilentlyContinue) '
             '{"running"} else {"closed"}')
    previous = None
    while True:
        state = {"time_utc": timestamp(), "models": sorted(selected),
                 "minimum_free_gpu_mib": 10240, "pid": os.getpid()}
        try:
            game = subprocess.run([
                "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                "-NoProfile", "-Command", probe], check=True, capture_output=True,
                text=True, timeout=10).stdout.strip()
            if game not in ("running", "closed"):
                raise ValueError("Unexpected game probe response")
            memory = subprocess.run([
                "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=10)
            free = int(memory.stdout.strip().splitlines()[0])
            state.update(game_running=game == "running", free_gpu_mib=free,
                         status="waiting_game" if game == "running" else
                         "waiting_gpu_memory" if free < 10240 else "ready")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            state.update(status="probe_error", error=type(error).__name__)
        save_state(directory / "launch_wait.json", state)
        if state["status"] != previous:
            print(f"Launch gate: {state['status']}", flush=True)
            previous = state["status"]
        if state["status"] == "ready":
            return
        time.sleep(30)


def snapshot(run_dir):
    files = sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "src/configs").rglob("*.yaml")) +
                   [ROOT / name for name in ("train.py", "train_suite.py", "train_queue.py", "sync_queue.py",
                                             "sync_comet.py", "preflight_suite.py", "requirements.txt")])
    checksum = hashlib.sha256()
    with zipfile.ZipFile(run_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            content = path.read_bytes()
            checksum.update(str(path.relative_to(ROOT)).encode())
            checksum.update(content)
            archive.writestr(str(path.relative_to(ROOT)), content)
    (run_dir / "source_sha256.txt").write_text(checksum.hexdigest() + "\n")


def audit_local(run_dir):
    config = OmegaConf.load(run_dir / "config.yaml")
    result = json.loads((run_dir / "training_result.json").read_text())
    assert config.datasets.train.segment_seconds is None
    assert config.trainer.epoch_len is None and not config.trainer.drop_last
    assert list(config.trainer.evaluation_partitions) == ["val", "test"]
    assert not config.metrics.train and config.dataloader.batch_size == 32
    assert result["epochs_completed"] == result["epochs_requested"] == 100
    assert result["epoch_len"] == 338
    records = [json.loads(line) for line in (run_dir / "epoch_metrics.jsonl").read_text().splitlines()]
    records = {row["epoch"]: row for row in records}
    assert set(records) == set(range(1, 101))
    for epoch, row in records.items():
        assert (row["samples"], row["batches"], row["val_samples"], row["test_samples"]) == (10802, 338, 770, 824)
        assert all(math.isfinite(value) for value in row.values())
        for name in ("val_PESQ", "val_STOI", "val_SI_SDR", "test_PESQ", "test_STOI", "test_SI_SDR"):
            assert name in row
    checkpoint = torch.load(run_dir / "checkpoint-epoch100.pth", map_location="cpu", weights_only=False)
    best = torch.load(run_dir / "model_best.pth", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 100 and checkpoint["epoch_complete"]
    for source in (checkpoint, best):
        assert all(torch.isfinite(value).all() for value in source["state_dict"].values())
        for state in source["optimizer"]["state"].values():
            assert all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor))
    setup_config(config)
    model = instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.load_state_dict(best["state_dict"], strict=True)
    minimum = min(row["val_loss"] for row in records.values())
    assert math.isclose(best["monitor_best"], minimum, abs_tol=1e-9)
    return records, {"epochs": 100, "best_epoch": best["epoch"], "best_val_loss": minimum,
                     "best_epoch_test_PESQ": records[best["epoch"]]["test_PESQ"],
                     "parameters": sum(p.numel() for p in model.parameters()), "local_passed": True}


def audit_remote(client, run_dir, records):
    state = json.loads((run_dir / "comet_sync.json").read_text())
    assert state["last_epoch"] == 100 and state.get("error") is None
    points = client.request("GET", "experiment/metrics/get-all", {"experimentKey": state["experiment_key"]})["metrics"]
    lookup = {}
    for point in points:
        if point.get("epoch") is not None:
            key = (int(point["epoch"]), point["metricName"])
            lookup.setdefault(key, []).append(point)
    checked = 0
    for epoch, row in records.items():
        metrics = comet_metric_names({k: v for k, v in row.items() if k != "epoch"}, ["val", "test"])
        for name, value in metrics.items():
            actual = lookup.get((epoch, name), [])
            assert actual, (epoch, name)
            for point in actual:
                assert int(point["step"]) == epoch * 338
                assert math.isclose(float(point["metricValue"]), value, rel_tol=1e-9, abs_tol=1e-9)
            checked += 1
    return {"url": state["url"], "remote_values_verified": checked, "remote_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=ROOT / "outputs/full_recordings_20261014")
    parser.add_argument("--models", nargs="+", choices=[name for name, _ in MODELS],
                        help="Run only these models, leaving the rest of the suite pending")
    parser.add_argument("--grnn-backend", choices=["auto", "torch", "triton"],
                        help="Override the backend for selected GRNN models")
    parser.add_argument("--wait-for-game", action="store_true",
                        help="Wait for ARC Raiders to close and at least 10 GiB GPU memory to be free")
    args = parser.parse_args()
    selected = set(args.models) if args.models else {name for name, _ in MODELS}
    directory = args.suite_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (directory / "controller.pid").write_text(str(os.getpid()) + "\n")
    if args.wait_for_game:
        wait_for_game(directory, selected)
    path = directory / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        assert [(r["name"], r["model_config"]) for r in manifest["runs"]] == MODELS
        for entry in manifest["runs"]:
            pid = entry.get("pid")
            command_line = Path(f"/proc/{pid}/cmdline")
            if pid and command_line.exists() and entry["run_name"] in command_line.read_text():
                raise RuntimeError(f"Existing training PID {pid} is still live; do not launch a duplicate")
            if entry["status"] == "training":
                entry.update(status="interrupted", interruption_detected_at_utc=timestamp())
        manifest.update(status="running", active_models=[name for name, _ in MODELS if name in selected])
        manifest.pop("error", None)
        save_state(path, manifest)
    else:
        manifest = {"created_at_utc": timestamp(), "status": "running",
                    "protocol": {"epochs": 100, "batch_size": 32, "full_recordings": True,
                                 "train_samples": 10802, "val_samples": 770, "test_samples": 824,
                                 "train_batches": 338, "drop_last": False,
                                 "evaluate_after_train_epoch": ["val", "test"]},
                    "runs": [{"name": name, "model_config": model, "status": "pending",
                              "run_name": name + "_full_20261014"} for name, model in MODELS]}
        save_state(path, manifest)
    client = CometRestClient((ROOT / ".comet_key").read_text().strip(), "windows")
    try:
        for entry in manifest["runs"]:
            if entry["name"] not in selected:
                continue
            run_dir = ROOT / "saved" / entry["run_name"]
            output = directory / entry["name"]
            output.mkdir(exist_ok=True)
            if entry["status"] == "complete":
                continue
            overrides = [f"model={entry['model_config']}", f"writer.run_name={entry['run_name']}"]
            if args.grnn_backend and entry["name"].startswith("grnn"):
                overrides.append(f"+model.grnn_backend={args.grnn_backend}")
            if not (run_dir / "training_result.json").exists():
                if run_dir.exists():
                    checkpoints = sorted(run_dir.glob("checkpoint-epoch*.pth"),
                                         key=lambda p: int(p.stem.split("epoch")[-1]), reverse=True)
                    complete = next((p for p in checkpoints if torch.load(p, map_location="cpu", weights_only=False).get("epoch_complete")), None)
                    if complete is None:
                        raise RuntimeError(f"Existing run has no complete checkpoint: {run_dir}")
                    overrides.append(f"trainer.resume_from={complete.name}")
                command = [sys.executable, "-u", "train.py", "-cn=full_recordings", *overrides]
                entry.update(status="training", started_at_utc=timestamp(), command=command)
                with (output / "train.log").open("a") as log:
                    train = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
                entry["pid"] = train.pid
                (output / "train.pid").write_text(str(train.pid) + "\n")
                save_state(path, manifest)
                print(f"Training {entry['name']} (PID {train.pid})", flush=True)
                while not (run_dir / "config.yaml").exists() and train.poll() is None:
                    time.sleep(1)
                if train.poll() is not None:
                    raise RuntimeError(f"Training failed during initialization: {entry['name']}")
                snapshot(run_dir)
                monitor_log = (output / "monitor.log").open("a")
                monitor = subprocess.Popen([sys.executable, "-u", "monitor_training.py", "--pid-file", str(output / "train.pid"),
                                            "--run-dir", str(run_dir), "--output-dir", str(output), "--status-file",
                                            str(ROOT / "outputs/dfnet_launch/status.json")],
                                           cwd=ROOT, stdout=monitor_log, stderr=subprocess.STDOUT)
                sync_log = (output / "comet.log").open("a")
                synchronizer = subprocess.Popen([sys.executable, "-u", "sync_comet.py", "--run-dir", str(run_dir),
                                                 "--epoch-len", "338", "--transport", "windows", "--follow",
                                                 "--pid-file", str(output / "train.pid")],
                                                cwd=ROOT, env=environment(), stdout=sync_log, stderr=subprocess.STDOUT)
                entry["sync_pid"] = synchronizer.pid
                save_state(path, manifest)
                while train.poll() is None:
                    if (run_dir / "comet_sync.json").exists() and "comet_url" not in entry:
                        state = json.loads((run_dir / "comet_sync.json").read_text())
                        try:
                            client.request("POST", "write/experiment/log-other", {
                                "experimentKey": state["experiment_key"], "key": "full_recording_protocol",
                                "value": json.dumps({"model_config": entry["model_config"], **manifest["protocol"]})})
                            entry["comet_url"] = state["url"]
                            save_state(path, manifest)
                        except (RuntimeError, TimeoutError):
                            client.close()
                            client = CometRestClient((ROOT / ".comet_key").read_text().strip(), "windows")
                    if synchronizer.poll() is not None:
                        synchronizer = subprocess.Popen([sys.executable, "-u", "sync_comet.py", "--run-dir", str(run_dir),
                                                         "--epoch-len", "338", "--transport", "windows", "--follow",
                                                         "--pid-file", str(output / "train.pid")],
                                                        cwd=ROOT, env=environment(), stdout=sync_log, stderr=subprocess.STDOUT)
                        entry["sync_pid"] = synchronizer.pid
                        save_state(path, manifest)
                    time.sleep(10)
                entry["exit_code"] = train.returncode
                (output / "train.exit").write_text(str(train.returncode) + "\n")
                save_state(path, manifest)
                if train.returncode != 0:
                    raise RuntimeError(f"Training {entry['name']} exited with {train.returncode}")
                monitor.wait(timeout=40)
                monitor_log.close()
                entry["status"] = "syncing"
                save_state(path, manifest)
                while True:
                    if (run_dir / "comet_sync.json").exists():
                        state = json.loads((run_dir / "comet_sync.json").read_text())
                        if state.get("last_epoch") == 100 and state.get("error") is None:
                            break
                    if synchronizer.poll() is not None:
                        synchronizer = subprocess.Popen([sys.executable, "-u", "sync_comet.py", "--run-dir", str(run_dir),
                                                         "--epoch-len", "338", "--transport", "windows"],
                                                        cwd=ROOT, env=environment(), stdout=sync_log, stderr=subprocess.STDOUT)
                    time.sleep(10)
                synchronizer.wait(timeout=30)
                sync_log.close()
            entry["status"] = "verifying"
            save_state(path, manifest)
            records, local = audit_local(run_dir)
            remote = audit_remote(client, run_dir, records)
            entry.update(status="complete", finished_at_utc=timestamp(), **local, **remote)
            save_state(output / "verification.json", dict(local, **remote))
            save_state(path, manifest)
            print(f"Completed {entry['name']}: {entry['url']}", flush=True)
        all_complete = all(entry["status"] == "complete" for entry in manifest["runs"])
        manifest.update(status="complete" if all_complete else "partial_complete")
        if all_complete:
            manifest["finished_at_utc"] = timestamp()
        save_state(path, manifest)
        print("All 11 models trained and verified" if all_complete else
              "Selected models trained and verified; other models remain unfinished", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_state(path, manifest)
        raise
    finally:
        client.close()
        lock.close()


if __name__ == "__main__":
    main()
