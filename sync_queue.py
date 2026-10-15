"""Retry Comet uploads independently from the local training queue."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from sync_comet import save_state

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.queue_dir.resolve()
    lock = (directory / "publisher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        manifest = json.loads((directory / "manifest.json").read_text())
        progress = []
        for entry in manifest["runs"]:
            run = ROOT / "saved" / entry["run_name"]
            item = {"name": entry["name"], "status": "waiting_training", "last_epoch": 0}
            progress.append(item)
            if not (run / "config.yaml").exists():
                continue
            rows = run / "epoch_metrics.jsonl"
            completed = 0
            if rows.exists():
                completed = max((json.loads(s)["epoch"] for s in rows.read_text().splitlines() if s.strip()), default=0)
            state_path = run / "comet_sync.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not state.get("url") or state.get("error") or state.get("last_epoch", 0) < completed:
                output = directory / entry["name"]
                output.mkdir(exist_ok=True)
                try:
                    command = [sys.executable, "-u", str(ROOT / "sync_comet.py"),
                               "--run-dir", str(run), "--epoch-len", "338", "--transport", "windows"]
                    log_state_path = output / "comet_logs_sync.json"
                    if not state.get("experiment_key") and log_state_path.exists():
                        log_state = json.loads(log_state_path.read_text())
                        if log_state.get("experiment_key"):
                            command.extend(["--experiment-key", log_state["experiment_key"]])
                    with (output / "comet.log").open("a") as log:
                        result = subprocess.run(command, cwd=ROOT,
                                                stdout=log, stderr=subprocess.STDOUT, timeout=900)
                    item["last_exit_code"] = result.returncode
                except subprocess.TimeoutExpired:
                    item["last_exit_code"] = "timeout"
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
            item.update(last_epoch=state.get("last_epoch", 0), local_epochs=completed, url=state.get("url"),
                        status="synced" if state.get("url") and not state.get("error") and
                        state.get("last_epoch", 0) >= completed else "pending")
        report = {"time_utc": datetime.now(timezone.utc).isoformat(), "runs": progress}
        save_state(directory / "comet_status.json", report)
        finished = manifest["status"] in ("trained", "finished_with_errors")
        if finished and all(r["status"] == "synced" or
                            (e["status"] == "failed" and not (ROOT / "saved" / e["run_name"]).exists())
                            for r, e in zip(progress, manifest["runs"])):
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
