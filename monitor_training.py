"""Monitor a training process without interrupting it; tolerate GPU probe failures."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()
    pid = int(args.pid_file.read_text())
    start_identity = Path(f"/proc/{pid}/stat").read_text().split()[21]
    with (args.output_dir / "monitor.jsonl").open("a", buffering=1) as output:
        while True:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().split()
                running = state[2] not in ("Z", "X") and state[21] == start_identity
            except FileNotFoundError:
                running = False
            log_path = args.output_dir / "train.log"
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 100000))
                tail = log.read().decode(errors="replace")
            progress = re.findall(r"(train|val|test):\s+(\d+)%[^\r\n]*?\| (\d+)/(\d+)", tail)
            epochs = re.findall(r"Train Epoch: (\d+)", tail)
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=8, check=True).stdout.strip()
                gpu_error = None
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as error:
                gpu, gpu_error = None, type(error).__name__
            checkpoints = [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime}
                           for p in args.run_dir.glob("*.pth")]
            record = {"time_utc": datetime.now(timezone.utc).isoformat(), "pid": pid,
                      "running": running, "run_dir": str(args.run_dir),
                      "epoch": int(epochs[-1]) if epochs else None,
                      "progress": list(progress[-1][:2]) if progress else None,
                      "iteration": int(progress[-1][2]) if progress else None,
                      "iterations_total": int(progress[-1][3]) if progress else None,
                      "gpu": gpu, "gpu_error": gpu_error, "checkpoints": checkpoints,
                      "error": bool(re.search(r"Traceback|OutOfMemoryError|Loss:\s*(nan|inf)", tail))}
            exit_file = args.output_dir / "train.exit"
            if not running and exit_file.exists():
                record["exit_code"] = int(exit_file.read_text())
            output.write(json.dumps(record) + "\n")
            temporary = args.status_file.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(args.status_file)
            if not running:
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
