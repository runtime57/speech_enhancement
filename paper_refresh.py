"""Refresh and publish paper artifacts when a new complete epoch-50 checkpoint appears."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from benchmark_cpu import MODELS, ROOT, file_hash
from sync_comet import save_state


ARTIFACTS = ["benchmarks/cpu_speech_enhancement_20261015.json",
             "benchmarks/training_curves_20261015.json", "paper/results/cpu_benchmark_20261015.md",
             "paper/graphics/cpu_quality_tradeoff_20261015.svg", "paper/graphics/cpu_quality_tradeoff_20261015.png",
             "paper/graphics/learning_curves_20261015.svg", "paper/graphics/learning_curves_20261015.png"]


def checkpoint_signature():
    result = {}
    for name, _, _, run in MODELS:
        directory = ROOT / "saved" / run
        checkpoint, journal = directory / "checkpoint-epoch50.pth", directory / "epoch_metrics.jsonl"
        if not checkpoint.exists() or not journal.exists():
            continue
        if any(json.loads(s)["epoch"] == 50 for s in journal.read_text().splitlines() if s.strip()):
            result[name] = file_hash(checkpoint)
    return result


def github_environment():
    env = dict(os.environ)
    # Use existing Linux SSH keys. Only TCP transport runs through Windows if WSL networking is unavailable.
    if Path("/mnt/c/Program Files/Python311/python.exe").exists():
        directory = ROOT / "outputs/paper_exports"
        source = directory / "github_tcp_bridge.py"
        source.write_text('''import os,socket,sys,threading,msvcrt
msvcrt.setmode(sys.stdin.fileno(),os.O_BINARY)
msvcrt.setmode(sys.stdout.fileno(),os.O_BINARY)
s=socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=20)
s.settimeout(None)
def upstream():
 try:
  while True:
   data=os.read(sys.stdin.fileno(),65536)
   if not data:break
   s.sendall(data)
  s.shutdown(socket.SHUT_WR)
 except OSError:pass
threading.Thread(target=upstream,daemon=True).start()
try:
 while True:
  data=s.recv(65536)
  if not data:break
  sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
finally:s.close()
''')
        winpath = subprocess.check_output(["wslpath", "-w", str(source)], text=True).strip()
        wrapper = directory / "github-ssh-bridge"
        wrapper.write_text("#!/bin/sh\nexec " + shlex.quote("/mnt/c/Program Files/Python311/python.exe") +
                           " -u " + shlex.quote(winpath) + ' "$@"\n')
        wrapper.chmod(0o700)
        proxy = f"ProxyCommand={shlex.quote(str(wrapper))} %h %p"
        env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes -o ConnectTimeout=20 -o " + shlex.quote(proxy)
    return env


def main():
    directory = ROOT / "outputs/paper_exports"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "refresh.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refresh(directory)


def refresh(directory):
    path = directory / "refresh_status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    signature = checkpoint_signature()
    previous = state.get("published_checkpoint_signature", {})
    if not signature or any(signature.get(name) != checksum for name, checksum in previous.items()):
        state.update(status="waiting_checkpoints", error="Previously published checkpoints are missing or changed" if previous else None)
        save_state(path, state)
        return
    if state.get("published_checkpoint_signature") == signature:
        return
    try:
        state.update(status="refreshing", started_at_utc=datetime.now(timezone.utc).isoformat())
        save_state(path, state)
        with (directory / "artifacts.lock").open("w") as artifacts:
            fcntl.flock(artifacts, fcntl.LOCK_EX)
            env = github_environment()
            checkout = directory / "publish_checkout"
            if state.get("generated_checkpoint_signature") != signature:
                subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, env=env, check=True, timeout=90)
                for script in ("benchmark_cpu.py", "paper_results.py"):
                    published_source = subprocess.check_output(["git", "show", f"origin/main:{script}"], cwd=ROOT)
                    if published_source != (ROOT / script).read_bytes():
                        raise ValueError("Publish changed measurement/rendering code before refreshing data")
                for script in ("benchmark_cpu.py", "paper_results.py"):
                    subprocess.run([sys.executable, "-u", str(ROOT / script)], cwd=ROOT, check=True)
                if not checkout.exists():
                    subprocess.run(["git", "worktree", "add", "--detach", str(checkout), "origin/main"], cwd=ROOT, check=True)
                else:
                    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True)
                    if dirty:
                        raise ValueError("Publication checkout has unexpected local changes")
                    subprocess.run(["git", "checkout", "--detach", "origin/main"], cwd=checkout, check=True)
                for artifact in ARTIFACTS:
                    target = checkout / artifact
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / artifact, target)
                subprocess.run(["git", "add", "--", *ARTIFACTS], cwd=checkout, check=True)
                changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *ARTIFACTS], cwd=checkout).returncode
                if changed:
                    subprocess.run(["git", "diff", "--cached", "--check", "--", *ARTIFACTS], cwd=checkout, check=True)
                    subprocess.run(["git", "commit", "-m", f"Update paper results for {len(signature)} completed models"],
                                   cwd=checkout, check=True)
                state.update(generated_checkpoint_signature=signature,
                             generated_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip())
                save_state(path, state)
            local = state["generated_commit"]
            subprocess.run(["git", "push", "origin", local + ":refs/heads/main"], cwd=checkout, env=env, check=True, timeout=120)
            remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"],
                                             cwd=ROOT, env=env, text=True, timeout=60).split()[0]
            assert local == remote
        state.update(status="published", published_checkpoint_signature=signature, commit=local,
                     updated_at_utc=datetime.now(timezone.utc).isoformat(), error=None)
        save_state(path, state)
        print("Published", len(signature), "completed model results", local, flush=True)
    except Exception as error:
        state.update(status="pending", error=type(error).__name__)
        save_state(path, state)
        raise


if __name__ == "__main__":
    main()
