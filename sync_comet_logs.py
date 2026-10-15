"""Upload local training logs to existing Comet experiments, independently of training."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time

from omegaconf import OmegaConf

from sync_comet import CometRestClient, ROOT, save_state


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def discover_runs(root=ROOT):
    outputs = {}
    failed = {}
    for manifest in sorted((root / "outputs").glob("*/manifest.json")):
        for entry in json.loads(manifest.read_text()).get("runs", []):
            outputs[entry["run_name"]] = manifest.parent / entry["name"]
            if (outputs[entry["run_name"]] / "preflight.log").exists() and (
                entry["status"] == "failed" or (outputs[entry["run_name"]] / "comet_logs_sync.json").exists()
            ):
                failed[entry["run_name"]] = outputs[entry["run_name"]]
    outputs.update({
        "dfnet_20261013": root / "outputs/dfnet_launch",
        "dfnet_audited_20261013": root / "outputs/dfnet_restart",
        "dfnet_full_20261014_interrupted_0130": root / "outputs/full_recordings_20261014/interrupted_start_20261014_0130/dfnet",
    })
    runs = []
    for path in sorted((root / "saved").glob("*/config.yaml")):
        config = OmegaConf.load(path)
        if config.writer.get("project_name") == "speech_enhancement":
            runs.append((path.parent, outputs.get(path.parent.name)))
    present = {run.name for run, output in runs}
    runs.extend((root / "saved" / name, output) for name, output in failed.items() if name not in present)
    return runs


def log_files(run, output):
    files = {name: run / name for name in
             ("info.log", "epoch_metrics.jsonl", "config.yaml", "training_result.json")}
    if output:
        files.update({name: output / name for name in
                      ("train.log", "monitor.jsonl", "preflight.log", "preflight.json", "verification.json")})
        files["comet_sync.log"] = output / "comet.log"
    return {name: path for name, path in files.items() if path.is_file()}


def snapshot(path, api_key):
    # A running logger may append while uploading; verify the exact bytes captured here.
    return path.read_bytes().replace(api_key.encode(), b"[REDACTED]")


def output_lines(data):
    text = data.decode("utf-8", errors="replace")
    # Never acknowledge a partial final line while the trainer is still writing it.
    complete = text.rsplit("\n", 1)[0] if "\n" in text else ""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", complete).splitlines()


def prefix_digest(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def experiment_for(run, output):
    for state in (run / "comet_sync.json", (output or run) / "comet_logs_sync.json"):
        if state.exists():
            item = json.loads(state.read_text())
            if item.get("experiment_key"):
                return item["experiment_key"], item.get("url")
    # The first baseline used Comet directly; its existing link is in stdout.
    for path in (output / "train.log" if output else None, run / "info.log"):
        if path and path.exists():
            match = re.search(r"https://www\.comet\.(?:com|ml)/([^/\s]+)/speech-enhancement/([a-zA-Z0-9]+)",
                              path.read_text(errors="replace"))
            if match:
                return match[2], f"https://www.comet.com/{match[1]}/speech-enhancement/{match[2]}"
    return None, None


def training_status(run, output):
    if output and (output.parent / "manifest.json").exists():
        manifest = json.loads((output.parent / "manifest.json").read_text())
        for entry in manifest.get("runs", []):
            if entry["run_name"] == run.name:
                if entry["status"] == "pending" and entry.get("attempts"):
                    return "queued_retry"
                if entry["status"] == "failed" and not (run / "config.yaml").exists():
                    return "failed_preflight"
                return entry["status"]
    if (run / "training_result.json").exists():
        return "trained"
    return None


def sync_output(client, experiment_key, lines, state, save):
    progress = state.setdefault("output", {})
    if progress and progress.get("format_version") != 2:
        previous_end = progress.get("offset_base", 0) + progress.get("synced_lines", 0)
        progress.clear()
        progress.update(format_version=2, offset_base=max(1_000_000, previous_end + 1))
        save()
    progress["format_version"] = 2
    count = progress.get("synced_lines", 0)
    if count > len(lines) or (count and progress.get("prefix_sha256") != prefix_digest(lines[:count])):
        raise ValueError("The previously uploaded training log prefix has changed")
    if "offset_base" not in progress:
        remote = client.request("GET", "experiment/output", {"experimentKey": experiment_key}).get("output", "")
        if lines and "\n".join(lines) in remote:
            progress.update(synced_lines=len(lines), prefix_sha256=prefix_digest(lines),
                            verified_lines=len(lines), offset_base=0)
            save()
            return
        # Keep replayed logs after any output already captured by the original SDK.
        progress["offset_base"] = 1_000_000 if remote else 0
        save()
    for start in range(count, len(lines), 200):
        chunk = lines[start:start + 200]
        client.request("POST", "write/experiment/output", {
            "experimentKey": experiment_key,
            "outputLines": [{"output": line + "\n", "stderr": False,
                             "localTimestamp": int(time.time() * 1000),
                             "offset": progress["offset_base"] + start + i}
                            for i, line in enumerate(chunk)],
        })
        count = start + len(chunk)
        progress.update(synced_lines=count, prefix_sha256=prefix_digest(lines[:count]))
        save()
    if lines and progress.get("verified_lines", 0) != len(lines):
        remote = client.request("GET", "experiment/output", {"experimentKey": experiment_key}).get("output", "")
        if "\n".join(lines) not in remote:
            raise RuntimeError("Comet Output does not contain the complete training log yet")
        progress["verified_lines"] = len(lines)
        save()


def sync_run(client, run, output):
    path = (output or run) / "comet_logs_sync.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.setdefault("assets", {})
    save = lambda: save_state(path, state)
    url = state.get("url")
    try:
        experiment_key, url = experiment_for(run, output)
        if not experiment_key:
            if (run / "config.yaml").exists() or not output or not (output / "preflight.log").exists():
                return {"run": run.name, "status": "waiting_experiment"}
            # Failed GPU preflights never reached trainer initialization, but their logs matter too.
            created = client.request("POST", "write/experiment/create", {
                "workspaceName": "runtime57", "projectName": "speech_enhancement", "experimentName": run.name})
            experiment_key = created["experimentKey"]
            state.update(experiment_key=experiment_key, run_dir=str(run.resolve()))
            save()
        if state.get("experiment_key") and state["experiment_key"] != experiment_key:
            raise ValueError("Log upload state belongs to another Comet experiment")
        if not url:
            metadata = client.request("GET", "experiment/metadata", {"experimentKey": experiment_key})
            url = f"https://www.comet.com/{metadata['workspaceName']}/{metadata['projectName']}/{experiment_key}"
        state.update(run_dir=str(run.resolve()), experiment_key=experiment_key, url=url)
        save()
        status = training_status(run, output)
        if status and state.get("training_status") != status:
            client.request("POST", "write/experiment/log-other", {
                "experimentKey": experiment_key, "key": "training_status", "value": status})
            state["training_status"] = status
            save()
        files = log_files(run, output)
        captured_info = None
        captured_source = None
        previous_source = state.get("output_source")
        if not previous_source and state.get("output"):
            previous_source = "info.log" if "info.log" in state["assets"] else "preflight.log"
        for name, source in files.items():
            data = snapshot(source, client.api_key)
            digest = hashlib.sha256(data).hexdigest()
            if name == "info.log" or (name == "preflight.log" and not (run / "info.log").exists()):
                captured_info = data
                captured_source = name
            previous = state["assets"].get(name, {})
            if previous.get("sha256") == digest and previous.get("verified"):
                continue
            item = client.upload_asset(experiment_key, name, data)
            state["assets"][name] = dict(item, uploaded_at_utc=timestamp())
            save()
            print(f"{run.name}: verified {name} ({len(data)} bytes)", flush=True)
        if captured_info is not None:
            if previous_source and previous_source != captured_source:
                prior = state.get("output", {})
                state.setdefault("output_history", []).append(dict(prior, source=previous_source))
                state["output"] = {"format_version": 2,
                                   "offset_base": prior.get("offset_base", 0) + prior.get("synced_lines", 0) + 1}
            state["output_source"] = captured_source
            sync_output(client, experiment_key, output_lines(captured_info), state, save)
        state.update(error=None, last_success_utc=timestamp())
        save()
        return {"run": run.name, "status": "synced", "url": url,
                "assets": len(state["assets"]), "output_lines": state.get("output", {}).get("verified_lines", 0)}
    except Exception as error:
        state.update(error=type(error).__name__, last_attempt_utc=timestamp())
        save()
        print(f"{run.name}: log upload pending ({type(error).__name__})", flush=True)
        return {"run": run.name, "status": "pending", "url": url, "error": type(error).__name__}


def queue_finished(root=ROOT):
    path = root / "outputs/report_queue_20261015/manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    if manifest["status"] not in ("trained", "finished_with_errors"):
        return False
    for entry in manifest["runs"]:
        run = root / "saved" / entry["run_name"]
        journal = run / "epoch_metrics.jsonl"
        if not journal.exists():
            if entry["status"] == "failed":
                continue
            return False
        completed = max((json.loads(s)["epoch"] for s in journal.read_text().splitlines() if s.strip()), default=0)
        path = run / "comet_sync.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get("error") or state.get("last_epoch", 0) < completed:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=180)
    parser.add_argument("--transport", choices=["native", "windows"], default="windows")
    args = parser.parse_args()
    directory = ROOT / "outputs/comet_logs"
    directory.mkdir(exist_ok=True)
    lock = (directory / "publisher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key = (ROOT / ".comet_key").read_text().strip()
    while True:
        runs = discover_runs()
        progress = [{"run": run.name, "status": "queued", "url": experiment_for(run, output)[1]}
                    for run, output in runs]
        report = {"time_utc": timestamp(), "runs": progress}
        save_state(directory / "status.json", report)
        client = CometRestClient(key, args.transport, timeout=20)
        try:
            for i, (run, output) in enumerate(runs):
                progress[i] = sync_run(client, run, output)
                report["time_utc"] = timestamp()
                save_state(directory / "status.json", report)
        finally:
            client.close()
        if not args.follow or (queue_finished() and all(p["status"] == "synced" for p in progress)):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
