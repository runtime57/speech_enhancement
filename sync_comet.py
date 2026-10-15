"""Replay and follow complete epoch summaries without changing a running trainer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import time

from omegaconf import OmegaConf
import requests


ROOT = Path(__file__).resolve().parent
BASE_URL = "https://www.comet.com/api/rest/v2/"

# The API key and request bodies travel over stdin, never the process command line.
# Windows networking can be used when WSL cannot reach Comet directly.
WINDOWS_WORKER = r'''
import base64, concurrent.futures, hashlib, json, sys, threading
import requests
local = threading.local()
def send(command):
    if not hasattr(local, "session"):
        local.session = requests.Session()
        local.session.trust_env = False
    try:
        digest = command.pop("_response_digest", False)
        asset = command.pop("_asset", None)
        if asset:
            command["files"] = {"file": (asset["name"], base64.b64decode(asset["data"]), "application/octet-stream")}
        response = local.session.request(**command)
        body = {"sha256": hashlib.sha256(response.content).hexdigest(), "bytes": len(response.content)} if digest else response.json()
        return {"status": response.status_code, "body": body}
    except Exception as error:
        return {"status": 0, "error": type(error).__name__}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for line in sys.stdin:
        commands = json.loads(line)
        print(json.dumps(list(pool.map(send, commands))), flush=True)
'''


class CometRestClient:
    def __init__(self, api_key, transport="native", timeout=15):
        if not api_key or "\n" in api_key or "\r" in api_key:
            raise ValueError("A nonempty Comet API key is required")
        self.api_key = api_key
        self.timeout = timeout
        self.worker = None
        if transport == "windows":
            executable = "/mnt/c/Program Files/Python311/python.exe"
            if not Path(executable).is_file():
                raise ValueError("Windows Python with requests is unavailable")
            self.worker = subprocess.Popen(
                [executable, "-u", "-c", WINDOWS_WORKER], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                encoding="utf-8", bufsize=1,
            )

    def request_many(self, commands):
        packets = [dict(method=method, url=BASE_URL + endpoint,
                        params=payload if method == "GET" else None,
                        json=payload if method != "GET" else None,
                        headers={"Authorization": self.api_key}, timeout=self.timeout)
                   for method, endpoint, payload in commands]
        return self._request_packets(packets)

    def _request_packets(self, packets):
        if self.worker is not None:
            if self.worker.poll() is not None:
                raise RuntimeError("Windows Comet transport has exited")
            self.worker.stdin.write(json.dumps(packets, allow_nan=False) + "\n")
            self.worker.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.worker.stdout, selectors.EVENT_READ)
                packet_timeout = max((packet.get("timeout", self.timeout) for packet in packets), default=self.timeout)
                if not selector.select(packet_timeout * math.ceil(len(packets) / 4) + 5):
                    self.worker.terminate()
                    raise TimeoutError("Windows Comet transport timed out")
            responses = json.loads(self.worker.stdout.readline())
        else:
            def send(packet):
                digest = packet.pop("_response_digest", False)
                asset = packet.pop("_asset", None)
                if asset:
                    packet["files"] = {"file": (asset["name"], base64.b64decode(asset["data"]), "application/octet-stream")}
                response = requests.request(**packet)
                body = {"sha256": hashlib.sha256(response.content).hexdigest(), "bytes": len(response.content)} if digest else response.json()
                return {"status": response.status_code, "body": body}
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(pool.map(send, packets))
        for response in responses:
            if response["status"] != 200:
                raise RuntimeError(f"Comet request failed (HTTP {response['status']})")
            body = response.get("body")
            if isinstance(body, dict) and body.get("code", 200) not in (0, 200):
                raise RuntimeError(f"Comet rejected request (code {body['code']})")
        return [response["body"] for response in responses]

    def request(self, method, endpoint, payload=None):
        return self.request_many([(method, endpoint, payload)])[0]

    def upload_asset(self, experiment_key, name, data):
        """Overwrite a named log asset and verify its downloaded bytes."""
        packet = dict(method="POST", url=BASE_URL + "write/experiment/upload-asset",
                      params={"experimentKey": experiment_key, "fileName": name,
                              "extension": Path(name).suffix, "type": "asset", "overwrite": "true"},
                      headers={"Authorization": self.api_key}, timeout=max(40, self.timeout),
                      _asset={"name": name, "data": base64.b64encode(data).decode()})
        uploaded = self._request_packets([packet])[0]
        asset_id = uploaded.get("assetId")
        if not asset_id:
            listing = self.request("GET", "experiment/asset/list", {"experimentKey": experiment_key, "type": "asset"})
            assets = [a for a in listing["assets"] if a["fileName"] == name]
            if not assets:
                raise RuntimeError("Uploaded log asset is absent from Comet")
            asset_id = assets[-1]["assetId"]
        downloaded = self._request_packets([dict(
            method="GET", url=BASE_URL + "experiment/asset/get-asset",
            params={"experimentKey": experiment_key, "assetId": asset_id},
            headers={"Authorization": self.api_key}, timeout=max(40, self.timeout),
            _response_digest=True)])[0]
        if downloaded["sha256"] != hashlib.sha256(data).hexdigest() or downloaded["bytes"] != len(data):
            raise RuntimeError("Comet log asset checksum mismatch")
        return dict(name=name, asset_id=asset_id, verified=True, **downloaded)

    def log_metrics(self, experiment_key, metrics, step, epoch, timestamp):
        return self.request_many([
            ("POST", "write/experiment/metric", {
                "experimentKey": experiment_key, "metricName": name,
                "metricValue": value, "step": step, "epoch": epoch,
                "timestamp": timestamp,
            }) for name, value in metrics.items()
        ])

    def close(self):
        if self.worker is not None:
            self.worker.stdin.close()
            try:
                self.worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                self.worker.wait(timeout=3)


def expected_metric_names(config, include_evaluation=True):
    losses = list(config.writer.loss_names)
    names = set(losses + ["grad_norm"])
    names.update(metric.name for metric in config.metrics.get("train", []))
    if config.trainer.get("record_epoch_counts", False):
        names.update(["samples", "batches", "seconds"])
    for part in config.trainer.get("evaluation_partitions", ["val", "test"]) if include_evaluation else []:
        if part in config.datasets and part != "train":
            names.update(f"{part}_{name}" for name in losses)
            names.update(f"{part}_{metric.name}" for metric in config.metrics.inference)
            if config.trainer.get("record_epoch_counts", False):
                names.update([f"{part}_samples", f"{part}_seconds"])
    return names


def read_epoch_summaries(log_path, required_names):
    """Only expose a summary when all expected metrics have been written."""
    line_pattern = re.compile(
        r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .* - INFO -\s+(\w+)\s+:\s+(\S+)\s*$"
    )
    summaries, current = [], None
    for line in log_path.read_text().splitlines():
        match = line_pattern.match(line)
        if not match:
            continue
        stamp, name, value = match.groups()
        if name == "epoch":
            current = {"epoch": int(value), "metrics": {},
                       "timestamp": int(datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f").timestamp() * 1000)}
        elif current is not None and name in required_names:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Nonfinite metric in epoch {current['epoch']}: {name}")
            current["metrics"][name] = number
            if required_names <= current["metrics"].keys():
                summaries.append(current)
                current = None
    return summaries


def comet_metric_names(metrics, partitions):
    result = {}
    for name, value in metrics.items():
        for part in partitions:
            if name.startswith(part + "_"):
                result[f"{name[len(part) + 1:]}_{part}"] = value
                break
        else:
            result[f"{name}_train"] = value
    return result


def read_completed_epoch_summaries(run_dir, required_names, evaluation_period=1,
                                   final_epoch=None, train_names=None):
    """Prefer committed epoch records so a later failed epoch cannot block replay."""
    records_path = run_dir / "epoch_metrics.jsonl"
    if not records_path.exists():
        return read_epoch_summaries(run_dir / "info.log", required_names)
    timestamps = {}
    pattern = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .* - INFO -\s+epoch\s+:\s+(\d+)\s*$")
    for line in (run_dir / "info.log").read_text().splitlines():
        match = pattern.match(line)
        if match:
            stamp, epoch = match.groups()
            timestamps[int(epoch)] = int(datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f").timestamp() * 1000)
    summaries = []
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        epoch = int(record.pop("epoch"))
        required = required_names if epoch % evaluation_period == 0 or epoch == final_epoch else train_names
        if required is None:
            raise ValueError("Train metric names are required for sparse evaluation")
        if not required <= record.keys():
            raise ValueError(f"Incomplete committed epoch {epoch}")
        metrics = {name: record[name] for name in required}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Nonfinite committed epoch {epoch}")
        summaries.append({"epoch": epoch, "metrics": metrics, "timestamp": timestamps[epoch]})
    return summaries


def infer_epoch_len(run_dir, config):
    if config.trainer.get("epoch_len") is not None:
        return int(config.trainer.epoch_len)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(run_dir / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    markers = events.Scalars("epoch_train")
    for first, second in zip(markers, markers[1:]):
        epoch_delta = round(second.value - first.value)
        step_delta = second.step - first.step
        if epoch_delta > 0 and step_delta > 0 and step_delta % epoch_delta == 0:
            return step_delta // epoch_delta
    raise ValueError("Cannot infer epoch length yet; supply --epoch-len")


def save_state(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def sync_summaries(client, state, state_path, summaries, epoch_len, partitions, prefix=""):
    for summary in summaries:
        epoch = summary["epoch"]
        if epoch <= state.get("last_epoch", 0):
            continue
        metrics = {prefix + name: value for name, value in
                   comet_metric_names(summary["metrics"], partitions).items()}
        client.log_metrics(state["experiment_key"], metrics, step=epoch * epoch_len,
                           epoch=epoch, timestamp=summary["timestamp"])
        # Advance only after the server acknowledges every metric in this epoch.
        state["last_epoch"] = epoch
        state["metrics_per_epoch"] = len(metrics)
        state["last_success_utc"] = datetime.now(timezone.utc).isoformat()
        state["error"] = None
        save_state(state_path, state)
        print(f"Comet: epoch {epoch}, {len(metrics)} metrics synced", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--project", default="speech_enhancement")
    parser.add_argument("--workspace")
    parser.add_argument("--experiment-key")
    parser.add_argument("--epoch-len", type=int)
    parser.add_argument("--metric-prefix", default="")
    parser.add_argument("--transport", choices=["native", "windows"], default="native")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    config = OmegaConf.load(args.run_dir / "config.yaml")
    required = expected_metric_names(config)
    train_names = expected_metric_names(config, include_evaluation=False)
    partitions = [part for part in config.datasets if part != "train"]
    epoch_len = args.epoch_len or infer_epoch_len(args.run_dir, config)
    if epoch_len <= 0:
        raise ValueError("Epoch length must be positive")
    state_path = args.state_file or args.run_dir / "comet_sync.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    # Guard against accidentally replaying another run into a saved experiment.
    identity = {"run_dir": str(args.run_dir.resolve()), "epoch_len": epoch_len,
                "metric_prefix": args.metric_prefix}
    if state and any(state.get(key) != value for key, value in identity.items()):
        raise ValueError("Saved sync state belongs to a different run or mapping")
    key = os.environ.get("COMET_API_KEY") or (ROOT / ".comet_key").read_text().strip()
    client = CometRestClient(key, args.transport)
    try:
        if not state.get("experiment_key"):
            workspace = args.workspace or client.request("GET", "account-details")["defaultWorkspaceName"]
            experiment_key = args.experiment_key
            if experiment_key is None:
                created = client.request("POST", "write/experiment/create", {
                    "workspaceName": workspace, "projectName": args.project,
                    "experimentName": config.writer.run_name,
                })
                experiment_key = created["experimentKey"]
            state = dict(identity, experiment_key=experiment_key, last_epoch=0)
            save_state(state_path, state)
        if not state.get("url"):
            experiment_key = state["experiment_key"]
            metadata = client.request("GET", "experiment/metadata", {"experimentKey": experiment_key})
            state["url"] = f"https://www.comet.com/{metadata['workspaceName']}/{metadata['projectName']}/{experiment_key}"
            save_state(state_path, state)
            client.request("POST", "write/experiment/log-other", {
                "experimentKey": experiment_key, "key": "epoch_metrics_source",
                "value": "Recorded epoch summaries from local info.log; explicit step and epoch",
            })
        print(state["url"], flush=True)
        while True:
            try:
                summaries = read_completed_epoch_summaries(
                    args.run_dir, required, int(config.trainer.get("evaluation_period", 1)),
                    int(config.trainer.n_epochs), train_names)
                sync_summaries(client, state, state_path, summaries, epoch_len,
                               partitions, args.metric_prefix)
            except (requests.RequestException, RuntimeError, TimeoutError, ValueError) as error:
                # Local training never waits for this process or the network.
                state["error"] = type(error).__name__
                save_state(state_path, state)
                print(f"Comet sync pending: {type(error).__name__}", flush=True)
                if not args.follow:
                    raise
                if client.worker is not None and client.worker.poll() is not None:
                    client.close()
                    client = CometRestClient(key, args.transport)
            if not args.follow:
                break
            if args.pid_file is not None:
                pid = int(args.pid_file.read_text().strip())
                if not Path(f"/proc/{pid}").exists() and state.get("error") is None:
                    break
            time.sleep(args.poll_seconds)
    finally:
        client.close()


if __name__ == "__main__":
    main()
