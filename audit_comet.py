"""Verify local epoch results in Comet and repair missing points without new experiments."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from sync_comet import CometRestClient, ROOT, comet_metric_names, read_completed_epoch_summaries, expected_metric_names, save_state
from omegaconf import OmegaConf


def audit(client, run, repair=False):
    state = json.loads((run / "comet_sync.json").read_text())
    config = OmegaConf.load(run / "config.yaml")
    summaries = read_completed_epoch_summaries(
        run, expected_metric_names(config), int(config.trainer.get("evaluation_period", 1)),
        int(config.trainer.n_epochs), expected_metric_names(config, False))
    points = client.request("GET", "experiment/metrics/get-all", {"experimentKey": state["experiment_key"]})["metrics"]
    lookup = {}
    for point in points:
        if point.get("epoch") is not None:
            lookup.setdefault((int(point["epoch"]), point["metricName"]), []).append(point)
    missing, repaired, verified = [], [], 0
    for summary in summaries:
        epoch = summary["epoch"]
        metrics = comet_metric_names(summary["metrics"], [p for p in config.datasets if p != "train"])
        invalid = {}
        for name, value in metrics.items():
            matches = lookup.get((epoch, name), [])
            if any(int(p.get("step", -1)) == epoch * state["epoch_len"] and
                   math.isclose(float(p["metricValue"]), value, rel_tol=1e-9, abs_tol=1e-9) for p in matches):
                verified += 1
            else:
                invalid[name] = value
        if invalid:
            if repair:
                client.log_metrics(state["experiment_key"], invalid, epoch * state["epoch_len"], epoch, summary["timestamp"])
                repaired.extend([epoch, name] for name in invalid)
            else:
                missing.extend([epoch, name] for name in invalid)
    return {"run": run.name, "url": state.get("url"), "local_epochs": max((s["epoch"] for s in summaries), default=0),
            "verified_metrics": verified, "remote_points": len(points), "missing": missing,
            "repaired": repaired, "passed": not missing and not repaired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--archive-unstarted", action="store_true")
    args = parser.parse_args()
    client = CometRestClient((ROOT / ".comet_key").read_text().strip(), "windows", timeout=20)
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "runs": [], "archived_unstarted": []}
    try:
        for path in sorted((ROOT / "saved").glob("*/epoch_metrics.jsonl")):
            if not (path.parent / "comet_sync.json").exists():
                continue
            result = audit(client, path.parent, args.repair)
            if result["repaired"]:
                result = audit(client, path.parent)
            report["runs"].append(result)
            print(json.dumps(result), flush=True)
        if args.archive_unstarted:
            manifest = json.loads((ROOT / "outputs/report_queue_20261015/manifest.json").read_text())
            for entry in manifest["runs"]:
                run = ROOT / "saved" / entry["run_name"]
                path = ROOT / "outputs/report_queue_20261015" / entry["name"] / "comet_logs_sync.json"
                if entry["status"] != "pending" or (run / "config.yaml").exists() or not path.exists():
                    continue
                state = json.loads(path.read_text())
                key = state.get("experiment_key")
                if not key:
                    continue
                summary = client.request("GET", "experiment/metrics/summary", {"experimentKey": key})
                if summary.get("values"):
                    continue
                client.request("GET", "write/experiment/archive", {"experimentKey": key})
                metadata = client.request("GET", "experiment/metadata", {"experimentKey": key})
                assert metadata["archived"]
                report["archived_unstarted"].append({"run": run.name, "experiment_key": key,
                                                     "reason": "No training epochs or metrics; retained preflight logs. Restored automatically when training starts."})
                print("Archived unstarted preflight:", run.name, flush=True)
        save_state(ROOT / "outputs/comet_logs/results_audit.json", report)
    finally:
        client.close()


if __name__ == "__main__":
    main()
