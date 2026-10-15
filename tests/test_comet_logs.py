import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from sync_comet import CometRestClient
from sync_comet_logs import output_lines, snapshot, sync_output, sync_run


class LogUploadTests(unittest.TestCase):
    def test_asset_is_acknowledged_only_after_download_checksum_matches(self):
        client = CometRestClient("test-key")
        client._request_packets = Mock(side_effect=[[{"assetId": "id"}], [{"sha256": "wrong", "bytes": 3}]])
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            client.upload_asset("experiment", "train.log", b"log")
        digest = hashlib.sha256(b"log").hexdigest()
        client._request_packets = Mock(side_effect=[[{"assetId": "id"}], [{"sha256": digest, "bytes": 3}]])
        self.assertTrue(client.upload_asset("experiment", "train.log", b"log")["verified"])

    def test_failed_asset_upload_is_retried_and_api_key_is_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "comet_sync.json").write_text(json.dumps({"experiment_key": "id", "url": "url"}))
            (run / "train.log").write_text("auth test-key\n")
            client = Mock(api_key="test-key")
            client.upload_asset.side_effect = RuntimeError("offline")
            self.assertEqual(sync_run(client, run, run)["status"], "pending")
            state = json.loads((run / "comet_logs_sync.json").read_text())
            self.assertEqual(state["assets"], {})
            client.upload_asset.side_effect = lambda key, name, data: {
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "verified": True}
            self.assertEqual(sync_run(client, run, run)["status"], "synced")
            self.assertEqual(client.upload_asset.call_args.args[2], b"auth [REDACTED]\n")
            count = client.upload_asset.call_count
            sync_run(client, run, run)
            self.assertEqual(client.upload_asset.call_count, count)

    def test_output_resumes_after_last_ack_and_appends_new_complete_lines(self):
        state, remote, posts = {}, {}, []
        lines = [f"line {i}" for i in range(450)]
        fail = [True]

        def request(method, endpoint, payload):
            if method == "GET":
                return {"output": "".join(remote[k] for k in sorted(remote))}
            if len(posts) == 1 and fail[0]:
                fail[0] = False
                raise RuntimeError("offline")
            posts.append(payload)
            remote.update({line["offset"]: line["output"] for line in payload["outputLines"]})

        client = Mock()
        client.request.side_effect = request
        with self.assertRaises(RuntimeError):
            sync_output(client, "id", lines, state, lambda: None)
        self.assertEqual(state["output"]["synced_lines"], 200)
        sync_output(client, "id", lines, state, lambda: None)
        self.assertEqual(state["output"]["verified_lines"], 450)
        self.assertEqual(len(posts), 3)
        sync_output(client, "id", lines + ["new"], state, lambda: None)
        self.assertEqual(len(posts), 4)
        self.assertEqual(posts[-1]["outputLines"][0]["offset"], 450)
        self.assertEqual(posts[-1]["outputLines"][0]["output"], "new\n")
        self.assertEqual(output_lines(b"ready\npartial"), ["ready"])

    def test_existing_console_log_is_verified_without_replaying_it(self):
        client = Mock()
        client.request.return_value = {"output": "old SDK output\na\nb\ntraceback"}
        state = {}
        sync_output(client, "id", ["a", "b"], state, lambda: None)
        self.assertEqual(client.request.call_count, 1)
        self.assertEqual(state["output"]["verified_lines"], 2)

    def test_recovered_preflight_and_training_logs_keep_both_output_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, output = root / "saved/model", root / "outputs/model"
            output.mkdir(parents=True)
            (output / "preflight.log").write_text("preflight failed\nCUDA unavailable\n")
            (output / "comet_logs_sync.json").write_text(json.dumps({"experiment_key": "id", "url": "url"}))
            remote = {}

            def request(method, endpoint, payload):
                if endpoint == "experiment/output":
                    return {"output": "".join(remote[k] for k in sorted(remote))}
                if endpoint == "write/experiment/output":
                    remote.update({p["offset"]: p["output"] for p in payload["outputLines"]})
                return {}

            client = Mock(api_key="test-key")
            client.request.side_effect = request
            client.upload_asset.side_effect = lambda key, name, data: {
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "verified": True}
            self.assertEqual(sync_run(client, run, output)["status"], "synced")
            run.mkdir(parents=True)
            (run / "config.yaml").write_text("trainer: {}\n")
            (run / "info.log").write_text("training started\nepoch 1\n")
            self.assertEqual(sync_run(client, run, output)["status"], "synced")
            state = json.loads((output / "comet_logs_sync.json").read_text())
            self.assertEqual(state["output_source"], "info.log")
            self.assertEqual(state["output_history"][0]["source"], "preflight.log")
            self.assertEqual(state["output"]["verified_lines"], 2)
            self.assertEqual("".join(remote[k] for k in sorted(remote)),
                             "preflight failed\nCUDA unavailable\ntraining started\nepoch 1\n")


if __name__ == "__main__":
    unittest.main()
