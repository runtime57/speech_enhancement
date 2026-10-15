import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from omegaconf import OmegaConf

from src.logger.cometml import CometMLWriter
from sync_comet import (comet_metric_names, expected_metric_names,
                        read_epoch_summaries, read_completed_epoch_summaries, sync_summaries)


class CometLoggingTests(unittest.TestCase):
    def test_sparse_evaluation_uploads_train_every_epoch_and_eval_on_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = "2026-10-15 00:00:00,123 - train - INFO -     "
            (root / "info.log").write_text("".join(prefix + f"epoch : {e}\n" for e in range(1, 8)))
            records = [{"epoch": e, "loss": .5, **({"val_PESQ": 2.5} if e in (5, 7) else {})}
                       for e in range(1, 8)]
            (root / "epoch_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            rows = read_completed_epoch_summaries(root, {"loss", "val_PESQ"}, 5, 7, {"loss"})
            client = Mock()
            state = {"experiment_key": "fake", "last_epoch": 0}
            sync_summaries(client, state, root / "state.json", rows, 338, ["val"])
            self.assertEqual(client.log_metrics.call_count, 7)
            for call in client.log_metrics.call_args_list:
                epoch = call.kwargs["epoch"]
                self.assertEqual(call.kwargs["step"], epoch * 338)
                self.assertEqual("PESQ_val" in call.args[1], epoch in (5, 7))

    def test_failed_later_epoch_does_not_block_committed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = "2026-10-14 00:00:00,123 - train - INFO -     "
            (root / "info.log").write_text(prefix + "epoch : 1\n" + prefix + "loss : 0.25\n" +
                                           prefix + "epoch : 2\n" + prefix + "loss : inf\n")
            (root / "epoch_metrics.jsonl").write_text(json.dumps({"epoch": 1, "loss": 0.25}) + "\n")
            summaries = read_completed_epoch_summaries(root, {"loss"})
            self.assertEqual([(row["epoch"], row["metrics"]) for row in summaries], [(1, {"loss": 0.25})])
            (root / "epoch_metrics.jsonl").write_text(json.dumps({"epoch": 1, "loss": float("inf")}) + "\n")
            with self.assertRaises(ValueError):
                read_completed_epoch_summaries(root, {"loss"})

    def test_train_and_validation_metrics_keep_distinct_steps_and_epochs(self):
        writer = CometMLWriter.__new__(CometMLWriter)
        writer.exp = Mock()
        writer.step, writer.epoch, writer.mode = 0, None, "train"
        from datetime import datetime
        writer.timer = datetime.now()
        writer.set_epoch(3)
        writer.set_step(674, "train")
        writer.add_scalar("loss", 0.3)
        writer.set_step(1011, "val")
        writer.add_scalars({"PESQ": 2.5, "STOI": 0.9})
        self.assertIn(unittest.mock.call({"loss_train": 0.3}, step=674, epoch=3),
                      writer.exp.log_metrics.call_args_list)
        self.assertIn(unittest.mock.call({"PESQ_val": 2.5, "STOI_val": 0.9}, step=1011, epoch=3),
                      writer.exp.log_metrics.call_args_list)
        writer.close()
        writer.exp.end.assert_called_once()

    def test_partial_epoch_is_not_uploaded_and_names_preserve_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "info.log"
            prefix = "2026-10-14 00:00:00,123 - train - INFO -     "
            path.write_text(prefix + "epoch : 2\n" + prefix + "loss : 0.25\n")
            required = {"loss", "val_PESQ"}
            self.assertEqual(read_epoch_summaries(path, required), [])
            with path.open("a") as log:
                log.write(prefix + "val_PESQ : 2.5\n" + prefix + "epoch : 3\n")
            summaries = read_epoch_summaries(path, required)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["epoch"], 2)
            self.assertEqual(comet_metric_names(summaries[0]["metrics"], ["val"]),
                             {"loss_train": 0.25, "PESQ_val": 2.5})

    def test_sync_retry_and_resume_never_skip_an_unacknowledged_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = {"experiment_key": "fake", "last_epoch": 0}
            client = Mock()
            client.log_metrics.side_effect = [RuntimeError("unavailable"), None]
            summaries = [{"epoch": 1, "metrics": {"val_PESQ": 2.5}, "timestamp": 1234}]
            with self.assertRaises(RuntimeError):
                sync_summaries(client, state, path, summaries, 337, ["val"])
            self.assertEqual(state["last_epoch"], 0)
            sync_summaries(client, state, path, summaries, 337, ["val"])
            self.assertEqual(json.loads(path.read_text())["last_epoch"], 1)
            self.assertEqual(client.log_metrics.call_args.kwargs,
                             {"step": 337, "epoch": 1, "timestamp": 1234})
            sync_summaries(client, state, path, summaries, 337, ["val"])
            self.assertEqual(client.log_metrics.call_count, 2)

    def test_expected_metrics_use_only_configured_evaluation_partitions(self):
        config = OmegaConf.create({"writer": {"loss_names": ["loss", "al"]},
                                  "trainer": {"evaluation_partitions": ["val"]},
                                  "datasets": {"train": {}, "val": {}, "test": {}},
                                  "metrics": {"inference": [{"name": "PESQ"}]}})
        self.assertEqual(expected_metric_names(config),
                         {"loss", "al", "grad_norm", "val_loss", "val_al", "val_PESQ"})


if __name__ == "__main__":
    unittest.main()
