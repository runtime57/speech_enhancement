import unittest
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch

from src.model.modules import GroupedGRNNLayer
from src.utils.config_utils import setup_config
from src.utils.training_suite import REPORT_MODELS
from train_queue import make_manifest, training_command, run_preflight, wait_for_running_runs

ROOT = Path(__file__).resolve().parents[1]


class TrainingQueueTests(unittest.TestCase):
    def test_existing_later_trainer_is_adopted_before_requeued_earlier_jobs(self):
        manifest = make_manifest()
        active = manifest["runs"][5]
        active.update(status="training", identity={"pid": 123})
        calls = [True, True, False]

        def live(identity, expected):
            return bool(identity) and calls.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            with patch("train_queue.is_live", side_effect=live), patch("train_queue.time.sleep") as sleep:
                adopted = wait_for_running_runs(manifest, Path(directory) / "manifest.json")
        self.assertEqual(adopted, [active])
        self.assertEqual(manifest["active_model"], "conv")
        sleep.assert_called_once_with(15)

    def test_preflight_retries_only_temporary_device_errors_and_keeps_logs(self):
        for errors, expected_calls, succeeds in [
            (["CUDA driver error: device not ready", ""], 2, True),
            (["CUDA driver error: device not ready"] * 3, 3, False),
            (["CUDA out of memory"], 1, False),
        ]:
            with self.subTest(errors=errors), tempfile.TemporaryDirectory() as directory:
                messages = iter(errors)

                def execute(command, **kwargs):
                    message = next(messages)
                    kwargs["stdout"].write(message + "\n")
                    return subprocess.CompletedProcess(command, 1 if message else 0)

                path = Path(directory) / "preflight.log"
                with patch("train_queue.subprocess.run", side_effect=execute) as run, patch("train_queue.time.sleep"):
                    if succeeds:
                        run_preflight(["preflight"], path)
                    else:
                        with self.assertRaises(subprocess.CalledProcessError):
                            run_preflight(["preflight"], path)
                    self.assertEqual(run.call_count, expected_calls)
                self.assertEqual(path.read_text(), "".join(e + "\n" for e in errors))

    def test_queue_has_only_remaining_report_models_and_fixed_protocol(self):
        plan = make_manifest()
        self.assertEqual(len(plan["runs"]), 9)
        self.assertEqual(len({r["run_name"] for r in plan["runs"]}), 9)
        self.assertFalse(any("emb_" in r["name"] for r in plan["runs"]))
        for entry in plan["runs"]:
            with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/configs")):
                config = compose(config_name="report_queue", overrides=training_command(entry)[4:])
            self.assertEqual(config.trainer.n_epochs, 50)
            self.assertEqual(config.trainer.evaluation_period, 5)
            self.assertEqual(config.dataloader.batch_size, 32)
            self.assertEqual(config.dataloader.multiprocessing_context, "spawn")
            self.assertIsNone(config.datasets.train.segment_seconds)
            self.assertEqual(config.model.grnn_backend, "triton")

    def test_all_six_convolutional_variants_forward_and_backward_on_cpu(self):
        torch.set_num_threads(1)
        for name, model_config in REPORT_MODELS[3:]:
            with self.subTest(name=name):
                with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/configs")):
                    config = compose(config_name="report_queue", overrides=[f"model={model_config}", "++model.grnn_backend=torch"])
                setup_config(config)
                model = instantiate(config.model)
                layers = [m for m in model.modules() if isinstance(m, GroupedGRNNLayer)]
                self.assertTrue(layers)
                self.assertTrue(all(m.backend == "torch" for m in layers))
                output = model(spec=torch.randn(2, 1, 16, 161, 2),
                               feat_erb=torch.randn(2, 1, 16, 24), feat_spec=torch.randn(2, 1, 16, 96, 2))
                self.assertEqual(output[0].shape, (2, 1, 16, 161, 2))
                sum(t.square().mean() for t in output).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
