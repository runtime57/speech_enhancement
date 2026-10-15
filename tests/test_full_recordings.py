import copy
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import torch
from torch.utils.data import DataLoader

from src.datasets.collate import collate_fn
from src.datasets.data_utils import get_dataloaders
from src.trainer.base_trainer import BaseTrainer


ROOT = Path(__file__).resolve().parents[1]


class Writer:
    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_step(self, step, mode="train"):
        self.step, self.mode = step, mode

    def add_scalar(self, name, value):
        pass


def items(start, count):
    return [{"noisy": torch.full((20,), float(i)), "clean": torch.full((20,), float(i))}
            for i in range(start, start + count)]


class FullRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/configs")):
            cls.config = compose(config_name="full_recordings")

    def test_config_uses_whole_recordings_and_all_evaluation_partitions(self):
        c = self.config
        self.assertIsNone(c.datasets.train.segment_seconds)
        self.assertIsNone(c.trainer.epoch_len)
        self.assertFalse(c.trainer.drop_last)
        self.assertFalse(c.trainer.skip_oom)
        self.assertEqual(list(c.trainer.evaluation_partitions), ["val", "test"])
        self.assertEqual(list(c.metrics.train), [])
        self.assertEqual(c.trainer.n_epochs, 100)

    def test_dataloader_covers_every_train_record_including_last_batch(self):
        from hydra.utils import instantiate as real_instantiate
        c = copy.deepcopy(self.config)
        c.dataloader.batch_size, c.dataloader.num_workers = 4, 0
        datasets = {"train": items(1, 7), "val": items(20, 5), "test": items(30, 6)}

        def instantiate(config, **kwargs):
            if config is c.datasets:
                return datasets
            return real_instantiate(config, **kwargs)

        with patch("src.datasets.data_utils.instantiate", side_effect=instantiate):
            loaders, _ = get_dataloaders(c, "cpu")
        records = [int(value) for batch in loaders["train"] for value in batch["noisy"][:, 0]]
        self.assertEqual(sorted(records), list(range(1, 8)))
        self.assertEqual(len(records), len(set(records)))
        self.assertEqual(len(loaders["train"]), 2)
        self.assertEqual(sorted(loaders["train"].dataset[0]), ["clean", "noisy"])

    def test_train_epoch_never_evaluates_and_evaluation_never_updates_optimizer(self):
        class ToyTrainer(BaseTrainer):
            def process_batch(self, batch, metrics, step):
                self.observations.append((self.is_train, self.writer.mode, len(batch["noisy"])))
                if self.is_train:
                    self.optimizer.zero_grad()
                    self.model.weight.square().sum().backward()
                    self.optimizer.step()
                metrics.update("loss", 1., n=len(batch["noisy"]))
                return dict(batch, loss=torch.tensor(1.))

            def _log_batch(self, *args):
                pass

            def _save_checkpoint(self, *args, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as directory:
            c = copy.deepcopy(self.config)
            c.trainer.n_epochs = 2
            c.trainer.save_dir = directory
            c.trainer.monitor = "off"
            c.writer.loss_names = ["loss"]
            loaders = {name: DataLoader(data, batch_size=4, collate_fn=collate_fn)
                       for name, data in {"train": items(1, 7), "val": items(20, 5), "test": items(30, 6)}.items()}
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3)
            trainer = ToyTrainer(model, None, {"train": [], "inference": []}, optimizer,
                                 scheduler, c, "cpu", loaders, logging.getLogger("full_test"), Writer())
            trainer.observations = []
            result = trainer._train_epoch(1)
            self.assertEqual(result["samples"], 7)
            self.assertEqual(result["batches"], 2)
            self.assertTrue(all(is_train for is_train, _, _ in trainer.observations))
            before = model.weight.detach().clone()
            self.assertEqual(trainer._evaluation_epoch(1, "val", loaders["val"])["samples"], 5)
            self.assertEqual(trainer._evaluation_epoch(1, "test", loaders["test"])["samples"], 6)
            torch.testing.assert_close(model.weight, before)
            trainer.observations = []
            trainer._train_process()
            self.assertEqual(sum(size for train, _, size in trainer.observations if train), 14)
            import json
            records = [json.loads(line) for line in (trainer.checkpoint_dir / "epoch_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([(r["samples"], r["val_samples"], r["test_samples"]) for r in records],
                             [(7, 5, 6), (7, 5, 6)])

    def test_amp_skipped_step_preserves_full_epoch_and_finite_gradient_summary(self):
        class ToyTrainer(BaseTrainer):
            def process_batch(self, batch, metrics, step):
                for parameter in self.model.parameters():
                    parameter.grad = torch.full_like(parameter, float("inf") if step == 0 else 1.)
                if step != 0:
                    self.optimizer.step()
                metrics.update("loss", 1., n=len(batch["noisy"]))
                return dict(batch, loss=torch.tensor(1.), optimizer_stepped=step != 0)

            def _log_batch(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory:
            c = copy.deepcopy(self.config)
            c.trainer.save_dir = directory
            c.trainer.monitor = "off"
            c.writer.loss_names = ["loss"]
            loader = DataLoader(items(1, 7), batch_size=4, collate_fn=collate_fn)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3)
            trainer = ToyTrainer(model, None, {"train": [], "inference": []}, optimizer,
                                 scheduler, c, "cpu", {"train": loader},
                                 logging.getLogger("amp_skip_test"), Writer())
            result = trainer._train_epoch(1)
            self.assertEqual((result["samples"], result["batches"]), (7, 2))
            self.assertAlmostEqual(result["grad_norm"], 2. ** .5, places=6)
            import json
            json.dumps(result, allow_nan=False)

    def test_sparse_evaluation_keeps_monitor_and_evaluates_final_epoch(self):
        class ToyTrainer(BaseTrainer):
            def process_batch(self, batch, metrics, step):
                if self.is_train:
                    self.optimizer.zero_grad()
                    self.model.weight.square().sum().backward()
                    self.optimizer.step()
                metrics.update("loss", 10. - self._last_epoch, n=len(batch["noisy"]))
                return dict(batch, loss=torch.tensor(1.))

            def _evaluation_epoch(self, epoch, part, dataloader):
                self.evaluations.append((epoch, part))
                return super()._evaluation_epoch(epoch, part, dataloader)

            def _log_batch(self, *args):
                pass

            def _save_checkpoint(self, *args, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as directory:
            from omegaconf import OmegaConf
            c = copy.deepcopy(self.config)
            OmegaConf.set_struct(c, False)
            c.trainer.evaluation_period = 5
            c.trainer.n_epochs = 7
            c.trainer.save_dir = directory
            c.writer.loss_names = ["loss"]
            loaders = {name: DataLoader(items(1, 2), batch_size=4, collate_fn=collate_fn)
                       for name in ("train", "val", "test")}
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3)
            trainer = ToyTrainer(model, None, {"train": [], "inference": []}, optimizer,
                                 scheduler, c, "cpu", loaders, logging.getLogger("sparse_test"), Writer())
            trainer.evaluations = []
            trainer._train_process()
            self.assertEqual(trainer.evaluations, [(5, "val"), (5, "test"), (7, "val"), (7, "test")])
            self.assertEqual(trainer.mnt_mode, "min")
            self.assertEqual(trainer.mnt_best, 3.)
            import json
            rows = [json.loads(line) for line in (trainer.checkpoint_dir / "epoch_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 7)
            self.assertEqual([r["epoch"] for r in rows if "val_loss" in r], [5, 7])


if __name__ == "__main__":
    unittest.main()
