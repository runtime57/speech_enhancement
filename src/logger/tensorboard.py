import json
from pathlib import Path

from src.utils.io_utils import ROOT_PATH


class TensorBoardWriter:
    """Local experiment logging, independent of network or service availability."""

    def __init__(self, logger, project_config, project_name, run_name, run_id=None,
                 log_dir=None, **kwargs):
        from torch.utils.tensorboard import SummaryWriter

        self.run_id = run_id
        self.step = 0
        self.mode = "train"
        directory = ROOT_PATH / project_config["trainer"]["save_dir"] / run_name / "tensorboard"
        self.log_dir = Path(log_dir) if log_dir is not None else directory
        self.summary = SummaryWriter(str(self.log_dir), flush_secs=10)
        self.summary.add_text("configuration", json.dumps(project_config, indent=2, default=str), 0)
        logger.info("TensorBoard logs: %s", self.log_dir)

    def set_step(self, step, mode="train"):
        self.step, self.mode = int(step), mode

    def _name(self, name):
        return f"{name}_{self.mode}"

    def add_scalar(self, name, value):
        self.summary.add_scalar(self._name(name), value, self.step)

    def add_scalars(self, values):
        for name, value in values.items():
            self.add_scalar(name, value)

    def add_text(self, name, text):
        self.summary.add_text(self._name(name), str(text), self.step)

    def add_image(self, name, value):
        self.summary.add_image(self._name(name), value, self.step)

    def add_audio(self, name, value, sample_rate=None):
        value = value.detach().cpu()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        self.summary.add_audio(self._name(name), value, self.step, sample_rate=sample_rate)

    def add_histogram(self, name, value, bins=None):
        self.summary.add_histogram(self._name(name), value, self.step, bins=bins or "tensorflow")

    def add_table(self, name, table):
        self.add_text(name, table.to_csv(index=False))

    def add_checkpoint(self, path, save_dir):
        self.add_text("checkpoint", str(path))

    def close(self):
        self.summary.flush()
        self.summary.close()
