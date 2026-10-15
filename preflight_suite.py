"""Verify a real worst-length batch, gradients and optimizer before suite training."""

import argparse
import json
import logging
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from safetensors import safe_open
import torch

from src.datasets.collate import collate_fn
from src.metrics.tracker import MetricTracker
from src.trainer import Trainer
from src.utils.config_utils import setup_config
from src.utils.init_utils import set_random_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="full_recordings")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(Path("src/configs").resolve())):
        config = compose(config_name=args.config, overrides=[
            f"model={args.model}", "datasets.train.rawdata_path=data",
            "datasets.val.rawdata_path=data", "datasets.test.rawdata_path=data", *args.override])
    setup_config(config)
    set_random_seed(config.trainer.seed)
    datasets = instantiate(config.datasets)
    lengths = []
    for index, entry in enumerate(datasets.train._index):
        with safe_open(entry["element_path"], framework="pt") as handle:
            lengths.append((handle.get_slice("noisy").get_shape()[0], index))
    longest = max(lengths)[1]
    # One longest record forces the same maximum padding shape possible in training.
    indices = [longest] + [i for i in range(31) if i != longest]
    if len(indices) < 32:
        indices.append(31)
    samples = [datasets.train[i] for i in indices]
    model = instantiate(config.model).cuda()
    criterion = instantiate(config.loss_function).cuda()
    optimizer = instantiate(config.optimizer, params=model.parameters())
    scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    trainer = Trainer(model, criterion, {"train": [], "inference": []}, optimizer,
                      scheduler, config, "cuda", {"train": [None]},
                      logging.getLogger("preflight"), None)
    original = next(model.parameters()).detach().clone()
    started = time.monotonic()
    for attempt in range(12):
        batch = trainer.process_batch(collate_fn(samples), MetricTracker(*config.writer.loss_names), attempt)
        if not torch.equal(original, next(model.parameters())):
            break
    else:
        raise RuntimeError("Optimizer did not update")
    torch.cuda.synchronize()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert batch["noisy"].shape[0] == 32
    assert int(batch["lengths"].max()) == max(lengths)[0]
    report = {"model": args.model, "passed": True, "batch_size": 32,
              "max_recording_seconds": max(lengths)[0] / config.stft.sr,
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2,
              "train_seconds": time.monotonic() - started,
              "loss": batch["loss"].item(), "amp_scale": trainer.grad_scaler.get_scale(),
              "dataset_sizes": {k: len(v) for k, v in datasets.items()},
              "finite_weights_and_gradients": True, "optimizer_updated": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
