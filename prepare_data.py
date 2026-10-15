"""Prepare the paired waveform cache without starting a tracking experiment."""

import hydra
import torch
from hydra.utils import instantiate

from src.utils.config_utils import setup_config


@hydra.main(version_base=None, config_path="src/configs", config_name="dfnet")
def main(config):
    torch.set_num_threads(1)
    setup_config(config)
    for partition, dataset_config in config.datasets.items():
        dataset = instantiate(dataset_config)
        if not len(dataset):
            raise ValueError(f"Empty dataset partition: {partition}")
        print(f"Prepared {partition}: {len(dataset)} paired waveforms", flush=True)


if __name__ == "__main__":
    main()
