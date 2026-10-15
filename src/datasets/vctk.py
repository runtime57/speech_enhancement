import os
from pathlib import Path
import tempfile
from typing import Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm
import torchaudio
import safetensors
import safetensors.torch
from omegaconf import OmegaConf

from src.utils.config_utils import get_config, stft_config

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.utils.vctk_utils import create_vctk_split
from src.utils.config_utils import get_config
from src.utils.io_utils import safe_torchaudio_load

class VCTKDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, rawdata_path, name="train", sr=16000, include_speakers=None,
        exclude_speakers=None, segment_seconds=None, *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        self.sr = sr
        self.segment_samples = None if segment_seconds is None else int(segment_seconds * sr)
        if self.segment_samples is not None and self.segment_samples <= 0:
            raise ValueError("segment_seconds must be positive")
        index_path = ROOT_PATH / "data" / "vctk" / name / "index.json"
        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
            if not index or any(
                Path(element["element_path"]).parent != index_path.parent
                or not Path(element["element_path"]).is_file()
                for element in index
            ):
                index = self._create_index(rawdata_path, name, sr)
        else:
            index = self._create_index(rawdata_path,  name, sr)

        included = None if include_speakers is None else set(include_speakers)
        excluded = set(exclude_speakers or [])
        index = [element for element in index if
                 (included is None or Path(element["original_clean"]).stem.split("_")[0] in included)
                 and Path(element["original_clean"]).stem.split("_")[0] not in excluded]
        if not index:
            raise ValueError(f"Empty {name} dataset after speaker filtering")
        super().__init__(index, *args, **kwargs)

    def _create_index(self, rawdata_path, name, sr=16000):
        index = []
        data_path = ROOT_PATH / "data" / "vctk"
        split_path = data_path / name
        split_path.mkdir(exist_ok=True, parents=True)
        
        if name != "onebatch":
            create_vctk_split(name, rawdata_path, data_path)

        split = read_json(split_path / "split.json")
        number_of_zeros = 8
        
        p = stft_config()

        print("Creating VCTK Dataset")
        for i, element in tqdm(enumerate(split), total=len(split)):
            path = split_path / f"{i:0{number_of_zeros}d}.pt"

            clean_path, noisy_path = element["clean_path"], element["noisy_path"]

            clean, _ = safe_torchaudio_load(clean_path, sr)
            noisy, _ = safe_torchaudio_load(noisy_path, sr)

            if clean.shape != noisy.shape or not clean.numel():
                raise ValueError(f"Invalid audio pair: {clean_path}, {noisy_path}")
            if not torch.isfinite(clean).all() or not torch.isfinite(noisy).all():
                raise ValueError(f"Non-finite audio pair: {clean_path}, {noisy_path}")

            safetensors.torch.save_file({
                "clean": clean,
                "noisy": noisy
            }, path)

            index.append({ 
                "element_path": str(path), 
                "original_clean": clean_path, 
                "original_noisy": noisy_path
            })

        write_json(index, str(split_path / "index.json"))
        return index


    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_path = data_dict["element_path"]
        obj = safetensors.torch.load_file(data_path)

        noisy = obj["noisy"]
        clean = obj["clean"]
        if self.segment_samples is not None and noisy.numel() > self.segment_samples:
            start = int(torch.randint(noisy.numel() - self.segment_samples + 1, ()).item())
            noisy = noisy[start:start + self.segment_samples]
            clean = clean[start:start + self.segment_samples]
        return self.preprocess_data({"noisy": noisy, "clean": clean})
        
