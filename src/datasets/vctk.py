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
# from src.utils.vctk_utils import create_vctk_split
from src.utils.config_utils import get_config
from src.utils.io_utils import safe_torchaudio_load

class VCTKDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, name="train", sr = 16000, *args, **kwargs
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
        index_path = ROOT_PATH / "data" / "vctk" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name, sr)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name, sr=16000):
        index = []
        data_path = ROOT_PATH / "data" / "vctk" / name
        data_path.mkdir(exist_ok=True, parents=True)
        
        # create_vctk_split(name)

        split = read_json(ROOT_PATH / "data" / "vctk" / name / "split.json")
        number_of_zeros = 8
        
        p = stft_config()

        print("Creating VCTK Dataset")
        for i, element in tqdm(enumerate(split)):
            path = data_path / f"{i:0{number_of_zeros}d}.pt"

            clean_path, noisy_path = element["clean_path"], element["noisy_path"]

            clean, _ = safe_torchaudio_load(clean_path, sr)
            noisy, _ = safe_torchaudio_load(noisy_path, sr)

            safetensors.torch.save_file({
                "clean": clean,
                "noisy": noisy
            }, path)

            index.append({ 
                "element_path": str(path), 
                "original_clean": clean_path, 
                "original_noisy": noisy_path
            })

        write_json(index, str(data_path / "index.json"))
        return index


    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_path = data_dict["element_path"]
        obj = safetensors.torch.load_file(data_path)

        noisy = obj["noisy"]
        clean = obj["clean"]
        noisy = self.preprocess_data(noisy)
        return {"noisy": noisy, "clean": clean}
        
