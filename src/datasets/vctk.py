import os
from pathlib import Path
import tempfile
from typing import Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm
import torchaudio
import safetensors
from omegaconf import OmegaConf

from utils.config_utils import get_config, stft_config

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.utils.vctk_utils import create_vctk_split
from src.utils.config_utils import get_config

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
        
        create_vctk_split(name)

        split = read_json(ROOT_PATH / "data" / "vctk" / name / "split.json")
        number_of_zeros = 8
        
        p = stft_config()

        print("Creating Example Dataset")
        for i, (noisy_path, clean_path) in tqdm(enumerate(split)):
            path = data_path / f"{i:0{number_of_zeros}d}.pt"

            clean, _ = self.safe_torchaudio_load(clean_path, sr)
            noisy, _ = self.safe_torchaudio_load(noisy_path, sr)

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
        data_dict = self.index[ind]
        data_path = data_dict["element_path"]
        obj = safetensors.torch.load_file(data_path)

        noisy = obj["noisy"]
        clean = obj["clean"]
        noisy = self.preprocess_data(noisy)
        return {"noisy": noisy, "clean": clean}


    def safe_torchaudio_load(self, path: Union[str, Path], 
                             target_sr: Optional[int] = None, mono: bool = True):
        waveform, sr = torchaudio.load(str(path))  # [C, T]
        waveform = waveform.to(torch.float32)

        if mono and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

        if target_sr is not None and sr != target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=target_sr,
                resampling_method="sinc_interp_hann",
                lowpass_filter_width=16,
            )
            sr = target_sr

        if mono:
            waveform = waveform.squeeze(0)  # [T]
        return waveform.contiguous(), sr
        
