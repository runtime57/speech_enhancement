import json
from collections import OrderedDict
from pathlib import Path
import torch
import torchaudio
from typing import Optional, Union

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(fname):
    """
    Read the given json file.

    Args:
        fname (str): filename of the json file.
    Returns:
        json (list[OrderedDict] | OrderedDict): loaded json.
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    """
    Write the content to the given json file.

    Args:
        content (Any JSON-friendly): content to write.
        fname (str): filename of the json file.
    """
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def safe_torchaudio_load(path: Union[str, Path], 
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

def safe_torchaudio_save(path: Path, waveform: torch.Tensor, sr: int) -> None:
    torchaudio.save(str(path), waveform, sample_rate=sr)
