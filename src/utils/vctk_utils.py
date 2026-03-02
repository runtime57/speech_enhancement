from src.utils.io_utils import ROOT_PATH, read_json, write_json
from sklearn.model_selection import train_test_split
from csv import DictReader
from random import shuffle
from pathlib import Path

def get_wav_paths(directory='.'):
    start_path = Path(directory)
    wav_list = [
        str(file.absolute())
        for file in start_path.rglob('*.wav')
    ]
    return list(sorted(wav_list))

def create_vctk_split(split_name, rawdata_path, dataset_path):
    assert split_name in ["train", "test"]

    rawdata_path = Path(rawdata_path)
    dataset_path = Path(dataset_path)

    clean_paths = get_wav_paths(rawdata_path / f"clean_{split_name}set_wav")
    noisy_paths = get_wav_paths(rawdata_path / f"noisy_{split_name}set_wav")

    split = []
    for clean, noisy in zip(clean_paths, noisy_paths):
        split.append({"clean_path": clean, "noisy_path": noisy})
    
    write_json(split, str(dataset_path / split_name / "split.json"))
