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
        if file.is_file()
    ]
    return list(sorted(wav_list))

def create_vctk_split(split_name, rawdata_path, dataset_path):
    assert split_name in ["train", "test"]

    rawdata_path = Path(rawdata_path).resolve()
    dataset_path = Path(dataset_path)

    if split_name == "train":
        ending = f"{split_name}set_28spk_wav"
    else:
        ending = f"{split_name}set_wav"
    clean_dir = rawdata_path / f"clean_{ending}"
    noisy_dir = rawdata_path / f"noisy_{ending}"
    for directory in [clean_dir, noisy_dir]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    clean_paths = {
        Path(path).relative_to(clean_dir): path
        for path in get_wav_paths(clean_dir)
    }
    noisy_paths = {
        Path(path).relative_to(noisy_dir): path
        for path in get_wav_paths(noisy_dir)
    }
    if not clean_paths or not noisy_paths:
        raise ValueError(f"No paired WAV files found in {clean_dir} and {noisy_dir}")
    if clean_paths.keys() != noisy_paths.keys():
        missing_clean = sorted(str(path) for path in noisy_paths.keys() - clean_paths.keys())
        missing_noisy = sorted(str(path) for path in clean_paths.keys() - noisy_paths.keys())
        raise ValueError(
            f"Unpaired WAV files: missing clean={missing_clean}, missing noisy={missing_noisy}"
        )

    split = []
    for path in sorted(clean_paths):
        split.append({"clean_path": clean_paths[path], "noisy_path": noisy_paths[path]})
    
    split_path = dataset_path / split_name
    split_path.mkdir(exist_ok=True, parents=True)
    write_json(split, str(split_path / "split.json"))
