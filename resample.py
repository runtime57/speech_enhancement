import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

from src.utils.config_utils import setup_config, stft_config
from src.utils.df_utils import as_complex, exp_unit_norm
from src.utils.erb_utils import compute_erb_feats_from_stft
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH, safe_torchaudio_load, safe_torchaudio_save
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="denoise")
def main(config) -> None:
    setup_config(config)
    p = stft_config()
    input_path = OmegaConf.select(config, "denoise.input_path")
    input_path = Path(to_absolute_path(str(input_path))).expanduser()
    wav, _ = safe_torchaudio_load(input_path, target_sr=int(p.sr), mono=True)  # [T]
    wav = wav.unsqueeze(0)
    safe_torchaudio_save("data/resampled.wav", wav, sr=int(p.sr))
    print(f"Saved enhanced audio to: data/resampled.wav")


if __name__ == "__main__":
    main()