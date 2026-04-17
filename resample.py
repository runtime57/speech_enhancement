import warnings
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from src.utils.config_utils import setup_config, stft_config
from src.utils.io_utils import safe_torchaudio_load, safe_torchaudio_save

warnings.filterwarnings("ignore", category=UserWarning)


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _list_audio_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {directory}")

    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    return files


@hydra.main(version_base=None, config_path="src/configs", config_name="resample")
def main(config) -> None:
    dirs = OmegaConf.select(config, "dirs")

    if not dirs:
        raise ValueError(
            "Set `resample.dirs` in config.\n"
            "Example:\n"
            "resample:\n"
            "  dirs: ['demo/audio/noisy', 'demo/audio/clean']"
        )

    target_sr = int(config.sr)

    for dir_path in dirs:
        dir_path = Path(to_absolute_path(str(dir_path))).expanduser()

        files = _list_audio_files(dir_path)

        print(f"\nProcessing directory: {dir_path}")
        print(f"Found {len(files)} file(s)")

        for input_path in files:
            wav, _ = safe_torchaudio_load(
                input_path,
                target_sr=target_sr,
                mono=True
            )  # [T]

            wav = wav.unsqueeze(0)  # [1, T]

            # ⚠️ перезаписываем тот же файл
            safe_torchaudio_save(input_path, wav, sr=target_sr)

            print(f"[OK] {input_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()