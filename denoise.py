import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

from src.utils.config_utils import setup_config, stft_config
from src.utils.df_utils import (
    audio_lengths_to_frame_lengths,
    exp_unit_norm,
    istft_with_df_config,
    stft_with_df_config,
)
from src.utils.erb_utils import compute_erb_feats_from_stft
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import safe_torchaudio_load, safe_torchaudio_save

warnings.filterwarnings("ignore", category=UserWarning)


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _resolve_device(config) -> str:
    device = OmegaConf.select(config, "denoise.device") or "auto"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _resolve_checkpoint_path(config) -> Path:
    checkpoint_path = OmegaConf.select(config, "denoise.from_pretrained")
    if not checkpoint_path:
        raise ValueError(
            "Set `denoise.from_pretrained` in the config, "
            "e.g. denoise.from_pretrained=models/model_best.pth"
        )
    return Path(to_absolute_path(str(checkpoint_path))).expanduser()


def _load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def _list_audio_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {directory}")

    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not files:
        raise ValueError(
            f"No audio files found in {directory}. "
            f"Supported extensions: {sorted(AUDIO_EXTENSIONS)}"
        )

    return files


@torch.no_grad()
def _denoise_waveform(model: torch.nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    p = stft_config()

    noisy_stft = stft_with_df_config(noisy)  # [B,F,T,2]

    def to_spec(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()  # [B,1,T,F,2]

    noisy_spec = to_spec(noisy_stft)
    lengths = torch.full(
        (noisy.shape[0],),
        noisy.shape[-1],
        device=noisy.device,
        dtype=torch.long,
    )
    frame_lengths = audio_lengths_to_frame_lengths(lengths, int(p.fft_size), int(p.hop_length))
    feat_erb = compute_erb_feats_from_stft(noisy_stft, lengths_frames=frame_lengths)

    feat_spec = noisy_spec[:, :, :, : int(p.nb_df), :]
    feat_spec, _ = exp_unit_norm(feat_spec, lengths=frame_lengths)

    enh, *_ = model(spec=noisy_spec, feat_erb=feat_erb, feat_spec=feat_spec)

    enh_spec = enh.squeeze(1).permute(0, 2, 1, 3).contiguous()  # [B,F,T,2]
    enh_wav = istft_with_df_config(enh_spec, length=noisy.shape[-1])
    return enh_wav


def _process_file(
    model: torch.nn.Module,
    input_path: Path,
    output_path: Path,
    device: str,
    sr: int,
) -> None:
    noisy_wav, _ = safe_torchaudio_load(input_path, target_sr=sr, mono=True)  # [T]
    noisy_wav = noisy_wav.unsqueeze(0).to(device)  # [B,T]

    enh_wav = _denoise_waveform(model, noisy=noisy_wav).cpu().clamp(-1.0, 1.0)  # [B,T]

    output_path.parent.mkdir(exist_ok=True, parents=True)
    safe_torchaudio_save(output_path, enh_wav, sr=sr)
    print(f"[OK] {input_path.name} -> {output_path}")


@hydra.main(version_base=None, config_path="src/configs", config_name="denoise")
def main(config) -> None:
    setup_config(config)

    if OmegaConf.select(config, "stft") is None:
        raise ValueError("Config must contain `stft` section.")

    seed = OmegaConf.select(config, "denoise.seed") or 1
    set_random_seed(int(seed))

    device = _resolve_device(config)

    model = instantiate(config.model).to(device)
    model.eval()

    checkpoint_path = _resolve_checkpoint_path(config)
    _load_model_weights(model, checkpoint_path=checkpoint_path, device=device)

    p = stft_config()
    sr = int(p.sr)

    noisy_dir = OmegaConf.select(config, "denoise.noisy_dir")
    output_dir = OmegaConf.select(config, "denoise.output_dir")

    if not noisy_dir or not output_dir:
        raise ValueError(
            "Set both `denoise.noisy_dir` and `denoise.output_dir` in the config.\n"
            "Example:\n"
            "  denoise.noisy_dir=demo/audio/noisy\n"
            "  denoise.output_dir=demo/audio/enhanced/grnn"
        )

    noisy_dir = Path(to_absolute_path(str(noisy_dir))).expanduser()
    output_dir = Path(to_absolute_path(str(output_dir))).expanduser()
    output_dir.mkdir(exist_ok=True, parents=True)

    input_files = _list_audio_files(noisy_dir)

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input directory: {noisy_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(input_files)} audio file(s)")

    for input_path in input_files:
        output_path = output_dir / input_path.name
        _process_file(
            model=model,
            input_path=input_path,
            output_path=output_path,
            device=device,
            sr=sr,
        )

    print("Done.")


if __name__ == "__main__":
    main()
