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


def _resolve_device(config) -> str:
    device = OmegaConf.select(config, "denoise.device") or "auto"

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _resolve_checkpoint_path(config) -> Path:
    explicit = OmegaConf.select(config, "denoise.from_pretrained")
    if explicit:
        return Path(to_absolute_path(str(explicit))).expanduser()

    save_dir = OmegaConf.select(config, "denoise.save_dir") or "saved"
    run_name = OmegaConf.select(config, "writer.run_name")
    if not run_name:
        raise ValueError(
            "Checkpoint path not found. Set `denoise.from_pretrained`, or set `writer.run_name` "
            "(then defaults to ${denoise.save_dir}/<run_name>/model_best.pth)."
        )

    return ROOT_PATH / str(save_dir) / str(run_name) / "model_best.pth"


def _load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)



@torch.no_grad()
def _denoise_waveform(model: torch.nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    p = stft_config()
    noisy_stft = torch.stft(
        noisy,
        n_fft=int(p.fft_size),
        hop_length=int(p.hop_length),
        return_complex=False,
    )  # [B,F,T,2]

    def to_spec(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()  # [B,1,T,F,2]

    noisy_spec = to_spec(noisy_stft)
    feat_erb = compute_erb_feats_from_stft(noisy_stft)

    feat_spec = noisy_spec[:, :, :, : int(p.nb_df), :]
    feat_spec, _ = exp_unit_norm(feat_spec)

    enh, *_ = model(spec=noisy_spec, feat_erb=feat_erb, feat_spec=feat_spec)

    enh_spec = enh.squeeze(1).permute(0, 2, 1, 3).contiguous()  # [B,F,T,2]
    enh_wav = torch.istft(
        as_complex(enh_spec),
        n_fft=int(p.fft_size),
        hop_length=int(p.hop_length),
        length=noisy.shape[-1],
    )
    return enh_wav


@hydra.main(version_base=None, config_path="src/configs", config_name="denoise")
def main(config) -> None:
    setup_config(config)
    if OmegaConf.select(config, "stft") is None:
        raise ValueError("Config must contain `stft` section (see src/configs/denoise.yaml).")

    seed = OmegaConf.select(config, "denoise.seed") or 1
    set_random_seed(int(seed))

    device = _resolve_device(config)
    model = instantiate(config.model).to(device)
    model.eval()

    checkpoint_path = _resolve_checkpoint_path(config)
    _load_model_weights(model, checkpoint_path=checkpoint_path, device=device)

    p = stft_config()
    input_path = OmegaConf.select(config, "denoise.input_path")
    output_path = OmegaConf.select(config, "denoise.output_path")
    if not input_path or not output_path:
        raise ValueError(
            "Set `denoise.input_path` and `denoise.output_path` in the config "
            "(e.g. src/configs/denoise.yaml), or override via CLI:\n"
            "  python3 denoise.py -cn=denoise denoise.input_path=<path> denoise.output_path=<path>"
        )

    input_path = Path(to_absolute_path(str(input_path))).expanduser()
    output_path = Path(to_absolute_path(str(output_path))).expanduser()
    output_path.parent.mkdir(exist_ok=True, parents=True)

    noisy_wav, _ = safe_torchaudio_load(input_path, target_sr=int(p.sr), mono=True)  # [T]
    noisy_wav = noisy_wav.unsqueeze(0).to(device)  # [B,T]

    enh_wav = _denoise_waveform(model, noisy=noisy_wav).cpu().clamp(-1.0, 1.0)  # [B,T]

    safe_torchaudio_save(output_path, enh_wav, sr=int(p.sr))
    print(f"Saved enhanced audio to: {output_path}")


if __name__ == "__main__":
    main()
