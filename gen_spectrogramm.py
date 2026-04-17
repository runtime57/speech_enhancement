from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def iter_audio_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def save_demo_spectrogram(
    audio_path: Path,
    image_path: Path,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    image_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis=None,
        y_axis=None,
        ax=ax,
    )
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(image_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    audio_root = Path("audio")
    image_root = Path("images")

    for audio_path in sorted(iter_audio_files(audio_root)):
        rel = audio_path.relative_to(audio_root)
        image_path = image_root / rel.with_suffix(".png")
        save_demo_spectrogram(audio_path, image_path)
        print(f"[OK] {image_path}")

main()