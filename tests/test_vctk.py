import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torchaudio

from src.datasets.vctk import VCTKDataset


class VCTKCacheTests(unittest.TestCase):
    def test_speaker_holdout_and_paired_random_crops(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "data/vctk/train"
            cache.mkdir(parents=True)
            clean = torch.arange(60000, dtype=torch.float32) / 100000
            index = []
            for i, speaker in enumerate(("p226", "p287", "p228")):
                path = cache / f"{i:08d}.pt"
                save_file({"clean": clean, "noisy": clean + 0.01}, path)
                index.append({"element_path": str(path),
                              "original_clean": str(root / f"{speaker}_001.wav"),
                              "original_noisy": str(root / f"{speaker}_001.wav")})
            (cache / "index.json").write_text(json.dumps(index))
            with patch("src.datasets.vctk.ROOT_PATH", root):
                train = VCTKDataset(root, name="train", exclude_speakers=["p226", "p287"],
                                    segment_seconds=3)
                val = VCTKDataset(root, name="train", include_speakers=["p226", "p287"])
                self.assertEqual((len(train), len(val)), (1, 2))
                self.assertTrue({e["element_path"] for e in train._index}.isdisjoint(
                    {e["element_path"] for e in val._index}))
                torch.manual_seed(37)
                sample = train[0]
                torch.manual_seed(37)
                repeated = train[0]
                self.assertEqual(sample["clean"].numel(), 48000)
                torch.testing.assert_close(sample["noisy"] - sample["clean"], torch.full((48000,), 0.01))
                torch.testing.assert_close(sample["clean"], repeated["clean"])
                self.assertEqual(val[0]["clean"].numel(), 60000)
                full_train = VCTKDataset(root, name="train", exclude_speakers=["p226", "p287"],
                                         segment_seconds=None)
                torch.testing.assert_close(full_train[0]["clean"], clean)

    def test_empty_and_missing_cache_are_rebuilt_from_paired_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            waveform = torch.linspace(-0.5, 0.5, 4800).unsqueeze(0)
            for kind in ("clean", "noisy"):
                audio_dir = raw / f"{kind}_testset_wav"
                audio_dir.mkdir(parents=True)
                torchaudio.save(str(audio_dir / "sample.wav"), waveform, 48000)
            cache = root / "data/vctk/test"
            cache.mkdir(parents=True)
            (cache / "index.json").write_text("[]")
            with patch("src.datasets.vctk.ROOT_PATH", root):
                dataset = VCTKDataset(raw, name="test")
                self.assertEqual(len(dataset), 1)
                sample = dataset[0]
                self.assertEqual(sample["clean"].shape, (1600,))
                torch.testing.assert_close(sample["clean"], sample["noisy"])
                cached_path = Path(json.loads((cache / "index.json").read_text())[0]["element_path"])
                cached_path.unlink()
                rebuilt = VCTKDataset(raw, name="test", instance_transforms={"noisy": lambda audio: audio * 0})
                self.assertTrue(cached_path.is_file())
                self.assertEqual(rebuilt[0]["noisy"].count_nonzero().item(), 0)
                torch.testing.assert_close(rebuilt[0]["clean"], sample["clean"])


if __name__ == "__main__":
    unittest.main()
