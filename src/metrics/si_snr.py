import torch
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class SI_SNR(BaseMetric):
    """
    Scale-Invariant Signal-to-Noise Ratio Improvement metric.
    Higher is better.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalNoiseRatio().to(self.device)

    def __call__(self, separated_audio, audio_target, audio_mix, **kwargs):
        if separated_audio.dim() == 2:
            separated_audio = separated_audio.unsqueeze(1)
        if audio_target.dim() == 2:
            audio_target = audio_target.unsqueeze(1)
        lengths = kwargs.get("lengths")
        if lengths is None:
            si_snr_separated = self.metric(separated_audio, audio_target)
            return torch.mean(si_snr_separated).item()
        scores = []
        for pred, target, length in zip(separated_audio, audio_target, lengths):
            n = int(length.item())
            scores.append(self.metric(pred[..., :n], target[..., :n]))
        si_snr_separated = torch.stack(scores)
        return torch.mean(si_snr_separated).item()
