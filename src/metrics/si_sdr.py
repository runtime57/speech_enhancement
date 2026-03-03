import torch
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

from src.metrics.base_metric import BaseMetric


class SI_SDR(BaseMetric):
    """
    Scale-Invariant Signal-to-Distortion Ratio metric.
    Higher is better.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalDistortionRatio().to(self.device)

    def __call__(self, separated_audio, audio_target, **kwargs):
        if separated_audio.dim() == 2:
            separated_audio = separated_audio.unsqueeze(1)
        if audio_target.dim() == 2:
            audio_target = audio_target.unsqueeze(1)

        si_sdr = self.metric(separated_audio, audio_target)
        return torch.mean(si_sdr).item()