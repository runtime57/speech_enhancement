import torch
from torchmetrics.audio.sdr import SignalDistortionRatio

from src.metrics.base_metric import BaseMetric


class SDRi(BaseMetric):
    """
    Signal-to-Distortion Ratio Improvement metric.
    Higher is better.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = SignalDistortionRatio().to(self.device)

    def __call__(self, separated_audio, audio_target, audio_mix, **kwargs):
        if separated_audio.dim() == 2:
            separated_audio = separated_audio.unsqueeze(1)
        if audio_target.dim() == 2:
            audio_target = audio_target.unsqueeze(1)

        sdr_separated = self.metric(separated_audio, audio_target)
        audio_mix_expanded = audio_mix.unsqueeze(1).expand(
            -1, separated_audio.shape[1], -1
        )
        sdr_mixture = self.metric(audio_mix_expanded, audio_target)

        return torch.mean(sdr_separated - sdr_mixture).item()