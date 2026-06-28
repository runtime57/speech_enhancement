import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    """
    Short-Time Objective Intelligibility metric.
    Higher is better.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ShortTimeObjectiveIntelligibility(16000).to(self.device)

    def __call__(self, separated_audio, audio_target, **kwargs):
        if separated_audio.dim() == 3:
            separated_audio = separated_audio.squeeze(1)
        if audio_target.dim() == 3:
            audio_target = audio_target.squeeze(1)
        lengths = kwargs.get("lengths")
        if lengths is None:
            return torch.mean(self.metric(separated_audio, audio_target)).item()
        scores = []
        for pred, target, length in zip(separated_audio, audio_target, lengths):
            n = int(length.item())
            scores.append(self.metric(pred[:n].unsqueeze(0), target[:n].unsqueeze(0)))
        return torch.mean(torch.stack(scores)).item()
