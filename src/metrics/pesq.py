import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    """
    Perceptual Evaluation of Speech Quality metric.
    Higher is better.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = PerceptualEvaluationSpeechQuality(16000, "wb").to(self.device)

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
