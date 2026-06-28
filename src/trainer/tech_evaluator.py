import json

import torch
from tqdm.auto import tqdm

from src.metrics.efficiency import (
    count_parameters,
    get_model_size,
    measure_flops,
    measure_macs,
    measure_inference_time,
    measure_peak_memory,
)
from src.trainer.base_trainer import BaseTrainer
from src.utils.config_utils import stft_config
from src.utils.df_utils import audio_lengths_to_frame_lengths, exp_unit_norm, stft_with_df_config
from src.utils.erb_utils import compute_erb_feats_from_stft


class TechEvaluator(BaseTrainer):
    """
    TechEvaluator (Like Inferencer but for Tech Evaluation) class.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.evaluator.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.evaluator
        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms
        self.model_metrics = {}
        self.model_metrics["n_parameters"] = count_parameters(model)
        self.model_metrics["model_size_bytes"] = get_model_size(
            config.evaluator.get("from_pretrained")
        )

        size_bytes = self.model_metrics["model_size_bytes"]
        self.model_metrics["model_size_mb"] = (
            round(size_bytes / (1024**2), 2) if size_bytes is not None else None
        )
        self.model_metrics["device"] = self.device

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path

        if not skip_model_load:
            self._from_pretrained(config.evaluator.get("from_pretrained"))

    def _prepare_batch_for_model(self, batch):
        batch = self.move_batch_to_device(batch)
        if "lengths" in batch:
            batch["lengths"] = batch["lengths"].to(self.device)
        batch = self.transform_batch(batch)

        p = stft_config()

        clean_stft = stft_with_df_config(batch["clean"])
        noisy_stft = stft_with_df_config(batch["noisy"])

        def to_spec(x):
            return x.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()

        clean_spec = to_spec(clean_stft)
        noisy_spec = to_spec(noisy_stft)

        frame_lengths = audio_lengths_to_frame_lengths(batch["lengths"], p.fft_size, p.hop_length)
        feat_erb = compute_erb_feats_from_stft(noisy_stft, lengths_frames=frame_lengths)

        feat_spec = noisy_spec[:, :, :, : p.nb_df, :]
        feat_spec, _ = exp_unit_norm(feat_spec, lengths=frame_lengths)

        batch.update(
            {
                "clean_spec": clean_spec,
                "noisy_spec": noisy_spec,
                "feat_erb": feat_erb,
                "feat_spec": feat_spec,
            }
        )
        return batch

    def _get_model_inputs(self, batch):
        return {
            "spec": batch["noisy_spec"],
            "feat_erb": batch["feat_erb"],
            "feat_spec": batch["feat_spec"],
        }

    def compute_model_metrics(self, batch):
        batch = self._prepare_batch_for_model(batch)
        model_inputs = self._get_model_inputs(batch)

        flops = measure_flops(self.model, model_inputs)
        self.model_metrics["flops"] = flops
        self.model_metrics["giga_flops"] = flops / 1e9 if flops is not None else None

        self.model_metrics["peak_memory_bytes"] = measure_peak_memory(
            self.model, model_inputs, device=self.device
        )
        self.model_metrics["avg_inference_time_sec"] = measure_inference_time(
            self.model, model_inputs, device=self.device
        )

        macs = measure_macs(self.model, model_inputs)
        self.model_metrics["macs"] = macs
        self.model_metrics["giga_macs"] = macs / 1e9 if macs is not None else None

        if self.save_path is not None:
            with open(self.save_path / f"{str(self.device)}.json", "w") as f:
                json.dump(self.model_metrics, f, indent=2)

        return self.model_metrics

    def _tech_evaluation_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                model_metrics = self.compute_model_metrics(batch=batch)
                break

        return model_metrics

    def run_tech_evaluation(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            tech_metrics = self._tech_evaluation_part(part, dataloader)
            part_logs[part] = tech_metrics
        return part_logs
