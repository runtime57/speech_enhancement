import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.config_utils import stft_config
from src.utils.df_utils import as_complex, exp_unit_norm
from src.utils.erb_utils import compute_erb_feats_from_stft

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        p = stft_config()
        clean_stft = torch.stft(
            batch["clean"],
            n_fft=p.fft_size,
            hop_length=p.hop_length,
            return_complex=False,
        )  # [B, F, T, 2]
        noisy_stft = torch.stft(
            batch["noisy"],
            n_fft=p.fft_size,
            hop_length=p.hop_length,
            return_complex=False,
        )  # [B, F, T, 2]

        # reshape to model convention [B, 1, T, F, 2]
        def to_spec(x):
            return x.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()

        clean_spec = to_spec(clean_stft)
        noisy_spec = to_spec(noisy_stft)

        feat_erb = compute_erb_feats_from_stft(noisy_stft)

        feat_spec = noisy_spec[:, :, :, : p.nb_df, :]
        feat_spec, _ = exp_unit_norm(feat_spec)


        enh, m, lsnr, _ = self.model(
            spec=noisy_spec,
            feat_erb=feat_erb,
            feat_spec=feat_spec,
        )

        batch.update(
            {
                "clean_spec": clean_spec,
                "noisy_spec": noisy_spec,
                "enh": enh,
                "m": m,
                "lsnr": lsnr,
            }
        )

        # Convert enhanced STFT back to waveform for waveform-based metrics/losses.
        # enh: [B, 1, T, F, 2] -> istft expects [B, F, T] complex
        def spec_to_waveform(spec: torch.Tensor, length: int) -> torch.Tensor:
            spec = spec.squeeze(1).permute(0, 2, 1, 3).contiguous()  # [B, F, T, 2]
            return torch.istft(
                as_complex(spec),
                n_fft=p.fft_size,
                hop_length=p.hop_length,
                length=length,
            )

        loss_on_waveform = bool(self.cfg_trainer.get("loss_on_waveform", False))
        enh_wav = spec_to_waveform(
            enh if loss_on_waveform else enh.detach(),
            length=batch["noisy"].shape[-1],
        )
        batch.update(
            {
                "enh_wav": enh_wav,
                "separated_audio": enh_wav,
                "audio_target": batch["clean"],
                "audio_mix": batch["noisy"],
            }
        )

        all_losses = self.criterion(
            clean=clean_spec,
            noisy=noisy_spec,
            enh=enh,
            m=m,
            lsnr=lsnr,
        )
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
