import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.config_utils import stft_config
from src.utils.df_utils import stft_with_df_config
from src.utils.enhancement import prepare_enhancement_inputs, synthesize_enhancement
from src.utils.precision import model_autocast


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, step):
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
        if "lengths" in batch:
            batch["lengths"] = batch["lengths"].to(self.device)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        p = stft_config()
        clean_stft = stft_with_df_config(batch["clean"], batch["lengths"])

        # reshape to model convention [B, 1, T, F, 2]
        def to_spec(x):
            return x.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()

        clean_spec = to_spec(clean_stft)
        inputs, frame_lengths = prepare_enhancement_inputs(batch["noisy"], batch["lengths"])
        noisy_spec = inputs["spec"]

        with model_autocast(self.device, self.amp_enabled, self.amp_dtype):
            outputs = self.model(**inputs)
        # FFTs, fractional spectral powers and waveform losses stay in float32.
        enh, m, lsnr, df_alpha = (tensor.float() for tensor in outputs)

        batch.update(
            {
                "clean_spec": clean_spec,
                "noisy_spec": noisy_spec,
                "enh": enh,
                "m": m,
                "lsnr": lsnr,
                "df_alpha": df_alpha,
            }
        )

        # Convert enhanced STFT back to waveform for waveform-based metrics/losses.
        # enh: [B, 1, T, F, 2] -> istft expects [B, F, T] complex
        loss_on_waveform = bool(self.cfg_trainer.get("loss_on_waveform", False))
        if not self.is_train or metric_funcs or loss_on_waveform:
            enh_wav = synthesize_enhancement(
                enh if loss_on_waveform else enh.detach(),
                batch["lengths"], output_length=batch["noisy"].shape[-1],
            )
            batch.update({"enh_wav": enh_wav, "separated_audio": enh_wav,
                          "audio_target": batch["clean"], "audio_mix": batch["noisy"]})

        all_losses = self.criterion(
            clean=clean_spec,
            noisy=noisy_spec,
            enh=enh,
            m=m,
            lsnr=lsnr,
            df_alpha=df_alpha,
            frame_lengths=frame_lengths,
            lengths=batch["lengths"],
        )
        batch.update(all_losses)

        if not torch.isfinite(batch["loss"]):
            raise FloatingPointError("Non-finite loss during batch processing")

        if self.is_train:
            self.grad_scaler.scale(batch["loss"]).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self._clip_grad_norm()
            previous_scale = self.grad_scaler.get_scale()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            optimizer_stepped = self.grad_scaler.get_scale() >= previous_scale
            batch["optimizer_stepped"] = optimizer_stepped
            if (
                optimizer_stepped
                and self.lr_scheduler is not None
                and self.cfg_trainer.get("lr_scheduler_interval", "legacy") == "legacy"
                and step is not None
                and step % 3 == 2
            ):
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=int(frame_lengths.sum().item()))

        for met in metric_funcs:
            metrics.update(met.name, met(**batch), n=batch["noisy"].shape[0])
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
