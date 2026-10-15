import warnings
import signal
import json
from datetime import datetime, timezone

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.config_utils import setup_config

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    # Background shell jobs may inherit SIGINT=SIG_IGN. Restore checkpoint-safe shutdown.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    project_config = OmegaConf.to_container(config)
    writer = instantiate(config.writer, logger, project_config)
    setup_config(config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)
    print(sum(p.numel() for p in model.parameters()))

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    try:
        trainer.train()
        result = {"epochs_completed": trainer._last_epoch,
                  "epochs_requested": trainer.epochs,
                  "stop_reason": "epoch_budget" if trainer._last_epoch == trainer.epochs else "early_stopping",
                  "epoch_len": trainer.epoch_len,
                  "evaluation_period": trainer.evaluation_period,
                  "dataset_sizes": {name: len(loader.dataset) for name, loader in dataloaders.items()},
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "run_name": config.writer.run_name}
        temporary = trainer.checkpoint_dir / "training_result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(trainer.checkpoint_dir / "training_result.json")
    finally:
        if hasattr(writer, "close"):
            writer.close()


if __name__ == "__main__":
    main()
