import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer.tech_evaluator import TechEvaluator
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="tech_eval")
def main(config):
    """
    Main script for model evaluation. Instantiates the model and
    dataloaders (with one example dataset). Runs tech_evaluator to calculate models' metrics and save them.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.evaluator.seed)

    if config.evaluator.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.evaluator.device

    dataloaders, batch_transforms = get_dataloaders(
        config, device
    )

    model = instantiate(
        config.model
    ).to(device)

    save_path = ROOT_PATH / "data" / "saved" / config.evaluator.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    evaluator = TechEvaluator(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        skip_model_load=True,
    )

    _ = evaluator.run_tech_evaluation()
    return


if __name__ == "__main__":
    main()