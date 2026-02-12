import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG = None

def setup_config(config):
    global CONFIG
    CONFIG = config

def get_config():
    global CONFIG
    if CONFIG is None:
        CONFIG = OmegaConf.load("src/configs/baseline.yaml")
    return CONFIG


def stft_config():
    global CONFIG
    if CONFIG is None:
        CONFIG = OmegaConf.load("src/configs/baseline.yaml")
    return CONFIG.stft