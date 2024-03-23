from pathlib import Path

from lightning.pytorch.core.saving import load_hparams_from_yaml
from omegaconf import DictConfig


def load_cfg(fpath: Path, cfg_name: str) -> DictConfig:
    all_configs = load_hparams_from_yaml(config_yaml=fpath, use_omegaconf=True)
    cfg = all_configs.get(cfg_name)
    if cfg is None:
        raise ValueError(f"Could not find config {cfg_name} in {fpath}")

    if hasattr(cfg, "lr"):  # convert exponential notation to float
        cfg.lr = float(cfg.lr)  # type: ignore

    return cfg
