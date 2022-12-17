from pathlib import Path

from lightning.pytorch.core.saving import load_hparams_from_yaml
from omegaconf import DictConfig


def load_cfg(fpath: Path, cfg_name: str) -> DictConfig:
    cfg = load_hparams_from_yaml(config_yaml=fpath, use_omegaconf=True)
    cfg = cfg.get(cfg_name)
    if hasattr(cfg, "lr"):
        cfg.lr = float(cfg.lr)  # type: ignore
    return cfg
