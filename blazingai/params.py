from pathlib import Path

from lightning.pytorch.core.saving import load_hparams_from_yaml
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError


def load_cfg(fpath: Path, cfg_name: str) -> DictConfig:
    cfg = load_hparams_from_yaml(config_yaml=fpath, use_omegaconf=True)
    cfg = cfg.get(cfg_name)
    try:
        if isinstance(cfg, str):
            cfg.lr = float(cfg.lr)
    except ConfigAttributeError:
        pass
    return cfg
