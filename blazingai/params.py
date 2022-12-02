from pathlib import Path

from lightning.pytorch.core.saving import load_hparams_from_yaml
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError


def load_cfg(fpath: Path, cfg_name: str) -> DictConfig:
    cfg: DictConfig = load_hparams_from_yaml(config_yaml=fpath, use_omegaconf=True)
    cfg = cfg.get(cfg_name)
    try:
        cfg.lr = float(cfg.lr)
    except ConfigAttributeError:
        pass
    return cfg
