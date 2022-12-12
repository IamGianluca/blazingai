import numpy as np
import torch
import torchmetrics
from omegaconf import DictConfig


def compute_oof_metric(cfg: DictConfig, y_true, y_pred) -> np.float32:
    metric = metric_factory(cfg=cfg)
    y_pred = torch.vstack(y_pred)
    y_true = torch.vstack(y_true)
    return np.float32(metric(y_pred, y_true))


class CrossValMetrics:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.metric = cfg.metric
        self._trgt = []
        self._pred = []
        self._val_scores = []
        self._trn_scores = []

    @property
    def trn_metric(self):
        return float(np.mean(self._trn_scores))

    @property
    def val_metric(self):
        return float(np.mean(self._val_scores))

    @property
    def oof_metric(self):
        return float(
            compute_oof_metric(cfg=self.cfg, y_pred=self._pred, y_true=self._trgt)
        )

    def add(self, trgt, pred, val_score, trn_score):
        self._trgt.extend(trgt)
        self._pred.extend(pred)
        self._val_scores.append(val_score)
        self._trn_scores.append(trn_score)


def metric_factory(cfg: DictConfig):
    if cfg.metric == "auc":
        # return torchmetrics.AUROC(pos_label=1)
        return torchmetrics.AUROC(
            pos_label=1,
            num_classes=cfg.num_classes,
        )
    elif cfg.metric == "mse":
        return torchmetrics.MeanSquaredError(squared=True)
    elif cfg.metric == "rmse":
        return torchmetrics.MeanSquaredError(squared=False)
    else:
        raise ValueError("Metric not supported yet.")
