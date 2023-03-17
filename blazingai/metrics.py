from typing import Any, Callable, Optional

import numpy as np
import torch
import torchmetrics
from joblib.parallel import FallbackToBackend
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.metric import Metric


def compute_oof_metric(cfg: DictConfig, y_true, y_pred) -> np.float32:
    metric = metric_factory(cfg=cfg)
    y_pred = torch.vstack(y_pred)
    y_true = torch.vstack(y_true)
    return np.float32(metric(y_pred, y_true))


class MeanColumnwiseRootMeanSquaredError(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.target = torchmetrics.CatMetric()
        self.preds = torchmetrics.CatMetric()

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.target.update(target)
        self.preds.update(preds)

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        preds = self.preds.compute()
        target = self.target.compute()

        rmse_scores = []
        for i in range(2):
            rmse_scores.append(
                mean_squared_error(
                    preds=preds,
                    target=target,
                    squared=False,  # rmse
                )
            )
        return torch.mean(torch.tensor(rmse_scores))


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
    elif cfg.metric == "mcrmse":
        return MeanColumnwiseRootMeanSquaredError()
    elif cfg.metric == "f1":
        return torchmetrics.F1Score()
    else:
        raise ValueError(f"{cfg.metric} is not supported yet.")
