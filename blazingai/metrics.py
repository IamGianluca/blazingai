from typing import Any, Callable, Optional

import numpy as np
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.metric import Metric


def compute_oof_metric(cfg: DictConfig, y_true, y_pred) -> float:
    metric = metric_factory(cfg=cfg)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    return metric(y_pred, y_true)


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
        return np.mean(self._trn_scores)

    @property
    def val_metric(self):
        return np.mean(self._val_scores)

    @property
    def oof_metric(self):
        return compute_oof_metric(cfg=self.cfg, y_pred=self._pred, y_true=self._trgt)

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
        return RootMeanSquaredError()
    else:
        raise ValueError("Metric not supported yet.")


# TODO: check if new upstream implementation works
# https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/regression/mse.py#L23
class RootMeanSquaredError(Metric):

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.target = torch.empty(0).cuda()
        self.preds = torch.empty(0).cuda()

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.target = torch.cat([self.target, target], 0)
        self.preds = torch.cat([self.preds, preds], 0)

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        preds = self.preds
        target = self.target

        # reset for next epoch
        self.preds = torch.empty(0).cuda()
        self.target = torch.empty(0).cuda()

        return mean_squared_error(
            preds=preds,
            target=target,
            squared=False,
        )
