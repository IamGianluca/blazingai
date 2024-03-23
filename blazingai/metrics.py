from typing import List

import numpy as np
import torch
import torchmetrics
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassF1Score
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

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.target.update(target)
        self.preds.update(preds)

    def compute(self) -> torch.Tensor:
        """Computes mean squared error over state."""
        preds = self.preds.compute()
        target = self.target.compute()

        rmse_scores = []
        for _ in range(2):
            rmse_scores.append(
                mean_squared_error(
                    preds=preds,
                    target=target,
                    squared=False,  # rmse
                )
            )
        return torch.mean(torch.tensor(rmse_scores))


class CrossValMetricsTracker:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.metric = cfg.metric
        self._trgt: List[torch.Tensor] = []
        self._pred: List[torch.Tensor] = []
        self._val_scores: List[torch.Tensor] = []
        self._trn_scores: List[torch.Tensor] = []

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

    def add(
        self,
        trgt: List[torch.Tensor],  # y_true test set
        pred: List[torch.Tensor],  # y_pred test set
        val_score: torch.Tensor,
        trn_score: torch.Tensor,
    ):
        self._trgt.extend(trgt)
        self._pred.extend(pred)
        self._val_scores.append(val_score)
        self._trn_scores.append(trn_score)


def metric_factory(cfg: DictConfig):
    if cfg.metric == "binary_accuracy":
        return torchmetrics.Accuracy(task="binary")
    if cfg.metric == "binary_auc":
        # return torchmetrics.AUROC(pos_label=1)
        return torchmetrics.AUROC(
            task="binary",
            pos_label=1,
            num_classes=cfg.num_classes,
        )
    elif cfg.metric == "mse":
        return torchmetrics.MeanSquaredError(squared=True)
    elif cfg.metric == "rmse":
        return torchmetrics.MeanSquaredError(squared=False)
    elif cfg.metric == "mcrmse":
        return MeanColumnwiseRootMeanSquaredError()
    elif cfg.metric == "multiclass_f1_macro":
        return MulticlassF1Score(average="macro", num_classes=2)
    else:
        raise ValueError(f"{cfg.metric} is not supported yet.")
