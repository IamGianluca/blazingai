import json
from pathlib import Path

import numpy as np
from blazingai.metrics import CrossValMetrics
from numpy.typing import ArrayLike


def save_pred(fpath: Path, pred: ArrayLike) -> None:
    pred = np.array(pred)
    np.save(fpath, pred)


def save_mtrc(fpath: Path, metrics: CrossValMetrics) -> None:
    data = {}
    data[f"train {metrics.metric}"] = round(metrics.trn_metric, 4)
    data[f"cv {metrics.metric}"] = round(metrics.val_metric, 4)
    data[f"oof {metrics.metric}"] = round(metrics.oof_metric, 4)
    with open(fpath, "w") as f:
        json.dump(data, f)


def print_mtrc(metric: str, trn_metric: float, val_metric: float) -> None:
    print(f"\nBest {metric}: Train {trn_metric:.4f}, Valid: {val_metric:.4f}")
