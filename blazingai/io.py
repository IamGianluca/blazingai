import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


def save_pred(fpath: Path, pred: ArrayLike) -> None:
    pred = np.array(pred)
    np.save(fpath, pred)


def save_metrics(
    fpath: Path,
    metric: str,
    trn_metric: float,
    val_metric: float,
    oof_metric: float,
) -> None:
    data = {}
    data[f"train {metric}"] = round(trn_metric, 4)
    data[f"cv {metric}"] = round(val_metric, 4)
    data[f"oof {metric}"] = round(oof_metric, 4)
    with open(fpath, "w") as f:
        json.dump(data, f)
