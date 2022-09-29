import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


def save_predictions(fpath: Path, preds: ArrayLike) -> None:
    preds = np.array(preds)
    np.save(fpath, preds)


def save_metrics(
    fpath: Path,
    metric: str,
    train_metric: float,
    cv_metric: float,
    oof_metric: float,
) -> None:
    data = {}
    data[f"train {metric}"] = round(train_metric, 4)
    data[f"cv {metric}"] = round(cv_metric, 4)
    data[f"oof {metric}"] = round(oof_metric, 4)
    with open(fpath, "w") as f:
        json.dump(data, f)
