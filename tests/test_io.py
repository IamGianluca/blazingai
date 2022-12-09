import numpy as np
import pytest
import torch
from blazingai.io import save_mtrc, save_pred


class CrossValMetricsFake:
    metric = "rmse"
    trn_metric = 0.1
    val_metric = 0.05
    oof_metric = 0.07


@pytest.fixture
def metric():
    return CrossValMetricsFake()


def test_save_metrics(tmp_path, metric):
    fpath = tmp_path / "model_one.score"
    save_mtrc(fpath=fpath, metrics=metric)
    with open(fpath) as f:
        assert f.readline() == '{"train rmse": 0.1, "cv rmse": 0.05, "oof rmse": 0.07}'


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_save_predictions(tmp_path, dtype):
    # given
    fpath = tmp_path / "model_weights.npy"
    preds = [
        torch.tensor([1, 2]).to(dtype),
        torch.tensor([3, 4]).to(dtype),
        torch.tensor([5, 6]).to(dtype),
    ]

    # when
    save_pred(fpath=fpath, pred=preds)

    # then
    result = np.load(fpath)
    np.testing.assert_array_almost_equal(result, np.array([[1, 2], [3, 4], [5, 6]]))
