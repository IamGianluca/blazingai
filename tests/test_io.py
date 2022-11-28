import numpy as np
from blazingai.io import save_metrics, save_pred


def test_save_metrics(tmp_path):
    fpath = tmp_path / "model_one.score"
    save_metrics(
        fpath=fpath,
        metric="rmse",
        trn_metric=0.10,
        val_metric=0.05,
        oof_metric=0.07,
    )
    with open(fpath) as f:
        assert f.readline() == '{"train rmse": 0.1, "cv rmse": 0.05, "oof rmse": 0.07}'


def test_save_predictions(tmp_path):
    fpath = tmp_path / "model_weights.npy"
    preds = [1, 2, 3, 4, 5, 6]
    save_pred(fpath=fpath, pred=preds)

    result = np.load(fpath)
    np.testing.assert_array_almost_equal(result, np.array(preds))
