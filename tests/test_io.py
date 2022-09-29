import numpy as np
from blazingai.io import save_metrics, save_predictions


def test_save_metrics(tmp_path):
    fpath = tmp_path / "model_one.score"
    save_metrics(
        fpath=fpath,
        metric="rmse",
        train_metric=0.10,
        cv_metric=0.05,
        oof_metric=0.07,
    )
    with open(fpath) as f:
        assert (
            f.readline()
            == '{"train rmse": 0.1, "cv rmse": 0.05, "oof rmse": 0.07}'
        )


def test_save_predictions(tmp_path):
    fpath = tmp_path / "model_weights.npy"
    preds = [1, 2, 3, 4, 5, 6]
    save_predictions(fpath=fpath, preds=preds)

    result = np.load(fpath)
    np.testing.assert_array_almost_equal(result, np.array(preds))
