import pytest
import torch
from blazingai.metrics import CrossValMetricsTracker, MeanColumnwiseRootMeanSquaredError
from omegaconf import DictConfig

from torchmetrics.regression.mse import MeanSquaredError


def test_oof_metric():
    # given
    cfg = DictConfig({"metric": "binary_accuracy"})  # default threshold: 0.5
    cv_metrics = CrossValMetricsTracker(cfg)

    # fold 1; first prediction is wrong
    trgt1 = [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])]
    pred1 = [torch.tensor([0.4, 0.2, 0.9]), torch.tensor([0.3, 0.7, 0.4])]
    trn_score1 = torch.tensor(0.92)
    val_score1 = torch.tensor(0.85)

    # fold 2
    trgt2 = [torch.tensor([1, 1, 0]), torch.tensor([1, 0, 1])]
    pred2 = [torch.tensor([0.9, 0.6, 0.1]), torch.tensor([0.7, 0.4, 0.8])]
    trn_score2 = torch.tensor(0.88)
    val_score2 = torch.tensor(0.80)

    # when
    cv_metrics.add(trgt1, pred1, val_score1, trn_score1)
    cv_metrics.add(trgt2, pred2, val_score2, trn_score2)

    # then
    assert cv_metrics.trn_metric == pytest.approx((0.92 + 0.88) / 2)
    assert cv_metrics.val_metric == pytest.approx((0.85 + 0.80) / 2)
    assert cv_metrics.oof_metric == pytest.approx(0.916666)


@pytest.mark.parametrize("squared,result", [(True, 0.002075), (False, 0.045552168)])
def test_mse_across_batches(squared, result):
    # given
    metric_fn = MeanSquaredError(squared=squared)
    batched_pred_and_trgt = [
        [torch.tensor([1.0, 1.0]), torch.tensor([1.00, 0.91])],
        [torch.tensor([0.9, 0.8]), torch.tensor([0.91, 0.79])],
    ]

    # when
    for batch in batched_pred_and_trgt:
        pred, trgt = batch
        metric_fn.update(preds=pred, target=trgt)

    # then
    metric = metric_fn.compute()
    assert metric.allclose(torch.tensor(result))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mcrmse(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping unit test case.")
    trgt = torch.tensor([[1.0, 0.9], [0.95, 1.0]]).to(device)
    pred = torch.tensor([[0.8, 1.1], [0.95, 1.0]]).to(device)

    metric_fn = MeanColumnwiseRootMeanSquaredError()
    metric_fn.update(pred, trgt)
    metric = metric_fn.compute()
    assert metric.allclose(torch.tensor(0.14142135))
