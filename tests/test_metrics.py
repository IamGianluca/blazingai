import pytest
import torch
from blazingai.metrics import MeanColumnwiseRootMeanSquaredError

from torchmetrics.regression.mse import MeanSquaredError


@pytest.mark.parametrize("squared,result", [(True, 0.002075), (False, 0.045552168)])
def test_mse_across_batches(squared, result):
    # given
    metric_fn = MeanSquaredError(squared=squared, compute_on_step=True)
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
