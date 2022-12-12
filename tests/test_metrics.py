import pytest
import torch

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
