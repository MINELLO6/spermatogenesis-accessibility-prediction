"""Boundary cases that previously failed or silently produced misleading results."""

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score

from developmental_accessibility.data import one_hot_encode, overlap_groups
from developmental_accessibility.metrics import regression_metrics


def test_encoding_numpy_input_and_unknown_base():
    x = one_hot_encode(np.array(["aN", "CT"]))
    assert x.shape == (2, 4, 2)
    assert x[0, :, 1].sum() == 0
    assert x[0, 0, 0] == 1


@pytest.mark.parametrize("sequences", [[], [""], ["A", "AA"]])
def test_invalid_sequences(sequences):
    with pytest.raises(ValueError):
        one_hot_encode(sequences)


def test_interleaved_unsorted_regions():
    groups = overlap_groups(["2", "1", "1", "2", "1"], [50, 600, 100, 100, 250])
    assert groups[0] == groups[3]
    assert groups[2] == groups[4]
    assert len(set(groups)) == 3


@pytest.mark.parametrize("position,gap", [([np.nan], 200), ([1], -1)])
def test_invalid_group_coordinates(position, gap):
    with pytest.raises(ValueError):
        overlap_groups(["1"], position, gap)


def test_metrics_match_sklearn_with_constant_columns():
    truth = np.array([[1.0, 4.0, 0.0], [1.0, 4.0, 2.0], [1.0, 4.0, 4.0]])
    pred = truth.copy()
    pred[:, 1] += 1
    pred[:, 2] += 0.5
    result = regression_metrics(truth, pred)
    np.testing.assert_allclose(result["r2_bins"], r2_score(truth, pred, multioutput="raw_values"))


@pytest.mark.parametrize("truth", [np.empty((0, 20)), np.ones((1, 20)), np.full((2, 20), np.nan)])
def test_invalid_metric_inputs(truth):
    with pytest.raises(ValueError):
        regression_metrics(truth, truth)


def test_sparse_motif_empty_regions_and_padding():
    from scripts.training.train_sparse_motif import MotifTransformer

    torch.manual_seed(12)
    model = MotifTransformer().eval()
    motif = torch.tensor([[1, 2], [0, 0]])
    pos = torch.tensor([[1, 3], [0, 0]])
    score = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    mask = motif != 0
    out = model(motif, pos, score, mask)
    assert out.shape == (2, 20)
    assert torch.isfinite(out).all()
    # Extra masked tokens must not change a region prediction.
    pad = lambda x: torch.nn.functional.pad(x, (0, 3))
    torch.testing.assert_close(model(pad(motif), pad(pos), pad(score), pad(mask)), out)
    out.square().mean().backward()
    assert model.region_token.grad is not None
