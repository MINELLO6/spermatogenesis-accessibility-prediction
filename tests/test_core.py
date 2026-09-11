import numpy as np
import torch

from developmental_accessibility.data import one_hot_encode, overlap_groups, reverse_complement
from developmental_accessibility.metrics import regression_metrics
from developmental_accessibility.models import FrozenNucleotideTransformerHead, RawMultiScaleResCNN


def test_raw_model_shape():
    model = RawMultiScaleResCNN().eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 4, 201)).shape == (2, 20)


def test_nt_head_shape():
    model = FrozenNucleotideTransformerHead().eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 34, 512), torch.ones(2, 34, dtype=torch.bool)).shape == (2, 20)


def test_encoding_and_reverse_complement():
    encoded = one_hot_encode(["ACGT"])
    assert torch.equal(reverse_complement(encoded), encoded)


def test_overlap_groups():
    groups = overlap_groups(["1", "1", "1", "2"], [100, 250, 600, 10])
    assert groups.tolist() == [0, 0, 1, 2]


def test_metrics_for_perfect_prediction():
    y = np.arange(60, dtype=float).reshape(3, 20)
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mean_r2"] == 1.0
