"""Check IA3 through the normal module call, without downloading a backbone."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.train_nt_ia3 import IA3Regressor


class TinyESM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 512)
        layer = nn.Module()
        layer.attention = nn.Module()
        layer.attention.self = nn.Module()
        layer.attention.self.key = nn.Linear(512, 512)
        layer.attention.self.value = nn.Linear(512, 512)
        layer.intermediate = nn.Linear(512, 2048)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([layer])

    def forward(self, input_ids, **kwargs):
        layer = self.encoder.layer[0]
        x = self.embedding(input_ids)
        x = layer.attention.self.key(x) + layer.attention.self.value(x)
        x = layer.intermediate(x).reshape(*x.shape[:2], 4, 512).mean(2)
        return SimpleNamespace(last_hidden_state=x)


def make_model():
    nt = nn.Module()
    nt.esm = TinyESM()
    return IA3Regressor(nt)


def test_standard_forward_and_frozen_parameters():
    model = make_model().eval()
    ids = torch.tensor([[0, 1, 2], [0, 3, 4]])
    mask = torch.ones_like(ids)
    output = model(ids, mask)
    torch.testing.assert_close(output, model.encoder_forward(ids, mask))
    assert output.shape == (2, 20)
    output.square().mean().backward()
    gates = [(n, p) for n, p in model.nt.named_parameters() if "ia3_scale" in n]
    assert sum(p.numel() for _, p in gates) == 3072
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in gates)
    assert all(
        not p.requires_grad and p.grad is None
        for n, p in model.nt.named_parameters()
        if "ia3_scale" not in n
    )


def test_incomplete_adapter_rejected():
    model = make_model()
    state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.load_adapter(state)
    state.pop(next(iter(state)))
    with pytest.raises(ValueError, match="missing"):
        model.load_adapter(state)
