"""Canonical model definitions shared by training, evaluation, and interpretation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class RawMultiScaleResCNN(nn.Module):
    """The selected 201-bp raw-sequence architecture."""

    def __init__(self, n_outputs: int = 20) -> None:
        super().__init__()

        def branch(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(4, 64, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

        self.branch7 = branch(7)
        self.branch15 = branch(15)
        self.branch31 = branch(31)
        self.project = nn.Sequential(
            nn.Conv1d(192, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.blocks = nn.Sequential(
            *(DilatedResidualBlock(128, dilation) for dilation in (1, 2, 4, 8))
        )
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.branch7(x), self.branch15(x), self.branch31(x)), dim=1)
        x = self.blocks(self.project(x))
        pooled = torch.cat((x.mean(dim=2), x.max(dim=2).values), dim=1)
        return self.head(pooled)


class FrozenNucleotideTransformerHead(nn.Module):
    """Attention-plus-maximum pooling head for 512-dimensional frozen token states."""

    def __init__(self, n_outputs: int = 20) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256))
        self.attn = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_outputs),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(tokens)
        scores = self.attn(x).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        attention_pool = (x * weights.unsqueeze(-1)).sum(dim=1)
        maximum_pool = x.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        return self.head(torch.cat((attention_pool, maximum_pool), dim=-1))

