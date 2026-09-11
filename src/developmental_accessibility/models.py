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
        self.project = nn.Sequential(nn.Conv1d(192, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
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


class MotifTokenEncoder(nn.Module):
    def __init__(self, d_model=128, use_position=True):

        super().__init__()

        self.use_position = use_position

        # 0 = padding
        # motif IDs: 1..1443
        self.motif_emb = nn.Embedding(1444, d_model, padding_idx=0)

        # 0 = padding
        # bins 1..10
        self.pos_emb = nn.Embedding(11, d_model, padding_idx=0)

        self.score_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, motif, pos, score, mask):

        x = self.motif_emb(motif)

        if self.use_position:
            x = x + self.pos_emb(pos)

        x = x + self.score_mlp(score.unsqueeze(-1))

        x = self.norm(x)

        x = x * mask.unsqueeze(-1)

        return x


class MotifBagMLP(nn.Module):
    def __init__(self, d_model=128):

        super().__init__()

        self.encoder = MotifTokenEncoder(d_model=d_model, use_position=True)

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model, 20)
        )

    def forward(self, motif, pos, score, mask):

        x = self.encoder(motif, pos, score, mask)

        m = mask.unsqueeze(-1)

        # mean
        denom = m.sum(dim=1).clamp(min=1)

        mean_pool = (x * m).sum(dim=1) / denom

        # max
        max_x = x.masked_fill(~m, -1e9)

        max_pool = max_x.max(dim=1).values

        empty = mask.sum(dim=1) == 0

        max_pool[empty] = 0

        z = torch.cat([mean_pool, max_pool], dim=-1)

        return self.head(z)


class MotifTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, ff_dim=512, dropout=0.1):

        super().__init__()

        self.token_encoder = MotifTokenEncoder(d_model=d_model, use_position=True)

        self.region_token = nn.Parameter(torch.zeros(1, 1, d_model))

        nn.init.normal_(self.region_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 20)
        )

    def forward(self, motif, pos, score, mask):

        x = self.token_encoder(motif, pos, score, mask)

        B = x.shape[0]

        cls = self.region_token.expand(B, -1, -1)

        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)

        full_mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=~full_mask)

        x = self.final_norm(x)

        cls_out = x[:, 0]

        token_x = x[:, 1:]

        m = mask.unsqueeze(-1)

        denom = m.sum(dim=1).clamp(min=1)

        mean_pool = (token_x * m).sum(dim=1) / denom

        max_pool = token_x.masked_fill(~m, -1e9).max(dim=1).values

        empty = mask.sum(dim=1) == 0

        max_pool[empty] = 0

        z = torch.cat([cls_out, mean_pool, max_pool], dim=-1)

        return self.head(z)
