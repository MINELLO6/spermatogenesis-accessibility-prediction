"""Data encoding and genomic grouping helpers."""

from collections.abc import Sequence

import numpy as np
import torch


def one_hot_encode(sequences: Sequence[str]) -> torch.Tensor:
    """Encode equal-length A/C/G/T strings as float32 tensors of shape N x 4 x L."""
    if not sequences:
        raise ValueError("At least one sequence is required")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("All sequences must have the same length")
    chars = np.asarray([list(sequence.upper()) for sequence in sequences])
    encoded = np.stack([chars == base for base in "ACGT"], axis=1)
    return torch.from_numpy(encoded.astype(np.float32, copy=False))


def reverse_complement(x: torch.Tensor) -> torch.Tensor:
    """Reverse positions and exchange A/T and C/G channels for N x 4 x L tensors."""
    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError("Expected input with shape (batch, 4, length)")
    return x[:, (3, 2, 1, 0), :].flip(dims=(2,))


def overlap_groups(chromosome: Sequence[str], position: Sequence[int], gap: int = 200) -> np.ndarray:
    """Assign consecutive sorted regions to groups using chromosome and maximum gap."""
    chromosome = np.asarray(chromosome)
    position = np.asarray(position)
    if len(chromosome) != len(position):
        raise ValueError("Chromosome and position arrays must have equal length")
    if len(position) == 0:
        return np.empty(0, dtype=np.int64)
    new_group = np.ones(len(position), dtype=bool)
    new_group[1:] = (chromosome[1:] != chromosome[:-1]) | (
        position[1:] - position[:-1] > gap
    )
    return np.cumsum(new_group, dtype=np.int64) - 1

