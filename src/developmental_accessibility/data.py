"""Data encoding and genomic grouping helpers."""

from collections.abc import Sequence

import numpy as np
import torch


def one_hot_encode(sequences: Sequence[str]) -> torch.Tensor:
    """Encode equal-length A/C/G/T strings as float32 tensors of shape N x 4 x L."""
    if len(sequences) == 0:
        raise ValueError("At least one sequence is required")
    length = len(sequences[0])
    if length == 0:
        raise ValueError("Sequences must not be empty")
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


def overlap_groups(
    chromosome: Sequence[str], position: Sequence[int], gap: int = 200
) -> np.ndarray:
    """Group overlapping windows, returning assignments in the original row order."""
    chromosome = np.asarray(chromosome)
    position = np.asarray(position)
    if chromosome.ndim != 1 or position.ndim != 1:
        raise ValueError("Chromosome and position must be one-dimensional")
    if gap < 0 or not np.isfinite(gap):
        raise ValueError("Gap must be finite and nonnegative")
    if not np.issubdtype(position.dtype, np.number) or not np.isfinite(position).all():
        raise ValueError("Positions must be finite numbers")
    if len(chromosome) != len(position):
        raise ValueError("Chromosome and position arrays must have equal length")
    if len(position) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((position, chromosome))
    chromosome, position = chromosome[order], position[order]
    new_group = np.ones(len(position), dtype=bool)
    new_group[1:] = (chromosome[1:] != chromosome[:-1]) | (position[1:] - position[:-1] > gap)
    groups = np.empty(len(position), dtype=np.int64)
    groups[order] = np.cumsum(new_group, dtype=np.int64) - 1
    return groups
