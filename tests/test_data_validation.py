import pickle

import numpy as np
import pytest

from scripts.diagnostics.validate_data import validate


def make_data(root):
    n = 12
    (root / "totmat_shape.txt").write_text(f"{n} 20\n")
    np.ones((n, 20), dtype="<f8").tofile(root / "totmat_f64.bin")
    (root / "fullseqs.txt").write_bytes((b"A" * 201 + b"\n") * n)
    dev = np.arange(10)
    folds = [(dev[dev % 5 != k], dev[dev % 5 == k]) for k in range(5)]
    with (root / "folds.pkl").open("wb") as handle:
        pickle.dump(folds, handle)
    return folds


def test_valid_partitions(tmp_path):
    make_data(tmp_path)
    result = validate(tmp_path)
    assert result["development"] == 10
    assert result["heldout"] == 2


def test_training_leak_rejected(tmp_path):
    folds = make_data(tmp_path)
    folds[0] = (np.append(folds[0][0], 11), folds[0][1])
    with (tmp_path / "folds.pkl").open("wb") as handle:
        pickle.dump(folds, handle)
    with pytest.raises(ValueError, match="same development set"):
        validate(tmp_path)


def test_windows_line_endings_rejected(tmp_path):
    make_data(tmp_path)
    (tmp_path / "fullseqs.txt").write_bytes((b"A" * 201 + b"\r\n") * 12)
    with pytest.raises(ValueError, match="byte size"):
        validate(tmp_path)
