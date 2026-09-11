"""Strictly load all 22 release checkpoints and report their content hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from developmental_accessibility.models import (
    FrozenNucleotideTransformerHead,
    MotifTransformer,
    RawMultiScaleResCNN,
)
from developmental_accessibility.paths import WEIGHTS_ROOT


def verify(root):
    records = []
    for family, prefix, model_type in [
        ("raw_rc", "raw_rc", RawMultiScaleResCNN),
        ("raw_no_rc", "raw", RawMultiScaleResCNN),
        ("frozen_nt", "nt_attnpool", FrozenNucleotideTransformerHead),
        ("motif", "motif_transformer", MotifTransformer),
    ]:
        for fold in range(1, 6):
            path = root / family / f"{prefix}_fold{fold}.pt"
            model = model_type()
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )
            records.append(
                {
                    "file": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    for name in ["raw_rc", "raw_no_rc"]:
        path = root / "full_development" / f"{name}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        # The release stores state dictionaries; original experiment files may wrap them.
        if "model_state" in state:
            state = state["model_state"]
        RawMultiScaleResCNN().load_state_dict(state, strict=True)
        records.append(
            {
                "file": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.weights_root)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
