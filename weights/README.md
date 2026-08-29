# Model checkpoints

This directory contains only checkpoints used in the final reported analyses.
Every file is below GitHub's 100 MB per-file limit. Integrity hashes are listed
in `SHA256SUMS`.

| Directory | Contents | Intended use |
|---|---|---|
| `raw_rc/` | Five grouped-fold raw-sequence models with reverse-complement augmentation | Principal held-out ensemble |
| `raw_no_rc/` | Five grouped-fold raw-sequence models without augmentation | Augmentation comparison |
| `frozen_nt/` | Five attention-pooling heads | Frozen Nucleotide Transformer ensemble; backbone downloaded separately |
| `motif/` | Five HOCOMOCO motif-transformer checkpoints | Motif representation comparison |
| `full_development/` | Raw+RC and raw-no-RC full-development fits | Post hoc interpretation; not principal model ranking |

PyTorch checkpoints can execute arbitrary Python during unsafe deserialisation.
Load these trusted project files with `weights_only=True` where their format
permits it. The full-development files also contain training metadata and may
require `weights_only=False`; do not use that mode for untrusted checkpoints.

