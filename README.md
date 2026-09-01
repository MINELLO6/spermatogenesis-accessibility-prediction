# Predicting Developmental Chromatin Accessibility from DNA Sequence

Reproducible code, compact results, and publication figures for comparing
task-specific convolutional models, a frozen Nucleotide Transformer, and
HOCOMOCO motif representations on mouse spermatogenesis accessibility profiles.

The repository also contains a matched row-normalized Poisson experiment on
960,329 temporally variable regions. After every 20-bin profile was rescaled to
100 pseudo-counts, the raw-sequence CNN achieved shape R2 0.1958, compared with
0.1086 for the dense motif CNN and 0.1003 for the dense motif Transformer.

## Main result

On 485,229 locked held-out regions, the five-fold reverse-complement raw-sequence
ensemble achieved a mean per-bin R2 of 0.611. Corresponding results were 0.591
without reverse-complement augmentation, 0.535 for the frozen Nucleotide
Transformer ensemble, and 0.443 for the motif transformer ensemble.

## Repository layout

```text
configs/                 Locked pre-test analysis plan
scripts/diagnostics/     Split and target diagnostics
scripts/training/        Raw, pretrained, and motif training programs
scripts/evaluation/      Locked ensemble evaluation and grouped bootstrap
scripts/interpretation/  Integrated gradients and in-silico mutagenesis
scripts/figures/         Figure-generation programs
src/                     Shared models, encoders, grouping, and metrics
tests/                   Unit tests for the shared package
results/                 Compact metrics and diagnostics (no individual data)
figures/                 PDF figures used in the dissertation
weights/                 Final fold ensembles and full-development checkpoints
paper/                   LaTeX draft and bibliography
```

## Reproducibility status

The repository contains analysis code, compact numerical outputs, and the 22
final model checkpoints used by the reported analyses. It does not redistribute
input genomic data, full prediction arrays, or per-region attribution arrays.
Those files are too large for Git and may be subject to data-sharing
restrictions. See `DATA.md` for the expected inputs.

The held-out test set was locked before final comparison. The main comparison
uses equal-weight averages of the five development-fold checkpoints. Uncertainty
for model differences uses 2,000 Poisson(1) multiplier-bootstrap replicates at
the 200-bp genomic overlap-group level.

## Environment

Python 3.10 or 3.11 is recommended. Install the analysis dependencies with:

```bash
python -m pip install -r requirements.txt
```

For an editable installation of the shared package and development tools:

```bash
python -m pip install -e ".[dev]"
pytest
```

A CUDA-capable PyTorch installation is required for model training and
interpretation. Paths in the original experiment scripts reflect the compute
server layout and should be adjusted through the constants at the top of each
script before running on another system.

## Typical workflow

1. Prepare the row-aligned inputs described in `DATA.md`.
2. Run development-set diagnostics and verify that genomic groups do not cross
   the development/test boundary.
3. Train the five grouped folds for each prespecified model.
4. Evaluate equal-weight ensembles once on the locked held-out indices.
5. Run the grouped bootstrap and generate final figures.
6. Fit the full-development raw model for post hoc attribution only.

The normalized-Poisson shape comparison can be reproduced with
`scripts/training/train_shape_shared_trunk.py`; compact five-fold results are in
`results/metrics/shape_normalized_poisson/summary.csv`.

## Interpretation caveat

Integrated-gradient magnitude showed weak positional agreement with exhaustive
single-base substitution effects. Attribution maps should therefore be treated
as model-sensitivity diagnostics and hypothesis generators, not as evidence that
individual bases or motifs are causally regulatory.
