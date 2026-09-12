# Predicting Developmental Chromatin Accessibility from DNA Sequence

Reproducible code, compact results, and publication figures for comparing
task-specific convolutional models, a frozen Nucleotide Transformer, and
HOCOMOCO motif representations on mouse spermatogenesis accessibility profiles.

The repository also contains a matched row-normalized Poisson experiment on
864,133 temporally variable development regions (960,329 before test-set exclusion).
After every 20-bin profile was rescaled to
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
interpretation. Run commands from the repository root using `python -m`, after
installing the shared package. The training programs target Linux CUDA systems.
Configure locations once, rather than editing individual scripts:

```bash
export ACCESSIBILITY_DATA_ROOT=/path/to/row_aligned_data
export ACCESSIBILITY_RUN_ROOT=/path/to/new_experiment_outputs
export ACCESSIBILITY_NT_PATH=/path/to/cached_nucleotide_transformer
```

Defaults are `data/`, `runs/`, and `models/nucleotide-transformer-v2-50m/` within
the repository. `ACCESSIBILITY_WEIGHTS_ROOT` optionally replaces `weights/`.
NT uses the locally cached `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
backbone (original revision `81b29e5786726d891dbf929404ef20adca5b36f1`).
It is not downloaded during training or tests. For PDF/draw.io figure conversion,
install the additional dependencies with `python -m pip install -e ".[figures]"`.

Check inputs before training:

```bash
python -m scripts.diagnostics.validate_data --check-groups
python -m scripts.diagnostics.verify_weights
python -m scripts.summarize_nt_layerwise
```

The validator checks array byte sizes, sequence record boundaries, finite targets,
all five train/validation partitions, and, with `--check-groups`, genomic-group
separation. It does not replace provenance checks on the biological inputs.

### Download the Nucleotide Transformer once

After installing the package, download the recorded snapshot with:

```bash
python -m scripts.download_nt_model
python -m scripts.download_nt_model --check-only
```

The destination follows `ACCESSIBILITY_NT_PATH`, or defaults to
`models/nucleotide-transformer-v2-50m/`. For a different location:

```bash
python -m scripts.download_nt_model --output-dir /path/to/nt-v2-50m
export ACCESSIBILITY_NT_PATH=/path/to/nt-v2-50m
python -m scripts.download_nt_model --check-only
python -m scripts.training.train_nt_heads --head attnpool --fold 0
```

The downloader pins revision `81b29e5786726d891dbf929404ef20adca5b36f1` of
[the InstaDeepAI NT-v2-50M multispecies model](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species).
It retrieves the tokenizer, configuration, custom Python modules, and Safetensors
weights, excluding the duplicate PyTorch weights and JAX checkpoint. It records
the requested version and completion status in `accessibility_snapshot.json`.
An interrupted download can be rerun. An existing unrecorded model directory is
left untouched: use a new directory for this helper, or continue using your
existing cache directly with the training scripts.

`--check-only` checks the record, required nonempty files, and core model
dimensions offline; it does not load the weights or prove their numerical
integrity. The downloader itself does not execute the model's custom code.
Training uses `AutoTokenizer` and `AutoModelForMaskedLM` with
`trust_remote_code=True` and `local_files_only=True`. Missing local files therefore
cause a failure rather than a different checkpoint being downloaded silently.
The regression head consumes token hidden states, not vocabulary logits, and
pooling excludes padding and the leading classification token. Frozen-feature
training uses `no_grad()` for the backbone; IA3 keeps the encoder gradient path
while freezing its original weights. Respect the upstream model's
CC-BY-NC-SA-4.0 licence when reusing it.

## Model entry points

Training fold arguments are zero-based (0–4). Each command accepts `--help`.

| Analysis | Module | Key arguments |
|---|---|---|
| Raw CNN | `scripts.training.train_models` | `--model raw --fold 0` |
| Raw CNN with RC | `scripts.training.train_raw_rc` | `--fold 0` |
| RC with attention pooling | `scripts.training.train_raw_rc_attn` | `--fold 0` |
| ConvNeXt comparison | `scripts.training.train_convnext` | `--fold 0` |
| Frozen NT | `scripts.training.train_nt_heads` | `--head attnpool --fold 0` |
| Frozen layer probe | `scripts.train_nt_layerwise` | `--fold 0` |
| IA3 | `scripts.train_nt_ia3` | `--fold 0` |
| Sparse motif Transformer (primary ensemble) | `scripts.training.train_sparse_motif` | `--mode train --model transformer --fold 0 --max-tokens 168` |
| Dense score/presence Transformer screen | `scripts.training.train_motif_transformer_allregions` | `--fold 0` |
| Dense score-only magnitude comparisons | `scripts.training.train_dense_motif_magnitude` | `--variant conv --fold 0` |
| Shared-trunk shape comparisons | `scripts.training.train_shape_shared_trunk` | `--modality motif --objective equal_ce --fold 0` |

For example, run `python -m scripts.training.train_raw_rc --fold 0`.
The sparse motif input can be constructed from `allmot_f64.bin` with
`python -m scripts.training.train_sparse_motif --mode prepare`.
The dense screen is a different model from the primary sparse Transformer.

New primary fold checkpoints are written to `$ACCESSIBILITY_RUN_ROOT/results`.
The ensemble evaluators deliberately load the published, frozen checkpoints in
`weights/`; evaluating newly trained checkpoints requires placing them in the
same family subdirectories under a separate `ACCESSIBILITY_WEIGHTS_ROOT`.
Post hoc IA3 and layer-mixture evaluation uses their saved run directories.

## Verification

```bash
python -m pytest -q
ruff format --check scripts src tests
ruff check scripts src tests --select E9,F63,F7,F82
python -m scripts.diagnostics.smoke_training --output-root runs/checks --include-backbone
```

The last command performs one-epoch integration runs on newly generated synthetic
data and writes a report and per-model logs in a unique directory. It needs a GPU;
omit `--include-backbone` to skip the three cached-NT checks. These checks exercise
data loading, optimization, validation and checkpoint writing. They are not a
reproduction of the full dissertation experiments. See `CODE_REVIEW.md` for the
scope of the source recovery and verification.

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

## September 2026 revision

The post hoc GELU versus Softplus encoder ablation, five-fold configurations,
metrics and reproducible audit are in
`results/shape_activation_ablation_20260911/`. Softplus reduced shape R2 in all
five paired folds, so the original Transformer results retain GELU internally.

`paper/dissertation_draft.tex` is an archived manuscript snapshot, not a live
mirror of the Overleaf project. This code revision does not change the manuscript
or replace its architecture figures. The approved model-comparison plot can be
regenerated with `python -m scripts.figures.plot_model_comparison`; its display
values and provenance note are in `results/figure_data/`.
