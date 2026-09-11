# Post hoc encoder-activation ablation (11 September 2026)

This development-set experiment changes the dense motif Transformer's internal
encoder activation from GELU to Softplus. It is not a new held-out test and does
not replace the original experiment or its recorded results.

The archived training script differed from the original server version of
`train_shape_shared_trunk.py` only in the encoder activation
(`activation=F.softplus`) and an explicit metadata field. It preserves that
historical source, including its server paths; the maintained entry point has
since received portability and input-validation fixes.
`--activation softplus` in the original script controls the CNN/head
activation; its Transformer encoder still uses GELU. Both runs use the same
positive Softplus rate link and normalized-Poisson objective.

| Encoder activation | Shape R2 | Within-region Pearson | Jensen-Shannon divergence |
|---|---:|---:|---:|
| GELU | 0.100303 ± 0.001320 | 0.454689 ± 0.001349 | 0.049848 ± 0.000076 |
| Softplus | 0.095968 ± 0.000779 | 0.451578 ± 0.001160 | 0.050032 ± 0.000087 |

Values are means and sample standard deviations across five grouped folds.
Softplus minus GELU mean paired R2 difference: **−0.0043348**; all five fold
differences are negative. Pearson also decreases in all five folds. This is
descriptive evidence for retaining the original activation in this setting,
not a general ranking of activations. Overlapping training sets mean that the
five folds are not five independent experiments; these SDs are not confidence
intervals.

## Data and protocol

The p < 0.001 filter retains 960,329 regions overall. Only the 864,133 filtered
development regions enter these training/validation runs. Validation fold sizes
are 173,199; 172,978; 172,744; 172,340; and 172,872. Both arms retain identical
recorded fold seeds, sample counts, architecture dimensions, optimizer settings,
early-stopping rule, positional reversal probability, and pseudo-count total.

`launch_manifest.json` preserves the actual server arguments. `full/` contains
the new Softplus configurations and metrics; `gelu/` contains the original
comparison configurations and metrics. `summary.json` records paired differences
for six metrics. Checkpoint and individual prediction arrays remain on the
compute server and are excluded from this compact archive.

Run the self-contained audit without GPU/data dependencies:

```bash
python results/shape_activation_ablation_20260911/audit.py
```

The audit checks matched recorded configurations and sample counts and recomputes
all six summaries directly from per-fold metrics. It does not independently
recompute metrics from prediction arrays or establish bitwise training reproducibility.

To repeat training, use the archived script with the arguments in the launch
manifest, setting `ROOT` to the aligned input directory described in `DATA.md`
and changing `--output-root` to a new directory. Fold arguments are zero-based.
The successful experiment ran under Python 3.12; the older general repository
environment guidance should not be treated as an exact frozen environment.
