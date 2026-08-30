# Final dense-motif shape position ablation

This directory contains the completed five-fold `motif_nopos + equal_ce + Softplus` experiment copied from the authorised training server before shutdown.

The no-position input max-pools each of the 1,443 motif scores over the ten position bins and broadcasts the pooled vector back across those bins. It therefore preserves motif identity and score while removing positional variation. All other settings match the position-aware Softplus experiment.

| Metric | Position-aware, mean (SD) | No position, mean (SD) | Paired position minus no-position |
|---|---:|---:|---:|
| Mean-bin shape R2 | 0.107990 (0.000562) | 0.103255 (0.000271) | +0.004736 |
| Within-region Pearson | 0.460076 (0.001474) | 0.456259 (0.000994) | +0.003817 |
| JS divergence | 0.049408 (0.000133) | 0.049699 (0.000063) | -0.000292 |
| Wasserstein distance, bins | 1.482047 (0.010529) | 1.491872 (0.005672) | -0.009824 |
| Centre-of-mass MAE, bins | 1.243033 (0.011852) | 1.252130 (0.006579) | -0.009097 |
| Peak-bin MAE | 5.529157 (0.034478) | 5.533968 (0.055612) | -0.004811 |
| Exact peak accuracy | 0.216359 (0.002262) | 0.214992 (0.001459) | +0.001367 |

The R2 difference was positive in all five paired folds (0.005305, 0.005550, 0.004053, 0.004363 and 0.004407). Position therefore contributes a small but reproducible amount of distributional shape information; the peak-localisation metrics do not show a comparably consistent gain.

The archive `shape_motif_nopos_softplus5.tar.gz` has SHA-256:

`1211A7443D5E7DF2C21BD9E618718C5A6D68B4BBBBE5FBB4350A385CCFBD3174`
