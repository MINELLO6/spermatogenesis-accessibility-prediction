# Row-normalized Poisson shape experiment

This directory contains compact five-fold summaries for the matched shape
comparison on the 960,329 regions with temporal-variation p-value below 0.001.
For every region, the observed 20-bin profile was rescaled to 100 pseudo-counts.
Models produced 20 positive rates through Softplus and minimized Poisson
negative log-likelihood. The predicted total was learned rather than fixed.

`summary.csv` reports the mean and sample standard deviation across the five
pre-existing genomic-group folds. Lower Jensen-Shannon (`js_mean`) is better;
higher shape R2 and within-region Pearson correlation are better. The reported
predicted total is the fold-averaged mean of the sum of the 20 fitted rates.

Training is implemented in
`scripts/training/train_shape_shared_trunk.py`. The original per-region
prediction arrays and checkpoints are intentionally omitted because they are
large; the script stores validation indices and predictions when run against
the source data described in `DATA.md`.
