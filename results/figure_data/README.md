# Model comparison plotting data

`model_comparison.json` contains the values used in the approved dissertation
comparison plot: five-fold means and sample standard deviations. Some older
model entries retain the rounding of the dissertation summary table; these are
display data, not replacements for fold-level metric files. The intervals show
standard deviations, not confidence intervals.

Regenerate the PDF, SVG and PNG with
`python -m scripts.figures.plot_model_comparison`. The command reads this JSON
without accessing genomic records or running models.
