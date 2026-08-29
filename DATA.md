# Data manifest

The analysis expects row-aligned files for 4,851,460 mouse genomic regions:

| Logical input | Expected content |
|---|---|
| coordinates | Chromosome and median genomic position for each region |
| sequences | One 201-bp reference sequence per region |
| targets | A 4,851,460 x 20 float matrix of accessibility values |
| folds | Five grouped development-fold index pairs |
| motif hits | Sparse HOCOMOCO v12 motif identity, position, and score records |
| checkpoints | Five trained checkpoints for every final ensemble |

Nearby regions on the same chromosome separated by at most 200 bp must share a
group. The locked held-out indices are not included because they reveal the
mapping to the source data and are unnecessary for reviewing the methods.

Large or restricted inputs should be obtained from the data owner or the source
study rather than committed to GitHub. The 22 final checkpoints are included in
`weights/`; do not commit raw sequences, response matrices, exploratory
checkpoints, prediction arrays, credentials, or SSH configuration.

