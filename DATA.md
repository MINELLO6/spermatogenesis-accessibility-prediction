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

## On-disk schema

Set `ACCESSIBILITY_DATA_ROOT` to a directory containing these files. Row ordering
must be identical across coordinates, sequences, targets, fold indices and motif
region IDs. A matching shape alone does not establish biological alignment.

| File | Encoding and interpretation |
|---|---|
| `totmat_shape.txt` | Two whitespace-separated integers: number of regions and 20 |
| `totmat_f64.bin` | Little-endian float64, column-major (Fortran) order, shape from the text file |
| `fullseqs.txt` | Exactly 201 ASCII bases and one LF byte per row; no FASTA headers or CRLF |
| `peaks.tsv` | Tab-separated table with `chr` and numeric `median_position`, in region order |
| `folds.pkl` | Five pairs of zero-based integer train/validation vectors; load only trusted files |
| `peaktest.tsv` | Row-aligned table with `p.value`, required for variable-region shape analysis |
| `allmot_shape.txt` | Number of hits and four columns for the original motif hit matrix |
| `allmot_f64.bin` | Float64 Fortran-order columns: one-based region ID, score, one-based position bin (1–10), one-based motif ID (1–1443) |
| `motif_csr/offsets.npy` | int64 offsets of length number-of-regions + 1 |
| `motif_csr/motif_ids.npy` | uint16 motif IDs; 0 is reserved for padding |
| `motif_csr/position_bins.npy` | uint8 position bins; 0 is reserved for padding |
| `motif_csr/scores.npy` | float16 motif scores aligned with IDs and positions |
| `motif_csr/stats.json` | Counts and token-cap recommendation written by the preparation command |

CSR preparation preserves the original hit scores; it does not perform a new
motif scan. The source reference genome, upstream accessibility processing and
HOCOMOCO scan outputs must be obtained with the original dataset. This repository
does not recreate these biological inputs from an accession alone.

The development set is the union of the five validation folds. Its complement
defines the held-out row indices only when the original row ordering and original
folds are supplied. `validate_data` checks that each fold partitions exactly this
same development set and reports a SHA-256 digest of the complement. It never
invents replacement folds or rewrites a locked split.
