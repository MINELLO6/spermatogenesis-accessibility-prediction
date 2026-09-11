# Code review and verification — 11 September 2026

The review covered the repository's training, evaluation, diagnostic,
interpretation and figure programs, their imports, input/output paths, and
checkpoint interfaces. The original server source was inspected where the
repository omitted an experiment. Existing final weights were not modified.

## Repairs

- The ordinary `IA3Regressor(...)` call now uses encoder features instead of
  raising an exception. Adapter loading requires every learned scale and head
  parameter; incomplete adapter files cannot silently leave parameters initialized.
- Data, output, backbone and published-weight locations share one configuration
  module. Internal imports resolve against repository modules. Programs no longer
  start training simply because they are imported; command-line help is available.
- The layer-probe summary reads the archived metrics directory, reports sample
  SD, and writes JSON only when an output is requested. Training and post hoc
  layer-mixture evaluation now agree on the run-directory name.
- Sparse motif training and locked evaluation share their model definitions.
  Raw and frozen-NT ensemble evaluation also use the shared definitions. Strict
  loading and matching hashes confirm compatibility with all published weights.
- Grouping handles unsorted/interleaved chromosomes. Encoding accepts NumPy
  string arrays. Regression metrics reject nonfinite/insufficient inputs and use
  the sklearn convention for constant targets.
- Shape training rejects incompatible linear/factorized settings and empty
  filtered splits, prevents output collisions, normalizes positive fractional
  totals correctly, and loads only the inputs required by its modality.
- The attribution sampler skips bins with fewer than two candidates instead of
  selecting the entire bin through a `[-0:]` slice.
- The position-ablation queue uses configured GPUs, invokes existing modules,
  waits for its own jobs and reports failures. Historical process IDs were removed.
- Python formatting and imports were standardized. CPU tests and format/static
  checks are included in GitHub Actions. Linux shell files retain LF endings.

Correcting fractional-total normalization can affect new shape runs whose row
totals lie between zero and one. Archived numerical results were not rewritten.
The shared metrics' constant-target convention is likewise explicit; it does
not change nonconstant-target R-squared.

## Recovered experiment sources

| Server source | Maintained entry point | Purpose |
|---|---|---|
| `train_motif_get.py` | `scripts.training.train_sparse_motif` | CSR preparation, sparse bag baseline and primary sparse Transformer |
| `train_dense_motif_magnitude.py` | `scripts.training.train_dense_motif_magnitude` | Score-only dense magnitude/position comparisons |
| `train_raw_rc_attn.py` | `scripts.training.train_raw_rc_attn` | RC CNN with additional attention pooling |
| `train_convnext.py` | `scripts.training.train_convnext` | ConvNeXt comparison |

These are recovered implementations with entry-point/path/style changes, not
new reconstructions inferred from figure captions. The dense score/presence
Transformer screen remains separately named and documented. Historical activation
ablation source under `results/` is retained as provenance, rather than presented
as the maintained command-line interface.

## Verification performed

- 22 CPU tests: standard model outputs, encoding/grouping/metric boundaries,
  sparse empty-region and padding behavior, backward gradients, IA3 freezing and
  checkpoint completeness, and invalid split/data layouts.
- 17 one-epoch GPU integration jobs using fresh synthetic inputs: raw CNN,
  ConvNeXt, RC attention CNN, sparse preparation and training, RC CNN, dense CNN,
  position-free dense model, dense score/presence Transformer, five shared-trunk
  shape configurations, frozen NT, IA3, and all-layer probing. The last three
  used the actual cached NT-v2-50M backbone.
- Strict loading of all 22 final checkpoints. Their SHA-256 hashes exactly match
  `results/weight_manifest.json`, computed from the repository's published files.
- Actual dataset validation: 4,851,460 regions, 4,366,231 development observations
  and 485,229 held-out observations; no overlap group crosses any train/validation
  or development/test boundary.
- All 34 command-help entries, static checks, archived layer-summary generation and the approved
  model-comparison figure generator were exercised separately.
- An additional fresh virtual environment installed the CPU PyTorch wheel and
  `.[dev]` from scratch. With PyTorch 2.14.0+cpu, NumPy 2.5.3, Transformers 4.57.6
  and pytest 8.4.2, the same 22 tests, formatting and static checks all passed.
  GitHub's hosted job did not start because of an account-side infrastructure
  restriction; these clean-environment checks were executed independently.

The GPU smoke tests verify execution and file interfaces, not model quality or
exact numerical reproduction. Full five-fold training, held-out inference and
per-region interpretation were not repeated in this code review. Original data,
the cached pretrained backbone, and prediction/attribution arrays remain external
requirements. `DATA.md` describes the required files and the remaining upstream
data-provenance dependency. No raw genomic observations or credentials are added.
