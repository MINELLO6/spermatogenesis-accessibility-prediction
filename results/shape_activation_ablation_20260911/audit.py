"""Recompute descriptive paired-fold summaries using only the Python standard library."""
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
KEYS = ("mean_bin_r2", "pearson_mean", "js_mean", "wasserstein_bins_mean",
        "peak_bin_mae", "peak_bin_accuracy")
MATCH = ("modality", "fold", "objective", "shape_total", "activation", "backbone",
         "transformer_d_model", "transformer_layers", "transformer_heads", "epochs",
         "patience", "batch_size", "accum_steps", "lr", "p_threshold", "max_train",
         "max_val", "rc_prob", "presence", "seed", "train_n", "val_n", "n_parameters")
soft, gelu, validation_n = [], [], []
for fold in range(1, 6):
    sp = ROOT / "full" / "motif_normalized_poisson" / f"fold{fold}"
    gp = ROOT / "gelu" / "motif_normalized_poisson" / f"fold{fold}"
    sc, gc = (json.loads((p / "config.json").read_text()) for p in (sp, gp))
    for key in MATCH:
        assert sc[key] == gc[key], (fold, key, sc[key], gc[key])
    sm, gm = (json.loads((p / "metrics.json").read_text()) for p in (sp, gp))
    assert sm["n"] == gm["n"] == sc["val_n"]
    assert sc["train_n"] + sc["val_n"] == 864133
    validation_n.append(sm["n"])
    soft.append(sm)
    gelu.append(gm)
assert sum(validation_n) == 864133
recorded = json.loads((ROOT / "summary.json").read_text())
for key in KEYS:
    sv, gv = [x[key] for x in soft], [x[key] for x in gelu]
    diffs = [s-g for s, g in zip(sv, gv)]
    computed = {"softplus_mean": mean(sv), "softplus_sd": stdev(sv),
                "gelu_mean": mean(gv), "gelu_sd": stdev(gv),
                "difference_mean": mean(diffs)}
    for field, value in computed.items():
        assert math.isclose(recorded[key][field], value, rel_tol=1e-10, abs_tol=1e-12)
    assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(diffs, recorded[key]["paired_differences"]))
    print(f"{key}: GELU {mean(gv):.8f}; Softplus {mean(sv):.8f}; paired difference {mean(diffs):+.8f}")
print("PASS: matched recorded configurations, fold sample counts and all six metric summaries.")
