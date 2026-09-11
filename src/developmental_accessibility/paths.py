"""Shared locations, configurable before starting an experiment.

Importing this module never creates directories or downloads a backbone.
"""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ACCESSIBILITY_DATA_ROOT", REPOSITORY / "data")).expanduser()
RUN_ROOT = Path(os.environ.get("ACCESSIBILITY_RUN_ROOT", REPOSITORY / "runs")).expanduser()
WEIGHTS_ROOT = Path(
    os.environ.get("ACCESSIBILITY_WEIGHTS_ROOT", REPOSITORY / "weights")
).expanduser()
NT_MODEL_PATH = Path(
    os.environ.get("ACCESSIBILITY_NT_PATH", REPOSITORY / "models" / "nucleotide-transformer-v2-50m")
).expanduser()
