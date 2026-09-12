"""Download the recorded Hugging Face NT snapshot for the offline training scripts."""

import argparse
import json
import os
from pathlib import Path

# Keep setup usable before PyTorch is installed. Match the training path default.
NT_MODEL_PATH = Path(
    os.environ.get(
        "ACCESSIBILITY_NT_PATH",
        Path(__file__).resolve().parents[1] / "models" / "nucleotide-transformer-v2-50m",
    )
)

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
MANIFEST = "accessibility_snapshot.json"
FILES = (
    "README.md",
    "config.json",
    "esm_config.py",
    "modeling_esm.py",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def check_files(directory):
    directory = Path(directory)
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete NT snapshot; missing: {', '.join(missing)}")
    empty = [name for name in FILES if (directory / name).stat().st_size == 0]
    if empty:
        raise ValueError(f"Empty NT files: {', '.join(empty)}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    for key, expected in {"hidden_size": 512, "num_hidden_layers": 12}.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected NT configuration: {key}={config.get(key)!r}")


def prepare_model(directory, *, check_only=False, downloader=None):
    directory = Path(directory).expanduser().resolve()
    marker = directory / MANIFEST
    record = {"repo_id": MODEL_ID, "revision": REVISION}
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in record.items()):
            raise ValueError(
                "Destination records a different model/version; choose a new directory"
            )
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(
            "Destination is not empty and has no snapshot record; choose a new directory"
        )

    if check_only:
        if not marker.exists():
            raise ValueError("No snapshot record found; run the download command first")
        if not existing.get("complete", False):
            raise ValueError("Download did not complete; rerun without --check-only")
        check_files(directory)
        return directory

    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    directory.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**record, "complete": False}, indent=2) + "\n", encoding="utf-8")
    downloader(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(directory),
        allow_patterns=list(FILES),
    )
    check_files(directory)
    marker.write_text(json.dumps({**record, "complete": True}, indent=2) + "\n", encoding="utf-8")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=NT_MODEL_PATH)
    parser.add_argument(
        "--check-only", action="store_true", help="Check local files without network access"
    )
    args = parser.parse_args()
    try:
        location = prepare_model(args.output_dir, check_only=args.check_only)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"NT setup failed: {exc}\n")
    print(f"NT snapshot ready: {location}")
    print(f"Model: {MODEL_ID}\nRevision: {REVISION}")
    print("Use this directory as ACCESSIBILITY_NT_PATH for training.")


if __name__ == "__main__":
    main()
