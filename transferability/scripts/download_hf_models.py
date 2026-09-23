#!/usr/bin/env python3
"""Download the frozen Transformer checkpoints into a project-local HF cache."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


TRANSFERABILITY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRANSFERABILITY_ROOT.parent
DEFAULT_CONFIG = TRANSFERABILITY_ROOT / "config" / "models.json"
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "huggingface" / "hub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["chemberta", "molformer"],
        help="Model aliases from config/models.json.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install the project environment first, "
            "or run: python -m pip install huggingface_hub"
        ) from exc

    args = parse_args()
    config_path = args.config.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open(encoding="utf-8") as handle:
        registry = json.load(handle)

    unknown = sorted(set(args.models) - set(registry))
    if unknown:
        raise SystemExit(f"Unknown model aliases: {', '.join(unknown)}")

    api = HfApi()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "models": {},
    }

    for alias in args.models:
        spec = registry[alias]
        repo_id = spec["repo_id"]
        requested_revision = spec.get("revision", "main")
        print(f"Downloading {alias}: {repo_id}@{requested_revision}", flush=True)

        # Exclude alternate framework weights. Both selected checkpoints provide
        # the PyTorch files needed by AutoModel.
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=requested_revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.py",
                "*.bin",
                "*.safetensors",
                "README.md",
            ],
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],
        )
        info = api.model_info(repo_id=repo_id, revision=requested_revision)
        manifest["models"][alias] = {
            **spec,
            "resolved_revision": info.sha,
            "snapshot_path": str(Path(snapshot_path).resolve()),
        }
        print(f"Cached at {snapshot_path}", flush=True)

    manifest_path = cache_dir.parent / "download_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
