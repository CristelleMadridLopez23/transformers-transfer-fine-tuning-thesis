"""Archive primary paper/code/tokenizer references with published checksum validation.

Download small public artifacts only; never download model weights. Existing
references are preserved. Extract the MIT-licensed paper code to a dedicated audit
directory and retain a manifest with source URLs and content hashes.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from workflow import LOGGER, cli, load_config, read_json, sha256, write_json

REFERENCES = {
    "zenodo_record.json": "https://zenodo.org/api/records/21151839",
    "MatText-main.zip": "https://zenodo.org/api/records/21151839/files/MatText-main.zip/content",
    "xtal2txt_tokenizer_2024.py": "https://raw.githubusercontent.com/lamalab-org/xtal2txt/1d9d0c6f8675efea885ef7c9c87b3195cd79aaed/src/xtal2txt/tokenizer.py",
    "slice_vocab.txt": "https://zenodo.org/records/11484062/files/slice_vocab.txt?download=1",
    "hf_checkpoint_api.json": "https://huggingface.co/api/models/n0w0f/MatText-slices-2m",
    "hf_checkpoint_commits.json": "https://huggingface.co/api/models/n0w0f/MatText-slices-2m/commits/main",
    "paper.html": "https://www.nature.com/articles/s43588-026-01045-2",
    "supplementary_information.pdf": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43588-026-01045-2/MediaObjects/43588_2026_1045_MOESM1_ESM.pdf",
    "github_issues.json": "https://api.github.com/repos/lamalab-org/MatText/issues?state=all&per_page=100",
    "hf_discussions.json": "https://huggingface.co/api/models/n0w0f/MatText-slices-2m/discussions",
    "issue99_comments.json": "https://api.github.com/repos/lamalab-org/MatText/issues/99/comments"
}
PUBLISHED_MD5 = {"MatText-main.zip": "3304d9a20704cd9a2538082d01aaaf63",
                 "slice_vocab.txt": "ea6ea28b3e7bf8a5a7e2f4f514968639"}


def download(path: Path, url: str) -> None:
    """Download public URL to new Path or reuse existing bytes; return None after checksum checks."""
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read()
        if path.name in PUBLISHED_MD5 and hashlib.md5(body).hexdigest() != PUBLISHED_MD5[path.name]:
            raise ValueError(f"Published checksum mismatch: {path.name}")
        path.write_bytes(body)
    if path.name in PUBLISHED_MD5 and hashlib.md5(path.read_bytes()).hexdigest() != PUBLISHED_MD5[path.name]:
        raise ValueError(f"Local original reference checksum mismatch: {path.name}")


def fetch(config: dict[str, Any]) -> None:
    """Fetch configured primary references and all checkpoint revision file trees; return None."""
    directory = config["paths"]["reports"] / "audit/reference"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in REFERENCES.items():
        path = directory / name
        download(path, url)
        records.append({"file": name, "url": url, "sha256": sha256(path),
                        "published_md5": PUBLISHED_MD5.get(name)})
    for item in read_json(directory / "hf_checkpoint_commits.json"):
        revision = item["id"]
        path = directory / f"hf_tree_{revision}.json"
        url = f"https://huggingface.co/api/models/n0w0f/MatText-slices-2m/tree/{revision}?recursive=true"
        download(path, url)
        records.append({"file": path.name, "url": url, "sha256": sha256(path)})
    base = directory / "paper_code"
    with zipfile.ZipFile(directory / "MatText-main.zip") as archive:
        for info in archive.infolist():
            target = (base / info.filename).resolve()
            if base.resolve() not in target.parents:
                raise ValueError("Unsafe archive member path")
            if info.is_dir():
                continue
            body = archive.read(info)
            if target.exists() and target.read_bytes() != body:
                raise ValueError(f"Archived source was modified: {target}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
    write_json(directory / "primary_reference_manifest.json", {"files": records,
               "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "policy": "Original public references preserved; checkpoint history checked without downloading weights"})
    LOGGER.info("primary_references_verified files=%d", len(records))


def main() -> None:
    """Fetch immutable/public evidence from CLI config; log descriptive failures and propagate errors."""
    try:
        fetch(load_config(cli(__doc__).parse_args().config))
    except Exception:
        LOGGER.exception("reference_fetch_failed")
        raise


if __name__ == "__main__":
    main()
