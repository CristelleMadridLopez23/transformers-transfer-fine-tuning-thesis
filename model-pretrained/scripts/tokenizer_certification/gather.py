"""Cache pinned public forensic evidence without touching checkpoints or datasets.

Downloads only source archives and small metadata/configuration files. Existing
paper archives are read in place. Every fetched byte stream receives a URL,
revision, acquisition time and SHA-256 in the evidence manifest. No jobs run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = Path("reports/exp02_only_hex_mattext_finetune/audit/tokenizer_certification")
LOGGER = logging.getLogger(__name__)
SNAPSHOTS = {
    "mattext_preupload": ("MatText", "b7ce8b3e071c18440e6720d682be6bd3e355f9c9"),
    "mattext_june28": ("MatText", "c7ad540d63bcf1d02e0529c26c3a09ec4ded81d5"),
    "xtal_preupload": ("xtal2txt", "3f7081c474ea8ab79f9ed5920cfb754fb14de1cc"),
    "xtal_june5": ("xtal2txt", "1d9d0c6f8675efea885ef7c9c87b3195cd79aaed"),
    "xtal_2026": ("xtal2txt", "c6cf9593596ce99f054f1ebbdc4346d7dbec99fb"),
}


def digest(path: Path) -> str:
    """Return SHA-256 of a file, streaming to bound memory use."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Serialize a JSON-compatible value with stable indentation and a newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch(url: str, path: Path, revision: str, offline: bool) -> dict[str, object]:
    """Cache a public URL or validate its existing cache; return provenance."""
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Offline evidence missing: {path}")
        LOGGER.info("fetch url=%s", url)
        request = urllib.request.Request(url, headers={"User-Agent": "MatText-thesis-forensic-audit"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {"path": str(path.relative_to(ROOT)), "url": url, "revision": revision,
            "sha256": digest(path), "bytes": path.stat().st_size,
            "cache_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def gather(output: Path, offline: bool = False) -> None:
    """Cache full pinned source snapshots, commit metadata and live HF identity."""
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for label, (repository, revision) in SNAPSHOTS.items():
        entries.append(fetch(f"https://codeload.github.com/lamalab-org/{repository}/zip/{revision}",
                             evidence / f"{label}.zip", revision, offline))
        entries.append(fetch(f"https://api.github.com/repos/lamalab-org/{repository}/commits/{revision}",
                             evidence / f"{label}_commit.json", revision, offline))
        # Extract only text evidence under a new report directory; reject traversal.
        with zipfile.ZipFile(evidence / f"{label}.zip") as archive:
            for member in archive.infolist():
                relative = Path(*Path(member.filename).parts[1:])
                if member.is_dir() or not relative.parts or ".." in relative.parts:
                    continue
                if relative.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".toml", ".lock"}:
                    target = evidence / label / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
    revision = "c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e"
    for name, url in {
        "hf_live_api.json": "https://huggingface.co/api/models/n0w0f/MatText-slices-2m?blobs=true",
        "hf_pinned_tree.json": f"https://huggingface.co/api/models/n0w0f/MatText-slices-2m/tree/{revision}?recursive=true&expand=true",
        "hf_pinned_config.json": f"https://huggingface.co/n0w0f/MatText-slices-2m/resolve/{revision}/config.json",
        "tokenizers_documentation.html": "https://lamalab-org.github.io/MatText/docs/tokenizers/",
    }.items():
        entries.append(fetch(url, evidence / name, revision if "pinned" in name else "observed_at_acquisition", offline))
    old = output.parent / "reference"
    for name in ["MatText-main.zip", "zenodo_record.json", "slice_vocab.txt", "hf_checkpoint_commits.json",
                 "primary_reference_manifest.json", "xtal2txt_tokenizer_history.json", "github_issues.json", "hf_discussions.json"]:
        path = old / name
        entries.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                        "origin": "existing pinned audit evidence; URL is in its reference manifest", "bytes": path.stat().st_size})
    write_json(output / "evidence_manifest.json", {"snapshots": SNAPSHOTS, "artifacts": entries})
    LOGGER.info("evidence_ready artifacts=%d", len(entries))


def main() -> None:
    """Validate CLI paths, cache evidence and log descriptive download failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "reports"):
        parser.error("--output must be inside project reports/")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s level=%(levelname)s message=%(message)s')
    try:
        gather(output, args.offline)
    except Exception:
        LOGGER.exception("evidence_acquisition_failed")
        raise


if __name__ == "__main__":
    main()
