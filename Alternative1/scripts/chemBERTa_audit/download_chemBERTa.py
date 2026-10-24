#!/usr/bin/env python3

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Descarga completa y trazable de ChemBERTa desde Hugging Face."
    )
    parser.add_argument(
        "--repo",
        default="DeepChem/ChemBERTa-77M-MLM",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--destino",
        default="model-pretrained/chemberta-77m-mlm",
    )
    args = parser.parse_args()

    api = HfApi()
    info = api.model_info(
        repo_id=args.repo,
        revision=args.revision,
    )
    commit = info.sha
    if not commit:
        raise RuntimeError("No se pudo resolver el commit del repositorio.")

    destination = Path(args.destino).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    # El manifiesto queda fuera de la copia original del repositorio.
    manifest_path = destination.parent / (
        destination.name + ".download-manifest.json"
    )

    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("repo_id") != args.repo
            or previous.get("resolved_commit") != commit
        ):
            raise RuntimeError(
                "El destino ya está asociado a otra revisión o repositorio. "
                "Usa otro --destino para no mezclar versiones."
            )

    remote_files = sorted(
        api.list_repo_files(
            repo_id=args.repo,
            revision=commit,
            repo_type="model",
        )
    )

    print(f"Repositorio: {args.repo}")
    print(f"Commit:      {commit}")
    print(f"Destino:     {destination}")
    print(f"Archivos:    {len(remote_files)}")

    snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        revision=commit,
        local_dir=str(destination),
        max_workers=4,
    )

    inventory = []
    missing = []

    for relative_name in remote_files:
        path = destination / relative_name
        if not path.is_file():
            missing.append(relative_name)
            continue

        inventory.append(
            {
                "path": relative_name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    if missing:
        raise RuntimeError(
            "Descarga incompleta. Faltan:\n" + "\n".join(missing)
        )

    if not (destination / "config.json").is_file():
        raise RuntimeError("No se encontró config.json.")

    weight_files = [
        name for name in remote_files
        if name.endswith(".safetensors")
        or Path(name).name.startswith("pytorch_model")
    ]
    if not weight_files:
        raise RuntimeError("No se encontraron pesos para PyTorch.")

    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
    }
    tokenizer_files = [
        row for row in inventory
        if Path(row["path"]).name in tokenizer_names
    ]

    manifest = {
        "repo_id": args.repo,
        "requested_revision": args.revision,
        "resolved_commit": commit,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "huggingface_hub_version": importlib.metadata.version(
            "huggingface_hub"
        ),
        "file_count": len(inventory),
        "total_bytes": sum(row["size_bytes"] for row in inventory),
        "files": inventory,
        "validation_scope": (
            "Repository download, file presence and local SHA-256 inventory. "
            "No model execution or SLICES tokenizer validation performed."
        ),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nDescarga completa.")
    print(f"Manifiesto: {manifest_path}")
    print("\nArchivos del tokenizador:")

    for row in tokenizer_files:
        print(f"  {row['path']}: {row['size_bytes']} bytes")

    empty_tokenizer_files = [
        row["path"] for row in tokenizer_files
        if row["size_bytes"] == 0
    ]
    if empty_tokenizer_files:
        print(
            "\nAVISO: hay archivos del tokenizador vacíos. "
            "Se conservaron tal como fueron publicados:"
        )
        for name in empty_tokenizer_files:
            print(f"  {name}")

    print(
        "\nEsta comprobación no certifica que el tokenizador "
        "sea adecuado para SLICES."
    )


if __name__ == "__main__":
    main()