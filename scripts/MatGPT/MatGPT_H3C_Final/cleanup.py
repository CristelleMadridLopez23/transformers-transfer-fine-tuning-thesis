"""Inventaría, respalda y archiva experimentos alternativos verificados.

El modo ``archive`` solo retira las rutas enumeradas tras copiar cada archivo,
comparar SHA-256 y confirmar que la reproducción H3-C y tests existen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from reproduce import project_root

LOG = logging.getLogger("h3c_cleanup")

ARCHIVE_ROOTS = (
    "scripts/MatGPT/MatGPT_frozen_3.0",
    "reports/MatGPT/MatGPT_frozen_3.0",
    "sh_files/MatGPT_frozen_3.0",
    "scripts/MatterGPT/MatterGPT_frozen_siamese",
    "reports/MatterGPT/MatterGPT_frozen_siamese",
    "model-pretrained/scripts/siamese_controlled_ablations",
    "reports/siamese_controlled_ablations",
    "experiments/siamese_controlled_ablations",
    "sh_files/siamese_controlled_ablations",
    "model-pretrained/scripts/siamese_energy_compression_audit",
    "reports/siamese_energy_compression_audit",
    "model-pretrained/scripts/both_siamese_Frozen",
    "reports/both_siamese_Frozen",
)

KEEP_ROOTS = (
    "scripts/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8",
    "reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8",
    "reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache",
    "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv",
    "reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv",
    "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth",
    "scripts/MatGPT_finetuning2/resource_exec.py",
    "scripts/MatGPT/MatGPT_H3C_Final",
    "reports/MatGPT/MatGPT_H3C_Final",
)


def digest(path: Path) -> str:
    """Devuelve SHA-256 de un archivo mediante lectura por bloques."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root: Path, archive: Path) -> dict[str, Any]:
    """Registra cada fuente, acción y hash antes de la retirada acotada."""
    report = root / "reports/MatGPT/MatGPT_H3C_Final"
    report.mkdir(parents=True, exist_ok=True)
    if any((root / rel).resolve() == archive.resolve() for rel in ARCHIVE_ROOTS):
        raise ValueError("El archivo coincide con una fuente")
    rows: list[dict[str, str | int]] = []
    counts: dict[str, int] = {}
    for rel in ARCHIVE_ROOTS:
        source = root / rel
        if not source.is_dir():
            raise FileNotFoundError(f"No existe raíz de archivo: {source}")
        paths = sorted(p for p in source.rglob("*") if p.is_file())
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError(f"Enlace simbólico inesperado: {rel}")
        counts[rel] = len(paths)
        for path in paths:
            relative = path.relative_to(root).as_posix()
            rows.append({"source": relative, "action": "ARCHIVE", "destination": str((archive / "files" / relative).relative_to(root)),
                         "bytes": path.stat().st_size, "sha256": digest(path), "reason": "Experimento alternativo cerrado; H3-C no importa este módulo"})
        LOG.info(json.dumps({"event": "inventoried", "root": rel, "files": len(paths)}))
    for rel in KEEP_ROOTS:
        source = root / rel
        if not source.exists():
            raise FileNotFoundError(f"Dependencia H3-C ausente: {source}")
        rows.append({"source": rel, "action": "KEEP", "destination": rel,
                     "bytes": source.stat().st_size if source.is_file() else 0,
                     "sha256": digest(source) if source.is_file() else "directory",
                     "reason": "Dependencia H3-C, datos originales o herramienta compartida"})
    csv_path = report / "cleanup_manifest.csv"
    if csv_path.exists():
        raise FileExistsError(f"Manifiesto ya existente: {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "action", "destination", "bytes", "sha256", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "-z"], cwd=root)
    inventory_text = (
        "# Inventario de limpieza H3-C\n\n"
        f"Referencia Git recuperable previa: `{head}`. Cambios de worktree existentes: {status.count(bytes([0]))} entradas de estado; "
        "se conservaron y no se usó reset/clean. La referencia registra el estado confirmado; los archivos modificados o no rastreados "
        "de las rutas retiradas se preservan mediante copias SHA-256 en el archivo.\n\n"
        "## Retirada acotada\n\n"
        "\n".join(f"- `{rel}`: {counts[rel]} archivos" for rel in ARCHIVE_ROOTS)
        + "\n\n## Conservación crítica\n\n"
        + "\n".join(f"- `{rel}`" for rel in KEEP_ROOTS)
        + "\n\nEl listado exacto de cada archivo y hash consta en `cleanup_manifest.csv`. "
        "La retirada se ejecuta únicamente tras copia y verificación de cada hash.\n"
    )
    (report / "cleanup_inventory.md").write_text(inventory_text, encoding="utf-8")
    payload = {"status": "INVENTORIED", "git_head": head, "archive": str(archive.relative_to(root)),
               "roots": counts, "archived_files": sum(counts.values()), "archived_bytes": sum(int(row["bytes"]) for row in rows if row["action"] == "ARCHIVE"),
               "manifest_sha256": digest(csv_path)}
    (report / "cleanup_backup_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def archive_verified(root: Path, archive: Path) -> dict[str, Any]:
    """Copia, verifica y retira solo raíces autorizadas tras controles PASS."""
    report = root / "reports/MatGPT/MatGPT_H3C_Final"
    for rel in ("reproduction/reproduction_manifest.json", "analysis/manifest.json"):
        state = json.loads((report / rel).read_text())
        if state.get("status") != "PASS":
            raise ValueError(f"Control H3-C no superado: {rel}")
    if not (report / "tests_resources.json").exists():
        raise FileNotFoundError("Falta corrida de tests H3-C")
    manifest_path = report / "cleanup_manifest.csv"
    backup_path = report / "cleanup_backup_manifest.json"
    old = json.loads(backup_path.read_text())
    if old.get("status") != "INVENTORIED":
        raise ValueError("Se requiere inventario nuevo sin limpieza anterior")
    if old["manifest_sha256"] != digest(manifest_path):
        raise ValueError("El manifiesto de limpieza cambió")
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["action"] == "ARCHIVE"]
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "git_head.ref").write_text(old["git_head"] + "\n", encoding="ascii")
    # La copia ocurre antes de retirar ninguna fuente. Cada archivo se verifica
    # tanto contra su hash de inventario como contra su copia recuperable.
    for index, row in enumerate(rows, 1):
        src = root / row["source"]
        dst = root / row["destination"]
        if not src.is_file() or digest(src) != row["sha256"]:
            raise ValueError(f"Fuente cambió desde el inventario: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if digest(dst) != row["sha256"]:
            raise ValueError(f"Copia no verificable: {dst}")
        if index % 100 == 0:
            LOG.info(json.dumps({"event": "backup_progress", "verified": index, "total": len(rows)}))
    for row in rows:
        if digest(root / row["destination"]) != row["sha256"]:
            raise ValueError(f"Verificación global falló: {row['destination']}")
    recovery = archive / "RECOVERY.md"
    recovery.write_text(
        "# Respaldo recuperable de experimentos H3-C descartados\n\n"
        "Cada archivo bajo `files/` conserva su ruta relativa original. El SHA-256 exacto "
        "está en `reports/MatGPT/MatGPT_H3C_Final/cleanup_manifest.csv` del árbol activo. "
        "`git_head.ref` identifica el commit previo. Para recuperar una carpeta, copiarla "
        "desde `files/` a su ruta original y verificar sus hashes; no se requiere reset de Git.\n",
        encoding="utf-8")
    payload = {**old, "status": "BACKUP_VERIFIED", "verified_files": len(rows), "recovery_sha256": digest(recovery)}
    backup_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    # Cada directorio se retira solamente después de verificar el respaldo total.
    for rel in ARCHIVE_ROOTS:
        shutil.rmtree(root / rel)
        LOG.info(json.dumps({"event": "archived_root", "root": rel}))
    payload["status"] = "ARCHIVED"
    payload["retired_roots"] = list(ARCHIVE_ROOTS)
    backup_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Limpieza controlada H3-C", "", "[HECHO VERIFICADO] La reproducción, los controles y los tests terminaron antes de retirar las rutas siguientes.",
             f"[HECHO VERIFICADO] {len(rows)} archivos y {old['archived_bytes']} bytes se copiaron y comprobaron por SHA-256 en `{archive.relative_to(root)}`.",
             f"[HECHO VERIFICADO] Referencia Git previa: `{old['git_head']}`. Los cambios no confirmados de estas rutas están incluidos en las copias verificadas.",
             "", "## Rutas retiradas del árbol activo y archivadas", ""]
    lines += [f"- `{rel}` ({old['roots'][rel]} archivos)" for rel in ARCHIVE_ROOTS]
    lines += ["", "## Dependencias y datos conservados", ""]
    lines += [f"- `{rel}`" for rel in KEEP_ROOTS]
    lines += ["", "El detalle archivo por archivo consta en `cleanup_manifest.csv`; no se retiraron datasets, checkpoints originales ni módulos compartidos.",
              "La recuperación está explicada en `archive/.../RECOVERY.md`. No se realizó `git reset` ni `git clean`.", ""]
    (report / "cleanup_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    """Selecciona inventario o archivo, con raíz configurable y logging JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "archive"))
    parser.add_argument("--repo", type=Path, default=project_root())
    parser.add_argument("--archive", type=Path, default=Path("archive/MatGPT_H3C_Final_20261028"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = args.repo.resolve()
    archive = (root / args.archive).resolve()
    if not archive.is_relative_to(root):
        parser.error("El respaldo debe estar dentro del workspace")
    result = inventory(root, archive) if args.mode == "inventory" else archive_verified(root, archive)
    LOG.info(json.dumps({"event": "cleanup_state", **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
