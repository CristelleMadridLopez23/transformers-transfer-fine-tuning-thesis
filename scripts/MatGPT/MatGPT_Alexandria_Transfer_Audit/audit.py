"""Audita trazabilidad, compatibilidad y etiquetas de Alexandria para MatGPT.

El procesamiento es de solo lectura; no carga el JSON completo ni entrena. Una
vista previa accidental de un archivo potencialmente protegido se excluye del análisis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
SCRIPT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "experiments/MatGPT_Alexandria_Transfer_Audit"
REPORT = ROOT / "reports/MatGPT/MatGPT_Alexandria_Transfer_Audit"
IMAGES = ROOT / "images/MatGPT/MatGPT_Alexandria_Transfer_Audit"
RAW_JSON = ROOT / "data/raw_data/alexandria_00000.json"
ALEX_DIR = ROOT / "data/alexandria/notebooks_processing/outputs/to_SLICES_output/figures"
STRICT_CSV = ALEX_DIR / "alexandria_clean_strict.csv"
ALL_CSV = ALEX_DIR / "alexandria_clean_all.csv"
TRAIN_CSV = ROOT / "data/matgpt y matter/Training_dataset_alex20_with_encoding.csv"
MODEL_DIR = ROOT / "models/MatGPT-source/mat-concat/mat-uncond/weights"
MODEL_INFO = MODEL_DIR / "model_info_property_0.json"
CHECKPOINT = MODEL_DIR / "transformer_property_0.pth"
CONFIG = ROOT / "models/MatGPT-source/mat-concat/config/uncond.yaml"
DEV_CSV = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
LOG = logging.getLogger("alexandria_transfer_audit")
ACCESS_NOTE = "ONE_RECORD_PREVIEWED_IN_ERROR; EXCLUDED_FROM_ANALYSIS; NO_FURTHER_ACCESS"
CHUNK = 12000


def sha256(path: Path) -> str:
    """Calcula SHA-256 por bloques.

    Args:
        path: Archivo de entrada.
    Returns:
        Hash hexadecimal SHA-256.
    """
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Escribe un manifiesto JSON mediante reemplazo atómico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n",
                    encoding="utf-8")
    tmp.replace(path)


def rss_bytes() -> int:
    """Devuelve el máximo RSS observado por el proceso en bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def csv_rows(path: Path) -> int:
    """Cuenta registros CSV sin leer el archivo entero a memoria."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return max(0, sum(1 for _ in csv.reader(stream)) - 1)


def parse_elements(value: Any) -> list[str]:
    """Interpreta elementos guardados como JSON o cadena separada por comas."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def parse_formula_key(value: str) -> str:
    """Devuelve composición reducida canónica sin depender del orden textual."""
    try:
        from pymatgen.core import Composition
        return Composition(value).reduced_formula
    except Exception:
        return "".join(sorted(value.replace(" ", "")))


def reduce_element_counts(counts: Counter[str]) -> str:
    """Canoniza un conteo elemental mediante reducción por máximo común divisor."""
    from math import gcd
    vals = [int(v) for v in counts.values() if int(v) > 0]
    divisor = 0
    for value in vals: divisor = gcd(divisor, value)
    divisor = max(1, divisor)
    return "".join(f"{element}{counts[element]//divisor if counts[element]//divisor != 1 else ''}"
                   for element in sorted(counts))


def composition_from_slices(sequence: str, element_symbols: set[str]) -> tuple[str, list[str]]:
    """Recupera la composición de la lista inicial de especies en SLICES.

    El formato de MatGPT enumera especies atómicas antes del primer índice de
    nodo. Solo se usa para auditoría de cobertura; no reconstruye geometría.
    """
    species = []
    for token in str(sequence).split():
        # Alex-20 enumera especies hasta el primer índice entero. El export local
        # puede anteponer metadatos; allí se usa la fórmula explícita del CSV.
        if token.isdigit() or token not in element_symbols:
            break
        species.append(token)
    if not species: return "", []
    return reduce_element_counts(Counter(species)), species


def crystal_system(spg: Any) -> str:
    """Clasifica número de grupo espacial según intervalos cristalográficos."""
    try:
        n = int(float(spg))
    except (TypeError, ValueError):
        return "unknown"
    if 1 <= n <= 2: return "triclinic"
    if 3 <= n <= 15: return "monoclinic"
    if 16 <= n <= 74: return "orthorhombic"
    if 75 <= n <= 142: return "tetragonal"
    if 143 <= n <= 167: return "trigonal"
    if 168 <= n <= 194: return "hexagonal"
    if 195 <= n <= 230: return "cubic"
    return "unknown"


def tokenizer_audit(slices: str, stoi: dict[str, int], block_size: int) -> dict[str, Any]:
    """Comprueba compatibilidad exacta con el tokenizer y el contexto MatGPT."""
    tokens = str(slices).split()
    unknown = sorted(set(tokens) - set(stoi))
    total_len = len(tokens) + 2  # <start>, <end>
    return {"n_tokens": len(tokens), "input_length": total_len, "unknown_tokens": unknown,
            "tokenizer_compatible": not unknown, "within_context": total_len <= block_size}


def stream_json_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Itera objetos de la lista `entries` de un JSON sin conservar el conjunto.

    Args:
        path: JSON Alexandria con raíz `{"entries": [...]}`.
    Yields:
        Un registro ComputedStructureEntry decodificado cada vez.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        prefix = ""
        while "[" not in prefix:
            part = stream.read(4096)
            if not part:
                raise ValueError("JSON no contiene la lista entries")
            prefix += part
        buffer = prefix[prefix.index("[") + 1:]
        eof = False
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                break
            if not buffer and eof:
                break
            try:
                entry, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                part = stream.read(1 << 20)
                if not part:
                    eof = True
                    if not buffer.strip(): break
                    raise ValueError("JSON truncado durante streaming")
                buffer += part
                continue
            if not isinstance(entry, dict):
                raise ValueError("Entrada JSON no es un objeto")
            yield entry
            buffer = buffer[end:]


def inventory_sources(stoi: dict[str, int], block_size: int) -> pd.DataFrame:
    """Crea inventario de archivos Alexandria/MatGPT sin abrir ABS3 test."""
    files = [RAW_JSON, ALL_CSV, STRICT_CSV,
             ALEX_DIR / "alexandria_00000_slices_all.csv",
             ALEX_DIR / "alexandria_00000_slices_energy.csv",
             ALEX_DIR / "alexandria_clean_basic.csv",
             ALEX_DIR / "alexandria_audit_all.csv",
             ALEX_DIR / "alexandria_singleton_compositions.csv",
             ROOT / "data/alexandria/notebooks_processing/outputs/cleaning_run_20261005_212323/alexandria_clean_strict.csv",
             TRAIN_CSV, MODEL_INFO, CHECKPOINT, CONFIG, DEV_CSV]
    out = []
    for path in files:
        if not path.is_file():
            out.append({"path": str(path.relative_to(ROOT)), "exists": False})
            continue
        rows = csv_rows(path) if path.suffix.lower() == ".csv" else None
        out.append({"path": str(path.relative_to(ROOT)), "exists": True,
                    "size_bytes": path.stat().st_size, "sha256": sha256(path), "rows": rows,
                    "file_type": path.suffix.lower(), "scope": "local source; no training or test labels read"})
    return pd.DataFrame(out)


def scan_csv(path: Path, slices_col: str, stoi: dict[str, int], block_size: int,
             retain_ids: bool = False) -> tuple[dict[str, Any], pd.DataFrame, set[str], set[str], pd.DataFrame]:
    """Resume estadísticas por chunks y exporta resumen por energía/elementos.

    Returns total de conteos, resumen de DFT row-wise, IDs, y hashes SLICES.
    """
    nrows = 0; valid_slices = 0; token_ok = 0; context_ok = 0; provisional_context_ok = 0; model_ready = 0
    n_atoms_sequence_mismatch = 0; n_atoms_sequence_comparable = 0
    lengths: list[int] = []; atom_counts: list[int] = []; spg_counts: Counter[str] = Counter()
    element_rows: Counter[str] = Counter(); element_structures: Counter[str] = Counter()
    n_element_counts: Counter[int] = Counter(); composition_counts: Counter[str] = Counter()
    ids: set[str] = set(); slice_hashes: set[str] = set(); formula_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    energy_rows: list[dict[str, Any]] = []; unknown_counts: Counter[str] = Counter()
    required = [slices_col]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in [slices_col, "mat_id", "reduced_formula", "formula", "composition_key", "elements",
                        "n_elements", "nsites", "spg", "space_group", "crystal_system", "prototype_id", "energy", "energy_total",
                        "energy_corrected", "energy_per_atom", "energy_corrected_per_atom", "e_form",
                        "e_above_hull", "correction", "slices_valid", "strict_valid_group", "location",
                        "parameters", "hubbards", "functional", "xc", "POTCAR"] if c in header]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        for row in chunk.to_dict(orient="records"):
            nrows += 1
            s = str(row.get(slices_col, ""))
            t = tokenizer_audit(s, stoi, block_size)
            is_valid = str(row.get("slices_valid", "True")).lower() in ("true", "1")
            valid_slices += int(is_valid)
            token_ok += int(t["tokenizer_compatible"])
            provisional_context_ok += int(t["within_context"])
            context_ok += int(t["tokenizer_compatible"] and t["within_context"])
            model_ready += int(is_valid and t["tokenizer_compatible"] and t["within_context"])
            lengths.append(t["input_length"])
            for token in t["unknown_tokens"]: unknown_counts[token] += 1
            sid = str(row.get("mat_id", ""))
            if retain_ids and sid: ids.add(sid)
            if s:
                slice_hashes.add(hashlib.sha256(" ".join(s.split()).encode()).hexdigest())
            try: n = int(float(row.get("nsites", 0)))
            except (TypeError, ValueError): n = 0
            if n: atom_counts.append(n)
            elems = parse_elements(row.get("elements"))
            slices_formula, slices_elements = composition_from_slices(s, set(stoi))
            if not elems: elems = sorted(set(slices_elements))
            formula = str(row.get("reduced_formula", row.get("formula", "")))
            if not formula and slices_formula: formula = slices_formula
            if n and slices_elements:
                n_atoms_sequence_comparable += 1
                if n != len(slices_elements): n_atoms_sequence_mismatch += 1
            for elem in elems:
                element_rows[elem] += 1
                if sid: element_structures[elem] += 1
            try: nelem = int(float(row.get("n_elements", len(elems))))
            except (TypeError, ValueError): nelem = len(elems)
            n_element_counts[nelem] += 1
            comp_key = parse_formula_key(formula) if formula else str(row.get("composition_key", ""))
            if comp_key: composition_counts[comp_key] += 1
            direct_system = str(row.get("crystal_system", "")).strip().lower()
            known_systems = {"triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"}
            if direct_system in known_systems:
                system = direct_system
            else:
                spg_value = row.get("spg")
                if spg_value is None or (isinstance(spg_value, float) and math.isnan(spg_value)):
                    spg_value = row.get("space_group")
                system = crystal_system(spg_value)
            spg_counts[system] += 1
            def number(col: str) -> float | None:
                try:
                    val = float(row.get(col))
                    return val if math.isfinite(val) else None
                except (TypeError, ValueError): return None
            et = number("energy_total")
            if et is None: et = number("energy")
            ec = number("energy_corrected")
            corr = number("correction")
            ep = number("energy_per_atom")
            epc = number("energy_corrected_per_atom")
            if ep is None and et is not None and n > 0: ep = et / n
            if epc is None and ec is not None and n > 0: epc = ec / n
            params = row.get("parameters")
            data_fields = ["functional", "xc", "hubbards", "POTCAR"]
            protocol_fields = [key for key in data_fields if row.get(key) not in (None, "", "{}", "nan")]
            energy_rows.append({"mat_id": sid, "reduced_formula": formula, "nsites": n,
                                "energy_total_ev": et, "energy_corrected_total_ev": ec,
                                "correction_ev": corr, "energy_per_atom_ev": ep,
                                "energy_corrected_per_atom_ev": epc,
                                "e_form_ev_atom": number("e_form"), "e_above_hull_ev_atom": number("e_above_hull"),
                                "energy_fields_present": int(et is not None),
                                "dft_protocol_fields_present": ";".join(protocol_fields),
                                "protocol_status": "PARTIAL" if protocol_fields else "UNKNOWN",
                                "tokenizer_compatible": t["tokenizer_compatible"],
                                "within_context": t["within_context"], "slices_valid": is_valid,
                                "model_ready": bool(is_valid and t["tokenizer_compatible"] and t["within_context"])})
            if formula and sid:
                formula_records[comp_key].append({"mat_id": sid, "formula": formula,
                    "slices_hash": hashlib.sha256(" ".join(s.split()).encode()).hexdigest() if s else "",
                    "prototype": str(row.get("prototype_id", "")), "energy_per_atom": ep,
                    "dft_status": "UNKNOWN" if not protocol_fields else "PARTIAL"})
    lens = np.asarray(lengths, dtype=float)
    summary = {"path": str(path.relative_to(ROOT)), "records": nrows,
        "valid_slices_rows": valid_slices, "tokenizer_compatible_rows": token_ok,
        "within_context_rows": context_ok, "provisional_whitespace_count_within_context_rows": provisional_context_ok,
        "model_ready_rows": model_ready,
        "unique_formulas": len(composition_counts), "unique_ids": len(ids) if retain_ids else None,
        "unique_slices_hashes": len(slice_hashes), "unique_formulas_with_multiple_records": sum(v > 1 for v in composition_counts.values()),
        "formula_record_counts": dict(Counter(composition_counts.values())),
        "elements": dict(element_rows), "n_elements": dict(n_element_counts),
        "crystal_systems_from_space_group": dict(spg_counts),
        "sequence_length_including_start_end": {q: float(np.quantile(lens, q)) for q in (.5,.9,.95,.99,1.0)} if len(lens) else {},
        "n_atoms": {q: float(np.quantile(atom_counts, q)) for q in (.5,.9,.95,.99,1.0)} if atom_counts else {},
        "n_atoms_over_20": int(np.sum(np.asarray(atom_counts)>20)),
        "n_atoms_vs_slices_species_mismatch_rows": n_atoms_sequence_mismatch,
        "n_atoms_vs_slices_species_comparable_rows": n_atoms_sequence_comparable,
        "unknown_token_occurrences_top": dict(unknown_counts.most_common(80)),
        "unknown_token_types": len(unknown_counts),
        "protocol_fields_available_in_csv": [x for x in ("functional","xc","hubbards","POTCAR") if x in header]}
    formula_table = pd.DataFrame([{"composition_key": key, "n_records": len(vals),
        "n_unique_ids": len({x['mat_id'] for x in vals}),
        "n_unique_slices": len({x['slices_hash'] for x in vals if x['slices_hash']}),
        "n_unique_prototypes": len({x['prototype'] for x in vals if x['prototype'] and x['prototype'] != "nan"}),
        "candidate_multistructure": len({x['slices_hash'] for x in vals if x['slices_hash']}) > 1,
        "dft_protocol_comparable": False,
        "gap_computation_status": "NOT_COMPUTED_PROTOCOL_UNKNOWN"} for key, vals in formula_records.items()])
    return summary, pd.DataFrame(energy_rows), ids, slice_hashes, formula_table


def scan_raw_json(path: Path, ids_in_training: set[str]) -> dict[str, Any]:
    """Cuenta y audita campos de ComputedStructureEntry por streaming."""
    start = time.monotonic(); n = 0; id_seen = set(); overlap = 0
    key_counts: Counter[str] = Counter(); data_key_counts: Counter[str] = Counter()
    protocol_records = 0; parameter_nonempty = 0; corrections = 0; structures = 0
    energy_equals_total = 0; nsites_counts: Counter[int] = Counter()
    for entry in stream_json_entries(path):
        n += 1
        data = entry.get("data", {}) or {}; params = entry.get("parameters", {}) or {}
        key_counts.update(entry.keys()); data_key_counts.update(data.keys())
        sid = str(data.get("mat_id", entry.get("entry_id", "")))
        if sid:
            id_seen.add(sid); overlap += int(sid in ids_in_training)
        protocol_keys = [k for k in params if any(s in k.lower() for s in ("xc", "gga", "hubbard", "potcar", "functional"))]
        protocol_records += int(bool(protocol_keys))
        parameter_nonempty += int(bool(params))
        corr = float(entry.get("correction", 0) or 0)
        corrections += int(abs(corr) > 1e-12)
        st = entry.get("structure") or {}
        sites = st.get("sites", [])
        structures += int(bool(sites))
        nsites_counts[len(sites)] += int(bool(sites))
        try: energy_equals_total += int(abs(float(entry.get("energy")) - float(data.get("energy_total"))) < 1e-10)
        except (TypeError, ValueError): pass
    return {"records": n, "unique_mat_ids": len(id_seen), "mat_id_overlap_with_matgpt_alex20": overlap,
        "top_entry_keys": dict(key_counts.most_common()), "top_data_keys": dict(data_key_counts.most_common()),
        "parameters_nonempty_records": parameter_nonempty, "dft_protocol_metadata_records": protocol_records,
        "nonzero_correction_records": corrections, "records_with_structure_sites": structures,
        "energy_matches_data_energy_total_records": energy_equals_total,
        "n_sites_distribution": dict(nsites_counts), "stream_wall_seconds": time.monotonic()-start,
        "dft_policy_status": "UNKNOWN unless calculation parameters/POTCAR/XC are present per entry"}


def system_summary(system: dict[str, Any], dev_formula_elements: dict[str, set[str]]) -> dict[str, Any]:
    """Deriva cobertura de elementos y fórmulas objetivo; no usa labels del test histórico."""
    alex_elements = set(system.get("elements", {}))
    target_elements = set().union(*dev_formula_elements.values()) if dev_formula_elements else set()
    return {"alexandria_elements": sorted(alex_elements), "development_abs3_elements": sorted(target_elements),
        "abs3_elements_missing_in_alexandria": sorted(target_elements - alex_elements),
        "abs3_elements_present_in_alexandria": sorted(target_elements & alex_elements),
        "sulfur_records": system.get("elements", {}).get("S", 0),
        "binary_ternary_multinary_record_counts": {str(k): v for k,v in system.get("n_elements", {}).items()},
        "development_formulas": len(dev_formula_elements),
        "historical_test_composition_exclusion_manifest_found": False,
        "historical_test_accessed": ACCESS_NOTE}


def make_figures(metrics: dict[str, Any], element_df: pd.DataFrame, strict_formula_df: pd.DataFrame,
                 strict_length: list[int], train_length: list[int], energy_df: pd.DataFrame) -> None:
    """Genera cinco figuras PNG de cobertura, tokenización y calidad de datos."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    strict = metrics["strict_csv"]; train = metrics["matgpt_training_csv"]
    # 1. Elements: local Alexandria strict source and development ABS3; MatGPT training is Alexandria-derived.
    symbols = sorted(element_df.loc[element_df.development_composition_count.gt(0), "element"].astype(str))
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(symbols)); width = .26
    tab=element_df.set_index("element").reindex(symbols,fill_value=0)
    local_pct=100*tab.structure_count/max(1,strict["records"])
    train_pct=100*tab.training_structure_count/max(1,train["records"])
    dev_pct=100*tab.development_composition_count/max(1,metrics["development"]["development_formulas"])
    ax.bar(x-width, train_pct, width, color="#2f7f79", label=f"Alex-20 preentrenamiento ({train['records']:,} estructuras)")
    ax.bar(x, local_pct, width, color="#5470a8", label=f"Alexandria local ({strict['records']:,} estructuras)")
    ax.bar(x+width, dev_pct, width, color="#d98e32", label=f"ABS₃ desarrollo ({metrics['development']['development_formulas']} fórmulas)")
    ax.set_xticks(x, symbols, rotation=45, ha="right"); ax.set_ylabel("Cobertura (%) dentro de cada fuente")
    ax.set_title("Cobertura elemental: Alexandria local y desarrollo ABS₃")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    ax.text(.01,-.27,"El mismo denominador se conserva dentro de cada fuente; una estructura o fórmula cuenta si contiene el elemento. ABS₃: solo desarrollo, test histórico excluido. Alexandria es un extracto local.",transform=ax.transAxes,fontsize=9)
    fig.tight_layout(); fig.savefig(IMAGES/"cobertura_elemental_alexandria_matgpt_abs3.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 2. Sequence lengths and exact context threshold.
    fig, ax = plt.subplots(figsize=(9,5.6))
    bins=np.linspace(0,500,51)
    ax.hist(train_length,bins=bins,density=True,histtype="step",linewidth=2,label="Alex-20 local usado por MatGPT",color="#2f7f79")
    ax.hist(strict_length,bins=bins,density=True,histtype="step",linewidth=2,label="Alexandria local convertida",color="#b64a48")
    info=json.loads(MODEL_INFO.read_text()); ax.axvline(info["block_size"],color="black",linestyle="--",label=f"Contexto MatGPT: {info['block_size']} tokens")
    ax.set_xlim(0,500)
    over_train=sum(v>500 for v in train_length); over_local=sum(v>500 for v in strict_length)
    ax.set_xlabel("Campos separados por espacios + inicio/cierre (tokens SLICES)"); ax.set_ylabel("Densidad")
    ax.set_title("Longitud SLICES y ventana del checkpoint MatGPT"); ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.text(.10,.025,f"Alex-20: longitud exacta del tokenizer. Alexandria: conteo provisional (tokens fuera de vocabulario). Vista ampliada a 500; fuera de escala: Alex-20 {over_train:,}, Alexandria {over_local:,}.",fontsize=8)
    fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.88)
    fig.savefig(IMAGES/"distribucion_longitud_slices_alexandria_matgpt.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 3. Structural/size distributions.
    fig, axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].hist(strict_formula_df.n_records,bins=np.arange(.5,max(2,strict_formula_df.n_records.max()+1.5),1),color="#5470a8")
    axes[0].set_xlabel("Registros por composición reducida"); axes[0].set_ylabel("Número de composiciones")
    axes[0].set_title("Repetición de composiciones")
    systems=metrics["strict_csv"]["crystal_systems_from_space_group"]
    axes[1].bar(list(systems),list(systems.values()),color="#679c71");axes[1].tick_params(axis="x",rotation=45)
    axes[1].set_ylabel("Estructuras");axes[1].set_title("Sistema cristalino registrado o inferido del grupo espacial")
    fig.suptitle("Estructuras Alexandria convertidas: distribución local")
    fig.text(.5,.01,"Filas de Alexandria local; no equivalen al universo completo. El recuento por composición no confirma polimorfos sin comparar estructuras.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(IMAGES/"distribucion_estructural_alexandria_local.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 4. Chemistry coverage as overlap fraction on dataset local.
    fig, ax=plt.subplots(figsize=(9,5.4))
    n=metrics["strict_csv"]["records"]; s=metrics["strict_csv"]["elements"].get("S",0)
    dev_s=metrics["development"]["development_formulas"]
    ax.bar(["Con azufre","Sin azufre"],[s,n-s],color=["#cf8e37","#7994bd"])
    ax.set_ylabel("Estructuras de Alexandria local");ax.set_title("Presencia de azufre en Alexandria local")
    ax.text(.02,.96,f"ABS₃ desarrollo: {metrics['development']['development_formulas']} fórmulas; elementos: {', '.join(metrics['development']['development_abs3_elements'])}\nFórmulas exactas en extracto Alexandria: {metrics['development'].get('exact_development_formula_overlap_count','NO_DETERMINADO')}; Alex-20: {metrics['development'].get('exact_development_formula_overlap_with_matgpt_alex20','NO_DETERMINADO')}",transform=ax.transAxes,va="top",fontsize=10)
    ax.text(.01,-.19,"Este gráfico no clasifica automáticamente compuestos ABX₃ como perovskitas; la similitud química no prueba similitud estructural.",transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(IMAGES/"cobertura_quimica_y_proximidad_abs3.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 5. Energy source distributions, raw and corrected separated.
    fig, ax=plt.subplots(figsize=(10,5.7))
    raw=pd.to_numeric(energy_df.energy_per_atom_ev,errors="coerce").dropna()
    corr=pd.to_numeric(energy_df.energy_corrected_per_atom_ev,errors="coerce").dropna()
    if len(raw): ax.hist(raw, bins=70,density=True,histtype="step",linewidth=2,label="Energía total cruda / átomo",color="#406da5")
    if len(corr): ax.hist(corr,bins=70,density=True,histtype="step",linewidth=2,label="Energía corregida / átomo",color="#bf4d4d")
    ax.set_xlabel("Energía (eV/átomo; distribuciones separadas)");ax.set_ylabel("Densidad")
    ax.set_title("Campos energéticos publicados en el extracto Alexandria local")
    ax.legend(frameon=False);ax.grid(alpha=.2)
    ax.text(.01,-.21,"No son energías comparables entre elementos ni se usan para ΔE: el protocolo XC/U/POTCAR no está completo por registro.",transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(IMAGES/"energias_alexandria_crudas_y_corregidas.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 6. Candidate groups per formula, without presenting unverified gaps.
    fig, ax=plt.subplots(figsize=(9,5.4))
    multi=strict_formula_df[strict_formula_df.n_unique_slices>1].n_unique_slices
    if len(multi): ax.hist(multi,bins=np.arange(1.5,max(3,multi.max()+1.5),1),color="#679c71")
    else: ax.text(.5,.5,"No hay grupos multiestructura verificables",ha="center",va="center",transform=ax.transAxes)
    ax.set_xlabel("SLICES distintos por composición reducida");ax.set_ylabel("Número de composiciones")
    ax.set_title("Candidatos a polimorfismo: conteo estructural preliminar")
    ax.text(.01,-.21,"Sin metadatos DFT homogéneos no se calculan ni ordenan gaps. Una cadena distinta no certifica estructura cristalográfica no equivalente.",transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(IMAGES/"grupos_multiestructura_y_gaps_no_verificados.png",dpi=300,bbox_inches="tight");plt.close(fig)


def write_report(summary: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Escribe una auditoría compacta con hechos, límites y decisión."""
    REPORT.mkdir(parents=True, exist_ok=True)
    m=metrics; strict=m["strict_csv"]; train=m["matgpt_training_csv"]; dev=m["development"]
    report=f'''# Auditoría Alexandria–MatGPT para transferencia energética

**Estado de auditoría:** `{summary['audit_status']}`  
**Viabilidad energética supervisada:** `{summary['energy_transfer_status']}`  
**Viabilidad intrafórmula:** `{summary['pairwise_transfer_status']}`  
**Continued pretraining:** `{summary['autoregressive_status']}`

## Resumen

**HECHO VERIFICADO.** El checkpoint auditado es el modelo **MatGPT incondicional** del repositorio `devalab/matgpt`, archivo `mat-concat/mat-uncond/weights/transformer_property_0.pth`, SHA-256 `{summary['checkpoint_sha256']}`. Tiene 12 bloques, dimensión oculta 768, 12 cabezas, vocabulario de {summary['vocab_size']} entradas y ventana `{summary['block_size']}`. El código y la configuración local `uncond.yaml` identifican `Training_dataset_alex20_with_encoding.csv` como entrada del entrenamiento sin propiedades condicionantes.

**EVIDENCIA DOCUMENTAL.** El README oficial de MatGPT describe Alex-20/MatterGPT como fuente de los datos de formación de energía y gap y publica el CSV de entrenamiento. El archivo local contiene {train['records']:,} filas y {train['unique_ids']:,} IDs. El modelo incondicional usa el corpus de SLICES para objetivo autoregresivo; este checkpoint no fue supervisado con energía durante ese entrenamiento. El hash del CSV no está incrustado en los pesos, por lo que la identidad exacta de cada ejemplo original se marca `{summary['pretraining_corpus_traceability']}`. Esto no debe confundirse con MatterGPT: ambos proyectos comparten procedencia Alex-20 documentada, pero sus checkpoints, implementaciones y objetivos son diferentes.

**RESULTADO ESTADÍSTICO.** El extracto local `alexandria_clean_strict.csv` tiene {strict['records']:,} filas, {strict['valid_slices_rows']:,} filas marcadas SLICES válidas y {strict['model_ready_rows']:,} secuencias compatibles con el tokenizer y ventana exactos de MatGPT sin truncamiento. El archivo no es la base Alexandria completa.

**RESULTADO DE SOLAPAMIENTO.** Se encontraron {summary['id_overlap_training']} IDs exactos en ambos: extracto local estricto y Alex-20 usado como corpus documentado. También coinciden exactamente {summary['slices_overlap_training']} hashes de SLICES normalizados. Por tanto, la fracción local ya vista por el preentrenamiento es {summary['overlap_fraction_local']:.1%} para IDs y {summary['slices_overlap_fraction_local']:.1%} para secuencias; la parte nueva exacta solo puede declararse para el extracto comparado, no para Alexandria completo.

**LIMITACIÓN DE ACCESO.** No se usó ninguna etiqueta ni resultado del test histórico. Durante el inventario inicial se mostró accidentalmente una vista previa de un solo registro de `data/raw_data/results(1).json`, antes de reconocer que podría contener datos ABS₃ protegidos. Se excluyó totalmente y no se volvió a abrir. No se puede afirmar que el archivo no se haya leído; la auditoría no incorpora ese contenido. El manifiesto autorizado de exclusión tampoco está disponible. Cualquier entrenamiento futuro debe permanecer bloqueado hasta verificar las exclusiones.

## Fuentes y procedencia del modelo

| Fuente | Observación |
|---|---|
| Checkpoint | `{summary['checkpoint_path']}`; SHA-256 `{summary['checkpoint_sha256']}` |
| Configuración | `mat-concat/config/uncond.yaml`; conditioning `unconditional`, propiedades seleccionadas vacías |
| Arquitectura | {summary['n_layers']} bloques, {summary['hidden_size']} dimensiones, {summary['n_heads']} cabezas; {summary['parameter_count']:,} parámetros verificados por carga histórica |
| Tokenizer | `{summary['model_info_path']}`, SHA-256 `{summary['model_info_sha256']}`; block size {summary['block_size']} |
| Corpus que declara la configuración | `Training_dataset_alex20_with_encoding.csv`, local, {train['records']:,} filas |
| Trazabilidad de la versión exacta | `{summary['pretraining_corpus_traceability']}`; no hay hash del CSV ligado a checkpoint |

El modelo es **MatGPT**, no MatterGPT. La publicación de MatterGPT reporta Alex-20 como corpus de SLICES para su entrenamiento; la documentación de MatGPT local también lo señala como CSV de entrada. En la ejecución que se audita, el YAML declara un modelo incondicional y `selected_properties: []`: la evidencia local apoya el aprendizaje de secuencias estructurales, no de energías como objetivo de ese checkpoint.

## Inventario y compatibilidad SLICES

- CSV MatGPT/Alex-20: {train['records']:,} registros; compatibles tokenizer: {train['tokenizer_compatible_rows']:,}; dentro de contexto: {train['within_context_rows']:,}; listos: {train['model_ready_rows']:,}. Fórmulas derivadas de la lista inicial de especies de SLICES, terminada en el primer índice entero: {train['unique_formulas']:,}.
- CSV Alexandria estricto local: {strict['records']:,}; SLICES marcado válido por el export: {strict['valid_slices_rows']:,}; compatible tokenizer: {strict['tokenizer_compatible_rows']:,}; secuencias que podrían caber por conteo de campos whitespace (no tokenizer válido): {strict['provisional_whitespace_count_within_context_rows']:,}; listo para MatGPT: {strict['model_ready_rows']:,}.
- Block size exacto: {summary['block_size']} incluidos `<start>` y `<end>`. No se truncaron secuencias.
- Estrategia de conversión local: no se halló un script de conversión/manifiesto versionado junto a los CSV. El formato del export local incluye prefijos y etiquetas adicionales; no se presume equivalencia con estrategia 4 de MatGPT. Se requiere verificar el código/versión de conversión antes de usarlo.
- Vocabulario original de MatGPT admite los tokens estructurales de Alex-20. La conversión Alexandria local muestra {strict['tokenizer_compatible_rows']:,}/{strict['records']:,} filas compatibles exactas. Tokens fuera de vocabulario frecuentes: `{list(strict['unknown_token_occurrences_top'].items())[:10]}`.

## Cobertura química y estructural

Elementos observados en extracto estricto: {', '.join(sorted(strict['elements']))}. Desarrollo ABS₃: {dev['development_formulas']} fórmulas; elementos {', '.join(dev['development_abs3_elements'])}. El registro de azufre cuenta {strict['elements'].get('S',0):,} estructuras, pero no basta para acreditar transferencia de ordenamiento polimórfico. Se encontraron {dev.get('exact_development_formula_overlap_count','NO_DETERMINADO')} fórmulas exactas del desarrollo ABS₃ en el extracto local. Esta comparación cubre únicamente desarrollo y no abre test.

El sistema cristalino usa la columna directa de Alex-20 cuando existe; para Alexandria se infiere del número de grupo espacial mediante intervalos cristalográficos estándar. No equivale a una etiqueta de prototipo. SLICES diferentes y prototype_id diferente son indicios para revisión, no prueba de estructuras no equivalentes. En el CSV de entrenamiento local, nsites llega a {train['n_atoms'][1.0]:.0f}; {train['n_atoms_over_20']:,} filas superan 20 sitios. La comprobación de nsites frente a la lista inicial de especies fue evaluable en {train['n_atoms_vs_slices_species_comparable_rows']:,} filas y discrepó en {train['n_atoms_vs_slices_species_mismatch_rows']:,}. No se debe interpretar nsites como número de átomos de celda primitiva sin confirmar el esquema del CSV. No se ejecutó StructureMatcher en todo el corpus; no se realizaron comparaciones cuadráticas.

## Energías DFT y polimorfismo

El CSV conserva `energy`/`energy_total`, `energy_corrected`, `correction`, `energy_per_atom`, `energy_corrected_per_atom`, `e_form` y `e_above_hull`. `e_form` y `e_above_hull` no se sustituyeron por energía total. Donde fue necesario, eV/átomo se auditó como E_total/N.

En el archivo estructural local, los campos por registro para XC, POTCAR y Hubbard U no están disponibles: el JSON original tiene `parameters` vacío en {m['raw_json']['parameters_nonempty_records']:,}/{m['raw_json']['records']:,} registros y no ofrece en el extracto un funcional/U verificable. `energy` coincide con `data.energy_total` en {m['raw_json']['energy_matches_data_energy_total_records']:,}/{m['raw_json']['records']:,} filas; hay correction distinta de cero en {m['raw_json']['nonzero_correction_records']:,}/{m['raw_json']['records']:,}. `energy_corrected` es campo separado. Se clasifica el protocolo como `DFT_PROTOCOL_UNKNOWN`, no como PBE homogéneo. Se mantienen crudas y corregidas separadas. No se computan gaps físicamente comparables ni rankings a partir de registros heterogéneos. Las estadísticas de rango y consistencia están en `metrics.json`; los valores eV/átomo se calcularon dividiendo el total correspondiente por N, nunca mezclando targets.

Conteo de composición repetida en el extracto: {strict['unique_formulas_with_multiple_records']:,} fórmulas tienen más de un registro; el número de entradas por fórmula y de hashes SLICES se exporta en `estadisticas_polimorfismo.csv`. Estos son **candidatos** a multiestructura, no un conteo certificado de polimorfos energéticamente comparables. El objetivo de aprendizaje intrafórmula no se puede aprobar con la evidencia DFT local actual.

## Duplicados, novedad y leakage

- IDs exactos compartidos con Alex-20: {summary['id_overlap_training']:,} de {strict['records']:,} filas del extracto ({summary['overlap_fraction_local']:.1%}).
- Hashes SLICES exactos compartidos tras normalizar espacios: {summary['slices_overlap_training']:,}.
- Composiciones reducidas compartidas entre Alexandria local y Alex-20 (Alex-20 derivado del prefijo de especies; Alexandria de fórmula explícita): {m['overlap']['alexandria_formula_count_also_in_matgpt_alex20']:,} de {strict['unique_formulas']:,}.
- De los IDs repetidos, la secuencia normalizada coincide exactamente en {summary['same_id_same_slices']:,}; en {summary['same_id_different_slices']:,} el ID es igual pero serialización SLICES difiere. Es probable que corresponda a conversiones/versiones distintas, no se deduce equivalencia solo del texto.
- Misma composición ABS₃ desarrollo: {dev.get('exact_development_formula_overlap_count','NO_DETERMINADO')} fórmulas en Alexandria local; esas fórmulas deben excluirse del entrenamiento intermedio inductivo.
- Equivalencia estructural: no verificada sistemáticamente para todo el extracto.
- Test histórico: no se usó ninguna etiqueta; ocurrió una vista previa accidental y excluida de un registro potencialmente protegido. Manifiesto de fórmulas excluidas no localizado. Estado futuro: `TRANSFER_TRAINING_BLOCKED_PENDING_HOLDOUT_EXCLUSION`.

La novedad respecto a MatGPT solo es cuantificable sobre la pareja local comparada (IDs/SLICES exactos). No se puede afirmar novedad frente al corpus Alexandria global, ni equivalencia del estado estructural por coincidencia solo de fórmula.

## Evaluación de estrategias futuras

| Estrategia | Estado | Fundamento |
|---|---|---|
| A. Continued pretraining autoregresivo | `{summary['autoregressive_status']}` | Alex-20 ya es el corpus estructural documentado de MatGPT y sus secuencias son compatibles. Alexandria local convertida tiene 0 filas compatibles con el tokenizer actual; solo cabe evaluar una conversión/reconversión compatible y deduplicada fuera de Alex-20. No repetir el corpus sin filtrarlo por solapamiento. |
| B. Adaptación energética supervisada amplia | `{summary['energy_transfer_status']}` | Existen campos de energía, pero protocolo XC/U/POTCAR no verificable por fila y no existe exclusión autorizada del test. `e_form` no es energía total. |
| C. Adaptación intrafórmula/pairwise | `{summary['pairwise_transfer_status']}` | Hay composiciones repetidas/SLICES candidatos, pero no se certifican equivalencia de estructura ni comparabilidad energética entre cálculos. No se pueden construir gaps DFT válidos todavía. |

## Subconjuntos candidatos

Los filtros y conteos están en `subconjuntos_candidatos.csv`. Ningún candidato queda habilitado para entrenamiento ABS₃ hasta resolver la exclusión de todas las fórmulas del test. Un subconjunto de secuencias podría ser útil para un estudio de continued pretraining solo si se deduplica por IDs/SLICES contra Alex-20 y se demuestra que la conversión usa la misma serialización/tokenizer. El subset energético necesita trazabilidad de campaña DFT antes de definir tamaño.

## Dictamen

**No conviene iniciar primero un nuevo continued pretraining con el Alexandria local en bruto:** el checkpoint MatGPT auditado ya se entrenó autoregresivamente en Alex-20, derivado de Alexandria, y una parte cuantificable del extracto local solapa exactamente con ese corpus. Un continued pretraining solo tendría hipótesis razonable si se identifica una fracción realmente novedosa, compatible, con serialización estable y exclusiones de evaluación completas.

**No se recomienda aún adaptación energética supervisada ni aprendizaje intrafórmula.** Los datos locales contienen labels de energía útiles como candidatos, pero no se verifican el protocolo DFT común y sus correcciones por registro y falta una lista autorizada para bloquear todas las composiciones de evaluación ABS₃. Primero hay que recuperar metadatos de cálculo/funcional/POTCAR y el manifiesto de exclusión del test; entonces volver a evaluar un subset con energías homogéneas.

### Clasificación de conclusiones

- **HECHO VERIFICADO:** checkpoint/hash, YAML `unconditional`, 172 tokens, block_size 442, configuración local apunta a Alex-20.
- **EVIDENCIA DOCUMENTAL:** MatGPT y MatterGPT describen Alex-20 como corpus derivado de Alexandria; no implica que ambos checkpoints o objetivos sean iguales.
- **RESULTADO ESTADÍSTICO:** tamaños, tokens listos, overlap exacto y cobertura vienen de las fuentes locales y CSV anexos.
- **INFERENCIA:** el nuevo continued pretraining sobre datos ya vistos tendría novedad limitada; supervisión DFT podría añadir tipo de señal distinto, si etiquetas homogéneas.
- **HIPÓTESIS:** un subset Alexandria no visto y energéticamente coherente podría ayudar a estabilizar adaptación ABS₃.
- **INFORMACIÓN NO DISPONIBLE:** corpus/hash exacto empaquetado en pesos; homogeneidad completa DFT; equivalencia estructural de candidatos; exclusión autorizada del test histórico.

## Figuras

![Cobertura elemental](../../../images/MatGPT/MatGPT_Alexandria_Transfer_Audit/cobertura_elemental_alexandria_matgpt_abs3.png)

![Longitud SLICES](../../../images/MatGPT/MatGPT_Alexandria_Transfer_Audit/distribucion_longitud_slices_alexandria_matgpt.png)

![Estructuras](../../../images/MatGPT/MatGPT_Alexandria_Transfer_Audit/distribucion_estructural_alexandria_local.png)

![Cobertura ABS3](../../../images/MatGPT/MatGPT_Alexandria_Transfer_Audit/cobertura_quimica_y_proximidad_abs3.png)

![Energías](../../../images/MatGPT/MatGPT_Alexandria_Transfer_Audit/energias_alexandria_crudas_y_corregidas.png)

![Grupos multiestructura](../../../images/MatGPT/MatGPT_Alexandria_Transfer_Audit/grupos_multiestructura_y_gaps_no_verificados.png)

## Referencias

1. MatGPT, repositorio y datos/checkpoints: https://github.com/devalab/matgpt y https://doi.org/10.6084/m9.figshare.34330655
2. MatterGPT: https://doi.org/10.48550/arXiv.2408.07608
3. Alexandria database: https://alexandria.icams.rub.de/ (PBE/PBEsol/SCAN y otras colecciones; no intercambiables sin control de protocolo).
4. SLICES: Xiao et al., https://doi.org/10.1038/s41467-023-42870-7
5. Material local: `models/MatGPT-source/README.md`, `mat-concat/config/uncond.yaml`, `mat-concat/mat-uncond/weights/model_info_property_0.json`.
'''
    (REPORT / "auditoria_alexandria_vs_matgpt.md").write_text(report, encoding="utf-8")
    readme = """# Auditoría Alexandria–MatGPT\n\nAuditoría de solo lectura, sin entrenamiento ni uso de etiquetas del test ABS₃. Una vista previa accidental de un registro potencialmente protegido se documenta y excluye.\n\n- Informe: `reports/MatGPT/MatGPT_Alexandria_Transfer_Audit/auditoria_alexandria_vs_matgpt.md`\n- Salidas compactas: `metrics.json`, `summary.json`, `resources.json`\n- Tablas: inventario, elementos, energía, duplicados, polimorfismo y subconjuntos\n- Figuras PNG: `images/MatGPT/MatGPT_Alexandria_Transfer_Audit/`\n\nEjecutar desde la raíz: `.venv/bin/python scripts/MatGPT/MatGPT_Alexandria_Transfer_Audit/audit.py --run`\nNo entrena modelos ni usa labels del test histórico; véase la nota de acceso accidental en el informe.\n"""
    (SCRIPT / "README.md").write_text(readme, encoding="utf-8")


def run_audit() -> dict[str, Any]:
    """Ejecuta auditoría local, produce tres JSON y tablas compactas."""
    started=time.monotonic(); cpu_started=time.process_time()
    for folder in (EXPERIMENT, REPORT, IMAGES): folder.mkdir(parents=True, exist_ok=True)
    model_meta=json.loads(MODEL_INFO.read_text(encoding="utf-8")); stoi=model_meta["stoi"]
    block_size=int(model_meta["block_size"])
    inv=inventory_sources(stoi,block_size); inv.to_csv(EXPERIMENT/"inventario_fuentes.csv",index=False)
    dev=pd.read_csv(DEV_CSV, usecols=["structure_id","split","reduced_formula"])
    dev=dev[dev.split.isin(["train","val"])].copy()
    if len(dev)!=210 or dev.structure_id.nunique()!=210: raise ValueError("Desarrollo ABS3 no tiene 210 IDs train/val")
    dev_formula_elements={}
    try:
        from pymatgen.core import Composition
        for formula in dev.reduced_formula.unique():
            comp=Composition(formula);dev_formula_elements[str(formula)]=set(comp.get_el_amt_dict())
    except ImportError:
        import re
        for formula in dev.reduced_formula.unique(): dev_formula_elements[str(formula)]=set(re.findall(r"[A-Z][a-z]?",formula))
    training,_,train_ids,train_hashes,train_formula=scan_csv(TRAIN_CSV,"SLICES",stoi,block_size,True)
    strict,energy_df,strict_ids,strict_hashes,formula_df=scan_csv(STRICT_CSV,"slices",stoi,block_size,True)
    all_stats,_,_,_,_=scan_csv(ALL_CSV,"slices",stoi,block_size,True)
    overlap_ids=train_ids & strict_ids; overlap_slices=train_hashes & strict_hashes
    # Match IDs exactly and compare their SLICES for direct corpus intersection.
    train_slice_by_id={};
    for chunk in pd.read_csv(TRAIN_CSV,usecols=["mat_id","SLICES"],chunksize=CHUNK,dtype=str,low_memory=False):
        for row in chunk.itertuples(index=False):
            if row.mat_id in strict_ids: train_slice_by_id[row.mat_id]=hashlib.sha256(" ".join(row.SLICES.split()).encode()).hexdigest()
    strict_slice_by_id={}
    for chunk in pd.read_csv(STRICT_CSV,usecols=["mat_id","slices"],chunksize=CHUNK,dtype=str,low_memory=False):
        for row in chunk.itertuples(index=False): strict_slice_by_id[row.mat_id]=hashlib.sha256(" ".join(row.slices.split()).encode()).hexdigest()
    matching_ids_same_slices=sum(train_slice_by_id[k]==strict_slice_by_id[k] for k in overlap_ids)
    matching_ids_different_slices=len(overlap_ids)-matching_ids_same_slices
    dev_keys={parse_formula_key(str(f)) for f in dev.reduced_formula.unique()}
    local_formula_keys=set(formula_df.composition_key.astype(str))
    training_formula_keys=set(train_formula.composition_key.astype(str)) if len(train_formula) else set()
    exact_formula_overlap=len(dev_keys & local_formula_keys)
    development=system_summary(strict,dev_formula_elements)
    development["exact_development_formula_overlap_count"]=exact_formula_overlap
    development["matching_development_formulas"] = sorted(dev_keys & local_formula_keys)
    development["exact_development_formula_overlap_with_matgpt_alex20"] = len(dev_keys & training_formula_keys)
    development["alexandria_formulas_also_in_matgpt_alex20"] = len(local_formula_keys & training_formula_keys)
    development["alexandria_formula_overlap_fraction_with_matgpt_alex20"] = len(local_formula_keys & training_formula_keys)/max(1,len(local_formula_keys))
    formula_df["abs3_development_formula"] = formula_df.composition_key.isin(dev_keys)
    formula_df.to_csv(EXPERIMENT/"estadisticas_polimorfismo.csv",index=False)
    energy_df.to_csv(EXPERIMENT/"auditoria_energia_dft.csv",index=False)
    def numeric_summary(column: str) -> dict[str, Any]:
        values=pd.to_numeric(energy_df[column],errors="coerce").dropna().to_numpy(float)
        return {"count":int(len(values)),"min":float(values.min()) if len(values) else None,
            "p01":float(np.quantile(values,.01)) if len(values) else None,
            "median":float(np.median(values)) if len(values) else None,
            "p99":float(np.quantile(values,.99)) if len(values) else None,
            "max":float(values.max()) if len(values) else None}
    raw_total=pd.to_numeric(energy_df.energy_total_ev,errors="coerce")
    corrected_total=pd.to_numeric(energy_df.energy_corrected_total_ev,errors="coerce")
    correction=pd.to_numeric(energy_df.correction_ev,errors="coerce")
    raw_per=pd.to_numeric(energy_df.energy_per_atom_ev,errors="coerce")
    corrected_per=pd.to_numeric(energy_df.energy_corrected_per_atom_ev,errors="coerce")
    n_atoms=pd.to_numeric(energy_df.nsites,errors="coerce")
    consistency={"raw_total_equals_corrected_plus_correction_rows":int(((corrected_total-raw_total-correction).abs()<1e-6).sum()),
        "rows_with_nonzero_correction":int((correction.abs()>1e-12).sum()),
        "raw_e_per_atom_consistent_with_total_over_n_rows":int(((raw_per-raw_total/n_atoms).abs()<1e-8).sum()),
        "corrected_e_per_atom_consistent_with_corrected_total_over_n_rows":int(((corrected_per-corrected_total/n_atoms).abs()<1e-8).sum())}
    # Candidate manifest reports all strict input IDs with explicit training block.
    cand=[]
    for chunk in pd.read_csv(STRICT_CSV,usecols=["mat_id","reduced_formula","slices_valid"],chunksize=CHUNK,low_memory=False):
        for row in chunk.to_dict(orient="records"):
            key=parse_formula_key(str(row["reduced_formula"]))
            cand.append({"mat_id":row["mat_id"],"reduced_formula":row["reduced_formula"],
                "matches_abs3_development_formula":key in dev_keys,
                "id_in_matgpt_alex20_pretraining_csv":str(row["mat_id"]) in train_ids,
                "eligible_for_future_abs3_transfer_training":False,
                "block_reason":"HISTORICAL_TEST_EXCLUSION_MANIFEST_MISSING; DFT_PROTOCOL_UNKNOWN"})
    candidates=pd.DataFrame(cand)
    candidates.to_csv(EXPERIMENT/"subconjuntos_candidatos.csv",index=False)
    duplicates=pd.DataFrame([{"comparison":"Alexandria strict local vs MatGPT Alex-20","unit":"mat_id",
        "matches":len(overlap_ids),"left_n":len(strict_ids),"right_n":len(train_ids),
        "interpretation":"Exact source-ID overlap; local rows already represented in documented pretraining corpus"},
        {"comparison":"Alexandria strict local vs MatGPT Alex-20","unit":"normalized SLICES SHA256",
        "matches":len(overlap_slices),"left_n":len(strict_hashes),"right_n":len(train_hashes),
        "interpretation":"Exact string overlap after collapsing whitespace; not structural equivalence"},
        {"comparison":"Same source ID, sequence comparison","unit":"SLICES hash equal",
        "matches":matching_ids_same_slices,"left_n":len(overlap_ids),"right_n":len(overlap_ids),
        "interpretation":"Same IDs, same exact normalized sequence"},
        {"comparison":"Same source ID, sequence comparison","unit":"SLICES hash differs",
        "matches":matching_ids_different_slices,"left_n":len(overlap_ids),"right_n":len(overlap_ids),
        "interpretation":"Same source ID but different conversion/serialization; review version before use"},
        {"comparison":"Alexandria local vs ABS3 development","unit":"reduced composition",
        "matches":exact_formula_overlap,"left_n":len(local_formula_keys),"right_n":len(dev_keys),
        "interpretation":"Development compositions only; test labels unused; one-record preview incident documented"}])
    duplicates.to_csv(EXPERIMENT/"duplicados_y_solapamiento.csv",index=False)
    element_rows=[]
    training_element_counts=training["elements"]
    for elem in sorted(set(strict["elements"]) | set(training["elements"]) |
                       set().union(*dev_formula_elements.values())):
        count=strict["elements"].get(elem,0)
        dev_formulas=sum(elem in es for es in dev_formula_elements.values())
        element_rows.append({"element":elem,"structure_count":count,
            "training_structure_count":training_element_counts.get(elem,0),
            "development_composition_count":dev_formulas,
            "token_in_matgpt_vocabulary":elem in stoi})
    element_df=pd.DataFrame(element_rows).sort_values("element")
    element_df.to_csv(EXPERIMENT/"cobertura_elemental.csv",index=False)
    # Length arrays are recomputed in a streaming pass for figures only.
    def lengths(path:Path,col:str)->list[int]:
        values=[]
        for chunk in pd.read_csv(path,usecols=[col],chunksize=CHUNK,dtype=str,low_memory=False):
            values.extend(chunk[col].fillna("").map(lambda s:len(str(s).split())+2).tolist())
        return values
    strict_lengths=lengths(STRICT_CSV,"slices"); train_lengths=lengths(TRAIN_CSV,"SLICES")
    raw_summary=scan_raw_json(RAW_JSON,train_ids)
    # The test is excluded from analysis; one accidental preview of a potentially
    # protected record occurred before recognizing its scope.
    summary={"status":"CONDITIONAL","audit_status":"CONDITIONAL",
        "energy_transfer_status":"NO_GO_PENDING_DFT_PROTOCOL_AND_TEST_EXCLUSION",
        "pairwise_transfer_status":"NO_GO_PENDING_COMPARABLE_MULTISTRUCTURE_DFT_AND_TEST_EXCLUSION",
        "autoregressive_status":"CONDITIONAL_GO_ONLY_FOR_VERIFIED_NOVEL_TOKENIZER_COMPATIBLE_SUBSET",
        "historical_test_accessed":ACCESS_NOTE,"checkpoint_path":str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256":sha256(CHECKPOINT),"model_info_path":str(MODEL_INFO.relative_to(ROOT)),
        "model_info_sha256":sha256(MODEL_INFO),"config_path":str(CONFIG.relative_to(ROOT)),
        "vocab_size":len(stoi),"block_size":block_size,"n_layers":12,"hidden_size":768,"n_heads":12,
        "parameter_count":114026668,"pretraining_corpus":"Alex-20 / Alexandria-derived; MatGPT unconditional run config points to local Alex-20 CSV",
        "pretraining_corpus_traceability":"PARTIALLY_VERIFIED (config/README + local Alex-20 file; checkpoint does not embed its dataset hash)",
        "alexandria_raw_json_records":raw_summary["records"],"matgpt_training_csv_records":training["records"],
        "alexandria_strict_csv_records":strict["records"],"id_overlap_training":len(overlap_ids),
        "slices_overlap_training":len(overlap_slices),"same_id_same_slices":matching_ids_same_slices,
        "same_id_different_slices":matching_ids_different_slices,
        "overlap_fraction_local":len(overlap_ids)/max(1,strict["records"]),
        "slices_overlap_fraction_local":len(overlap_slices)/max(1,strict["unique_slices_hashes"]),
        "test_exclusion_status":"TRANSFER_TRAINING_BLOCKED_PENDING_HOLDOUT_EXCLUSION",
        "development_abs3_formula_count":len(dev_keys),"development_exact_formula_overlap":exact_formula_overlap}
    metrics={"matgpt_training_csv":training,"strict_csv":strict,"all_csv":all_stats,
        "raw_json":raw_summary,"development":development,
        "overlap":{"exact_mat_id":len(overlap_ids),"exact_slices_hash":len(overlap_slices),
            "same_id_same_slices":matching_ids_same_slices,"same_id_different_slices":matching_ids_different_slices,
            "unique_formula_count_matgpt_alex20":len(training_formula_keys),
            "alexandria_formula_count_also_in_matgpt_alex20":len(local_formula_keys & training_formula_keys),
            "local_fraction_id_seen_in_pretraining":summary["overlap_fraction_local"],
            "local_fraction_slices_seen_in_pretraining":summary["slices_overlap_fraction_local"]},
        "dft_protocol":{"status":"DFT_PROTOCOL_UNKNOWN",
            "csv_has_xc_or_functional_fields":bool(strict["protocol_fields_available_in_csv"]),
            "raw_entries_with_parameter_protocol":raw_summary["dft_protocol_metadata_records"],
            "raw_entry_parameters_nonempty":raw_summary["parameters_nonempty_records"],
            "energy_columns":["energy","energy_total","energy_corrected","correction","energy_per_atom","energy_corrected_per_atom","e_form","e_above_hull"],
            "raw_total_ev":numeric_summary("energy_total_ev"),
            "corrected_total_ev":numeric_summary("energy_corrected_total_ev"),
            "raw_ev_per_atom":numeric_summary("energy_per_atom_ev"),
            "corrected_ev_per_atom":numeric_summary("energy_corrected_per_atom_ev"),
            "formation_energy_ev_per_atom":numeric_summary("e_form_ev_atom"),
            "energy_field_consistency":consistency,
            "gaps_computed":False,"reason":"No per-record verifiable homogeneous XC/U/POTCAR policy"},
        "candidates":{"strict_rows":len(candidates),"matches_abs3_development_formula":int(candidates.matches_abs3_development_formula.sum()),
            "training_eligible":0,"blocked_reason":"historical test exclusion manifest absent; DFT protocol unknown"},
        "historical_test_accessed":ACCESS_NOTE}
    resources={"status":"COMPLETE","mode":"read_only_audit","training_executed":False,
        "historical_test_accessed":ACCESS_NOTE,"cpu_count":os.cpu_count(),"cpu_seconds":time.process_time()-cpu_started,
        "wall_seconds":time.monotonic()-started,"peak_rss_bytes":rss_bytes(),
        "files_processed":len(inv),"alexandria_raw_records_streamed":raw_summary["records"],
        "records_scanned_csv":{"alex20":training["records"],"alexandria_strict":strict["records"],"alexandria_all":all_stats["records"]}}
    make_figures(metrics,element_df,formula_df,strict_lengths,train_lengths,energy_df)
    write_report(summary,metrics)
    atomic_json(EXPERIMENT/"metrics.json",metrics);atomic_json(EXPERIMENT/"summary.json",summary);atomic_json(EXPERIMENT/"resources.json",resources)
    LOG.info("%s",{"event":"audit_complete","status":summary["audit_status"],
        "Alex20_rows":training["records"],"strict_rows":strict["records"],"id_overlap":len(overlap_ids),
        "slices_overlap":len(overlap_slices),"model_ready":strict["model_ready_rows"],
        "wall_seconds":resources["wall_seconds"],"peak_rss_bytes":resources["peak_rss_bytes"]})
    return summary


def main() -> None:
    """Valida argumentos, configura logging y ejecuta la auditoría de solo lectura."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",help="ejecuta auditoría local de solo lectura")
    args=parser.parse_args()
    if not args.run: parser.error("Indica --run para iniciar la auditoría")
    EXPERIMENT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),logging.FileHandler(EXPERIMENT/"audit.log",encoding="utf-8")])
    run_audit()


if __name__=="__main__": main()
