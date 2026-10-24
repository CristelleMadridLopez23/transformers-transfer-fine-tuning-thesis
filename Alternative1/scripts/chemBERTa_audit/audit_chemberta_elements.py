#!/usr/bin/env python3
"""Audita símbolos químicos en el vocabulario y tokenización real de ChemBERTa.

Ejemplos:
  python audit_chemberta_elements.py
  python audit_chemberta_elements.py --model seyonec/ChemBERTa-zinc-base-v1
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from transformers import AutoTokenizer
from pymatgen.core.periodic_table import Element


def probe(tokenizer, text: str) -> dict:
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    decoded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    unknown = (tokenizer.unk_token_id is not None and
               tokenizer.unk_token_id in ids)
    return {
        "tokens": " | ".join(map(str, tokens)),
        "n_tokens": len(ids),
        "unk": unknown,
        "roundtrip": decoded == text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--output-dir", default="03_reports/chemberta_element_audit")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    vocab = tok.get_vocab()
    elements = [Element.from_Z(z).symbol for z in range(1, 119)]
    rows = []

    for symbol in elements:
        unbracketed = symbol in vocab
        bracketed = f"[{symbol}]" in vocab
        # Tokens como [Fe+2], [Ti+4], [13C], [SiH], [Ni+2].
        # No se confunden prefijos de símbolos: C frente a Cl, S frente a Sr.
        atom_pat = re.compile(
            r"^\[(?:\d+)?" + re.escape(symbol) + r"(?=[H@+\-\]])[^]]*\]$"
        )
        other_forms = sorted(
            (t for t in vocab if atom_pat.match(t)),
            key=lambda t: vocab[t],
        )
        raw = probe(tok, symbol)  # Como un elemento textual en SLICES
        after_space = probe(tok, " " + symbol)  # Otra posición frecuente en SLICES
        bracket = probe(tok, f"[{symbol}]")  # Forma atómica SMILES
        rows.append({
            "model": args.model,
            "Z": Element(symbol).Z,
            "element": symbol,
            "raw_vocab_exact": unbracketed,
            "raw_vocab_id": vocab.get(symbol, ""),
            "bracket_vocab_exact": bracketed,
            "bracket_vocab_id": vocab.get(f"[{symbol}]", ""),
            "smiles_vocab_forms_count": len(other_forms),
            "smiles_vocab_forms": "; ".join(other_forms),
            "slices_tokens": raw["tokens"],
            "slices_n_tokens": raw["n_tokens"],
            "slices_unk": raw["unk"],
            "slices_roundtrip": raw["roundtrip"],
            "slices_one_token": raw["n_tokens"] == 1 and raw["roundtrip"] and not raw["unk"],
            "slices_after_space_tokens": after_space["tokens"],
            "slices_after_space_n_tokens": after_space["n_tokens"],
            "slices_after_space_unk": after_space["unk"],
            "slices_after_space_roundtrip": after_space["roundtrip"],
            "smiles_bracket_tokens": bracket["tokens"],
            "smiles_bracket_n_tokens": bracket["n_tokens"],
            "smiles_bracket_unk": bracket["unk"],
            "smiles_bracket_roundtrip": bracket["roundtrip"],
        })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", args.model)
    outfile = out / f"{name}_elements.csv"
    with outfile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Modelo: {args.model} | vocab: {len(vocab)} entradas")
    print(f"Elementos con símbolo exacto en vocabulario: {sum(r['raw_vocab_exact'] for r in rows)} / 118")
    print(f"Elementos con [Símbolo] exacto: {sum(r['bracket_vocab_exact'] for r in rows)} / 118")
    print(f"Elementos con alguna variante SMILES en vocabulario: {sum(r['smiles_vocab_forms_count'] > 0 for r in rows)} / 118")
    print(f"Elementos SLICES de un token aislado, sin pérdida: {sum(r['slices_one_token'] for r in rows)} / 118")
    print("\nEjemplos relevantes para Alexandria:")
    watch = {"Li", "Na", "Mg", "Al", "Si", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Sr", "Ba", "Zr", "Mo", "Cd", "Rh"}
    for row in rows:
        if row["element"] in watch:
            print(f"  {row['element']:>2}: [{row['element']}] ID={str(row['bracket_vocab_id']):>3} "
                  f"| SLICES='{row['element']}' -> {row['slices_tokens']} "
                  f"| roundtrip={row['slices_roundtrip']}")
    print(f"\nCSV: {outfile}")
    print("AVISO: vocabulario/tokenización NO prueba frecuencia ni aparición en el corpus de 77M SMILES.")


if __name__ == "__main__":
    main()
