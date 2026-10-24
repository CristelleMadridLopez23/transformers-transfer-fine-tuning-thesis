#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clearance PAD para ChemBERTa-RoBERTa + tokenizer lexico SLICES v1.

SOLUCION CANDIDATA (no reemplaza checkpoints originales):
  - Mantener model.config.pad_token_id == 1 y sus posiciones aprendidas.
  - Utilizar el token heredado [unused1] (ID 1) como PAD SOLO en el
    nuevo tokenizer SLICES. [PAD] conserva su ID original 0.
  - Verificar mapeos, padding, posiciones, forward, MLM labels y serializacion.

Crea un directorio tokenizer_pad_aligned SOLO si todos los controles pasan.
No guarda pesos; el modelo redimensionado se usa exclusivamente en RAM.

Ejemplo:
 python scripts/chemBERTa_audit/clearance_chemberta_pad.py \
  --checkpoint model-pretrained/chemberta-77m-mlm \
  --tokenizer reports/slices_tokenizer_v1_smoke/candidate_tokenizer \
  --input data/notebooks_processing/outputs/to_SLICES_output/figures/alexandria_clean_all.csv \
  --column slices --require-valid \
  --out reports/chemberta_pad_clearance

Referencias:
 [PAD-01] https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
 [PAD-02] https://huggingface.co/docs/transformers/main_classes/tokenizer
 [PAD-03] https://huggingface.co/docs/transformers/main_classes/data_collator
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

TRUE = {"true", "1", "yes", "y", "t"}


def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def samples_from_csv(path, field, filter_valid, limit, tokenizer):
    """Elegir dos entradas reales de longitud diferente para contrastar padding.
    No toma decisiones sobre splits ni examina etiquetas energéticas.
    """
    options = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        cols = set(reader.fieldnames or [])
        missing = ({field, "slices_basic_valid", "slices_valid"} if filter_valid else {field}) - cols
        if missing:
            raise RuntimeError(f"Faltan columnas en CSV: {sorted(missing)}")
        for idx, row in enumerate(reader):
            if filter_valid and not (str(row["slices_basic_valid"]).strip().lower() in TRUE
                                     and str(row["slices_valid"]).strip().lower() in TRUE):
                continue
            text = " ".join((row.get(field) or "").split())
            if not text:
                continue
            units = text.split()
            # El tokenizer v1 es WordLevel: longitud ~= unidades+2.
            if 28 <= len(units) <= 170:
                options.append((len(units), text, row.get("mat_id", str(idx))))
            if idx >= limit or len(options) >= 500:
                break
    if len(options) < 2:
        raise RuntimeError("No hay al menos dos cadenas SLICES cortas para forward.")
    options.sort(key=lambda x: x[0])
    a = options[max(0, len(options)//5)]
    b = options[min(len(options)-1, len(options)*4//5)]
    if a[0] == b[0]:
        for o in options:
            if o[0] != a[0]:
                b = o
                break
    if a[0] == b[0]:
        raise RuntimeError("Se requieren dos longitudes diferentes para probar padding.")
    return [{"mat_id":a[2], "length_units":a[0], "text":a[1]},
            {"mat_id":b[2], "length_units":b[0], "text":b[1]}]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True, help="candidate_tokenizer generado por build_slices_tokenizer_v1.py")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--column", default="slices")
    p.add_argument("--require-valid", action="store_true")
    p.add_argument("--out", type=Path, default=Path("reports/chemberta_pad_clearance"))
    p.add_argument("--scan-rows", type=int, default=3000)
    p.add_argument("--max-diff", type=float, default=5e-4,
                   help="Máxima discrepancia absoluta entre logits válidos, float32 CPU")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {"status":"STARTED", "source_checkpoint":str(a.checkpoint),
               "source_tokenizer":str(a.tokenizer), "source_corpus":str(a.input),
               "checks": {}, "refs": ["PAD-01", "PAD-02", "PAD-03"]}
    checks = summary["checks"]
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
        for path in (a.checkpoint, a.tokenizer):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if not a.input.is_file():
            raise FileNotFoundError(a.input)
        torch.manual_seed(20261021)
        tok = AutoTokenizer.from_pretrained(str(a.tokenizer), use_fast=True, local_files_only=True)
        before_vocab = tok.get_vocab().copy()
        before_len = len(tok)
        before_specials = {
            k: getattr(tok, f"{k}_token_id", None)
            for k in ("cls", "sep", "mask", "unk", "bos", "eos", "pad")
        }
        cfg = json.loads((a.checkpoint / "config.json").read_text(encoding="utf-8"))
        model_pad = int(cfg["pad_token_id"])
        old_pad = tok.pad_token_id
        unused_id = tok.convert_tokens_to_ids("[unused1]")
        checks["source_config_expected"] = (
            model_pad == 1 and unused_id == 1 and old_pad == 0
            and tok.unk_token_id == 11 and tok.cls_token_id == 12
            and tok.sep_token_id == 13 and tok.mask_token_id == 14
            and before_len == 896
        )
        if not checks["source_config_expected"]:
            raise RuntimeError("Config fuente no coincide con la version auditada. ABORTADO para evitar cambios de IDs.")

        # El ID 1 ya existe. add_special_tokens debe cambiar el rol, no el vocabulario.
        added = tok.add_special_tokens({"pad_token": "[unused1]"})
        # Hugging Face puede devolver 1 al registrar como especial un token que
        # ya existe en el vocabulario. Ese contador no implica una nueva fila.
        checks["special_token_registration_count"] = int(added)
        checks["id_mapping_unchanged"] = (len(tok) == before_len
                                          and tok.get_vocab() == before_vocab)
        checks["pad_matches_model"] = (tok.pad_token_id == model_pad == 1)
        checks["other_specials_unchanged"] = all(
            getattr(tok, f"{k}_token_id", None) == before_specials[k]
            for k in ("cls", "sep", "mask", "unk", "bos", "eos")
        )
        if not all(checks[k] for k in (
            "id_mapping_unchanged", "pad_matches_model", "other_specials_unchanged")):
            raise RuntimeError("Falla de mapeo del tokenizer después de cambiar PAD a [unused1].")
        print("[PAD] ID del modelo y tokenizer ahora = 1; IDs originales intactos.", flush=True)

        selected = samples_from_csv(a.input, a.column, a.require_valid, a.scan_rows, tok)
        texts = [s["text"] for s in selected]
        # Debe ser exactamente el mismo tokenizado antes/después del cambio de PAD.
        from transformers import AutoTokenizer as LoadTokenizer
        old_tok = LoadTokenizer.from_pretrained(str(a.tokenizer), use_fast=True, local_files_only=True)
        checks["unpad_input_ids_equal"] = all(
            old_tok.encode(text, add_special_tokens=True) == tok.encode(text, add_special_tokens=True)
            for text in texts
        )
        checks["reserved_pad_not_in_real_tokens"] = all(
            model_pad not in tok.encode(text, add_special_tokens=True) for text in texts
        )
        checks["lexical_roundtrip_unchanged"] = all(
            " ".join(tok.decode(tok.encode(text, add_special_tokens=False),
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False).split()) == text
            for text in texts
        )
        summary["examples"] = [{"mat_id":s["mat_id"], "length_units":s["length_units"]}
                               for s in selected]
        batch = tok(texts, padding=True, truncation=False, return_tensors="pt",
                    return_special_tokens_mask=True)
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        mask_pad = attn.eq(0)
        checks["padding_present"] = bool(mask_pad.any().item())
        checks["pad_ids_correct"] = bool((ids[mask_pad] == model_pad).all().item())
        checks["nonpad_never_pad_id"] = bool((ids[attn.bool()] != model_pad).all().item())
        pos_from_ids = ids.ne(model_pad).long()
        pos_from_ids = torch.cumsum(pos_from_ids, dim=1) * pos_from_ids + model_pad
        pos_from_mask = (torch.cumsum(attn.long(), dim=1)*attn.long()) + model_pad
        checks["automatic_positions_match_attention_mask"] = bool(
            torch.equal(pos_from_ids, pos_from_mask))
        checks["position_bounds"] = int(pos_from_ids.max()) < int(cfg["max_position_embeddings"])
        summary["positions"] = {
            "model_pad_id": model_pad, "original_tokenizer_pad_id":old_pad,
            "aligned_tokenizer_pad_id":tok.pad_token_id,
            "max_position_embeddings":int(cfg["max_position_embeddings"]),
            "position_id_max":int(pos_from_ids.max()),
            "position_pad_unique":torch.unique(pos_from_ids[mask_pad]).tolist(),
            "valid_position_first_sample":pos_from_ids[0][attn[0].bool()][:8].tolist(),
        }
        print("[POS] Posiciones automáticas coinciden con attention_mask.", flush=True)

        print("[MODELO] Cargando MLM con pesos reales y redimensionando SOLO en RAM...", flush=True)
        model = AutoModelForMaskedLM.from_pretrained(str(a.checkpoint), local_files_only=True)
        orig_embed = model.get_input_embeddings().weight.detach().cpu().clone()
        old_rows = orig_embed.shape[0]
        checks["model_word_pad_idx_equals_config"] = (
            model.get_input_embeddings().padding_idx == model_pad)
        try:
            model.resize_token_embeddings(len(tok), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tok))
        checks["old_embedding_rows_preserved"] = torch.equal(
            orig_embed, model.get_input_embeddings().weight[:old_rows].detach().cpu())
        checks["new_embedding_size"] = model.get_input_embeddings().num_embeddings == len(tok)
        model.eval()
        with torch.no_grad():
            # Sin position_ids explícitos: este es el control central.
            logits_batch = model(input_ids=ids, attention_mask=attn).logits
            solo0 = tok(texts[0], truncation=False, return_tensors="pt")
            solo1 = tok(texts[1], truncation=False, return_tensors="pt")
            logits0 = model(**solo0).logits[0]
            logits1 = model(**solo1).logits[0]
        L0 = int(solo0["input_ids"].shape[1])
        L1 = int(solo1["input_ids"].shape[1])
        diff0 = float((logits_batch[0,:L0] - logits0).abs().max())
        diff1 = float((logits_batch[1,:L1] - logits1).abs().max())
        summary["forward"] = {
            "batch_shape": list(logits_batch.shape),
            "single_lengths": [L0, L1], "max_abs_diff_single_vs_batch": [diff0,diff1],
            "tolerance":a.max_diff,
            "input_embedding_rows_original":old_rows,
            "input_embedding_rows_resized":model.get_input_embeddings().num_embeddings,
            "explicit_position_ids_passed":False,
        }
        checks["forward_finite"] = bool(torch.isfinite(logits_batch).all().item())
        checks["same_logits_solo_vs_batch"] = (diff0 <= a.max_diff and diff1 <= a.max_diff)
        print(f"[FORWARD] max |solo-batch|={max(diff0,diff1):.3g}; tolerancia={a.max_diff:g}", flush=True)

        # Comprobar MLM real: el padding y los tokens especiales no deben ser supervisados.
        features = [tok(text, truncation=False, add_special_tokens=True,
                        return_special_tokens_mask=True) for text in texts]
        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tok, mlm=True, mlm_probability=0.15, return_tensors="pt")
        mlm = mlm_collator(features)
        mmask = mlm["attention_mask"].eq(0)
        masked = mlm["labels"].ne(-100)
        checks["mlm_padding_not_supervised"] = bool((mlm["labels"][mmask] == -100).all())
        checks["mlm_has_nonpad_targets"] = bool(masked.any().item())
        checks["mlm_labels_come_only_from_valid_tokens"] = bool(
            (masked & mmask).sum() == 0)
        checks["mlm_random_replacement_not_pad_in_this_batch"] = bool(
            (mlm["input_ids"][mlm["attention_mask"].bool()] != model_pad).all())
        # El collator estándar podría muestrear aleatoriamente un ID especial en otros batches.
        summary["mlm"] = {
            "masked_targets_count": int(masked.sum()),
            "padding_label_values": torch.unique(mlm["labels"][mmask]).tolist(),
            "note": "El collator de HF puede reemplazar tokens enmascarados aleatoriamente. "
                    "Antes de DAPT, auditar/sanitizar reemplazos aleatorios de IDs reservados y especiales."
        }
        with torch.no_grad():
            out_mlm = model(input_ids=mlm["input_ids"],
                            attention_mask=mlm["attention_mask"], labels=mlm["labels"])
        checks["mlm_loss_finite"] = bool(torch.isfinite(out_mlm.loss).item())
        summary["mlm"]["loss_test_value_untrained_new_embeddings"] = float(out_mlm.loss)
        print("[MLM] Forward con labels, sin actualizar pesos: OK.", flush=True)

        # Persistencia: guardar SOLO el tokenizer. Los pesos originales siguen intactos.
        critical = [
            "source_config_expected", "id_mapping_unchanged", "pad_matches_model",
            "other_specials_unchanged", "unpad_input_ids_equal", "reserved_pad_not_in_real_tokens",
            "lexical_roundtrip_unchanged", "padding_present", "pad_ids_correct",
            "nonpad_never_pad_id", "automatic_positions_match_attention_mask",
            "position_bounds", "model_word_pad_idx_equals_config",
            "old_embedding_rows_preserved", "new_embedding_size", "forward_finite",
            "same_logits_solo_vs_batch", "mlm_padding_not_supervised",
            "mlm_has_nonpad_targets", "mlm_labels_come_only_from_valid_tokens",
            "mlm_random_replacement_not_pad_in_this_batch", "mlm_loss_finite"
        ]
        failures = [k for k in critical if not checks.get(k,False)]
        if failures:
            summary["status"]="PAD_CLEARANCE_BLOCKED"
            summary["critical_failures"]=failures
        else:
            aligned_dir = a.out/"tokenizer_pad_aligned"
            aligned_dir.mkdir(parents=True, exist_ok=True)
            tok.save_pretrained(str(aligned_dir))
            tok_saved = AutoTokenizer.from_pretrained(str(aligned_dir), use_fast=True, local_files_only=True)
            checks["save_reload_pad_matches"] = tok_saved.pad_token_id == model_pad
            checks["save_reload_vocab_unchanged"] = tok_saved.get_vocab() == before_vocab
            checks["save_reload_encoding_unchanged"] = all(
                tok.encode(s["text"]) == tok_saved.encode(s["text"]) for s in selected)
            if not all(checks[k] for k in (
                "save_reload_pad_matches", "save_reload_vocab_unchanged",
                "save_reload_encoding_unchanged")):
                summary["status"]="PAD_CLEARANCE_BLOCKED_SERIALIZATION"
            else:
                summary["status"]="PAD_CLEARANCE_PASS"
                summary["aligned_tokenizer_dir"] = str(aligned_dir)
        summary["mlm_training_ready"] = False
        summary["remaining_clearances"] = [
            "Delimitar las 1,675 secuencias que exceden longitud y preservar integridad SLICES",
            "Comprobar oficialmente la gramática SLICES strategy=4",
            "Controlar duplicación/equivalencia estructural y splits",
            "Asegurar que el MLM no inserte PAD/otros tokens especiales como sustituciones aleatorias",
            "Diseñar adaptación de las 296 nuevas filas de embeddings antes de downstream",
        ]
    except Exception as ex:
        summary["status"]="PAD_CLEARANCE_ERROR"
        summary["error"]= f"{type(ex).__name__}: {ex}"
        summary["traceback"] = traceback.format_exc(limit=7)
        print(summary["traceback"], file=sys.stderr)
    finally:
        save_json(a.out/"summary.json",summary)
    print("\n=== CLEARANCE PAD ===")
    print("Estado:",summary["status"])
    print("Reportes:",(a.out/"summary.json").resolve())
    return 0 if summary["status"] == "PAD_CLEARANCE_PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
