"""Audita invariancia Ridge/L03 y restauración de un checkpoint LoRA completo."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

import experiment as ex
from lora_matgpt import insert_qv_lora

LOG = logging.getLogger("matgpt_lora_invariance")


def audit(run: Path, output: Path) -> dict[str, object]:
    """Compara antes/después sobre un triplete inner-train sin optimizar pesos."""
    torch.set_num_threads(2)
    frame = ex.development(); folds = ex.assignments(frame)
    model, stoi, block_size = ex.model_and_tokens()
    tokens = ex.encode(frame, stoi, block_size)
    first = folds[(folds.outer_fold == 0) & (folds.inner_role == "inner_train")]
    ids = ex.formula_groups(frame, set(first.structure_id))[0][1]
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["checkpoint_sha256"] != ex.sha256(ex.CHECKPOINT):
        raise ValueError("Checkpoint base distinto")
    captured: list[torch.Tensor] = []
    hook = model.transformer_decoder.layers[3].register_forward_hook(
        lambda _module, _args, value: captured.append(value.detach().clone()))
    model.eval()
    with torch.inference_mode():
        original = ex.final_end(model, ids, tokens, stoi["<pad>"])
    l03_before = captured[-1]
    base_before = {index: model.transformer_decoder.layers[index].self_attn.in_proj_weight.detach().clone()
                   for index in (10, 11)}
    adapters = insert_qv_lora(model)
    for adapter, state in zip(adapters, checkpoint["adapters"]):
        missing, unexpected = adapter.load_state_dict(state, strict=False)
        if unexpected or set(missing) != set(k for k in adapter.state_dict() if k.startswith("base.")):
            raise ValueError("Estado LoRA mal restaurado")
    model.eval()
    with torch.inference_mode():
        adapted = ex.final_end(model, ids, tokens, stoi["<pad>"])
        repeated = ex.final_end(model, ids, tokens, stoi["<pad>"])
    l03_after = captured[-1]
    hook.remove()
    l03_difference = float((l03_before - l03_after).abs().max())
    base_difference = max(float((base_before[index] - adapter.base.in_proj_weight).abs().max())
                          for index, adapter in zip((10, 11), adapters))
    drift = float((adapted - original).abs().max())
    repeat = float((adapted - repeated).abs().max())
    norm = float(np.sqrt(sum(float(torch.sum(adapter.q_b.detach() ** 2 + adapter.v_b.detach() ** 2))
                             for adapter in adapters)))
    status = "PASS" if l03_difference <= 1e-7 and base_difference == 0 and repeat <= 1e-7 and norm > 0 else "FAIL"
    result = {"status": status, "run": str(run), "structure_ids": ids,
              "l03_max_abs_difference": l03_difference,
              "base_in_proj_max_abs_difference": base_difference,
              "final_end_max_abs_drift": drift, "repeat_max_abs_difference": repeat,
              "trained_lora_b_norm": norm,
              "ridge_prediction_invariant_by_identical_l03": l03_difference == 0,
              "original_checkpoint_sha256": ex.sha256(ex.CHECKPOINT)}
    ex.atomic_json(output, result)
    LOG.info("%s", {"event": "invariance_audit", **result})
    if status != "PASS":
        raise ValueError("Invariancia de rama Ridge falló")
    return result


def main() -> None:
    """Verifica CLI y guarda auditoría JSON sin entrenar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ex.REPORT / "runs/lora_fold0_seed42")
    parser.add_argument("--output", type=Path, default=ex.REPORT / "preflight/invariance_after_training.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(audit(args.run.resolve(), args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
