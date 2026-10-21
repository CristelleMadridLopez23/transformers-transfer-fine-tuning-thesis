"""Run gated full-encoder BERT regression in a Slurm GPU allocation.

Match the paper's CLS regression, 2e-4 learning rate and early-stopping protocol;
preserve original composition splits. Optional fivefold CV uses training groups
only. Write new artifacts, never modify the pretrained MatText checkpoint.
"""
from __future__ import annotations

import csv
import os
import time
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from finetuning import FrameDataset, collate, load_rows, model, pad_id, partition
from workflow import LOGGER, cli, fingerprint, load_config, require_audit, sha256, write_json


def validation_rmse(network: Any, loader: DataLoader, device: str) -> float:
    """Evaluate network on validation-only loader; return finite RMSE in original eV/atom."""
    network.eval()
    errors = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = network(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.squeeze(-1)
            errors.extend((prediction - batch["labels"]).double().square().cpu().tolist())
    result = float(np.sqrt(np.mean(errors)))
    if not np.isfinite(result):
        raise RuntimeError("Nonfinite validation RMSE")
    return result


def fit(config: dict[str, Any], fold: int = -1) -> dict[str, Any]:
    """Train one fixed/cross-validation fold after critical audit; return artifact manifest dict."""
    require_audit(config)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Full fine-tuning requires an allocated Slurm GPU, never a login node")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    settings = config["training"]
    rows = load_rows(config)
    training, validation = partition(rows, fold, config["seed"], settings["paper_cv_folds"])
    name = "final" if fold == -1 else f"fold_{fold}"
    output = config["paths"]["experiment"] / "artifacts" / name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing training run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    collator = partial(collate, pad_id=pad_id(config), maximum=config["tokenizer"]["max_length"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(FrameDataset(training), batch_size=settings["batch_size"],
                              shuffle=True, generator=generator, collate_fn=collator)
    val_loader = DataLoader(FrameDataset(validation), batch_size=settings["batch_size"],
                            shuffle=False, collate_fn=collator)
    network = model(config).to("cuda")
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    steps = max(1, len(train_loader) * settings["epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1. - step / steps))
    best, bad_epochs, best_epoch = float("inf"), 0, 0
    started = time.monotonic()
    before = fingerprint(config)
    history = []
    for epoch in range(settings["epochs"]):
        network.train()
        total_loss, count = 0., 0
        for batch in train_loader:
            batch = {key: value.to("cuda") for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = network(**batch)
            if not torch.isfinite(result.loss):
                raise RuntimeError("Nonfinite training loss")
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(result.loss.detach()) * len(batch["labels"])
            count += len(batch["labels"])
        score = validation_rmse(network, val_loader, "cuda")
        history.append({"epoch": epoch + 1, "train_mse_ev2_atom2": total_loss / count,
                        "validation_rmse_ev_atom": score, "learning_rate": scheduler.get_last_lr()[0]})
        bad_epochs = 0 if score < best - settings["early_stopping_threshold_ev_atom"] else bad_epochs + 1
        if score < best:
            best, best_epoch = score, epoch + 1
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in network.state_dict().items()},
                        "epoch": best_epoch, "validation_rmse_ev_atom": best,
                        "fingerprint": before, "fold": fold}, output / "best.pt")
        with (output / "training_curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        LOGGER.info("finetune_epoch fold=%d epoch=%d val_rmse_ev_atom=%.8f", fold, epoch + 1, score)
        if bad_epochs >= settings["early_stopping_patience"]:
            break
    if fingerprint(config) != before:
        raise RuntimeError("Audited inputs or original weights changed during training")
    manifest = {"status": "completed", "fingerprint": before, "fold": fold,
                "train_rows": len(training), "validation_rows": len(validation),
                "train_compositions": sorted({r["reduced_formula"] for r in training}),
                "validation_compositions": sorted({r["reduced_formula"] for r in validation}),
                "test_used_for_selection": False, "best_epoch": best_epoch,
                "best_validation_rmse_ev_atom": best, "artifact_sha256": sha256(output / "best.pt"),
                "total_parameters": sum(p.numel() for p in network.parameters()),
                "trainable_encoder_parameters": sum(p.numel() for p in network.bert.parameters() if p.requires_grad),
                "seconds": time.monotonic() - started, "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "settings": settings}
    write_json(output / "training_manifest.json", manifest)
    return manifest


def main() -> None:
    """Train fixed split (-1) or an optional train-only CV fold (0..4); log failures descriptively."""
    parser = cli(__doc__)
    parser.add_argument("--fold", type=int, default=-1)
    args = parser.parse_args()
    try:
        fit(load_config(args.config), args.fold)
    except Exception:
        LOGGER.exception("finetuning_failed")
        raise


if __name__ == "__main__":
    main()
