#!/usr/bin/env python3
"""Fine-tune MatterGPT F4 on the explicitly tokenizer-compatible Alexandria subset.

The MatterGPT backbone is frozen for a 20-epoch MLP8 warmup, then only
Transformer blocks 8–11 and the energy head are updated. The property vector
remains fixed at [1.0, 2.0] for every structure; energy labels are never passed
to the generative backbone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
MODEL_DIR = ROOT / "models/MatterGPT"
DEFAULT_EXPERIMENT = ROOT / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset"
CONDITION = (1.0, 2.0)
EXPECTED_CHECKPOINT_SHA = "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"
LOG = logging.getLogger("mattergpt_alexandria_f4")


class EnergyHead(nn.Module):
    """MLP8 residual energy head with dimensions 768→8→1."""

    def __init__(self) -> None:
        """Initialize GELU and dropout layers."""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, embedding: Tensor) -> Tensor:
        """Predict normalized energy per atom for a batch of embeddings."""
        return self.network(embedding).squeeze(-1)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest.

    Args:
        path: Existing file to hash.
    Returns:
        Hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so an interruption cannot leave a partial file.

    Args:
        path: Destination JSON path.
        payload: Serializable content.
    Returns:
        None.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def peak_rss_bytes() -> int:
    """Return this process's peak resident memory in bytes.

    Args:
        None.
    Returns:
        Peak resident set size.
    """
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_model() -> tuple[nn.Module, dict[str, int], int]:
    """Load MatterGPT strictly and verify checkpoint, vocabulary, and dimensions.

    Args:
        None.
    Returns:
        Model, token-to-index mapping, and maximum input length.
    """
    checkpoint = MODEL_DIR / "pytorch_model.pt"
    config_path = MODEL_DIR / "config.json"
    vocab_path = MODEL_DIR / "Voc_prior"
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("El checkpoint MatterGPT no coincide con el SHA-256 auditado")
    sys.path.insert(0, str(MODEL_DIR))
    from model import GPT, GPTConfig

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = GPTConfig(config["vocab_size"], config["block_size"], **{
        key: value for key, value in config.items()
        if key not in {"model_type", "architectures", "vocab_size", "block_size"}
    })
    model = GPT(model_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint MatterGPT incompleto: {incompatible}")
    vocab = sorted(set(vocab_path.read_text(encoding="utf-8").splitlines() + ["<", ">"]))
    stoi = {token: index for index, token in enumerate(vocab)}
    if (len(vocab), config["block_size"], config["n_layer"], config["n_embd"], config["num_props"]) != (132, 397, 12, 768, 2):
        raise ValueError("La arquitectura MatterGPT no coincide con la configuración auditada")
    if len(vocab) != config["vocab_size"]:
        raise ValueError("El tamaño del tokenizer MatterGPT no coincide con el checkpoint")
    model.eval()
    return model, stoi, int(config["block_size"])


def read_sequences(frame: pd.DataFrame, stoi: dict[str, int], block_size: int) -> list[list[int]]:
    """Encode unmodified SLICES and reject any OOV or over-context sequence.

    Args:
        frame: One split's compatible Alexandria rows.
        stoi: MatterGPT token-to-index mapping.
        block_size: Checkpoint context length.
    Returns:
        Encoded sequences including MatterGPT start and end markers.
    """
    sequences: list[list[int]] = []
    for row in frame.itertuples(index=False):
        tokens = str(row.slices_matgpt).split()
        missing = sorted(set(tokens) - set(stoi))
        ids = [stoi[">"]] + [stoi[token] for token in tokens if token in stoi] + [stoi["<"]]
        if missing:
            raise ValueError(f"OOV en {row.mat_id}: {','.join(missing)}")
        if len(ids) > block_size:
            raise ValueError(f"Secuencia excede block_size={block_size}: {row.mat_id} ({len(ids)})")
        if int(row.matter_sequence_length) != len(ids):
            raise ValueError(f"Longitud de secuencia cambió para {row.mat_id}")
        sequences.append(ids)
    return sequences


def make_batch(sequences: list[list[int]], indices: np.ndarray) -> tuple[Tensor, Tensor]:
    """Right-pad a batch and return IDs plus final structural-token positions.

    Args:
        sequences: Encoded SLICES rows.
        indices: Row indices in this minibatch.
    Returns:
        Padded token IDs and per-row terminal-token positions.
    """
    selected = [sequences[int(index)] for index in indices]
    width = max(map(len, selected))
    ids = torch.zeros((len(selected), width), dtype=torch.long)
    for row, sequence in enumerate(selected):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return ids, torch.tensor([len(sequence) - 1 for sequence in selected], dtype=torch.long)


def contextualized_final_end(model: nn.Module, ids: Tensor, positions: Tensor) -> Tensor:
    """Return FINAL_END after property prefix, Transformer blocks, and ln_f.

    Args:
        model: MatterGPT backbone.
        ids: Padded batch of structural token IDs.
        positions: Position of the terminal structural token in each row.
    Returns:
        Contextualized vectors with shape [batch, 768].
    """
    batch, length = ids.shape
    device = ids.device
    token = model.tok_emb(ids)
    positional = model.pos_emb[:, :length, :]
    sequence_type = model.type_emb(torch.ones((batch, length), dtype=torch.long, device=device))
    hidden = model.drop(token + positional + sequence_type)
    property_type = model.type_emb(torch.zeros((batch, 1), dtype=torch.long, device=device))
    condition = torch.tensor(CONDITION, dtype=torch.float32, device=device).view(1, 1, 2).expand(batch, 1, 2)
    property_hidden = model.prop_nn(condition) + property_type
    hidden = torch.cat([property_hidden, hidden], dim=1)
    for block in model.blocks:
        hidden, _ = block(hidden)
    hidden = model.ln_f(hidden)
    rows = torch.arange(batch, device=device)
    return hidden[rows, positions.to(device) + 1]


def extract_embeddings(model: nn.Module, sequences: list[list[int]], batch_size: int,
                       device: torch.device) -> np.ndarray:
    """Extract frozen 768-dimensional FINAL_END vectors without gradients.

    Args:
        model: Frozen MatterGPT backbone.
        sequences: Encoded train or validation rows.
        batch_size: Extraction minibatch size.
        device: CPU, CUDA, or Metal device.
    Returns:
        Float32 embedding matrix of shape [rows, 768].
    """
    model.eval()
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            indices = np.arange(start, min(start + batch_size, len(sequences)))
            ids, positions = make_batch(sequences, indices)
            result.extend(contextualized_final_end(model, ids.to(device), positions.to(device))
                          .cpu().numpy().astype(np.float32))
    return np.stack(result)


def evaluate(model: nn.Module, head: EnergyHead, sequences: list[list[int]], targets: np.ndarray,
             batch_size: int, device: torch.device) -> tuple[float, np.ndarray]:
    """Compute full-validation normalized SmoothL1 and energy predictions.

    Args:
        model: MatterGPT backbone.
        head: Energy regression head.
        sequences: Validation sequences.
        targets: Train-normalized validation energies.
        batch_size: Evaluation minibatch size.
        device: CPU, CUDA, or Metal device.
    Returns:
        Mean SmoothL1 loss and normalized predictions.
    """
    model.eval(); head.eval()
    predictions: list[np.ndarray] = []
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            indices = np.arange(start, min(start + batch_size, len(sequences)))
            ids, positions = make_batch(sequences, indices)
            pred = head(contextualized_final_end(model, ids.to(device), positions.to(device)))
            target = torch.as_tensor(targets[indices], dtype=torch.float32, device=device)
            losses.extend(F.smooth_l1_loss(pred, target, beta=1.0, reduction="none").cpu().numpy().tolist())
            predictions.extend(pred.cpu().numpy())
    return float(np.mean(losses)), np.asarray(predictions, dtype=np.float64)


def train(args: argparse.Namespace) -> None:
    """Run the two-stage F4 protocol on the prepared Alexandria-compatible subset.

    Args:
        args: Validated CLI configuration.
    Returns:
        None.
    """
    experiment = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    train_path = data_dir / "alexandria_matter_compatible_train.csv"
    valid_path = data_dir / "alexandria_matter_compatible_validation.csv"
    for path in (train_path, valid_path, data_dir / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Falta manifiesto requerido: {path}")
    train_frame, val_frame = pd.read_csv(train_path), pd.read_csv(valid_path)
    if (len(train_frame), len(val_frame)) != (17245, 1718):
        raise ValueError("Los tamaños de cohorte no coinciden con el subset MatterGPT auditado")
    if set(train_frame.reduced_formula) & set(val_frame.reduced_formula):
        raise ValueError("Fuga de fórmulas entre train y validation")
    model, stoi, block_size = load_model()
    train_sequences = read_sequences(train_frame, stoi, block_size)
    val_sequences = read_sequences(val_frame, stoi, block_size)
    target_train = train_frame.energy_per_atom_ev.to_numpy(dtype=np.float64)
    target_val = val_frame.energy_per_atom_ev.to_numpy(dtype=np.float64)
    target_mean, target_std = float(target_train.mean()), float(target_train.std(ddof=0))
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("Desviación del target de entrenamiento inválida")
    y_train = ((target_train - target_mean) / target_std).astype(np.float32)
    y_val = ((target_val - target_mean) / target_std).astype(np.float32)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS solicitado pero no disponible")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado pero no disponible")
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)
    if device.type == "mps": torch.mps.manual_seed(args.seed)
    model.to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    head = EnergyHead().to(device)
    started = time.monotonic(); started_cpu = time.process_time()
    experiment.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    history_path = experiment / "historial_entrenamiento.csv"
    atomic_json(experiment / "resources.json", {"status": "RUNNING", "device": str(device),
        "threads": args.threads, "batch_size": args.batch_size, "train_rows": len(train_frame),
        "validation_rows": len(val_frame), "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "started_unix": time.time(), "peak_rss_bytes": peak_rss_bytes()})
    try:
        train_embeddings = extract_embeddings(model, train_sequences, args.batch_size, device)
        val_embeddings = extract_embeddings(model, val_sequences, args.batch_size, device)
        rng = np.random.default_rng(args.seed)
        warm_optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(1, 21):
            epoch_start = time.monotonic(); head.train(); losses: list[float] = []
            order = rng.permutation(len(train_sequences))
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                x = torch.from_numpy(train_embeddings[indices]).to(device)
                y = torch.from_numpy(y_train[indices]).to(device)
                warm_optimizer.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(head(x), y, beta=1.0)
                if not torch.isfinite(loss): raise FloatingPointError("Pérdida no finita en warmup")
                loss.backward(); warm_optimizer.step(); losses.append(float(loss.detach().item()))
            head.eval()
            with torch.inference_mode():
                val_pred = head(torch.from_numpy(val_embeddings).to(device)).cpu().numpy()
            val_loss = float(F.smooth_l1_loss(torch.from_numpy(val_pred.astype(np.float32)),
                                               torch.from_numpy(y_val), beta=1.0).item())
            history.append({"stage":"warmup_frozen","epoch":epoch,"train_smoothl1":float(np.mean(losses)),
                            "validation_smoothl1":val_loss,"epoch_wall_seconds":time.monotonic()-epoch_start})
            pd.DataFrame(history).to_csv(history_path,index=False)
        del train_embeddings, val_embeddings
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for block in model.blocks[8:12]:
            for parameter in block.parameters(): parameter.requires_grad_(True)
        trainable = [parameter for block in model.blocks[8:12] for parameter in block.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW([{"params":trainable,"lr":3e-6,"weight_decay":0.01},
                                       {"params":list(head.parameters()),"lr":1e-4,"weight_decay":1e-4}])
        best_loss, best_epoch, stale = float("inf"), 0, 0
        checkpoint_path = experiment / "checkpoints/mattergpt_alexandria_f4_energy_best.pt"
        checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
        for epoch in range(1,args.max_epochs+1):
            epoch_start=time.monotonic();model.train();head.train();order=rng.permutation(len(train_sequences))
            total=0.0;seen=0
            for start in range(0,len(order),args.batch_size):
                indices=order[start:start+args.batch_size]
                ids,positions=make_batch(train_sequences,indices);ids,positions=ids.to(device),positions.to(device)
                target=torch.as_tensor(y_train[indices],dtype=torch.float32,device=device)
                optimizer.zero_grad(set_to_none=True)
                pred=head(contextualized_final_end(model,ids,positions))
                loss=F.smooth_l1_loss(pred,target,beta=1.0)
                if not torch.isfinite(loss):raise FloatingPointError(f"Pérdida no finita en F4 época {epoch}")
                loss.backward()
                gradients=[parameter.grad for parameter in trainable+list(head.parameters()) if parameter.grad is not None]
                finite_flags=torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
                if not bool(finite_flags.all()):
                    raise FloatingPointError(f"Gradiente no finito en F4 época {epoch}, lote {start//args.batch_size}")
                optimizer.step();total+=float(loss.detach().item())*len(indices);seen+=len(indices)
            val_loss,_=evaluate(model,head,val_sequences,y_val,args.validation_batch_size,device)
            history.append({"stage":"finetune_f4","epoch":epoch,"train_smoothl1":total/seen,
                            "validation_smoothl1":val_loss,"epoch_wall_seconds":time.monotonic()-epoch_start})
            pd.DataFrame(history).to_csv(history_path,index=False)
            if val_loss<best_loss:
                best_loss,best_epoch,stale=val_loss,epoch,0
                state={"base_checkpoint_sha256":EXPECTED_CHECKPOINT_SHA,"condition_vector":CONDITION,
                    "selected_epoch":epoch,"validation_smoothl1":val_loss,"target_mean_ev_atom":target_mean,
                    "target_std_ev_atom":target_std,"head_state_dict":{k:v.detach().cpu() for k,v in head.state_dict().items()},
                    "blocks_8_11":{name:value.detach().cpu() for name,value in model.state_dict().items()
                                   if any(name.startswith(f"blocks.{i}.") for i in range(8,12))}}
                temporary=checkpoint_path.with_suffix(".pt.tmp")
                torch.save(state,temporary);os.replace(temporary,checkpoint_path)
            else: stale+=1
            atomic_json(experiment/"resources.json",{"status":"RUNNING","device":str(device),"threads":args.threads,
                "batch_size":args.batch_size,"completed_f4_epochs":epoch,"best_epoch":best_epoch,
                "best_validation_smoothl1":best_loss,"total_wall_seconds":time.monotonic()-started,
                "total_cpu_seconds":time.process_time()-started_cpu,"peak_rss_bytes":peak_rss_bytes(),
                "mps_driver_allocated_bytes":int(torch.mps.driver_allocated_memory()) if device.type=="mps" else 0})
            LOG.info(json.dumps({"event":"epoch_completed","stage":"finetune_f4","epoch":epoch,
                "train_smoothl1":total/seen,"validation_smoothl1":val_loss,"best_epoch":best_epoch,
                "stale_epochs":stale},sort_keys=True))
            if stale>=args.patience: break
    except KeyboardInterrupt:
        atomic_json(experiment/"resources.json",{"status":"PARTIAL_INTERRUPTED","device":str(device),
            "total_wall_seconds":time.monotonic()-started,"peak_rss_bytes":peak_rss_bytes()})
        raise
    if best_epoch==0 or not checkpoint_path.is_file():
        raise RuntimeError("No se guardó un checkpoint F4 seleccionado por validación")
    best=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    model.load_state_dict(best["blocks_8_11"],strict=False);head.load_state_dict(best["head_state_dict"],strict=True)
    _,prediction_norm=evaluate(model,head,val_sequences,y_val,args.validation_batch_size,device)
    prediction=prediction_norm*target_std+target_mean
    error=prediction-target_val
    metrics={"validation_rows":len(val_frame),"unit":"eV/atom","mae":float(np.mean(np.abs(error))),
             "rmse_global":float(np.sqrt(np.mean(error**2))),"bias":float(np.mean(error)),
             "selected_epoch":best_epoch,"best_validation_smoothl1_normalized":best_loss,
             "target_mean_ev_atom_train":target_mean,"target_std_ev_atom_train":target_std}
    atomic_json(experiment/"metrics.json",metrics)
    atomic_json(experiment/"summary.json",{"status":"COMPLETED","experiment":experiment.name,
        "training_executed":True,"backbone":"MatterGPT","backbone_frozen_except_blocks_8_11":True,
        "condition_vector":CONDITION,"train_rows":len(train_frame),"validation_rows":len(val_frame),
        "metrics":metrics,"checkpoint":str(checkpoint_path.relative_to(ROOT)),"checkpoint_sha256":sha256(checkpoint_path)})
    atomic_json(experiment/"resources.json",{"status":"COMPLETED","device":str(device),"threads":args.threads,
        "best_epoch":best_epoch,"total_wall_seconds":time.monotonic()-started,
        "total_cpu_seconds":time.process_time()-started_cpu,"peak_rss_bytes":peak_rss_bytes(),
        "mps_driver_allocated_bytes":int(torch.mps.driver_allocated_memory()) if device.type=="mps" else 0})


def main() -> None:
    """Parse and validate local F4 parameters, then run the requested training."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_EXPERIMENT)
    parser.add_argument("--device",choices=("auto","cpu","cuda","mps"),default="auto")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--validation-batch-size",type=int,default=8)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--max-epochs",type=int,default=100)
    parser.add_argument("--patience",type=int,default=15)
    args=parser.parse_args()
    if min(args.batch_size,args.validation_batch_size,args.threads,args.max_epochs,args.patience)<=0:
        parser.error("Batch, hilos, épocas y paciencia deben ser positivos")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(args.output_dir/"training.log",encoding="utf-8")])
    train(args)


if __name__ == "__main__":
    main()
