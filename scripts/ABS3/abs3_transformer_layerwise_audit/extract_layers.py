"""Extrae estados terminales por capa de MatGPT y MatterGPT para ABS₃.

Los backbones originales se cargan estrictamente, se mantienen en evaluación y
se procesan sin gradientes. FINAL se contrasta numéricamente con las cachés
históricas antes de aceptar las nuevas matrices.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
OUT = ROOT / "experiments/abs3_transformer_layerwise_audit"
DATA = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
EMBED = ROOT / "experiments/abs3_matgpt_mattergpt_original_f4_71/embeddings"
MAT_ROOT = ROOT / "models/MatGPT-source/mat-concat"
MAT_CKPT = MAT_ROOT / "mat-uncond/weights/transformer_property_0.pth"
MAT_INFO = MAT_CKPT.with_name("model_info_property_0.json")
MAT_CONFIG = MAT_ROOT / "config/uncond.yaml"
MATTER_ROOT = ROOT / "models/MatterGPT"
EXPECTED = {
    "MAT": "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414",
    "MATTER": "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21",
}
PROPERTY_VECTOR = (1.0, 2.0)
LOG = logging.getLogger("abs3.layerwise.extract")


def sha256(path: Path) -> str:
    """Calcula SHA-256 en bloques.

    Args:
        path: Archivo que se va a resumir.
    Returns:
        Hash hexadecimal del contenido.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    """Escribe un JSON mediante reemplazo atómico.

    Args:
        path: Ruta final.
        data: Objeto serializable.
    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_data() -> pd.DataFrame:
    """Lee y valida las 210 SLICES de desarrollo.

    Args:
        None.
    Returns:
        Tabla ordenada por composición y familia.
    """
    data = pd.read_csv(DATA, dtype={"structure_id": str, "composition_id": str}, low_memory=False)
    if len(data) != 210 or data.structure_id.nunique() != 210 or data.composition_id.nunique() != 70:
        raise ValueError("Dataset ABS₃ inválido: se requieren 210 estructuras y 70 composiciones")
    if data.groupby("composition_id").size().ne(3).any() or data.structure_id.duplicated().any():
        raise ValueError("Hay tripletes incompletos o IDs estructurales duplicados")
    if "SLICES" not in data or data.SLICES.isna().any():
        raise ValueError("Faltan cadenas SLICES válidas")
    return data.sort_values(["composition_id", "family"], kind="stable").reset_index(drop=True)


def load_matgpt() -> tuple[torch.nn.Module, dict[str, int], int, dict[str, Any]]:
    """Carga MatGPT original y verifica su estado estricto.

    Args:
        None.
    Returns:
        Modelo, vocabulario, tamaño de contexto y manifiesto de arquitectura.
    """
    if sha256(MAT_CKPT) != EXPECTED["MAT"]:
        raise ValueError("El SHA-256 del checkpoint MatGPT no coincide")
    sys.path.insert(0, str(MAT_ROOT))
    from utils.models import ModifiedTransformerDecoder

    info = json.loads(MAT_INFO.read_text(encoding="utf-8"))
    config = yaml.safe_load(MAT_CONFIG.read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in info["stoi"].items()}
    model_cfg = config["model"]
    if int(model_cfg["n_layer"]) != 12 or int(model_cfg["n_embd"]) != 768:
        raise ValueError("La arquitectura MatGPT no es la esperada de 12×768")
    model = ModifiedTransformerDecoder(vocab_size=len(stoi), n_embd=768, n_head=int(model_cfg["n_head"]),
        n_layer=12, block_size=int(info["block_size"]), dropout=float(model_cfg["dropout"]),
        stoi=stoi, num_properties=0, conditioning_mode="unconditional")
    result = model.load_state_dict(torch.load(MAT_CKPT, map_location="cpu", weights_only=True), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("Carga estricta MatGPT incompleta")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    arch = {"model_name": "MatGPT", "checkpoint_sha256": EXPECTED["MAT"], "number_of_layers": 12,
        "hidden_dimension": 768, "block_names": [f"transformer_decoder.layers.{i}" for i in range(12)],
        "normalization_modules": ["ln_f"], "terminal_token_definition": "último token SLICES <end>, posición len(ids)-1",
        "conditioning_strategy": "unconditional", "extraction_points": ["L0_INPUT", *[f"L{i}_BLOCK_{i-1}" for i in range(1,13)], "L12_PRE_LN_F", "FINAL_POST_LN_F"],
        "strict_load_missing_keys": result.missing_keys, "strict_load_unexpected_keys": result.unexpected_keys,
        "decoder_norm": str(model.transformer_decoder.norm)}
    return model, stoi, int(info["block_size"]), arch


def load_mattergpt() -> tuple[torch.nn.Module, dict[str, int], int, dict[str, Any]]:
    """Carga MatterGPT original con tokenizador y conditioning históricos.

    Args:
        None.
    Returns:
        Modelo, vocabulario, tamaño de contexto y manifiesto de arquitectura.
    """
    checkpoint = MATTER_ROOT / "pytorch_model.pt"
    if sha256(checkpoint) != EXPECTED["MATTER"]:
        raise ValueError("El SHA-256 del checkpoint MatterGPT no coincide")
    sys.path.insert(0, str(MATTER_ROOT))
    from model import GPT, GPTConfig

    config = json.loads((MATTER_ROOT / "config.json").read_text(encoding="utf-8"))
    kwargs = {k: v for k, v in config.items() if k not in {"model_type", "architectures", "vocab_size", "block_size"}}
    model = GPT(GPTConfig(config["vocab_size"], config["block_size"], **kwargs))
    result = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("Carga estricta MatterGPT incompleta")
    vocabulary = sorted(set((MATTER_ROOT / "Voc_prior").read_text(encoding="utf-8").splitlines()) | {"<", ">"})
    stoi = {token: i for i, token in enumerate(vocabulary)}
    if len(stoi) != int(config["vocab_size"]):
        raise ValueError("El vocabulario MatterGPT no coincide con config.json")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    arch = {"model_name": "MatterGPT", "checkpoint_sha256": EXPECTED["MATTER"],
        "number_of_layers": len(model.blocks), "hidden_dimension": int(config["n_embd"]),
        "block_names": [f"blocks.{i}" for i in range(len(model.blocks))], "normalization_modules": ["ln_f"],
        "terminal_token_definition": "último token SLICES <, índice len(ids)-1 más prefijo conditioning",
        "conditioning_strategy": "PROPERTY_VECTOR [1.0, 2.0] prepend mediante prop_nn + type_emb",
        "extraction_points": ["L0_INPUT", *[f"L{i}_BLOCK_{i-1}" for i in range(1,len(model.blocks)+1)], "L12_PRE_LN_F", "FINAL_POST_LN_F"],
        "strict_load_missing_keys": result.missing_keys, "strict_load_unexpected_keys": result.unexpected_keys}
    if len(model.blocks) != 12 or int(config["n_embd"]) != 768:
        raise ValueError("La arquitectura MatterGPT no es la esperada de 12×768")
    return model, stoi, int(config["block_size"]), arch


def encode(data: pd.DataFrame, stoi: dict[str, int], block_size: int, matter: bool) -> list[list[int]]:
    """Tokeniza SLICES y falla ante OOV o truncamiento.

    Args:
        data: Dataset en orden canónico.
        stoi: Tokenizador del checkpoint.
        block_size: Longitud máxima.
        matter: Si añade la convención de prefijo/sufijo MatterGPT.
    Returns:
        IDs de tokens por estructura.
    """
    sequences: list[list[int]] = []
    for row in data.itertuples():
        tokens = str(row.SLICES).split()
        if matter:
            tokens = [">", *tokens, "<"]
        else:
            tokens = ["<start>", *tokens, "<end>"]
        missing = sorted(set(tokens).difference(stoi))
        if missing:
            raise ValueError(f"OOV en {row.structure_id}: {missing[:5]}")
        ids = [stoi[token] for token in tokens]
        if len(ids) > block_size:
            raise ValueError(f"Secuencia {row.structure_id} excede contexto: {len(ids)}>{block_size}")
        sequences.append(ids)
    return sequences


def padded(sequences: list[list[int]], rows: list[int], pad_id: int) -> tuple[Tensor, Tensor]:
    """Construye lote con padding derecho y posiciones terminales.

    Args:
        sequences: Secuencias tokenizadas.
        rows: Índices de estructuras del lote.
        pad_id: ID de padding.
    Returns:
        Tensor de IDs y posiciones terminales locales.
    """
    width = max(len(sequences[i]) for i in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for j, i in enumerate(rows):
        ids[j, :len(sequences[i])] = torch.tensor(sequences[i], dtype=torch.long)
    positions = torch.tensor([len(sequences[i])-1 for i in rows], dtype=torch.long)
    return ids, positions


def extract_matgpt(model: torch.nn.Module, sequences: list[list[int]], pad_id: int,
                    batch_size: int) -> dict[str, np.ndarray]:
    """Captura L0, salidas de bloques y FINAL de MatGPT.

    Args:
        model: Transformer original congelado.
        sequences: IDs MatGPT.
        pad_id: Token de padding.
        batch_size: Tamaño de lote de extracción.
    Returns:
        Matriz float32 por punto de extracción.
    """
    layers = model.transformer_decoder.layers
    if len(layers) != 12 or model.transformer_decoder.norm is not None:
        raise ValueError("Estructura TransformerDecoder MatGPT inesperada")
    names = ["L0_INPUT", *[f"L{i}_BLOCK_{i-1}" for i in range(1,13)], "L12_PRE_LN_F", "FINAL_POST_LN_F"]
    result = {name: np.empty((len(sequences), 768), np.float32) for name in names}
    captured: dict[int, list[Tensor]] = {i: [] for i in range(12)}
    hooks = [layer.register_forward_hook(lambda _m, _i, out, k=i: captured[k].append(out)) for i,layer in enumerate(layers)]
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                rows = list(range(start, min(start+batch_size,len(sequences))))
                ids, pos = padded(sequences, rows, pad_id); batch,length=ids.shape
                x=(model.token_embedding_table(ids)+model.position_embedding_table(torch.arange(length))).permute(1,0,2)
                result["L0_INPUT"][rows]=x.permute(1,0,2)[torch.arange(batch),pos].cpu().numpy()
                memory=torch.zeros((length,batch,model.n_embd));mask=model.generate_square_subsequent_mask(length)
                model.transformer_decoder(x,memory,tgt_mask=mask)
                if any(len(captured[i]) != 1 for i in range(12)):
                    raise RuntimeError("No se capturó una salida única por bloque MatGPT")
                pre_final=None
                for i in range(12):
                    h=captured[i].pop().permute(1,0,2)
                    result[f"L{i+1}_BLOCK_{i}"][rows]=h[torch.arange(batch),pos].cpu().numpy()
                    if i==11:
                        pre_final=h
                if pre_final is None:
                    raise RuntimeError("No se recuperó la última capa MatGPT")
                result["L12_PRE_LN_F"][rows]=pre_final[torch.arange(batch),pos].cpu().numpy()
                final=model.ln_f(pre_final)
                result["FINAL_POST_LN_F"][rows]=final[torch.arange(batch),pos].cpu().numpy()
    finally:
        for hook in hooks: hook.remove()
    return result


def extract_mattergpt(model: torch.nn.Module, sequences: list[list[int]], batch_size: int) -> dict[str, np.ndarray]:
    """Captura input, bloques, pre-ln_f y FINAL de MatterGPT.

    Args:
        model: Transformer original congelado.
        sequences: IDs con prefix y terminal MatterGPT.
        batch_size: Tamaño máximo del lote.
    Returns:
        Matriz float32 por punto de extracción.
    """
    names=["L0_INPUT", *[f"L{i}_BLOCK_{i-1}" for i in range(1,13)], "L12_PRE_LN_F", "FINAL_POST_LN_F"]
    result={name:np.empty((len(sequences),768),np.float32) for name in names}
    captures: dict[str,list[Tensor]]={}
    handles=[]
    for i,block in enumerate(model.blocks):
        name=f"L{i+1}_BLOCK_{i}"
        captures[name]=[]
        handles.append(block.register_forward_hook(lambda _m,_i,out,key=name: captures[key].append(out[0])))
    captures["LN_F"]=[]
    handles.append(model.ln_f.register_forward_hook(lambda _m,_i,out: captures["LN_F"].append(out)))
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0,len(sequences),batch_size):
                rows=list(range(start,min(start+batch_size,len(sequences))))
                width=max(len(sequences[i]) for i in rows);ids=torch.zeros((len(rows),width),dtype=torch.long)
                for j,i in enumerate(rows):ids[j,:len(sequences[i])]=torch.tensor(sequences[i])
                prop=torch.tensor(PROPERTY_VECTOR,dtype=torch.float32).view(1,1,2).expand(len(rows),1,2)
                _,_,_,embed=model(ids,prop=prop)
                terminal=torch.tensor([len(sequences[i])-1 for i in rows],dtype=torch.long)+1
                batch_index=torch.arange(len(rows))
                result["L0_INPUT"][rows]=embed[batch_index,terminal-1].cpu().numpy()
                for i in range(12):
                    key=f"L{i+1}_BLOCK_{i}"
                    if len(captures[key])!=1:raise RuntimeError(f"Captura inválida de {key}")
                    result[key][rows]=captures[key].pop()[batch_index,terminal].cpu().numpy()
                if len(captures["LN_F"])!=1:raise RuntimeError("Captura ln_f MatterGPT inválida")
                final=captures["LN_F"].pop()
                result["L12_PRE_LN_F"][rows]=result["L12_BLOCK_11"][rows]
                result["FINAL_POST_LN_F"][rows]=final[batch_index,terminal].cpu().numpy()
    finally:
        for handle in handles:handle.remove()
    return result


def main() -> None:
    """Extrae cachés, compara FINAL histórico y escribe manifiestos."""
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size",type=int,default=4)
    args=parser.parse_args()
    if args.batch_size<1:parser.error("--batch-size debe ser positivo")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(1);started=time.perf_counter();data=load_data();OUT.mkdir(parents=True,exist_ok=True)
    models={};tokens={};architectures={};
    model,stoi,ctx,arch=load_matgpt();models["MAT"]=model;tokens["MAT"]=encode(data,stoi,ctx,False);architectures["MAT"]=arch
    mat=extract_matgpt(model,tokens["MAT"],stoi["<pad>"],args.batch_size);del model
    model,stoi,ctx,arch=load_mattergpt();models["MATTER"]=model;tokens["MATTER"]=encode(data,stoi,ctx,True);architectures["MATTER"]=arch
    matter=extract_mattergpt(model,tokens["MATTER"],args.batch_size);del model
    cache_manifest={};alignment={};layer_manifest={}
    for ext,states,filename in [("MAT",mat,"embeddings_matgpt_layers.npz"),("MATTER",matter,"embeddings_mattergpt_layers.npz")]:
        historical_name="matgpt_original_final_end.npz" if ext=="MAT" else "mattergpt_original_final_end.npz"
        with np.load(EMBED/historical_name,allow_pickle=False) as z:
            old=z["embeddings"].astype(np.float64);oldids=z["structure_id"].astype(str)
        ids=data.structure_id.to_numpy(str)
        if not np.array_equal(oldids,ids):
            mapping={sid:i for i,sid in enumerate(oldids)};old=old[[mapping[s] for s in ids]]
        new=states["FINAL_POST_LN_F"].astype(np.float64)
        diff=np.abs(new-old);den=np.maximum(np.abs(old),1e-8)
        cosine=np.sum(new*old,axis=1)/(np.linalg.norm(new,axis=1)*np.linalg.norm(old,axis=1)+1e-12)
        audit={"shape_match":new.shape==old.shape,"max_abs_diff":float(diff.max()),"mean_abs_diff":float(diff.mean()),
            "max_relative_diff":float((diff/den).max()),"min_cosine":float(cosine.min()),"mean_cosine":float(cosine.mean()),
            "max_norm_abs_diff":float(np.max(np.abs(np.linalg.norm(new,axis=1)-np.linalg.norm(old,axis=1)))),
            "tolerance_max_abs":1e-4,"tolerance_cosine_min":0.999999}
        if not audit["shape_match"] or audit["max_abs_diff"]>audit["tolerance_max_abs"] or audit["min_cosine"]<audit["tolerance_cosine_min"]:
            raise RuntimeError(f"FINAL {ext} no reproduce caché histórica: {audit}")
        cache=OUT/filename;np.savez_compressed(cache,structure_id=ids,**states)
        cache_manifest[ext]={"path":str(cache.relative_to(ROOT)),"sha256":sha256(cache),"structure_count":210,
            "hidden_dimension":768,"layers":list(states),"checkpoint_sha256":EXPECTED[ext],"dtype":"float32"}
        alignment[ext]=audit
        layer_manifest[ext]={"extraction_points":list(states),"terminal_positions":"último token válido; MatterGPT suma prefijo de propiedad al índice contextualizado",
            "padding":"derecho; estados extraídos en posición del terminal real","batch_size":args.batch_size}
    atomic_json(OUT/"architecture_manifest.json",architectures)
    atomic_json(OUT/"checkpoint_manifest.json",{e:{"sha256":EXPECTED[e],"status":"PASS_SHA256_RECOMPUTED"} for e in EXPECTED})
    atomic_json(OUT/"token_alignment_audit.json",alignment);atomic_json(OUT/"layer_extraction_manifest.json",layer_manifest)
    atomic_json(OUT/"embedding_cache_manifest.json",cache_manifest)
    atomic_json(OUT/"extract_resources.json",{"wall_seconds":time.perf_counter()-started,"device":"cpu","cpu_threads":1,
        "torch_threads":torch.get_num_threads(),"test_accessed":False,"backbones_modified":False})
    LOG.info("Extracción layerwise completada. compatibilidad FINAL: %s",json.dumps(alignment))


if __name__=="__main__":
    main()
