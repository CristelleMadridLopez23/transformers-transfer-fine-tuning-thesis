"""Extrae L03_MEAN y FINAL_END para MatGPT aleatorio congelado.

Reutiliza pesos aleatorios ya fijados por semilla en Frozen 3.0 y el mismo
tokenizador y decoder oficial. No lee etiquetas de evaluación ni entrena pesos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reproduce import EXPECTED_CHECKPOINT, project_root, read_development, sha256

LOG = logging.getLogger("h3c_random_extraction")


def extract(root: Path, seed: int) -> dict[str, Any]:
    """Extrae dos espacios históricos para 210 IDs de desarrollo en CPU."""
    if seed not in (42,123,2026):raise ValueError("Semilla fuera de protocolo")
    sys.path.insert(0,str(root))
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import (
        load_model,encode_sequence,hidden_and_logits)
    report=root/"reports/MatGPT/MatGPT_H3C_Final"
    output=report/f"controls/random_seed_{seed}"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Control existente; no sobrescribir: {output}")
    dataset_path=root/"reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
    data=read_development(dataset_path)
    source=root/"models/MatGPT-source/mat-concat"
    checkpoint=source/"mat-uncond/weights/transformer_property_0.pth"
    model_info=checkpoint.with_name("model_info_property_0.json")
    if sha256(checkpoint)!=EXPECTED_CHECKPOINT:raise ValueError("Checkpoint original alterado")
    torch.set_num_threads(2)
    model,stoi,context,torch_mod=load_model(source,model_info,checkpoint,source/"config/uncond.yaml")
    lock=json.loads((Path(__file__).with_name("random_initialization_lock.json")).read_text())
    torch.manual_seed(seed+500000)
    fresh=model.__class__(vocab_size=len(stoi),n_embd=768,n_head=12,n_layer=12,
                          block_size=context,dropout=.1,stoi=stoi,num_properties=0,
                          conditioning_mode="unconditional")
    random_state=fresh.state_dict()
    digest=hashlib.sha256()
    for name,tensor in sorted(random_state.items()):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    tensor_hash=digest.hexdigest()
    if tensor_hash!=lock["state_tensor_sha256"][str(seed)] or sum(p.numel() for p in fresh.parameters())!=lock["parameter_count"]:
        raise ValueError("Inicialización aleatoria no coincide con el control bloqueado")
    model=fresh;model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    sequences=[encode_sequence(s,stoi,context) for s in data.slices]
    captured:dict[str,torch.Tensor]={}
    handle=model.transformer_decoder.layers[2].register_forward_hook(
        lambda _module,_inputs,result:captured.__setitem__("l03",result.detach()))
    l03=[];final=[];pad=stoi["<pad>"];end=stoi["<end>"]
    try:
        with torch.inference_mode():
            for start in range(0,len(sequences),4):
                batch=sequences[start:start+4]
                ids=torch.full((len(batch),max(map(len,batch))),pad,dtype=torch.long)
                for i,sequence in enumerate(batch):
                    if sequence.count(end)!=1 or sequence[-1]!=end:
                        raise ValueError("Secuencia sin <end> final único")
                    ids[i,:len(sequence)]=torch.tensor(sequence,dtype=torch.long)
                hidden,_=hidden_and_logits(model,ids,torch_mod)
                for i,sequence in enumerate(batch):
                    stop=len(sequence)-1
                    l03.append(captured["l03"][1:stop,i,:].mean(dim=0).cpu().float().numpy().copy())
                    final.append(hidden[i,stop,:].cpu().float().numpy().copy())
    finally:handle.remove()
    l03_arr=np.stack(l03).astype(np.float32)
    final_arr=np.stack(final).astype(np.float32)
    if l03_arr.shape!=(210,768) or final_arr.shape!=(210,768):
        raise ValueError("Dimensiones aleatorias incorrectas")
    max_difference=None
    old_archive=root/f"reports/MatGPT/MatGPT_frozen_3.0/features/random_seed_{seed}/features.npz"
    if old_archive.is_file():
        with np.load(old_archive,allow_pickle=False) as prior:
            prior_lookup={sid:i for i,sid in enumerate(prior["structure_id"].astype(str))}
            if set(prior_lookup)!=set(data.structure_id):raise ValueError("IDs aleatorios distintos")
            old=np.stack([prior["h_end"][prior_lookup[sid]] for sid in data.structure_id])
        max_difference=float(np.max(np.abs(final_arr-old)))
        if max_difference>1e-5:raise ValueError(f"No reproduce caché random previo: {max_difference}")
    output.mkdir(parents=True,exist_ok=True)
    archive=output/"features.npz"
    np.savez_compressed(archive,structure_id=data.structure_id.to_numpy(dtype="U256"),
                        L03_MEAN=l03_arr,FINAL_END=final_arr)
    result={"status":"PASS","seed":seed,"n_structures":210,"dimension":768,
            "random_weight_tensor_sha256":tensor_hash,"feature_sha256":sha256(archive),
            "pretrained_checkpoint_sha256":sha256(checkpoint),"tokenizer_sha256":sha256(model_info),
            "max_abs_difference_final_end_vs_prior":max_difference,
            "backbone_trainable_parameters":0,"historical_test_accessed":False}
    (output/"manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    LOG.info(json.dumps({"event":"random_features","seed":seed,"status":"PASS"}))
    return result


def main() -> None:
    """Acepta raíz y semilla de extracción."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=project_root())
    parser.add_argument("--seed",type=int,choices=[42,123,2026],required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    extract(args.repo.resolve(),args.seed)


if __name__=="__main__":main()
