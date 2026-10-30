"""Caracteriza distancias y rangos de embeddings H3-C congelados, sin entrenar."""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reproduce import FAMILIES,project_root,read_development,sha256

LOG=logging.getLogger("h3c_final_representations")


def effective_rank(x:np.ndarray) -> float:
    """Calcula rango efectivo entrópico de la covarianza centrada."""
    singular=np.linalg.svd(x.astype(np.float64)-x.mean(axis=0),compute_uv=False,full_matrices=False)
    weights=singular**2
    p=weights[weights>0]/weights.sum()
    return float(np.exp(-np.sum(p*np.log(p))))


def inspect(root:Path) -> dict[str,Any]:
    """Mide pares intrafórmula en espacios pretrained/random con escala comparable."""
    report=root/"reports/MatGPT/MatGPT_H3C_Final"
    data=read_development(root/"reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv")
    old=root/"reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz"
    archives=[("pretrained",None,old)]+[("random",seed,report/f"controls/random_seed_{seed}/features.npz") for seed in (42,123,2026)]
    summary=[];pairs=[]
    pred=pd.read_csv(report/"reproduction/replayed_predictions.csv")
    pred42=pred[pred.seed==42].set_index("structure_id")
    for kind,seed,path in archives:
        with np.load(path,allow_pickle=False) as archive:
            ids=archive["structure_id"].astype(str).tolist()
            if ids!=data.structure_id.tolist():raise ValueError("IDs de embeddings desalineados")
            spaces={name:np.array(archive[name],dtype=np.float64,copy=True)
                    for name in ("L03_MEAN","FINAL_END")}
        for layer,x in spaces.items():
            centered=x-x.mean(axis=0)
            std=np.maximum(x.std(axis=0),1e-6)
            normalized=centered/std
            for formula,part in data.reset_index().groupby("composition_id"):
                group=part.set_index("polymorph_family").loc[list(FAMILIES)]
                for i,j in itertools.combinations(range(3),2):
                    a=group.iloc[i];b=group.iloc[j]
                    v=x[int(a["index"])];w=x[int(b["index"])]
                    z=normalized[int(a["index"])];q=normalized[int(b["index"])]
                    denominator=np.linalg.norm(v)*np.linalg.norm(w)
                    cosine=float(np.dot(v,w)/denominator) if denominator>0 else np.nan
                    dft_gap=1000*abs(float(a.energy_per_atom_dft_ev)-float(b.energy_per_atom_dft_ev))
                    pred_gap=1000*abs(float(pred42.loc[a.structure_id,"energy_per_atom_pred_ev"])-
                                      float(pred42.loc[b.structure_id,"energy_per_atom_pred_ev"]))
                    pairs.append({"backbone":kind,"seed":seed if seed is not None else -1,
                                  "feature":layer,"composition_id":formula,
                                  "structure_a":a.structure_id,"structure_b":b.structure_id,
                                  "cosine_similarity":cosine,
                                  "euclidean_raw":float(np.linalg.norm(v-w)),
                                  "euclidean_standardized":float(np.linalg.norm(z-q)),
                                  "dft_abs_gap_mev_atom":dft_gap,
                                  "pretrained_pred_abs_gap_mev_atom":pred_gap})
            sub=pd.DataFrame([item for item in pairs if item["backbone"]==kind and item["seed"]==(seed if seed is not None else -1) and item["feature"]==layer])
            summary.append({"backbone":kind,"seed":seed if seed is not None else -1,"feature":layer,
                            "n_structures":210,"n_compositions":70,"n_pairs":210,
                            "mean_dimension_variance":float(np.mean(x.var(axis=0))),
                            "effective_rank":effective_rank(x),
                            "median_cosine_similarity":float(sub.cosine_similarity.median()),
                            "mean_standardized_distance":float(sub.euclidean_standardized.mean()),
                            "spearman_distance_vs_abs_dft_gap":float(spearmanr(sub.euclidean_standardized,sub.dft_abs_gap_mev_atom).statistic)})
    pairs_df=pd.DataFrame(pairs);summary_df=pd.DataFrame(summary)
    focus=pairs_df[(pairs_df.backbone=="pretrained")&(pairs_df.feature=="FINAL_END")].copy()
    close=focus.nsmallest(10,"euclidean_standardized")
    divergent=focus[(focus.euclidean_standardized>focus.euclidean_standardized.median())&
                    (focus.pretrained_pred_abs_gap_mev_atom<5)].nsmallest(10,"pretrained_pred_abs_gap_mev_atom")
    examples=pd.concat([close.assign(reason="closest embeddings"),
                        divergent.assign(reason="distant embeddings; predicted gap <5")],ignore_index=True)
    out=report/"analysis"
    pairs_df.to_csv(out/"embedding_pairs.csv",index=False)
    summary_df.to_csv(out/"embedding_summary.csv",index=False)
    examples.to_csv(out/"embedding_examples.csv",index=False)
    result={"status":"PASS","rows":len(pairs_df),"source_feature_hashes":{str(p.relative_to(root)):sha256(p) for _,_,p in archives},
            "tables":{p.name:sha256(p) for p in (out/"embedding_pairs.csv",out/"embedding_summary.csv",out/"embedding_examples.csv")},
            "interpretation":"descriptive, not causal","historical_test_accessed":False}
    (out/"representation_manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    LOG.info(json.dumps({"event":"representation_diagnostic","status":"PASS","pairs":len(pairs_df)}))
    return result


def main() -> None:
    """Ejecuta diagnóstico de dos representaciones con raíz configurable."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=project_root())
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    inspect(args.repo.resolve())


if __name__=="__main__":main()
