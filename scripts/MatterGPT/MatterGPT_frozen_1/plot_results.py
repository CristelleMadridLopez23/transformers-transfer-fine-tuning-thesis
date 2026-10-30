"""Render MatterGPT development/test figures as PNG only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from common import REPORT_DIR, logger, sha256, write_json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

COLORS = {"M0": "#4c78a8", "M2": "#f58518", "M4": "#54a24b", "M3-A": "#b279a2"}
MARKERS = {"dist_perovskite": "o", "hexagonal": "^", "needle": "s"}


def _save(fig: plt.Figure, name: str, description: str, statistics: dict[str, Any] | None = None) -> None:
    """Save a publication-readable PNG and its numeric regression table."""
    path = REPORT_DIR / "figures" / name
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest_path = REPORT_DIR / "figures/figure_notes.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    existing[name] = {"description": description, "statistics": statistics or {}, "sha256": sha256(path)}
    write_json(manifest_path, existing)


def parity_figure(frame: pd.DataFrame, pred_cols: dict[str, str], name: str, partition: str) -> None:
    """Plot DFT on x and predictions on y with free-intercept OLS per model."""
    n = len(pred_cols); fig, axes = plt.subplots(1, n, figsize=(6.1*n, 5.7), squeeze=False)
    stats = {}
    truth_all = frame.energy_dft_ev_atom.to_numpy(float) * 1000
    pred_all = np.concatenate([frame[col].to_numpy(float) * 1000 for col in pred_cols.values()])
    limits = (float(min(truth_all.min(), pred_all.min())), float(max(truth_all.max(), pred_all.max())))
    pad = .05 * (limits[1] - limits[0] or 1); lo, hi = limits[0] - pad, limits[1] + pad
    for ax, (model, pred) in zip(axes[0], pred_cols.items()):
        x = frame.energy_dft_ev_atom.to_numpy(float) * 1000
        y = frame[pred].to_numpy(float) * 1000
        for family, family_rows in frame.groupby("polymorph_family", sort=False):
            family_mask=frame.polymorph_family.to_numpy()==family
            ax.scatter(x[family_mask], y[family_mask], s=42, alpha=.78, color=COLORS[model], marker=MARKERS.get(family,"o"), edgecolor="white", linewidth=.35)
        ax.plot([lo,hi],[lo,hi],"--",color="#222222",lw=1.5,label="Acuerdo ideal: y = x")
        lr=LinearRegression().fit(x[:,None],y); yh=lr.predict(x[:,None]); mae=float(np.mean(np.abs(x-y)))
        ax.plot([lo,hi],lr.predict(np.array([[lo],[hi]])),color="#d45087",lw=1.7,label="OLS con intercepto libre")
        r2=float(r2_score(y,yh)); stats[model]={"n_structures":len(x),"n_compositions":int(frame.composition_id.nunique()),"ols_slope_dimensionless":float(lr.coef_[0]),"ols_intercept_mev_atom":float(lr.intercept_),"ols_r2":r2,"mae_vs_dft_mev_atom":mae,"predictive_r2_y_equals_x":float(1-np.sum((y-x)**2)/np.sum((x-x.mean())**2))}
        ax.text(.04,.96,f"N = {len(x)} estructuras | {frame.composition_id.nunique()} composiciones\nOLS m = {lr.coef_[0]:.3f}; b = {lr.intercept_:.1f} meV/átomo\nR² OLS = {r2:.3f}\nMAE vs DFT = {mae:.1f} meV/átomo",transform=ax.transAxes,va="top",ha="left",fontsize=9,bbox={"boxstyle":"round","fc":"white","ec":"#cbd2d9","alpha":.96})
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="DFT, energía estática (meV/átomo)",ylabel="MatterGPT, energía predicha (meV/átomo)",title=model)
        ax.grid(alpha=.18)
        legend_handles=[Line2D([0],[0],color=COLORS[model],lw=2,label=f"Predicción {model}")]
        legend_handles += [Line2D([0],[0],marker=MARKERS[f],color="none",markerfacecolor="#6b7280",markeredgecolor="white",markersize=8,label=f) for f in MARKERS]
        legend_handles += [Line2D([0],[0],color="#222222",lw=1.5,ls="--",label="Acuerdo ideal y=x"),Line2D([0],[0],color="#d45087",lw=1.7,label="OLS libre")]
        ax.legend(handles=legend_handles,loc="lower right",fontsize=7.5)
    fig.suptitle(f"{partition}: paridad de energía absoluta",fontsize=16,y=1.02)
    fig.text(.5,-.02,"Cada punto representa una estructura; los valores OOF promedian folds y semillas. R² OLS describe la recta ajustada; R² predictivo usa y=x.",ha="center",fontsize=9)
    fig.tight_layout(); _save(fig,name,"DFT en x, predicción en y; los paneles comparten unidades; se muestran todas las estructuras elegibles.",stats)


def confusion_figure(frame: pd.DataFrame, prediction_col: str, name: str, title: str, pairwise: bool = False) -> None:
    """Draw a count-and-row-percent confusion matrix from physical triplets."""
    classes = ["perovskita", "hexagonal", "needle"]
    fam_index = {"dist_perovskite":0,"hexagonal":1,"needle":2}
    counts = np.zeros((3,4), dtype=int) if not pairwise else np.zeros((3,3),dtype=int)
    n_physical = n_ties = n_errors = 0
    for _, group in frame.groupby("composition_id"):
        group=group.sort_values("polymorph_family",key=lambda s:s.map(fam_index))
        y=group.energy_dft_ev_atom.to_numpy(float)*1000;p=group[prediction_col].to_numpy(float)*1000
        if not pairwise:
            if np.partition(y,1)[1]-np.min(y)<=1.0: continue
            gi=int(np.argmin(y)); gap=np.partition(p,1)[1]-np.min(p)
            if gap<=1.0: pj=3
            else: pj=int(np.argmin(p))
            counts[gi,pj]+=1;n_physical+=1;n_errors+=int(pj!=gi)
        else:
            for i,j in ((0,1),(0,2),(1,2)):
                d=(y[i]-y[j]); q=(p[i]-p[j])
                c_y=0 if d < -1 else (2 if d > 1 else 1)
                c_p=0 if q < -1 else (2 if q > 1 else 1)
                counts[c_y,c_p]+=1;n_physical+=1;n_ties+=int(c_y==1);n_errors+=int(c_y!=c_p)
    fig,ax=plt.subplots(figsize=(7.3,6.2)); row=counts.sum(axis=1,keepdims=True); pct=np.divide(counts,row,out=np.zeros_like(counts,dtype=float),where=row!=0)*100
    im=ax.imshow(pct,cmap="Blues",vmin=0,vmax=100)
    labels=classes+["tie/abstain"] if not pairwise else ["i < j","tie","j < i"]
    ax.set_xticks(range(len(labels)),labels,rotation=22,ha="right");ax.set_yticks(range(3),["perovskita","hexagonal","needle"] if not pairwise else ["i < j","tie","j < i"])
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]): ax.text(j,i,f"{counts[i,j]}\n{pct[i,j]:.1f}%",ha="center",va="center",color="black" if pct[i,j]<65 else "white",fontsize=10)
    ax.set_xlabel("Predicción"+(" del ganador DFT" if not pairwise else " de relación (pares P-H, P-N, H-N)"));ax.set_ylabel("Relación DFT" if pairwise else "Ground state DFT")
    ax.set_title(title);cb=fig.colorbar(im,ax=ax);cb.set_label("Porcentaje dentro de la fila (%)")
    fig.text(.5,.01,f"Observaciones físicas: {n_physical}; empates DFT (pairwise): {n_ties}; errores/abstenciones: {n_errors}. Umbral ±1 meV/átomo.",ha="center",fontsize=9)
    _save(fig,name,"Matriz OOF agregada por estructura/composición; conteos enteros y porcentajes normalizados por fila.",{"physical_observations":n_physical,"dft_ties":n_ties,"errors_or_abstentions":n_errors,"counts":counts.tolist()})


def run() -> dict[str, Any]:
    """Create development and test PNG figures from original prediction CSVs."""
    oof=pd.read_csv(REPORT_DIR/"h3a/oof_m3a_predictions.csv")
    test_path=REPORT_DIR/"test/test_predictions_with_dft.csv"
    if not test_path.exists(): raise FileNotFoundError("Puntúe test bloqueado antes de graficar")
    test=pd.read_csv(test_path)
    for frame in (oof,test):
        for model,col in (("M2","m2_prediction_ev_atom"),("M3-A","m3a_prediction_ev_atom")):
            if col not in frame:
                old="M2_prediction_ev_atom" if model=="M2" else "M3A_prediction_ev_atom"
                frame[col]=frame[old]
    oof_ensemble=oof.groupby(["structure_id","composition_id","polymorph_family","energy_dft_ev_atom"],as_index=False)[["m2_prediction_ev_atom","m3a_prediction_ev_atom"]].mean()
    parity_figure(oof_ensemble,{"M2":"m2_prediction_ev_atom","M3-A":"m3a_prediction_ev_atom"},"01_development_oof_absolute_parity.png","Development OOF")
    parity_figure(test,{"M2":"m2_prediction_ev_atom","M3-A":"m3a_prediction_ev_atom"},"02_test_absolute_parity.png","Test bloqueado (evaluación única)")
    confusion_figure(test,"m3a_prediction_ev_atom","03_test_ground_state_confusion.png","MatterGPT M3-A: selección del estado fundamental")
    confusion_figure(test,"m3a_prediction_ev_atom","04_test_pairwise_confusion.png","MatterGPT M3-A: ordenamiento por pares",pairwise=True)
    fig,ax=plt.subplots(figsize=(11,5.2)); err=(test.m3a_prediction_ev_atom-test.energy_dft_ev_atom).abs()*1000
    ordered=pd.DataFrame({"composition_id":test.composition_id,"absolute_error":err}).groupby("composition_id").absolute_error.mean().sort_values(ascending=False)
    ax.bar(range(len(ordered)),ordered,color="#b279a2");ax.set_xticks(range(len(ordered)),ordered.index,rotation=45,ha="right");ax.set_ylabel("MAE absoluto (meV/átomo)");ax.set_title("Test: error absoluto medio por composición (8 tripletes)");ax.grid(axis="y",alpha=.2)
    _save(fig,"05_test_absolute_error_by_composition.png","Cada barra representa el promedio de errores absolutos de tres polimorfos de una composición.",{"compositions":len(ordered),"values_mev_atom":ordered.to_dict()})
    # Anchor both series to the same known DFT ground-state polymorph.
    rel = test.copy()
    rel["dft_delta_mev"] = rel.groupby("composition_id").energy_dft_ev_atom.transform(lambda s:(s-s.min())*1000)
    rel["pred_delta_mev"] = rel.groupby("composition_id").m3a_prediction_ev_atom.transform(lambda s:(s-s.loc[rel.loc[s.index,"energy_dft_ev_atom"].idxmin()])*1000)
    fig,ax=plt.subplots(figsize=(7.5,5.2));
    for family,group in test.groupby("polymorph_family"):
        group_rel=rel[rel.polymorph_family==family]
        ax.scatter(group_rel.dft_delta_mev,group_rel.pred_delta_mev,label=family,marker=MARKERS[family],s=48,alpha=.8)
    ax.set_xlabel("DFT-GS-anchored ΔE (meV/átomo)");ax.set_ylabel("MatterGPT M3-A, anclado al mismo DFT-GS (meV/átomo)");ax.set_title("Test: ΔE anclada al estado fundamental DFT");ax.grid(alpha=.2)
    x=rel.dft_delta_mev.to_numpy(float);y=rel.pred_delta_mev.to_numpy(float);lr=LinearRegression().fit(x[:,None],y);r2=float(r2_score(y,lr.predict(x[:,None])));mae=float(np.mean(np.abs(y-x)))
    lo=float(min(x.min(),y.min()));hi=float(max(x.max(),y.max()));pad=.06*(hi-lo or 1);lo-=pad;hi+=pad
    ax.plot([lo,hi],[lo,hi],"--",color="black",label="Acuerdo ideal y=x");ax.plot([lo,hi],lr.predict(np.array([[lo],[hi]])),color="#d45087",label="OLS con intercepto libre")
    ax.set(xlim=(lo,hi),ylim=(lo,hi))
    ax.text(.04,.96,f"N = {len(x)} estructuras | {test.composition_id.nunique()} composiciones\nOLS m = {lr.coef_[0]:.3f}; b = {lr.intercept_:.2f} meV/átomo\nR² OLS = {r2:.3f}; MAE vs DFT = {mae:.2f} meV/átomo",transform=ax.transAxes,va="top",fontsize=9,bbox={"boxstyle":"round","fc":"white","ec":"#ccd2d9"})
    ax.legend(loc="lower right",title="Familia / referencia",fontsize=8)
    _save(fig,"06_test_dft_gs_anchored_delta.png","Cada marcador es una estructura; forma indica familia; el mismo polimorfo DFT-GS ancla ambas ΔE; se conserva ΔE predicha negativa. OLS se ajusta sobre todas las estructuras mostradas.",{"n_structures":len(test),"n_compositions":test.composition_id.nunique(),"ols_slope_dimensionless":float(lr.coef_[0]),"ols_intercept_mev_atom":float(lr.intercept_),"ols_r2":r2,"mae_mev_atom":mae})
    return {"status":"PASS","png_count":6,"files":sorted(p.name for p in (REPORT_DIR/"figures").glob("*.png"))}


def main() -> None:
    """Render all configured PNG figure families."""
    log=logger("mattergpt_figures")
    try: log.info("mattergpt_figures_complete %s",run())
    except Exception as exc: log.exception("mattergpt_figures_failed error=%s",exc);raise


if __name__=="__main__": main()
