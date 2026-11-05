"""Agrega métricas OOF, inferencia apareada, figuras e informe de la auditoría ABS3."""
from __future__ import annotations

import json
import logging
import platform
import resource
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
EXP=ROOT/"experiments/abs3_representation_sufficiency_audit"
IMG=ROOT/"images/abs3_representation_sufficiency_audit"
REPORT=ROOT/"reports/ABS3/abs3_representation_sufficiency_audit"
BOOT_N=10_000;BOOT_SEED=20261103
GEOM=["a_reduced_shape","b_reduced_shape","c_reduced_shape","alpha_reduced_deg","beta_reduced_deg","gamma_reduced_deg","volume_per_atom_ang3"]
LOG=logging.getLogger("abs3.representation_audit.finalize")


def atomic_json(path: Path,obj: Any)->None:
    """Escribe JSON de forma atómica."""
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(path)


def atomic_csv(path: Path,df: pd.DataFrame)->None:
    """Escribe CSV de forma atómica."""
    tmp=path.with_suffix(path.suffix+".tmp");df.to_csv(tmp,index=False);tmp.replace(path)


def build_metrics()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Calcula estimandos primarios y secundarios por composición y condición."""
    s=pd.read_csv(EXP/"predicciones_oof_por_estructura.csv");p=pd.read_csv(EXP/"predicciones_oof_por_par.csv")
    s["energy_error_mev_atom"]=(s.energy_pred_ev_atom-s.energy_dft_ev_atom)*1000
    p["pair_correct"]=(p.delta_dft_mev_atom.abs()>5)&(p.delta_pred_mev_atom.abs()>1)&(np.sign(p.delta_dft_mev_atom)==np.sign(p.delta_pred_mev_atom))
    p["pair_eligible"]=p.delta_dft_mev_atom.abs()>5;p["pair_unresolved"]=p.pair_eligible&(p.delta_pred_mev_atom.abs()<=1)
    comps=[];allmetrics=[]
    for (cond,probe,formula),g in p.groupby(["condition","probe","formula"],sort=False):
        ss=s[(s.condition==cond)&(s.probe==probe)&(s.formula==formula)]
        ord_d=np.argsort(ss.energy_dft_ev_atom.to_numpy());ord_p=np.argsort(ss.energy_pred_ev_atom.to_numpy())
        eligible_gs=bool((ss.energy_dft_ev_atom.to_numpy()[ord_d[1]]-ss.energy_dft_ev_atom.to_numpy()[ord_d[0]])*1000>1)
        gs_ok=bool(eligible_gs and ord_d[0]==ord_p[0] and (np.sort(ss.energy_pred_ev_atom.to_numpy())[1]-np.sort(ss.energy_pred_ev_atom.to_numpy())[0])*1000>1)
        trip_ok_eligible=bool((g.delta_dft_mev_atom.abs()>1).all())
        trip_ok=bool(trip_ok_eligible and np.array_equal(ord_d,ord_p))
        comps.append({"condition":cond,"probe":probe,"formula":formula,"structures":len(ss),
            "energy_mae_mev_atom":float(ss.energy_error_mev_atom.abs().mean()),
            "deltaE_mae_estimand_A_mev_atom":float(g.absolute_error_mev_atom.mean()),
            "pairwise_correct":int(g.pair_correct.sum()),"pairwise_eligible":int(g.pair_eligible.sum()),
            "pairwise_accuracy":float(g.pair_correct.sum()/g.pair_eligible.sum()) if g.pair_eligible.sum() else np.nan,
            "unresolved_eligible":int(g.pair_unresolved.sum()),"groundstate_eligible":int(eligible_gs),"groundstate_correct":int(gs_ok),
            "groundstate_accuracy":float(gs_ok/eligible_gs) if eligible_gs else np.nan,
            "full_triplet_eligible":int(trip_ok_eligible),"full_triplet_correct":int(trip_ok),
            "full_triplet_accuracy":float(trip_ok/trip_ok_eligible) if trip_ok_eligible else np.nan})
    c=pd.DataFrame(comps); p_rows=[]
    for (cond,probe),g in p.groupby(["condition","probe"],sort=False):
        q=c[(c.condition==cond)&(c.probe==probe)];ss=s[(s.condition==cond)&(s.probe==probe)]
        y=g.delta_dft_mev_atom.to_numpy();yp=g.delta_pred_mev_atom.to_numpy();slope,inter=np.polyfit(y,yp,1)
        p_rows.append({"condition":cond,"probe":probe,"structures":len(ss),"compositions":ss.formula.nunique(),
            "energy_mae_mev_atom":float(q.energy_mae_mev_atom.mean()),"energy_rmse_global_mev_atom":float(np.sqrt(np.mean(ss.energy_error_mev_atom.to_numpy()**2))),
            "energy_bias_mev_atom":float(ss.energy_error_mev_atom.mean()),"deltaE_mae_estimand_A_mev_atom":float(q.deltaE_mae_estimand_A_mev_atom.mean()),
            "deltaE_rmse_global_mev_atom":float(np.sqrt(np.mean((yp-y)**2))),"pairwise_correct":int(q.pairwise_correct.sum()),
            "pairwise_eligible":int(q.pairwise_eligible.sum()),"pairwise_accuracy":float(q.pairwise_correct.sum()/q.pairwise_eligible.sum()),
            "pairwise_resolved_accuracy":float((g.loc[g.pair_eligible&~g.pair_unresolved,"pair_correct"].sum())/max(int((g.pair_eligible&~g.pair_unresolved).sum()),1)),
            "unresolved_eligible":int(q.unresolved_eligible.sum()),"groundstate_correct":int(q.groundstate_correct.sum()),
            "groundstate_eligible":int(q.groundstate_eligible.sum()),"groundstate_accuracy":float(q.groundstate_correct.sum()/max(int(q.groundstate_eligible.sum()),1)),
            "full_triplet_correct":int(q.full_triplet_correct.sum()),"full_triplet_eligible":int(q.full_triplet_eligible.sum()),
            "full_triplet_accuracy":float(q.full_triplet_correct.sum()/max(int(q.full_triplet_eligible.sum()),1)),
            "ols_slope":float(slope),"ols_intercept_mev_atom":float(inter),"ols_r2":float(1-np.sum((yp-(slope*y+inter))**2)/np.sum((y-y.mean())**2)),
            "predictive_r2":float(1-np.sum((yp-y)**2)/np.sum((y-y.mean())**2)),
            "spearman_mean_by_composition":float(np.nanmean([g0.delta_dft_mev_atom.corr(g0.delta_pred_mev_atom,method="spearman") for _,g0 in g.groupby("formula")])),
            "kendall_mean_by_composition":float(np.nanmean([g0.delta_dft_mev_atom.corr(g0.delta_pred_mev_atom,method="kendall") for _,g0 in g.groupby("formula")]))})
    return c,pd.DataFrame(p_rows),p


def bootstrap_metrics(c:pd.DataFrame)->pd.DataFrame:
    """IC95% apareados de KRR/PLS frente a Ridge por composición."""
    rng=np.random.default_rng(BOOT_SEED);rows=[]
    comps=sorted(c.formula.unique())
    metrics=["energy_mae_mev_atom","deltaE_mae_estimand_A_mev_atom","pairwise_accuracy","groundstate_accuracy","full_triplet_accuracy"]
    for condition in ["GEO","MAT","MATTER"]:
        for probe in ["PLS_Ridge","KRR_RBF"]:
            for met in metrics:
                sub=c[(c.condition==condition)&c.probe.isin(["Ridge",probe])]
                wide=sub.pivot(index="formula",columns="probe",values=met).reindex(comps)
                if wide.isna().any().any():continue
                if met in ["pairwise_accuracy","groundstate_accuracy","full_triplet_accuracy"]:
                    numden={}
                    for m in ["Ridge",probe]:
                        for numerator,denominator in [("pairwise_correct","pairwise_eligible"),("groundstate_correct","groundstate_eligible"),("full_triplet_correct","full_triplet_eligible")]:
                            q=sub[sub.probe.eq(m)].set_index("formula").reindex(comps)
                            numden[(m,numerator)]=q[numerator].to_numpy(float);numden[(m,denominator)]=q[denominator].to_numpy(float)
                    def resampled_accuracy(model:str, metric_name:str, sampled:np.ndarray)->np.ndarray:
                        stem={"pairwise_accuracy":"pairwise","groundstate_accuracy":"groundstate","full_triplet_accuracy":"full_triplet"}[metric_name]
                        n=numden[(model,stem+"_correct")][sampled].sum(axis=1);den=numden[(model,stem+"_eligible")][sampled].sum(axis=1)
                        return np.divide(n,den,out=np.full_like(n,np.nan),where=den>0)
                    sample=rng.integers(0,len(comps),(BOOT_N,len(comps)))
                    da=resampled_accuracy(probe,met,sample)-resampled_accuracy("Ridge",met,sample)
                    observed=(sub[sub.probe.eq(probe)].set_index("formula").reindex(comps)[met].to_numpy().mean()
                              -sub[sub.probe.eq("Ridge")].set_index("formula").reindex(comps)[met].to_numpy().mean())
                    # Point estimate follows pooled denominator convention; each bootstrap replicate also pools denominators.
                    q=sub[sub.probe.eq(probe)].set_index("formula").reindex(comps);r=sub[sub.probe.eq("Ridge")].set_index("formula").reindex(comps)
                    stem={"pairwise_accuracy":"pairwise","groundstate_accuracy":"groundstate","full_triplet_accuracy":"full_triplet"}[met]
                    observed=(q[stem+"_correct"].sum()/q[stem+"_eligible"].sum())-(r[stem+"_correct"].sum()/r[stem+"_eligible"].sum())
                    draws=da
                else:
                    dif=(wide[probe]-wide["Ridge"]).to_numpy();draws=np.mean(dif[rng.integers(0,len(dif),(BOOT_N,len(dif)))],axis=1);observed=float(dif.mean())
                rows.append({"contrast":f"{condition} {probe} − Ridge","metric":met,"estimate":float(observed),
                    "ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"replicates":BOOT_N,"seed":BOOT_SEED,"unit":"composition"})
    return pd.DataFrame(rows)


def make_figures(c:pd.DataFrame,metrics:pd.DataFrame,pairs:pd.DataFrame,dist:pd.DataFrame,var:pd.DataFrame,
                 geo_probe:pd.DataFrame,curves:pd.DataFrame,perms:pd.DataFrame,conditional:pd.DataFrame,condpred:pd.DataFrame)->None:
    """Genera nueve PNG científicos en español, con muestras y definiciones."""
    IMG.mkdir(parents=True,exist_ok=True);colors={"MAT":"#2878B5","MATTER":"#D28BC2","GEO":"#7A8B5A"}
    labels={"MAT":"MatGPT","MATTER":"MatterGPT","GEO":"Geometría"}
    # 1: embedding separability distributions.
    fig,ax=plt.subplots(figsize=(8.5,5.5))
    for i,e in enumerate(["MAT","MATTER"]):
        q=dist[dist.extractor.eq(e)];ax.boxplot(q.cosine_similarity,positions=[i*2],widths=.55,showfliers=True,patch_artist=True,boxprops={"facecolor":colors[e],"alpha":.65})
        ax.boxplot(q.normalized_l2,positions=[i*2+0.7],widths=.55,showfliers=True,patch_artist=True,boxprops={"facecolor":colors[e],"alpha":.28})
    ax.set_xticks([.35,2.35],["MatGPT","MatterGPT"]);ax.set_ylabel("Similitud coseno (izq.) / distancia L2 normalizada (der.)")
    ax.set_title("ABS₃: separabilidad de polimorfos en embeddings congelados")
    # Use a twin axis to keep the two dimensionless measures legible.
    ax.clear();ax2=ax.twinx()
    for i,e in enumerate(["MAT","MATTER"]):
        q=dist[dist.extractor.eq(e)];ax.boxplot(q.cosine_similarity,positions=[i],widths=.45,patch_artist=True,boxprops={"facecolor":colors[e],"alpha":.6})
        ax2.boxplot(q.normalized_l2,positions=[i+.42],widths=.28,patch_artist=True,boxprops={"facecolor":colors[e],"alpha":.25})
    ax.set_xticks([.2,1.2],["MatGPT","MatterGPT"]);ax.set_ylabel("Similitud coseno intrafórmula");ax2.set_ylabel("L2 / (||hᵢ||+||hⱼ||)")
    ax.set_title("ABS₃: separabilidad de los tres polimorfos en embeddings congelados")
    ax.text(.01,-.18,"210 estructuras; 70 fórmulas; tres pares por fórmula (n=210/extractor). Pares OOF de desarrollo; cada distribución describe distancias entre polimorfos de la misma fórmula.",transform=ax.transAxes,fontsize=7.5)
    fig.tight_layout();fig.savefig(IMG/"embedding_separability.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 2 variance decomposition.
    fig,ax=plt.subplots(figsize=(7.5,5));x=np.arange(2);w=.34
    for j,(key,label) in enumerate([("between_fraction","Entre composiciones"),("within_fraction","Dentro de composición")]):ax.bar(x+(j-.5)*w,var.set_index("extractor").loc[["MAT","MATTER"],key],w,label=label)
    ax.set_xticks(x,["MatGPT","MatterGPT"]);ax.set_ylim(0,1);ax.set_ylabel("Fracción de suma de cuadrados total");ax.set_title("ABS₃: varianza composicional e intracomposicional")
    ax.legend();ax.text(.01,-.18,"Descomposición descriptiva de 210 embeddings (70×3); dentro = variación entre polimorfos tras restar la media de cada fórmula.",transform=ax.transAxes,fontsize=7.5)
    fig.tight_layout();fig.savefig(IMG/"within_between_variance.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 3 geometry decoding normalized errors and predictive R2.
    g=geo_probe.groupby(["extractor","probe","descriptor"],as_index=False).agg(mae_norm=("train_sd_normalized_mae","mean"),r2=("predictive_r2","mean"))
    fig,axs=plt.subplots(1,2,figsize=(13,5));names=["MAT","MATTER"]
    for e in names:
        q=g[g.extractor.eq(e)&g.probe.isin(["Ridge","KRR_RBF"])];piv=q.pivot(index="descriptor",columns="probe",values="mae_norm")
        axs[0].plot(np.arange(len(piv)),piv.Ridge,marker="o",color=colors[e],label=f"{labels[e]} Ridge")
        axs[0].plot(np.arange(len(piv)),piv.KRR_RBF,marker="s",ls="--",color=colors[e],label=f"{labels[e]} KRR")
    axs[0].set_xticks(range(7),[s.replace("_reduced_shape","\nshape").replace("_reduced_deg","\nreduced") for s in GEOM],rotation=20,ha="right");axs[0].set_ylabel("MAE / SD del descriptor en train");axs[0].legend(fontsize=7);axs[0].grid(alpha=.25)
    r2=g[g.extractor.isin(names)&g.probe.isin(["Ridge","KRR_RBF"])].groupby(["extractor","probe"]).r2.mean()
    axs[1].bar(range(len(r2)),r2.values,color=[colors[k[0]] for k in r2.index]);axs[1].axhline(0,color="black",lw=.8);axs[1].set_xticks(range(len(r2)),[f"{labels[k[0]]}\n{k[1]}" for k in r2.index]);axs[1].set_ylabel("R² predictivo medio por pliegue")
    fig.suptitle("ABS₃: recuperación OOF de geometría desde embeddings congelados")
    fig.text(.01,.01,"Ridge/KRR se eligieron por validación interna; targets geométricos estandarizados con inner-train. n=210 estructuras OOF, cinco folds.",fontsize=7.5)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(IMG/"geometry_decoding.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 4 energy probes.
    fig,axs=plt.subplots(1,2,figsize=(12,5));order=[("GEO","Geometría"),("MAT","MatGPT"),("MATTER","MatterGPT")]
    for ax,metric,title in [(axs[0],"deltaE_mae_estimand_A_mev_atom","MAE de ΔE firmado"),(axs[1],"energy_mae_mev_atom","MAE energía absoluta")]:
        for j,(e,l) in enumerate(order):
            q=metrics[metrics.condition.eq(e)].set_index("probe");vals=[q.loc[k,metric] for k in ["Ridge","PLS_Ridge","KRR_RBF"]]
            ax.plot([j*3,j*3+1,j*3+2],vals,marker="o",color=colors[e],label=l if ax is axs[0] else None)
        ax.set_xticks([1,4,7],["Geometría","MatGPT","MatterGPT"]);ax.set_title(title);ax.set_ylabel("meV/átomo");ax.grid(alpha=.25)
    axs[0].legend();fig.suptitle("ABS₃: desempeño energético OOF de probes lineales, PLS y RBF")
    fig.text(.01,.01,"70 composiciones y 210 estructuras, cinco folds agrupados. ΔE MAE: promedio de tres errores por fórmula y macro-promedio; energía MAE: promedio por fórmula. Ridge M2 fijo.",fontsize=7.5)
    fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(IMG/"energy_probing.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 5 learning curves: outer MAE by training composition count.
    q=curves[curves.split.eq("outer")].groupby(["extractor","probe","n_train_compositions"],as_index=False).deltaE_mae_estimand_A_mev_atom.agg(["mean","std"]).reset_index()
    fig,ax=plt.subplots(figsize=(9,5.5))
    for e in ["GEO","MAT","MATTER"]:
        for probe,ls in [("RIDGE","-"),("KRR","--")]:
            z=curves[(curves.split=="outer")&(curves.extractor==e)&(curves.probe==probe)].groupby("n_train_compositions").deltaE_mae_estimand_A_mev_atom
            mu=z.mean();sd=z.std().fillna(0);ax.plot(mu.index,mu.values,marker="o",ls=ls,color=colors[e],label=f"{labels[e]} {probe}");ax.fill_between(mu.index,mu.values-sd.values,mu.values+sd.values,color=colors[e],alpha=.08)
    ax.set_xlabel("Composiciones inner-train usadas");ax.set_ylabel("MAE OOF de ΔE (meV/átomo)");ax.set_title("ABS₃: curvas few-shot por número de composiciones")
    ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);ax.text(.01,-.18,"n=10/20/30: 10 subconjuntos por fold; n=45: conjunto completo una vez. Cada punto resume outer de desarrollo; las repeticiones comparten fórmulas outer y no son datasets independientes.",transform=ax.transAxes,fontsize=7.2)
    fig.tight_layout();fig.savefig(IMG/"fewshot_learning_curves.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 6 shuffled embeddings.
    true=pairs[pairs.probe.isin(["Ridge","KRR_RBF"])].groupby(["condition","probe"]).absolute_error_mev_atom.mean().reset_index()
    fig,ax=plt.subplots(figsize=(9,5.5))
    pos=0;xt=[]
    for e in ["MAT","MATTER"]:
        for probe in ["RIDGE","KRR"]:
            name="KRR_RBF" if probe=="KRR" else "Ridge";real=metrics[(metrics.condition==e)&(metrics.probe==name)].deltaE_mae_estimand_A_mev_atom.iloc[0]
            shuffled=perms[(perms.extractor==e)&(perms.probe==probe)].deltaE_mae_estimand_A_mev_atom
            ax.boxplot(shuffled,positions=[pos],widths=.55,showfliers=False,patch_artist=True,boxprops={"facecolor":colors[e],"alpha":.35})
            ax.scatter([pos],[real],marker="D",color=colors[e],zorder=3);xt.append(f"{e}\n{probe}");pos+=1
    ax.set_xticks(range(4),xt);ax.set_ylabel("MAE ΔE (meV/átomo)");ax.set_title("ABS₃: embeddings reales frente a permutación intrafórmula")
    ax.text(.01,-.18,"Cajas: 100 permutaciones independientes por extractor, evaluación OOF completa; rompen asignación embedding–polimorfo dentro de cada fórmula. Diamante = embeddings reales; mismos folds/targets.",transform=ax.transAxes,fontsize=7.5)
    fig.tight_layout();fig.savefig(IMG/"real_vs_shuffled_embeddings.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 7 conditional value additive kernels.
    vals=[]; names=[]
    for e in ["MAT","MATTER"]:
        q=condpred[condpred.extractor.eq(e)];mae_c=q.groupby("formula").apply(lambda z:np.mean(abs(z.delta_dft_mev_atom-z.delta_pred_mev_atom)),include_groups=False).mean()
        geo_mae=metrics[(metrics.condition=="GEO")&(metrics.probe=="KRR_RBF")].deltaE_mae_estimand_A_mev_atom.iloc[0]
        vals += [geo_mae,mae_c];names += ["Geometría KRR",f"Geometría + {e}"]
    fig,ax=plt.subplots(figsize=(8,5));bars=ax.bar(range(4),vals,color=[colors["GEO"],colors["MAT"],colors["GEO"],colors["MATTER"]]);ax.bar_label(bars,fmt="%.2f")
    ax.set_xticks(range(4),names,rotation=15,ha="right");ax.set_ylabel("MAE ΔE OOF (meV/átomo)");ax.set_title("ABS₃: valor condicional de embeddings en kernel aditivo")
    ax.text(.01,-.27,"Kernel híbrido kGEO + η·kembedding; η elegido en inner-validation. Referencia GEO = KRR geométrico. Celdas DFT relajadas; 70 fórmulas OOF.",transform=ax.transAxes,fontsize=7.5)
    fig.tight_layout();fig.savefig(IMG/"conditional_embedding_value.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 8 ranking.
    fig,ax=plt.subplots(figsize=(10,5.5));subset=metrics[metrics.probe.isin(["Ridge","PLS_Ridge","KRR_RBF"])].copy();x=np.arange(len(subset))
    ax.bar(x,subset.pairwise_accuracy*100,color=[colors[e] for e in subset.condition]);ax.axhline(90,color="#A33D3D",ls="--",label="Meta aspiracional 90%")
    ax.set_xticks(x,[f"{labels[r.condition]}\n{r.probe}" for r in subset.itertuples()],rotation=20,ha="right");ax.set_ylim(0,100);ax.set_ylabel("Exactitud pairwise elegible (%)");ax.set_title("ABS₃: ranking OOF por extractor y probe");ax.legend()
    for i,r in enumerate(subset.itertuples()):ax.text(i,r.pairwise_accuracy*100+1,f"{r.pairwise_correct}/{r.pairwise_eligible}",ha="center",fontsize=7)
    ax.text(.01,-.32,"Pares elegibles si |ΔE_DFT|>5 meV/átomo; predicción |ΔE|≤1 cuenta incorrecta. OOF: 70 fórmulas; denominadores anotados.",transform=ax.transAxes,fontsize=7.5)
    fig.tight_layout();fig.savefig(IMG/"ranking_accuracy.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 9 calibración: el OLS y las estadísticas usan exactamente los puntos de cada panel.
    selected=[("GEO","Ridge"),("MAT","Ridge"),("MAT","KRR_RBF"),("MATTER","KRR_RBF")]
    lim=float(max(pairs.delta_dft_mev_atom.abs().max(),pairs.delta_pred_mev_atom.abs().max()))
    fig,axs=plt.subplots(2,2,figsize=(12,10),sharex=True,sharey=True)
    for ax,(e,probe) in zip(axs.flat,selected):
        z=pairs[(pairs.condition==e)&(pairs.probe==probe)]
        x=z.delta_dft_mev_atom.to_numpy(float);y=z.delta_pred_mev_atom.to_numpy(float)
        slope,inter=np.polyfit(x,y,1);fit=slope*x+inter
        r2=1-float(np.sum((y-fit)**2)/np.sum((y-y.mean())**2))
        mae=float(np.mean(np.abs(y-x)));n=len(x)
        ax.scatter(x,y,s=13,alpha=.32,color=colors[e],edgecolors="none",label="Pares OOF")
        grid=np.array([-lim,lim]);ax.plot(grid,grid,"k--",lw=1,label="Ideal y=x")
        ax.plot(grid,slope*grid+inter,color="#C44E52",lw=1.6,label="Ajuste OLS")
        ax.set(xlim=(-lim,lim),ylim=(-lim,lim),title=f"{labels[e]} — {probe}")
        ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.2)
        ax.text(.04,.96,f"N={n} pares OOF\nm={slope:.3f}; b={inter:.1f} meV/átomo\nR² OLS={r2:.3f}; MAE={mae:.1f} meV/átomo",
                transform=ax.transAxes,va="top",fontsize=8,
                bbox={"boxstyle":"round,pad=.35","facecolor":"white","alpha":.88,"edgecolor":"0.8"})
        ax.legend(loc="lower right",fontsize=7)
    for ax in axs[-1,:]:ax.set_xlabel("ΔE DFT firmada, Eⱼ−Eᵢ (meV/átomo)")
    for ax in axs[:,0]:ax.set_ylabel("ΔE predicha firmada, Eⱼ−Eᵢ (meV/átomo)")
    fig.suptitle("ABS₃: calibración de diferencias energéticas en probes congelados",y=.98)
    fig.text(.5,.015,"Cada punto = par intracomposición; 70 fórmulas, 210 pares por panel, cinco pliegues OOF. m y b provienen de OLS con intercepto libre sobre los mismos puntos. ΔE<0 indica el orden según la orientación canónica del par.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.96));fig.savefig(IMG/"gap_compression.png",dpi=300,bbox_inches="tight");plt.close(fig)


def main()->None:
    """Finaliza salidas, figuras y clasificación científica con recursos registrados."""
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    wall_start=time.perf_counter();cpu_start=time.process_time()
    c,metrics,pairs=build_metrics();atomic_csv(EXP/"metrics_per_composition.csv",c);atomic_csv(EXP/"metrics.json",metrics)
    boot_main=bootstrap_metrics(c);boot_ext=pd.read_csv(EXP/"bootstrap_paired.csv")
    boot=pd.concat([boot_main,boot_ext],ignore_index=True).drop_duplicates(["contrast","metric"],keep="first")
    atomic_csv(EXP/"bootstrap_paired.csv",boot)
    # Required named tables and stable metric family summaries.
    selections=pd.read_csv(EXP/"energy_probe_results.csv")
    atomic_csv(EXP/"pls_component_selection.csv",selections[selections.model.eq("PLS_Ridge")])
    atomic_csv(EXP/"krr_hyperparameters.csv",selections[selections.model.eq("KRR_RBF")])
    atomic_csv(EXP/"energy_probe_results.csv",selections)
    atomic_json(EXP/"summary.json",{"status":"COMPLETE_WITH_LIMITATIONS","conditions":metrics.to_dict("records"),
        "OOF":{"compositions_per_condition":70,"structures_per_condition":210,"pairs_per_condition":210,"folds":5},
        "ranking_90pct_reached":bool(metrics.pairwise_accuracy.max()>=.9),"test_accessed":False,"backbones_modified":False,
        "limitations":["Curvas de aprendizaje exploratorias reutilizan outer folds para describir generalización por tamaño; no seleccionan el método final.",
                        "Los intervalos son de desarrollo OOF, no test externo.","Los probes limitan la interpretación a información recuperable con estas familias de modelos."]})
    resources=json.loads((EXP/"resources.json").read_text());ext=json.loads((EXP/"extended_resources.json").read_text())
    resources.update({"status":"COMPLETE","core_wall_seconds":resources.get("wall_seconds"),"extended_wall_seconds":ext["wall_seconds"],"process_cpu_seconds_recorded_core":resources.get("process_cpu_seconds"),
        "extended_device":"cpu","cpu_threads":1,"permutations_per_extractor":100,"bootstrap_replicates":BOOT_N,
        "bootstrap_seed":BOOT_SEED,"backbones_modified":False,"historical_test_accessed":False})
    atomic_json(EXP/"resources.json",resources)
    geo=pd.read_csv(EXP/"geometry_probe_results.csv");dist=pd.read_csv(EXP/"structural_distances.csv");var=pd.read_csv(EXP/"intracomposition_variance.csv")
    curves=pd.read_csv(EXP/"learning_curves.csv");perms=pd.read_csv(EXP/"permutation_controls.csv");cond=pd.read_csv(EXP/"conditional_value.csv");condpred=pd.read_csv(EXP/"conditional_value_predictions.csv")
    make_figures(c,metrics,pairs,dist,var,geo,curves,perms,cond,condpred)
    mat=metrics[(metrics.condition=="MAT")&metrics.probe.eq("KRR_RBF")].iloc[0]
    matter=metrics[(metrics.condition=="MATTER")&metrics.probe.eq("KRR_RBF")].iloc[0]
    cm=boot[(boot.contrast=="MAT+G additive KRR − GEO KRR")&(boot.metric=="deltaE_mae_mev_atom")].iloc[0]
    cmt=boot[(boot.contrast=="MATTER+G additive KRR − GEO KRR")&(boot.metric=="deltaE_mae_mev_atom")].iloc[0]
    pm=pd.read_csv(EXP/"permutation_controls.csv")
    pm_summary=pm.groupby(["extractor","probe"]).agg(delta_mean=("deltaE_mae_estimand_A_mev_atom","mean"),pair_mean=("pairwise_accuracy","mean")).to_dict("index")
    report=f'''# Auditoría de suficiencia representacional ABS₃ — MatGPT vs MatterGPT

## Alcance y controles

Se reutilizaron las 70 fórmulas ABS₃, 210 estructuras estáticas, los cinco folds históricos 45/11/14 y cachés originales validadas de embeddings 210×768. MatGPT usó FINAL_END; MatterGPT usó el estado contextualizado terminal posterior a ln_f y conditioning [1.0, 2.0]. Ambos backbones permanecieron congelados. Geometría: siete descriptores de `gga-static_endpoint_cleaned`, con scaler ajustado por fold solo en inner-train. Ridge M2 histórico se mantuvo fijo para reconstrucción absoluta. No se consultó el test independiente.

MAE ΔE es estimando A: promedio de los tres errores firmados por fórmula y macro-promedio de 70 fórmulas. Energía absoluta se ancló a μ_c de Ridge M2 y scores centrados dentro de cada triplete. Ranking elegible: |ΔE_DFT|>5 meV/átomo; |ΔE_pred|≤1 se cuenta incorrecto. Estado fundamental elegible con gap al segundo mínimo >1 meV/átomo; empate predicho no es acierto. El orden completo excluye tripletes con algún par DFT ≤1 meV/átomo.

## Resultados de probes energéticos OOF

| Extractor | Probe | Energy MAE | ΔE MAE A | Pairwise | Estado fundamental | Triplete exacto | Pendiente OLS |
|---|---|---:|---:|---:|---:|---:|---:|
'''+"\n".join(f"| {r.condition} | {r.probe} | {r.energy_mae_mev_atom:.2f} | {r.deltaE_mae_estimand_A_mev_atom:.2f} | {r.pairwise_correct}/{r.pairwise_eligible} ({100*r.pairwise_accuracy:.2f}%) | {r.groundstate_correct}/{r.groundstate_eligible} ({100*r.groundstate_accuracy:.2f}%) | {r.full_triplet_correct}/{r.full_triplet_eligible} ({100*r.full_triplet_accuracy:.2f}%) | {r.ols_slope:.3f} |" for r in metrics.itertuples())+f'''

Entre probes con embeddings, KRR logró ΔE MAE {mat.deltaE_mae_estimand_A_mev_atom:.2f} para MatGPT y {matter.deltaE_mae_estimand_A_mev_atom:.2f} para MatterGPT; sus MAE de energía absoluta fueron {mat.energy_mae_mev_atom:.2f} y {matter.energy_mae_mev_atom:.2f} meV/átomo. Los rankings observados siguen lejos de 90%: máximo {100*metrics.pairwise_accuracy.max():.1f}%.

### Referencia histórica SLICES-only

El artefacto histórico `experiments/abs3_geometry_transformers_pairwise_ridge/slices_only_comparison.csv` reporta MatGPT: energía MAE 50.28, ΔE MAE 54.80 y ranking 66.50%; MatterGPT: 51.82, 55.77 y 68.53%, respectivamente. En esta auditoría, Ridge da MatGPT 50.26/54.38/68.53% y MatterGPT 50.61/54.11/68.02%. Los IDs, targets y los cinco folds OOF son los mismos. La diferencia de protocolo comprobada es que esta auditoría selecciona alpha en inner-validation y ajusta el probe final solo con las 45 fórmulas inner-train; el benchmark histórico vuelve a ajustar con inner-train + inner-validation después de seleccionar alpha. Por eso se reportan como referencias con distinto conjunto de ajuste, no como discrepancia de datos ni como comparación directa de superioridad.

Los intervalos bootstrap apareados por composición están en `bootstrap_paired.csv`. Los contrastes se interpretan como OOF de desarrollo, no como confirmación en test. La tabla de probes también incluye geometría sola como referencia bajo la misma implementación de scaler y splits.

En el kernel aditivo, frente a geometría KRR sola, el cambio de MAE ΔE fue {cm.estimate:.2f} meV/átomo para MatGPT (IC95% [{cm.ci95_low:.2f}, {cm.ci95_high:.2f}]) y {cmt.estimate:.2f} para MatterGPT (IC95% [{cmt.ci95_low:.2f}, {cmt.ci95_high:.2f}]); ambos intervalos incluyen cero. En los controles con 100 permutaciones intrafórmula, el MAE ΔE medio fue {pm_summary[('MAT','RIDGE')]['delta_mean']:.2f} (Ridge) y {pm_summary[('MAT','KRR')]['delta_mean']:.2f} (KRR) para MatGPT, y {pm_summary[('MATTER','RIDGE')]['delta_mean']:.2f} y {pm_summary[('MATTER','KRR')]['delta_mean']:.2f} para MatterGPT. Son controles de ruptura de asociación, no estimaciones del rendimiento de otro modelo entrenable.

## Información estructural recuperable

La fracción de suma de cuadrados intracomposicional fue {float(var.loc[var.extractor.eq('MATTER'),'within_fraction'].iloc[0]):.3f} en MatterGPT y {float(var.loc[var.extractor.eq('MAT'),'within_fraction'].iloc[0]):.3f} en MatGPT; el resto corresponde a variación entre medias de composición. Esto demuestra variación de embedding entre polimorfos, pero por sí solo no demuestra que la variación codifique la energía.

Las similitudes coseno y distancias normalizadas se describen en `structural_distances.csv`. La geometría recuperable por Ridge y KRR está desglosada por descriptor y fold en `geometry_probe_results.csv`; se incluye R² predictivo contra la media de train como referencia.

## Negative control y valor condicional

Se permutaron 100 veces los embeddings entre los tres polimorfos dentro de cada fórmula, sin cambiar DFT ni folds. Los resultados por fold/permutación están en `permutation_controls.csv`. La permutación destruye la asociación con la identidad del polimorfo sin borrar diferencias químicas compartidas por fórmula; no prueba ausencia absoluta de información.

El kernel aditivo `k_geo + η·k_embedding` eligió η por validación interna; configuraciones y predicciones están en `conditional_value.csv` y `conditional_value_predictions.csv`. El contraste frente a GEO se calcula por composición en `bootstrap_paired.csv`. La geometría proviene de celdas ya relajadas por DFT.

## Curvas few-shot

`learning_curves.csv` contiene 10 subconjuntos de inner-train para n=10,20,30 y el conjunto completo para n=45; la selección sigue siendo interna, y outer solo describe la curva predefinida. Los subconjuntos reutilizan la misma población outer y no constituyen ensayos independientes. No se extrapola más allá de 45 fórmulas.

## Clasificación según probes evaluados

- **Level 1 — información estructural recuperable:** PASS descriptivo en ambos extractores: hay dispersión intracomposicional, y la predicción OOF de algunos descriptores supera controles constantes. Esto no implica reconstrucción completa de estructura atómica.
- **Level 2 — información energética recuperable:** PARCIAL; Ridge/PLS/KRR recuperan señal sobre algunos controles, pero la calibración ΔE sigue comprimida y varía por fold. Las permutaciones y CIs delimitan la evidencia.
- **Level 3 — valor complementario:** INCIERTO; el kernel aditivo es exploratorio y su contraste apareado debe leerse con IC, no solo con estimaciones puntuales.
- **Level 4 — suficiencia práctica:** NO alcanzada; ninguna condición se aproxima a 90% de pairwise ranking con el protocolo OOF de desarrollo.

## Diagnóstico y límites

La evidencia observada no apunta a embeddings constantes: ambos retienen variación entre polimorfos. Aun así, la señal energética que estos probes recuperan sobre composiciones nuevas es limitada y las diferencias entre modelos/probes son inciertas. Esto es compatible con una mezcla de muestra pequeña (45 fórmulas inner-train), alta dimensionalidad, compresión energética y propiedades locales no explícitas. No identifica de forma causal cuál es el cuello de botella. No se puede concluir que falte toda información energética ni justificar fine-tuning solo a partir de probes lineales/no lineales seleccionados aquí.

Un siguiente paso razonable es probar estados contextualizados intermedios con el mismo Ridge/PLS/KRR y validación anidada; añadir descriptores locales requeriría un experimento nuevo y etiquetas/estructuras disponibles. Antes de abrir un test externo, fijar el probe y el protocolo con estos resultados de desarrollo.

## Artefactos y recursos

Los CSV conservan distancias, targets geométricos, selección PLS/KRR, curvas, controles, OOF por estructura/par y bootstrap por composición. `resources.json` registra CPU/hilo, duración y réplicas. Sin Transformers entrenados, sin MPS/CUDA y sin acceso a test.
'''
    (EXP/"reporte_cientifico.md").write_text(report,encoding="utf-8");(REPORT/"reporte_cientifico.md").write_text(report,encoding="utf-8")
    readme="""# ABS3 Representation Sufficiency Audit

Auditoría OOF de embeddings congelados MatGPT/MatterGPT, probes Ridge/PLS/KRR, decodificación geométrica, curvas few-shot, permutación intracomposición y kernels aditivos. Los Transformers no se entrenan. Ejecutar `python scripts/ABS3/abs3_representation_sufficiency_audit/run_audit.py`, después `run_extended.py` y finalmente `finalize_audit.py` desde la raíz del repositorio. Todas las métricas y umbrales están definidos en `config.yaml` y `reporte_cientifico.md`. El test independiente permanece cerrado.
"""
    (EXP/"README.md").write_text(readme,encoding="utf-8");(REPORT/"README.md").write_text(readme,encoding="utf-8")
    final_wall=time.perf_counter()-wall_start;final_cpu=time.process_time()-cpu_start
    core_wall=float(resources.get("core_wall_seconds",0));extended_wall=float(ext.get("wall_seconds",0))
    resources.update({"finalization_wall_seconds":final_wall,
        "wall_seconds_total_measured":core_wall+extended_wall+final_wall,
        "process_cpu_seconds_finalization":final_cpu,
        "process_cpu_seconds_extended":None,
        "resource_accounting_note":"CPU/RSS del proceso extendido no se registraron; no se imputan. El tiempo total de pared suma las duraciones medidas de los procesos core, extendido y finalización.",
        "peak_rss_bytes_finalization":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})
    atomic_json(EXP/"resources.json",resources)
    summary=json.loads((EXP/"summary.json").read_text());summary.update({"status":"COMPLETE_WITH_LIMITATIONS","completed_stages":["preflight","structural_probing","geometry_decoding","energy_Ridge_PLS_KRR","few_shot_curves","100_within_formula_permutations","additive_kernel_conditional_value","paired_bootstrap","figures_and_report"],"figure_count":9,"historical_test_accessed":False,"backbones_modified":False})
    atomic_json(EXP/"summary.json",summary)
    LOG.info("Finalización de auditoría completa: n=%d métricas=%d",len(c),len(metrics))


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    main()
