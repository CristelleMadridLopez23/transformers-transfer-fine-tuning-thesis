"""Preflight read-only: verifica artefactos MatGPT H3-C bloqueados y datos geométricos."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from core import EXPECTED_CHECKPOINT_HASH,EXPECTED_CACHE_HASH,sha256,load_development,load_features,load_folds,load_geometry,GEOM_COLS
from entrenar import PATHS

def inspect(root,geo):
    root=Path(root).resolve(); geo=Path(geo).resolve()
    paths={k:root/rel for k,rel in PATHS.items() if k!='output'}
    for key in ('dataset','folds','cache','checkpoint','historic'):
        if not paths[key].exists():raise FileNotFoundError(f'Archivo faltante {key}: {paths[key]}')
    if sha256(paths['checkpoint'])!=EXPECTED_CHECKPOINT_HASH:raise ValueError('Checkpoint incorrecto')
    data=load_development(paths['dataset']);feat=load_features(paths['cache'],data.structure_id.tolist())
    folds=load_folds(paths['folds'],data);g=load_geometry(geo,data)
    df=pd.read_csv(geo)
    provenance=df.geometry_source.value_counts().to_dict()
    if len(provenance)>1:
        raise ValueError('Fuentes geométricas heterogéneas: '+str(provenance)+
                         '. Homogeneizar antes del experimento para evitar un factor de confusión.')
    geo_index=pd.DataFrame(g,columns=GEOM_COLS)
    geo_index['formula']=data.composition_id.to_numpy()
    variable_by_formula={formula:bool((part[list(GEOM_COLS)].max()-part[list(GEOM_COLS)].min()).max()>1e-7)
                         for formula,part in geo_index.groupby('formula')}
    if not any(variable_by_formula.values()):
        raise ValueError('Los descriptores son idénticos entre polimorfos de todas las fórmulas; no aportan señal intrafórmula')
    base=pd.read_csv(paths['historic'])
    if len(base)!=630 or not set(base.structure_id)==set(data.structure_id):raise ValueError('OOF histórico no verificado')
    for f in range(5):
        ridge=paths['ridge']/f'fold_{f}/ridge.joblib'
        if not ridge.exists():raise FileNotFoundError(ridge)
    out=dict(status='PASS',n_structures=len(data),n_compositions=data.composition_id.nunique(),
             geometry_columns=list(GEOM_COLS),geometry_sources=provenance,
             formulas_with_geometry_variation=sum(variable_by_formula.values()),
             geometry_ranges={k:[float(np.min(g[:,i])),float(np.max(g[:,i]))] for i,k in enumerate(GEOM_COLS)},
             no_test_energies_read=True,checkpoint_sha256=sha256(paths['checkpoint']),
             features_sha256=sha256(paths['cache']),geometry_sha256=sha256(geo),
             feature_cache_bitwise_historical=sha256(paths['cache'])==EXPECTED_CACHE_HASH,
             feature_cache_numeric_audit=(json.loads(paths['cache'].with_suffix('.numeric_audit.json').read_text())
               if sha256(paths['cache'])!=EXPECTED_CACHE_HASH else None),
             note='El baseline histórico emplea energía por átomo, no energía total como target.')
    print(json.dumps(out,indent=2,ensure_ascii=False))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--geometry-csv',type=Path,required=True)
    a=p.parse_args();inspect(a.repo_root,a.geometry_csv)
if __name__=='__main__':main()
