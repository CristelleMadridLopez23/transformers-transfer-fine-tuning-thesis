"""Extrae descriptores exclusivamente de geometría de celda cristalina real.
Requiere pymatgen: pip install pymatgen.
Entradas alternativas:
  --manifest-cif CSV con structure_id,path,geometry_source (CIF/POSCAR/JSON Structure)
  --structures-jsonl JSONL con structure_id,structure (pymatgen dict),geometry_source
No consulta etiquetas DFT ni crea geometrías artificiales.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import pandas as pd
from core import GEOM_COLS,load_development,sha256

def structure_descriptor(structure):
    # N original se valida antes de reducir la celda.
    primitive=structure.get_primitive_structure(tolerance=0.01)
    lattice=primitive.lattice.get_niggli_reduced_lattice(tol=1e-5)
    a,b,c=lattice.abc
    alpha,beta,gamma=lattice.angles
    base=lattice.volume**(1.0/3.0)
    if not np.isfinite(base) or base<=0:raise ValueError('Volumen inválido')
    # Shape adimensional + escala volumétrica por átomo original.
    return dict(zip(GEOM_COLS,[a/base,b/base,c/base,alpha,beta,gamma,
                                structure.volume/len(structure)]))

def export(dataset,output,manifest_cif=None,structures_jsonl=None):
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise SystemExit('Falta pymatgen: pip install pymatgen') from exc
    d=load_development(Path(dataset))
    idset=set(d.structure_id)
    entries={}
    if (manifest_cif is None)==(structures_jsonl is None):
        raise ValueError('Elegir exactamente uno de --manifest-cif o --structures-jsonl')
    if manifest_cif:
        manifest=Path(manifest_cif).resolve()
        df=pd.read_csv(manifest)
        for _,r in df.iterrows():
            sid=str(r['structure_id'])
            if sid not in idset:continue
            if sid in entries:raise ValueError('ID duplicado '+sid)
            path=Path(str(r['path']))
            if not path.is_absolute():path=(manifest.parent/path).resolve()
            entries[sid]=(Structure.from_file(str(path)),str(r['geometry_source']),str(path),sha256(path))
    else:
        path=Path(structures_jsonl).resolve()
        with path.open(encoding='utf-8') as f:
            for lineno,line in enumerate(f,1):
                if not line.strip():continue
                row=json.loads(line);sid=str(row.get('structure_id'))
                if sid not in idset:continue
                if sid in entries:raise ValueError('ID duplicado '+sid)
                if 'structure' not in row:raise ValueError('Sin structure '+sid)
                structure=Structure.from_dict(row['structure'])
                entries[sid]=(structure,str(row['geometry_source']),f'{path}:line:{lineno}',sha256(path))
    missing=idset-set(entries)
    if missing:raise ValueError(f'Faltan {len(missing)} estructuras; ejemplos: {sorted(missing)[:8]}')
    ref=d.set_index('structure_id')
    out=[]
    for sid in d.structure_id:
        structure,kind,path,src_hash=entries[sid]
        if len(structure)!=int(ref.loc[sid,'n_atoms']):
            raise ValueError(f'N_atoms distintos para {sid}: structure {len(structure)} vs dataset {ref.loc[sid,"n_atoms"]}')
        if not kind or kind.lower() in ('none','nan','unknown'):
            raise ValueError('Declarar geometry_source real para '+sid)
        desc=structure_descriptor(structure)
        out.append(dict(structure_id=sid,n_atoms=len(structure),geometry_source=kind,
                        source_path=path,source_sha256=src_hash,**desc))
    result=pd.DataFrame(out)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(f'No sobrescribir geometría existente: {output}')
    result.to_csv(output,index=False)
    manifest=dict(status='PASS',entries=len(result),source_counts=result.geometry_source.value_counts().to_dict(),
                  geometry_csv_sha256=sha256(output),descriptor_names=list(GEOM_COLS),
                  cell_reduction='pymatgen primitive then Niggli reduction',
                  shape='a,b,c divided by cubic root of reduced primitive volume',
                  volume='original cell volume per atom',
                  note='Only metric lattice descriptors, not atomic local distances or forces')
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest-cif',type=Path)
    inputs.add_argument('--structures-jsonl',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();export(a.dataset,a.output,a.manifest_cif,a.structures_jsonl)
if __name__=='__main__':main()
