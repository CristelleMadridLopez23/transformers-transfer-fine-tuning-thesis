#!/usr/bin/env python3
"""Audita diversidad y colisiones energéticas en SLICES de trayectorias ABS3."""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]; ALT=ROOT/'Alternative2'
SOURCE=ALT/'01_data'/'abs3'/'processed'/'only_hex_new_csv_slices'/'configurations_slices.csv'
OUT=ALT/'reports'/'MatGPT_finetuning'/'matgpt_e2_t0'
def main():
    """Genera auditoría descriptiva con fuentes locales y sin entrenamiento."""
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=SOURCE); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    d=pd.read_csv(a.input,low_memory=False)
    seqcol=next((c for c in ('matgpt_slices','SLICES','slices_model_input') if c in d),None)
    ecol=next((c for c in ('energy_per_atom_ev','energy_dft_ev_atom','energy_per_atom_eV') if c in d),None)
    if seqcol is None or ecol is None: raise ValueError('Se requieren SLICES y energía DFT por átomo identificables.')
    d=d[d[seqcol].notna() & pd.to_numeric(d[ecol],errors='coerce').notna()].copy(); d[ecol]=pd.to_numeric(d[ecol],errors='coerce')
    d['sequence']=d[seqcol].astype(str).str.strip(); d['token_length']=d.sequence.map(lambda x:len(x.split()))
    d['sequence_sha256']=d.sequence.map(lambda x:__import__('hashlib').sha256(x.encode()).hexdigest())
    key=next((c for c in ('config_id','structure_id','source_key') if c in d),None)
    if key is None: d['frame_key']=d.index.astype(str); key='frame_key'
    groupcol=next((c for c in ('reduced_formula','formula','composition') if c in d),None)
    if groupcol is None: d['composition_group']='unknown'; groupcol='composition_group'
    collisions=d.groupby('sequence_sha256').agg(n_frames=(key,'size'),energy_min=(ecol,'min'),energy_max=(ecol,'max'),composition=(groupcol,'first'),sequence=('sequence','first')).reset_index()
    collisions['delta_e_collision_mev_atom']=(collisions.energy_max-collisions.energy_min)*1000
    collisions['classification']=np.where(collisions.n_frames.eq(1),'unique_sequence',np.where(collisions.delta_e_collision_mev_atom<1,'same_sequence_energy_near_equal','same_sequence_distinct_energy'))
    d['tokenization_status']=np.where(d.token_length.le(442),'length_within_block_size','exceeds_block_size')
    a.out.mkdir(parents=True,exist_ok=True); d.to_csv(a.out/'frame_tokenization.csv',index=False); collisions.to_csv(a.out/'slices_collisions.csv',index=False)
    by=d.groupby(groupcol).agg(frames=(key,'size'),unique_sequences=('sequence_sha256','nunique'),mean_token_length=('token_length','mean')).reset_index(); by.to_csv(a.out/'trajectory_statistics.csv',index=False)
    valid=len(d); uniq=d.sequence_sha256.nunique(); summary={'status':'LIMITED','input':str(a.input.resolve()),'frames_valid':valid,'unique_token_sequences':int(uniq),'unique_ratio':float(uniq/valid) if valid else None,'frames_over_442_tokens':int((d.token_length>442).sum()),'collision_groups':int((collisions.n_frames>1).sum()),'distinct_energy_collision_groups':int((collisions.delta_e_collision_mev_atom>=1).sum()),'topology_equivalence':'not certified by token equality; periodic graph comparison remains pending','label_source_column':ecol,'interpretation':'SLICES idénticos con energía distinta son colisiones no distinguibles por el predictor basado solo en IDs.'}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n'); (a.out/'E2_T0_report.md').write_text('# E2-T0 — auditoría de trayectorias\n\nEstado: LIMITED. La identidad de secuencia no certifica equivalencia de grafos periódicos. Se usaron etiquetas disponibles en la fuente; no se entrenó, no se generaron etiquetas y no se usó NequIX.\n\n```json\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n```\n')
    logging.info('trajectory_audit_complete valid=%d unique=%d collisions=%d',valid,uniq,summary['collision_groups'])
if __name__=='__main__': main()
