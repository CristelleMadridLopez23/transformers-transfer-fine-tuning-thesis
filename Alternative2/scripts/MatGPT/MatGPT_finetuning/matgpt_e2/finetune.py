#!/usr/bin/env python3
"""Preflight y entrenamiento E2-A de MatGPT con un único bloque ajustable."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, logging, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[4]; ALT=ROOT/'Alternative2'
OUT=ALT/'reports'/'MatGPT_finetuning'/'matgpt_e2'
E13=ALT/'reports'/'MatGPT_frozen'/'matgpt_e1_3'
DATA=ALT/'reports'/'MatGPT_frozen'/'matgpt_e1_v2'/'dataset.csv'
CKPT=ALT/'models'/'MatGPT-source'/'mat-concat'/'mat-uncond'/'weights'/'transformer_property_0.pth'
INFO=CKPT.with_name('model_info_property_0.json'); SOURCE=ALT/'models'/'MatGPT-source'/'mat-concat'
EXPECTED='4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414'
SOURCE_MANIFEST=ALT/'models'/'MatGPT-source.download-manifest.json'
SEEDS=(42,123,2026); LOG=logging.getLogger('matgpt_e2')
def sync_device(device):
    """Sincroniza CUDA/MPS antes de tomar tiempos de pared."""
    if device.type=='cuda': torch.cuda.synchronize(device)
    elif device.type=='mps' and hasattr(torch,'mps'): torch.mps.synchronize()
class Head(nn.Module):
    """Cabeza MLP8 con GELU y dropout histórico."""
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(768,8),nn.GELU(),nn.Dropout(.1),nn.Linear(8,1))
    def forward(self,x): return self.net(x).squeeze(-1)
def sha(p):
    """Calcula SHA-256 de un archivo."""
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def setup():
    """Carga clase del commit local fijado y crea decoder unconditional estricto."""
    sys.path.insert(0,str(SOURCE)); from utils.models import ModifiedTransformerDecoder
    info=json.loads(INFO.read_text()); stoi=info['stoi']; cfg={'vocab_size':len(stoi),'n_embd':768,'n_head':12,'n_layer':12,'block_size':int(info['block_size']),'dropout':.1,'stoi':stoi,'conditioning_mode':'unconditional'}
    model=ModifiedTransformerDecoder(**cfg)
    raw=torch.load(CKPT,map_location='cpu',weights_only=True)
    state=raw
    model.load_state_dict(state,strict=True); return model,stoi
def encode(text,stoi,device):
    """Tokeniza con el vocabulario histórico e incorpora tokens especiales."""
    ids=[stoi['<start>']]+[stoi[t] for t in str(text).split()]+[stoi['<end>']]
    return ids
def end_positions(ids,end_token_id):
    """Devuelve la posición real de un único <end> por secuencia."""
    end_mask=ids.eq(end_token_id)
    counts=end_mask.sum(dim=1)
    if not bool(torch.all(counts.eq(1)).item()):
        raise RuntimeError('Cada secuencia debe contener exactamente un <end>.')
    positions=end_mask.to(dtype=torch.long).argmax(dim=1)
    if bool(torch.any(positions.eq(0)).item()):
        raise RuntimeError('La posición <end> no puede coincidir con <start> en índice 0.')
    return positions
def forward_hidden(model,ids,pad_id,trainable):
    """Replica el forward causal original y devuelve vector del token end."""
    B,T=ids.shape; dev=ids.device
    x=model.token_embedding_table(ids)+model.position_embedding_table(torch.arange(T,device=dev))
    x=x.permute(1,0,2); mask=model.generate_square_subsequent_mask(T).to(dev); mem=torch.zeros((T,B,model.n_embd),device=dev)
    for i,layer in enumerate(model.transformer_decoder.layers):
        if i==len(model.transformer_decoder.layers)-1 and trainable: x=layer(x,mem,tgt_mask=mask)
        else:
            with torch.no_grad(): x=layer(x,mem,tgt_mask=mask)
    x=model.ln_f(x.permute(1,0,2)); positions=end_positions(ids,model.stoi['<end>'])
    return x[torch.arange(B,device=dev),positions]
def batch(ids,indices,pad,device):
    """Crea minibatch con padding derecho y mantiene grupos completos."""
    width=max(len(ids[int(i)]) for i in indices); out=torch.full((len(indices),width),pad,dtype=torch.long,device=device)
    for j,i in enumerate(indices): out[j,:len(ids[int(i)])]=torch.tensor(ids[int(i)],dtype=torch.long,device=device)
    return out
def smoke_representation_checks(model,head,sequences,stoi,device,trainable):
    """Comprueba end pooling, diversidad de tres entradas y padding derecho."""
    encoded={str(s):encode(str(s),stoi,device) for s in pd.unique(pd.Series(sequences).dropna())}
    choices=sorted(encoded.items(),key=lambda item:len(item[1]))
    if len(choices)<3 or len(choices[0][1])==len(choices[-1][1]):
        raise RuntimeError('Smoke requiere tres SLICES distintos y longitudes variadas para probar padding.')
    chosen=[choices[0],choices[len(choices)//2],choices[-1]]
    seq_ids=[item[1] for item in chosen]; pad=stoi['<pad>']; xb=batch(seq_ids,np.arange(3),pad,device)
    positions=end_positions(xb,stoi['<end>'])
    expected=torch.tensor([len(item)-1 for item in seq_ids],device=device)
    if not torch.equal(positions,expected): raise RuntimeError('Smoke: el pooling no apunta a la posición real de <end>.')
    model.eval(); head.eval()
    with torch.no_grad():
        hidden=forward_hidden(model,xb,pad,trainable); pred=head(hidden)
        single=batch(seq_ids,np.array([0]),pad,device)
        single_hidden=forward_hidden(model,single,pad,trainable); single_pred=head(single_hidden)
    hidden_spread=float(torch.pdist(hidden.float().cpu()).max())
    prediction_spread=float((pred.max()-pred.min()).cpu())
    pad_hidden_error=float((single_hidden[0]-hidden[0]).abs().max().cpu())
    pad_prediction_error=float((single_pred[0]-pred[0]).abs().cpu())
    if hidden_spread<=1e-6 or prediction_spread<=1e-8:
        raise RuntimeError(f'Smoke: colapso en entradas distintas; embedding_spread={hidden_spread:.3e}, prediction_spread={prediction_spread:.3e}.')
    if pad_hidden_error>1e-5 or pad_prediction_error>1e-5:
        raise RuntimeError(f'Smoke: padding derecho alteró el embedding/predicción; hidden_error={pad_hidden_error:.3e}, prediction_error={pad_prediction_error:.3e}.')
    return {'sample_sequence_lengths':[len(item) for item in seq_ids],'end_positions':positions.cpu().tolist(),'embedding_max_pairwise_distance':hidden_spread,'prediction_range':prediction_spread,'padding_hidden_max_abs_error':pad_hidden_error,'padding_prediction_abs_error':pad_prediction_error}
def objective(pred,y,groups,target_scale,pair_scale):
    """Huber de energía normalizada y pares intracomposición según E1.3."""
    absolute=nn.functional.smooth_l1_loss((pred-y)/target_scale,torch.zeros_like(y),beta=1.)
    terms=[]
    for g in sorted(set(groups)):
        ix=[i for i,x in enumerate(groups) if x==g]
        if len(ix)>1:
            a,b=torch.triu_indices(len(ix),len(ix),1,device=pred.device)
            terms.append(nn.functional.smooth_l1_loss(((pred[ix][a]-pred[ix][b])-(y[ix][a]-y[ix][b]))/pair_scale,torch.zeros_like(a,dtype=pred.dtype),beta=1.))
    return absolute+(torch.stack(terms).mean() if terms else pred.sum()*0)
def fit(data,assign,stoi,device,fold,seed,mode,phase):
    """Entrena una cabeza con último bloque opcionalmente ajustable; sin test."""
    phase_dir={'smoke':'smoke_endtoken_v2','pilot':'pilot_endtoken_v2','cv':'runs_endtoken_v2'}.get(phase,'runs_endtoken_v2')
    dest=OUT/phase_dir/str(device)/mode/f'fold_{fold}_seed_{seed}.pt'
    if dest.exists(): raise RuntimeError(f'Ya existe {dest}; se conserva para evitar sobrescritura.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model,_=setup(); model.to(device)
    for p in model.parameters(): p.requires_grad=False
    partial=mode=='partial_ft'
    if partial:
        for p in model.transformer_decoder.layers[-1].parameters(): p.requires_grad=True
    head=Head().to(device); pad=stoi['<pad>']; end=stoi['<end>']
    ids=[encode(s,stoi,device) for s in data.matgpt_slices]
    if max(map(len,ids))>int(json.loads(INFO.read_text())['block_size']): raise RuntimeError('Secuencia excede block_size.')
    y=torch.tensor(data.energy_per_atom_ev.to_numpy(float),dtype=torch.float32,device=device)
    groups=data.composition_group.astype(str).tolist()
    role=assign.set_index('structure_id')
    inner_train=np.flatnonzero(data.structure_id.astype(str).map(role.inner_role).eq('inner_train').to_numpy())
    inner_val=np.flatnonzero(data.structure_id.astype(str).map(role.inner_role).eq('inner_validation').to_numpy())
    outer=np.flatnonzero(data.structure_id.astype(str).map(role.outer_role).eq('outer_validation').to_numpy())
    if (len(inner_train),len(inner_val),len(outer))!=(135,33,42): raise RuntimeError(f'Conteos fold {fold} inválidos.')
    trgroups=set(data.iloc[inner_train].composition_group); vgroups=set(data.iloc[inner_val].composition_group); ogroups=set(data.iloc[outer].composition_group)
    if trgroups&vgroups or trgroups&ogroups or vgroups&ogroups: raise RuntimeError('Fuga de composición.')
    ytr=data.iloc[inner_train].energy_per_atom_ev.to_numpy(float); mean=float(ytr.mean()); scale=max(float(ytr.std()),1e-6)
    diffs=[]
    for _,g in data.iloc[inner_train].groupby('composition_group'):
        z=g.energy_per_atom_ev.to_numpy(float); diffs.extend(z[i]-z[j] for i in range(len(z)) for j in range(i+1,len(z)))
    pair=max(float(np.sqrt(np.mean(np.square(diffs)))),.001)
    opt=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'lr':1e-5},{'params':head.parameters(),'lr':1e-3}],weight_decay=1e-4)
    best=float('inf'); best_epoch=0; best_state=None; wait=0; history=[]; rng=np.random.default_rng(seed)
    gradient_audit={}
    epochs=3 if phase=='smoke' else 100; patience=15 if phase=='smoke' else 20
    sync_device(device); training_started=time.perf_counter()
    for epoch in range(1,epochs+1):
        sync_device(device); epoch_started=time.perf_counter()
        model.eval(); head.train();
        if partial: model.transformer_decoder.layers[-1].train()
        order=rng.permutation(sorted(trgroups)); losses=[]
        for st in range(0,len(order),4):
            gs=set(order[st:st+4]); ix=np.array([i for i in inner_train if groups[i] in gs],dtype=int)
            xb=batch(ids,ix,pad,device); opt.zero_grad(set_to_none=True)
            h=forward_hidden(model,xb,pad,partial); pred=head(h)*scale+mean
            loss=objective(pred,y[ix],[groups[i] for i in ix],scale,pair)
            if not bool(torch.isfinite(loss).item()):
                diagnostics={'variant':mode,'fold':fold,'seed':seed,'epoch':epoch,'device':str(device),
                    'ids':data.iloc[ix].structure_id.astype(str).tolist(),
                    'hidden_all_finite':bool(torch.isfinite(h).all().item()),
                    'prediction_all_finite':bool(torch.isfinite(pred).all().item()),
                    'target_all_finite':bool(torch.isfinite(y[ix]).all().item()),
                    'target_mean':mean,'target_scale':scale,'pair_scale':pair,
                    'hidden_minmax':([float(h.min().detach().cpu()),float(h.max().detach().cpu())] if bool(torch.isfinite(h).all().item()) else None),
                    'prediction_values':(pred.detach().cpu().tolist() if bool(torch.isfinite(pred).all().item()) else None),
                    'target_values':y[ix].detach().cpu().tolist()}
                raise RuntimeError(f'Loss NaN/Inf; diagnóstico del primer lote: {json.dumps(diagnostics,ensure_ascii=False)}')
            loss.backward()
            if phase=='smoke' and epoch==1 and st==0:
                block_grads=[p.grad for p in model.transformer_decoder.layers[-1].parameters()]
                if not any(g is not None and torch.isfinite(g).all() and float(g.norm())>0 for g in block_grads): raise RuntimeError('Smoke: el último bloque no recibió gradiente finito y no nulo.')
                if any(param.grad is not None for name,param in model.named_parameters() if not name.startswith('transformer_decoder.layers.11.')): raise RuntimeError('Smoke: se detectó gradiente en parámetros congelados.')
                head_grads=[p.grad for p in head.parameters()]
                if not any(g is not None and torch.isfinite(g).all() and float(g.norm())>0 for g in head_grads): raise RuntimeError('Smoke: la cabeza no recibió gradiente finito y no nulo.')
                gradient_audit={'last_block_gradient_norm':float(torch.sqrt(sum(g.detach().float().pow(2).sum() for g in block_grads if g is not None)).cpu()),
                    'head_gradient_norm':float(torch.sqrt(sum(g.detach().float().pow(2).sum() for g in head_grads if g is not None)).cpu()),
                    'frozen_parameters_with_grad':0}
                before=[p.detach().clone() for p in model.transformer_decoder.layers[-1].parameters()]
                head_before=[p.detach().clone() for p in head.parameters()]
            torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters())+list(head.parameters()) if p.requires_grad],1.); opt.step()
            if phase=='smoke' and epoch==1 and st==0:
                block_updates=[(a-b.detach()).float().pow(2).sum() for a,b in zip(before,model.transformer_decoder.layers[-1].parameters())]
                head_updates=[(a-b.detach()).float().pow(2).sum() for a,b in zip(head_before,head.parameters())]
                gradient_audit['last_block_update_norm']=float(torch.sqrt(sum(block_updates)).cpu())
                gradient_audit['head_update_norm']=float(torch.sqrt(sum(head_updates)).cpu())
                if gradient_audit['last_block_update_norm']==0 or gradient_audit['head_update_norm']==0: raise RuntimeError('Smoke: la cabeza o el último bloque no cambió después del optimizer.step().')
            losses.append(float(loss.detach()))
        model.eval(); head.eval()
        with torch.no_grad():
            xv=batch(ids,inner_val,pad,device); pv=head(forward_hidden(model,xv,pad,partial))*scale+mean
            vl=float(objective(pv,y[inner_val],[groups[i] for i in inner_val],scale,pair))
        sync_device(device); epoch_seconds=time.perf_counter()-epoch_started
        history.append({'variant':mode,'fold':fold,'seed':seed,'epoch':epoch,'train_objective':float(np.mean(losses)),'inner_validation_objective':vl,'target_mean':mean,'target_scale':scale,'pair_scale':pair,'epoch_wall_seconds':epoch_seconds,'training_elapsed_seconds':time.perf_counter()-training_started})
        if vl<best-1e-12:
            best=vl; best_epoch=epoch; wait=0
            best_state={'backbone':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'head':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}
        else: wait+=1
        if wait>=patience: break
    if best_state is None: raise RuntimeError('No se seleccionó estado por inner validation.')
    sync_device(device); training_seconds=time.perf_counter()-training_started
    model.load_state_dict(best_state['backbone'],strict=True); head.load_state_dict(best_state['head'],strict=True); model.eval(); head.eval()
    smoke_audit=None
    if phase=='smoke': smoke_audit={**gradient_audit,**smoke_representation_checks(model,head,data.matgpt_slices,stoi,device,partial)}
    with torch.no_grad():
        pred=head(forward_hidden(model,batch(ids,outer,pad,device),pad,partial))*scale+mean
    part=data.iloc[outer][['structure_id','composition_group','reduced_formula','polymorph_family','energy_per_atom_ev']].copy()
    part['configuration']=mode; part['fold']=fold; part['seed']=seed; part['best_epoch']=best_epoch; part['energy_pred_ev_atom']=pred.cpu().numpy()
    part['training_wall_seconds']=training_seconds
    part['error_ev_atom']=part.energy_pred_ev_atom-part.energy_per_atom_ev
    part['delta_e_dft_mev_atom']=part.groupby('composition_group').energy_per_atom_ev.transform(lambda v:(v-v.min())*1000)
    part['delta_e_pred_mev_atom']=part.groupby('composition_group').energy_pred_ev_atom.transform(lambda v:(v-v.min())*1000)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise RuntimeError(f'Ya existe {dest}; se conserva para evitar sobrescritura.')
    torch.save({'model':best_state['backbone'] if partial else None,'head':best_state['head'],'variant':mode,'fold':fold,'seed':seed,'best_epoch':best_epoch,'inner_validation_objective':best,'target_mean':mean,'target_scale':scale,'pair_scale':pair,'training_wall_seconds':training_seconds,'checkpoint_sha256':EXPECTED},dest)
    if smoke_audit is not None:
        audit_path=OUT/'smoke_endtoken_v2'/str(device)/'representation_integrity.json'; audit_path.parent.mkdir(parents=True,exist_ok=True)
        audit_path.write_text(json.dumps({'status':'PASS',**smoke_audit},indent=2)+'\n')
    for row in history: row['training_wall_seconds']=training_seconds
    return history,part
def main():
    """Valida entradas, luego ejecuta fase solicitada y registra artefactos."""
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['preflight','smoke','pilot','cv'],required=True); ap.add_argument('--device',default='auto'); ap.add_argument('--fold',type=int,default=0); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--model',choices=['both','frozen','partial_ft'],default='both'); a=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    for p in (DATA,CKPT,INFO,SOURCE/'utils'/'models.py',E13/'fold_assignments.csv'):
        if not p.is_file(): raise FileNotFoundError(p)
    if sha(CKPT)!=EXPECTED: raise RuntimeError('Checkpoint SHA-256 no coincide con E0.')
    if SOURCE_MANIFEST.is_file():
        manifest=json.loads(SOURCE_MANIFEST.read_text())
        if manifest.get('resolved_commit')!='a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e': raise RuntimeError('El código MatGPT local difiere del commit verificado E0.')
    data=pd.read_csv(DATA); folds=pd.read_csv(E13/'fold_assignments.csv')
    if len(data)!=234 or data.structure_id.astype(str).duplicated().any(): raise RuntimeError('Dataset E1 no íntegro.')
    dev=data[data.split.isin(['train','val'])].copy().reset_index(drop=True)
    folds=folds[folds.historical_split.isin(['train','val'])]
    expected_ids=set(dev.structure_id.astype(str))
    if len(dev)!=210: raise RuntimeError(f'Dataset de desarrollo inesperado: {len(dev)} estructuras; se esperaban 210.')
    if folds.structure_id.astype(str).nunique()!=210 or folds.outer_fold.nunique()!=5:
        raise RuntimeError('La tabla E1.3 debe contener las 210 estructuras de desarrollo en cinco folds.')
    if folds.groupby('structure_id').composition_group.nunique().max()!=1:
        raise RuntimeError('Una estructura recibe más de una composición entre folds.')
    for fold_id, fold_rows in folds.groupby('outer_fold'):
        fold_ids=set(fold_rows.structure_id.astype(str))
        role_counts=(int(fold_rows.outer_role.eq('outer_validation').sum()),
                     int(fold_rows.inner_role.eq('inner_train').sum()),
                     int(fold_rows.inner_role.eq('inner_validation').sum()))
        if fold_ids!=expected_ids or len(fold_rows)!=210:
            raise RuntimeError(f'Fold {fold_id} no cubre exactamente los mismos 210 IDs de desarrollo.')
        if role_counts!=(42,135,33):
            raise RuntimeError(f'Fold {fold_id} tiene conteos outer/inner inesperados: {role_counts}; esperado (42,135,33).')
    group_map=folds.drop_duplicates('structure_id').set_index('structure_id').composition_group
    dev['composition_group']=dev.structure_id.astype(str).map(group_map)
    if dev.composition_group.isna().any(): raise RuntimeError('Falta composición canonizada en folds.')
    if a.phase=='preflight':
        m,stoi=setup(); trainable=[]
        for n,p in m.named_parameters(): p.requires_grad=False
        for n,p in m.transformer_decoder.layers[-1].named_parameters(): p.requires_grad=True; trainable.append('transformer_decoder.layers.11.'+n)
        n_total=sum(p.numel() for p in m.parameters()); n_train=sum(p.numel() for p in m.parameters() if p.requires_grad)
        OUT.mkdir(parents=True,exist_ok=True); (OUT/'preflight.json').write_text(json.dumps({'status':'PASS','checkpoint_sha256':EXPECTED,'folds':int(folds.outer_fold.nunique()),'development_structures':len(dev),'historical_test_excluded':24,'source_commit':'a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e','parameters_total':n_total,'parameters_last_block':n_train,'trainable_tensor_names':['transformer_decoder.layers.11.'+n for n,p in m.transformer_decoder.layers[-1].named_parameters() if p.requires_grad]},indent=2)+'\n')
        LOG.info('preflight PASS sha=%s folds=%d dev=%d test_excluded=%d parameters_total=%d last_block=%d head=%d',EXPECTED,folds.outer_fold.nunique(),len(dev),24,n_total,n_train,sum(p.numel() for p in Head().parameters())); return
    device=torch.device('mps' if a.device=='auto' and torch.backends.mps.is_available() else 'cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    _,stoi=setup()
    marker=OUT/'preflight.json'
    if not marker.is_file(): raise RuntimeError('Primero ejecuta --phase preflight y revisa su auditoría.')
    if a.phase=='smoke': folds_to=[(0,42)]; modes=['partial_ft']
    elif a.phase=='pilot': folds_to=[(a.fold,a.seed)]; modes=['frozen','partial_ft']
    else: folds_to=[(f,s) for f in range(5) for s in SEEDS]; modes=['frozen','partial_ft']
    allh=[]; allp=[]
    phase_dir={'smoke':'smoke_endtoken_v2','pilot':'pilot_endtoken_v2','cv':'runs_endtoken_v2'}[a.phase]
    sub=OUT/phase_dir/str(device)
    if (sub/'oof_predictions.csv').exists(): raise RuntimeError(f'{sub} ya contiene resultados; se conserva esta versión, no se sobrescribe.')
    sync_device(device); phase_started=time.perf_counter()
    for f,s in folds_to:
        fa=folds[folds.outer_fold.eq(f)]
        fd=dev[dev.structure_id.astype(str).isin(fa.structure_id.astype(str))].reset_index(drop=True)
        for mode in (modes if a.model=='both' else [a.model]):
            h,p=fit(fd,fa,stoi,device,f,s,mode,a.phase); allh.extend(h); allp.append(p)
    sub.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(allh).to_csv(sub/'training_history.csv',index=False)
    predictions=pd.concat(allp,ignore_index=True); predictions.to_csv(sub/'oof_predictions.csv',index=False)
    sync_device(device); phase_seconds=time.perf_counter()-phase_started
    fit_times=predictions.groupby(['configuration','fold','seed']).training_wall_seconds.first().reset_index().to_dict(orient='records')
    (sub/'runtime_summary.json').write_text(json.dumps({'phase':a.phase,'device':str(device),'fit_count':len(folds_to)*len(modes),'phase_wall_seconds':phase_seconds,'phase_wall_minutes':phase_seconds/60,'per_fit_training_seconds':fit_times},indent=2)+'\n')
    LOG.info('%s_complete fits=%d predictions=%d device=%s wall_seconds=%.2f wall_minutes=%.2f',a.phase,len(folds_to)*len(modes),len(predictions),device,phase_seconds,phase_seconds/60)
if __name__=='__main__': main()
