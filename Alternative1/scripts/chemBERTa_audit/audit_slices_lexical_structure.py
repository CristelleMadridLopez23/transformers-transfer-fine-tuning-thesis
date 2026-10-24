#!/usr/bin/env python3
"""Auditoría exploratoria de corpus SLICES strategy=4.

No modifica corpus/tokenizer/checkpoint. Requiere Python 3.9+; transformers opcional.

Ejemplo:
 python audit_slices_lexical_structure.py --input alexandria_clean_all.csv \
 --tokenizer model-pretrained/chemberta-77m-mlm --require-valid \
 --out reports/slices_lexical_structure

Clasificación de segmentos de SLICES (strategy=4) es heurística y se reportan
anomalías para inspección; no pretende reemplazar el parser SLICES oficial.
"""
import argparse, collections, csv, hashlib, json, math, re, statistics
from pathlib import Path

PERIODIC = re.compile(r'^[o+\-]{3}$')
INTEGER = re.compile(r'^\d+$')
ELEMENT = re.compile(r'^(?:[A-Z][a-z]?)$')
BOOLEAN_TRUE = {'true','1','yes','y','t'}
HEADER_RE = re.compile(r'^[A-Za-z+\-]+$')

def truth(v): return str(v).strip().lower() in BOOLEAN_TRUE

def dump_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def percentile(vals,q):
    if not vals:return None
    x=sorted(vals); pos=(len(x)-1)*q;lo=int(pos);hi=min(lo+1,len(x)-1)
    return round(x[lo]+(x[hi]-x[lo])*(pos-lo),2)

def unit_kind(token):
    if PERIODIC.fullmatch(token):return 'periodic_triplet'
    if INTEGER.fullmatch(token):return 'integer'
    if ELEMENT.fullmatch(token):return 'element_candidate'
    if HEADER_RE.fullmatch(token):return 'header_or_other_alpha'
    return 'other'

def segment(tokens):
    """Localiza el primer bloque plausible de aristas: i j o+o.
    La cabecera completa previa se conserva sin asumir gramática exacta.
    """
    start=None
    for k in range(len(tokens)-2):
        if INTEGER.fullmatch(tokens[k]) and INTEGER.fullmatch(tokens[k+1]) and PERIODIC.fullmatch(tokens[k+2]):
            start=k;break
    if start is None:
        return tokens, [], ['no_edge_triple_detected']
    header=tokens[:start];tail=tokens[start:];flags=[]
    if len(tail)%3:flags.append('edge_tail_not_multiple_of_3')
    for k in range(0,len(tail)-2,3):
        if not (INTEGER.fullmatch(tail[k]) and INTEGER.fullmatch(tail[k+1]) and PERIODIC.fullmatch(tail[k+2])):
            flags.append('nonconforming_edge_triplet');break
    return header,tail,flags

def signature(values):
    return hashlib.sha256('\x1f'.join(values).encode('utf8')).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input',required=True,help='CSV con columna SLICES')
    ap.add_argument('--column',default='slices')
    ap.add_argument('--require-valid',action='store_true')
    ap.add_argument('--tokenizer',help='Ruta del tokenizer (opcional)')
    ap.add_argument('--out',default='reports/slices_lexical_structure')
    ap.add_argument('--max-examples',type=int,default=200,help='Máximo de ejemplos de anomalías')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tok=None;tok_error=None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,use_fast=True)
            if not getattr(tok,'is_fast',False):tok_error='Tokenizer cargado sin offsets: requiere fast tokenizer para verificar offsets.';tok=None
        except Exception as e:tok_error=f'{type(e).__name__}: {e}'
    data=Path(args.input)
    total=valid_n=excluded=empty=0
    global_counts=collections.Counter();section_counts=collections.defaultdict(collections.Counter)
    max_index_counts=collections.Counter();sizes=[];long_ids=[];flags_counter=collections.Counter()
    ids=collections.defaultdict(list);texts=collections.defaultdict(list)
    compositions=collections.defaultdict(list);prototypes=collections.defaultdict(list)
    fingerprint=collections.defaultdict(list);fingerprint_column=None
    examples=[];lexical_failed=[];offset_errors=[];collisions=collections.defaultdict(list)
    tok_summary=collections.Counter()
    with data.open('r',encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames or []
        if args.column not in fields:raise ValueError(f'No existe columna {args.column!r}. Columnas: {fields}')
        if args.require_valid and not {'slices_basic_valid','slices_valid'}.issubset(fields):
            raise ValueError('Faltan slices_basic_valid y/o slices_valid para --require-valid')
        id_col=next((v for v in ['mat_id','material_id','entry_id','id'] if v in fields),None)
        comp_col=next((v for v in ['reduced_formula','formula_pretty','formula','composition','composition_reduced'] if v in fields),None)
        proto_col=next((v for v in ['prototype_id','prototype','spacegroup_number','space_group'] if v in fields),None)
        fingerprint_column=next((v for v in ['structure_hash','structure_fingerprint','canonical_structure_hash'] if v in fields),None)
        for row_no,row in enumerate(reader,2):
            total+=1
            if args.require_valid and not (truth(row['slices_basic_valid']) and truth(row['slices_valid'])):
                excluded+=1;continue
            raw=(row.get(args.column) or '').strip()
            if not raw:empty+=1;continue
            valid_n+=1;units=raw.split(); global_counts.update(units)
            h,tail,flags=segment(units)
            section_counts['header'].update(h)
            for i in range(0,len(tail),3):
                block=tail[i:i+3]
                for j,u in enumerate(block): section_counts[('edge_index_a','edge_index_b','periodic_triplet')[j]].update([u])
            flags_counter.update(flags)
            if flags and len(examples)<args.max_examples:
                examples.append({'row':row_no,'id':row.get(id_col) if id_col else None,'flags':flags,'slices':raw[:1200]})
            inds=[int(u) for i,u in enumerate(tail) if i%3!=2 and INTEGER.fullmatch(u)]
            max_index_counts[max(inds) if inds else -1]+=1
            length=len(units)+2;sizes.append(length)
            if length>512 and len(long_ids)<args.max_examples:long_ids.append({'row':row_no,'id':row.get(id_col) if id_col else None,'length':length})
            key=raw
            identifier=str(row.get(id_col) or '').strip() if id_col else ''
            if identifier:ids[identifier].append(row_no)
            texts[key].append((row_no,identifier))
            if comp_col and row.get(comp_col):compositions[row[comp_col].strip()].append(identifier or str(row_no))
            if proto_col and row.get(proto_col):prototypes[row[proto_col].strip()].append(identifier or str(row_no))
            if fingerprint_column and row.get(fingerprint_column):fingerprint[row[fingerprint_column].strip()].append(identifier or str(row_no))
            if tok is not None:
                try:
                    enc=tok(raw,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
                    tids=enc['input_ids'];offs=enc['offset_mapping'];tok_summary['observations']+=1
                    # Byte-level tokenize may produce offsets with overlaps; reconstruct source
                    # by covered character positions, not by comparing decode whitespace.
                    covered=set()
                    if len(tids)!=len(offs): tok_summary['mismatched_offset_lengths']+=1
                    for a,b in offs:
                        if 0<=a<b<=len(raw):covered.update(range(a,b))
                    required={i for i,c in enumerate(raw) if not c.isspace()}
                    missing=sorted(required-covered)
                    if missing:
                        tok_summary['samples_missing_nonspace_chars']+=1
                        tok_summary['missing_characters']+=len(missing)
                        if len(offset_errors)<args.max_examples:
                            offset_errors.append({'row':row_no,'id':identifier,'missing_chars':''.join(raw[i] for i in missing[:100]),'missing_positions':missing[:100],'text':raw[:1200]})
                    # Real independent lexical test: tokenizer IDs produced for an isolated unit
                    # are not compared with IDs in context; contextual BPE may vary.
                    tok_summary['unknown_tokens']+=sum(1 for t in tids if t==tok.unk_token_id)
                    collisions[signature([str(x) for x in tids])].append((row_no,identifier,raw,tuple(tids)))
                except Exception as e:
                    tok_summary['tokenize_error']+=1
                    if len(lexical_failed)<args.max_examples:lexical_failed.append({'row':row_no,'id':identifier,'error':str(e)})
            if valid_n%10000==0:print(f'[PROGRESO] {valid_n:,} cadenas auditadas',flush=True)
    collision_rows=[];colliding_inputs=0;collision_group_count=0
    for items in collisions.values():
        unique_by_text={s:(line,ident,code) for line,ident,s,code in items}
        if len(unique_by_text)>1:
            collision_group_count+=1;colliding_inputs+=len(unique_by_text)
            if len(collision_rows)<args.max_examples:
                sample=list(unique_by_text.items())[:3]
                collision_rows.append({'distinct_texts':len(unique_by_text),'examples':[
                    {'row':r,'id':ident,'slices':s[:1200]} for s,(r,ident,_) in sample]})
    vocab_rows=[]
    for u,n in global_counts.most_common():
        row={'unit':u,'category_lexical':unit_kind(u),'frequency':n,'header_frequency':section_counts['header'][u],
             'index_a_frequency':section_counts['edge_index_a'][u],
             'index_b_frequency':section_counts['edge_index_b'][u],
             'periodicity_frequency':section_counts['periodic_triplet'][u]}
        if tok is not None:
            row['in_exact_vocab']=int(u in tok.get_vocab())
            ids_unit=tok(u,add_special_tokens=False)['input_ids']
            row['isolated_token_count']=len(ids_unit)
            row['isolated_has_unknown']=int(tok.unk_token_id in ids_unit)
            row['isolated_decoded']=tok.decode(ids_unit,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        vocab_rows.append(row)
    csv_cols=['unit','category_lexical','frequency','header_frequency','index_a_frequency','index_b_frequency','periodicity_frequency']
    if tok is not None:csv_cols+=['in_exact_vocab','isolated_token_count','isolated_has_unknown','isolated_decoded']
    dump_csv(out/'lexical_units_by_role.csv',vocab_rows,csv_cols)
    dump_csv(out/'max_edge_index_distribution.csv', [{'max_edge_index':k,'structures':v} for k,v in sorted(max_index_counts.items())],['max_edge_index','structures'])
    dupids=[{'id':k,'rows':v[:20],'count':len(v)} for k,v in ids.items() if len(v)>1]
    dupstrings=[{'rows':[i for i,_ in v[:20]],'count':len(v),'ids':[i for _,i in v[:20]],'slices':s[:1200]} for s,v in texts.items() if len(v)>1]
    duplicate_summaries={
        'duplicate_mat_id_groups':len(dupids),'duplicate_exact_slices_groups':len(dupstrings),
        'same_composition_groups':sum(len(v)>1 for v in compositions.values()),
        'same_prototype_groups':sum(len(v)>1 for v in prototypes.values()),
        'duplicate_provided_structure_fingerprint_groups':sum(len(v)>1 for v in fingerprint.values()),
        'fingerprint_column':fingerprint_column,
        'structural_duplicates_verified':False,
        'note':'Coincidencia de composición/prototipo/SLICES no demuestra duplicación cristalográfica; para ello se requiere equivalencia estructural (p.ej. pymatgen StructureMatcher) y estructuras disponibles.'
    }
    summary={
        'input':str(data),'column':args.column,'filter_valid':args.require_valid,'rows_total':total,
        'excluded_by_validity':excluded,'eligible_nonempty':valid_n,'eligible_empty':empty,
        'unique_lexical_units':len(global_counts),'tokenizer_loaded':tok is not None,'tokenizer_load_error':tok_error,
        'length_hypothetical_one_unit_one_token_plus_2':{
            'median':percentile(sizes,.5),'p95':percentile(sizes,.95),'p99':percentile(sizes,.99),'max':max(sizes,default=0),
            'exceed_512':sum(x>512 for x in sizes),
            'exceed_512_pct':round(100*sum(x>512 for x in sizes)/len(sizes),3) if sizes else 0},
        'segmentation_flags':dict(flags_counter),'tokenizer_offset_tests':dict(tok_summary),
        'different_strings_same_token_ids':{'groups':collision_group_count,'distinct_affected_strings':colliding_inputs,'examples_capped':True},
        'duplication_review':duplicate_summaries,
        'caveats':['La posición inicial del bloque de aristas se detecta heurísticamente como índice índice tripleta periódica.',
                   'No confundir unidades ausentes del vocabulario con pérdida de caracteres.',
                   'La cobertura por offsets es una prueba de caracteres, no prueba de reconstrucción de semántica o ausencia de colisión.',
                   'Las longitudes son hipotéticas para 1 unidad léxica = 1 token; se debe medir longitud real del tokenizer candidato.',
                   'La igualdad de SLICES, composición o prototipo no es verificación cristalográfica de duplicados.']}
    for name,obj in [('summary.json',summary),('segmentation_examples.json',examples),('offset_missing_characters.json',offset_errors),('token_id_collisions.json',collision_rows),('duplicate_ids.json',dupids[:args.max_examples]),('duplicate_slices.json',dupstrings[:args.max_examples]),('tokenizer_errors.json',lexical_failed),('long_sequences_examples.json',long_ids)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    print('\n=== RESUMEN ===')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f'\nReportes en: {out.resolve()}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
