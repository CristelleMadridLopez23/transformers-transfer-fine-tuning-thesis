#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clearance 05 (PROVISIONAL): audit eligible SLICES and make composition-disjoint
split candidates. No crystallographic equivalence checking is claimed.

Requires the original Alexandria clean CSV and the frozen Clearance 03 manifest.
All structures within a connected component of composition_key + reduced_formula
are assigned to exactly one split. This is deliberately stricter than mat_id split.
Do NOT treat outputs as approved before structure and vocabulary provenance audits.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ('train', 'val', 'test')

def hfile(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def truth(v):
    return str(v).strip().lower() in {'1', 'true', 'yes'}

def csv_rows(path):
    with Path(path).open('r', newline='', encoding='utf-8-sig') as f:
        yield from csv.DictReader(f)

def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

class DSU:
    def __init__(self):
        self.p = {}
    def find(self, x):
        if x not in self.p:
            self.p[x] = x
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            # Stable representative, independent of original data order.
            self.p[max(a, b)] = min(a, b)

def split_groups(keys, seed, train, val):
    keys = sorted(keys, key=lambda k: hashlib.sha256(f'{seed}|{k}'.encode()).hexdigest())
    n = len(keys)
    nt = int(round(n * train))
    nv = int(round(n * val))
    if nt + nv > n:
        nv = n - nt
    return {k: ('train' if j < nt else 'val' if j < nt+nv else 'test') for j,k in enumerate(keys)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Alexandria clean CSV original')
    p.add_argument('--manifest', required=True, help='Clearance 03 dapt_candidate_manifest_UNSPLIT.csv')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-frac', type=float, default=0.8)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--expect-fit', type=int, default=96719)
    args = p.parse_args()
    if not (0 < args.train_frac < 1 and 0 < args.val_frac < 1 and args.train_frac + args.val_frac < 1):
        p.error('Los porcentajes deben ser positivos y train+val < 1')
    source, manifest = Path(args.input), Path(args.manifest)
    if not source.is_file() or not manifest.is_file():
        p.error(f'Archivos inexistentes: source={source.exists()}, manifest={manifest.exists()}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = {}
    duplicate_manifest_ids = []
    for r in csv_rows(manifest):
        mid = r.get('mat_id', '').strip()
        if not mid:
            p.error('El manifest contiene mat_id vacío')
        if mid in wanted:
            duplicate_manifest_ids.append(mid)
        wanted[mid] = r
    if not wanted:
        p.error('Manifest vacío')

    eligible = []
    seen = Counter()
    mismatches = []
    source_fields = None
    with source.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        source_fields = set(reader.fieldnames or [])
        required = {'mat_id','slices','composition_key','reduced_formula'}
        if not required.issubset(source_fields):
            p.error(f'Faltan columnas del corpus: {sorted(required-source_fields)}')
        for r in reader:
            mid = r.get('mat_id','').strip()
            if mid not in wanted:
                continue
            seen[mid] += 1
            m = wanted[mid]
            if m.get('composition_key','').strip() != r.get('composition_key','').strip():
                mismatches.append({'mat_id':mid, 'issue':'composition_key mismatch'})
            if m.get('prototype_id','').strip() != r.get('prototype_id','').strip():
                mismatches.append({'mat_id':mid, 'issue':'prototype_id mismatch'})
            if not r.get('slices','').strip():
                mismatches.append({'mat_id':mid, 'issue':'empty SLICES'})
            if 'length_with_specials' in m:
                try:
                    if int(m['length_with_specials']) > 512:
                        mismatches.append({'mat_id':mid, 'issue':'length exceeds 512 in FIT manifest'})
                except ValueError:
                    mismatches.append({'mat_id':mid, 'issue':'bad length'})
            eligible.append(r)

    missing = sorted(set(wanted)-set(seen))
    repeated_source = sorted(mid for mid,c in seen.items() if c > 1)
    dsu = DSU()
    comp_to_reduced = defaultdict(set)
    reduced_to_comp = defaultdict(set)
    for r in eligible:
        ck = r.get('composition_key','').strip()
        rf = r.get('reduced_formula','').strip()
        if not ck:
            mismatches.append({'mat_id':r['mat_id'], 'issue':'empty composition_key'})
        if not rf:
            mismatches.append({'mat_id':r['mat_id'], 'issue':'empty reduced_formula'})
        if ck and rf:
            dsu.union('CK:'+ck, 'RF:'+rf)
            comp_to_reduced[ck].add(rf)
            reduced_to_comp[rf].add(ck)
        elif ck:
            dsu.find('CK:'+ck)
        else:
            dsu.find('MISSING:'+r['mat_id'])
    # Canonical grouping using both composition_key and reduced_formula aliases.
    grouped = defaultdict(list)
    for r in eligible:
        ck = r.get('composition_key','').strip()
        group_key = dsu.find('CK:'+ck) if ck else dsu.find('MISSING:'+r['mat_id'])
        grouped[group_key].append(r)
    polymorph_keys = [k for k,rs in grouped.items() if len(rs) >= 2]
    singleton_keys = [k for k,rs in grouped.items() if len(rs) == 1]
    assignment = {}
    for keys in (polymorph_keys, singleton_keys):
        assignment.update(split_groups(keys, args.seed, args.train_frac, args.val_frac))

    duplicate_slices = defaultdict(list)
    prototype_by_group = defaultdict(Counter)
    quality_counts = Counter()
    assignments = []
    for group_key, rs in grouped.items():
        part = assignment[group_key]
        for r in rs:
            mid = r['mat_id'].strip()
            s = r['slices'].strip()
            duplicate_slices[s].append((mid,part))
            prototype_by_group[group_key][r.get('prototype_id','').strip()] += 1
            if truth(r.get('has_quality_flags','')):
                quality_counts[part] += 1
            assignments.append({
                'mat_id':mid,'source_row_index':r.get('source_row_index',''),
                'composition_key':r.get('composition_key',''),
                'reduced_formula':r.get('reduced_formula',''),
                'composition_group_id':group_key,
                'prototype_id':r.get('prototype_id',''),
                'split':part,
                'group_size':len(rs),
                'polymorph_group_candidate':int(len(rs)>=2),
                'slices':s,
            })
    cross_split_duplicate_slices = [
        {'mat_ids':'|'.join(mid for mid,_ in values[:10]),
         'splits':'|'.join(sorted({split for _,split in values})),
         'occurrences':len(values)}
        for values in duplicate_slices.values()
        if len({split for _,split in values})>1
    ]
    repeated_slices = [
        {'mat_ids':'|'.join(mid for mid,_ in vals[:10]),'occurrences':len(vals)}
        for vals in duplicate_slices.values() if len(vals)>1
    ]

    by_split = Counter(r['split'] for r in assignments)
    polygroups_by_split = Counter(assignment[k] for k in polymorph_keys)
    rows_polymorph_by_split = Counter(
        x['split'] for x in assignments if x['polymorph_group_candidate'])
    group_by_split = Counter(assignment.values())
    issues = []
    if missing: issues.append(f'{len(missing)} mat_id del manifest faltan en el CSV fuente')
    if repeated_source: issues.append(f'{len(repeated_source)} mat_id repetidos en CSV fuente')
    if duplicate_manifest_ids: issues.append(f'{len(duplicate_manifest_ids)} mat_id repetidos en manifest')
    if mismatches: issues.append(f'{len(mismatches)} inconsistencias con manifest o metadatos')
    if cross_split_duplicate_slices: issues.append(f'{len(cross_split_duplicate_slices)} cadenas SLICES idénticas en distintos splits')
    if len(assignments)!=args.expect_fit: issues.append(f'Esperados {args.expect_fit} elegibles, hay {len(assignments)}')
    if not all(polygroups_by_split[p] for p in SPLITS):
        issues.append('Algún split no tiene grupos polimórficos')

    a_fields = ['mat_id','source_row_index','composition_key','reduced_formula','composition_group_id','prototype_id','split','group_size','polymorph_group_candidate']
    write_csv(out/'composition_disjoint_split_PROVISIONAL.csv',a_fields, sorted(assignments,key=lambda x:x['mat_id']))
    for part in ('train','val'):
        write_csv(out/f'dapt_{part}_PROVISIONAL.csv',a_fields+['slices'],
                  (r for r in assignments if r['split']==part))
    write_csv(out/'downstream_polymorph_candidates_PROVISIONAL.csv',a_fields,
              (r for r in assignments if r['polymorph_group_candidate']))
    write_csv(out/'cross_split_duplicate_slices.csv', ['mat_ids','splits','occurrences'], cross_split_duplicate_slices)
    write_csv(out/'duplicate_slices_anywhere.csv', ['mat_ids','occurrences'], repeated_slices)
    write_csv(out/'consistency_issues.csv', ['mat_id','issue'], mismatches)
    aliases = [
        {'composition_key':key,'reduced_formulas':'|'.join(sorted(values)), 'count':len(values)}
        for key,values in comp_to_reduced.items() if len(values)>1
    ]
    write_csv(out/'ambiguous_composition_aliases.csv', ['composition_key','reduced_formulas','count'], aliases)

    status = 'GROUP_SPLIT_BLOCKED_INPUT_INCONSISTENT' if issues else 'GROUP_SPLIT_PROVISIONAL_STRUCTURE_PENDING'
    result = {
        'status':status,
        'scope':'Proposed composition-disjoint train/val/test; NOT an approved split or crystallographic deduplication',
        'source_csv':str(source),'source_sha256':hfile(source),
        'eligible_manifest':str(manifest),'manifest_sha256':hfile(manifest),
        'seed':args.seed,'ratio_by_groups':{'train':args.train_frac,'val':args.val_frac,'test':round(1-args.train_frac-args.val_frac,4)},
        'eligible_structures':len(assignments),
        'composition_groups':len(grouped),
        'provisional_polymorph_groups':len(polymorph_keys),
        'candidate_polymorph_structures':sum(1 for x in assignments if x['polymorph_group_candidate']),
        'splits': {s:{'structures':by_split[s], 'groups':group_by_split[s],
                      'polymorph_groups':polygroups_by_split[s],
                      'polymorph_structures':rows_polymorph_by_split[s],
                      'rows_with_quality_flags':quality_counts[s]} for s in SPLITS},
        'duplicate_manifest_ids':len(duplicate_manifest_ids),
        'repeated_mat_ids_in_source':len(repeated_source),
        'identical_slices_groups':len(repeated_slices),
        'identical_slices_cross_split_groups':len(cross_split_duplicate_slices),
        'composition_key_to_multiple_reduced_formula':len(aliases),
        'reduced_formula_to_multiple_composition_keys':sum(len(v)>1 for v in reduced_to_comp.values()),
        'manifest_consistency_issues':len(mismatches),
        'structural_equivalence':'NOT_VERIFIED; original CSV lacks full Structure objects',
        'vocabulary_provenance':'PENDING: candidate v1 lexical units were inventoried on all 98,394 valid structures before this split; strict inductive evaluation should use a grammar-predefined vocabulary or training-only lexical inventory, not tune vocabulary using held-out test.',
        'label_definition':'PENDING: 0-K relative polymorph stability requires energetic labels from eligible structures within each composition; not computed here.',
        'remaining_gates':['Retrieve original Alexandria structures to run pymatgen StructureMatcher within composition groups',
            'Decide and document token-vocabulary provenance to protect inductive test',
            'Check energy labels/equality tolerance and reduce duplicates before FINAL split',
            'Audit collator MLM on aligned tokenizer, including special/reserved random replacements'],
        'issues':issues,
        'new_references':{
            'R21':'Witman & Schindler, MatFold, Digital Discovery 2025 DOI:10.1039/D4DD00250D',
            'R22':'pymatgen StructureMatcher official API https://pymatgen.org/pymatgen.analysis.html',
            'R23':'scikit-learn common pitfalls: data leakage https://scikit-learn.org/stable/common_pitfalls.html',
            'R24':'scikit-learn model_selection GroupShuffleSplit API https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html'
        }
    }
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Estado:', status)
    print('Estructuras:', len(assignments),'| grupos químicos:',len(grouped),'| grupos polimórficos candidatos:',len(polymorph_keys))
    print('Splits:',json.dumps(result['splits'],ensure_ascii=False))
    print('Duplicados exactos entre splits:',len(cross_split_duplicate_slices),'| inconsistencias:',len(mismatches))
    print('IMPORTANTE: equivalencia geométrica y procedencia del vocabulario pendientes.')
    print('Resultados:',out.resolve())
    if issues:
        print('Bloqueos:',issues)
        raise SystemExit(2)

if __name__ == '__main__':
    main()
