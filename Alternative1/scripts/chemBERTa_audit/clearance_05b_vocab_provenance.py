#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clearance 05B: audit candidate SLICES tokenizer vocabulary provenance.

Uses lexical split() for exact WordLevel SLICES units. Joins the original CSV
with the provisional composition-disjoint split. No weights or data are changed.
The 15k/45k comparison is a coverage study, NOT a model-performance estimate.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PARTS = ('train', 'val', 'test')

def truth(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def file_sha(path):
    dig = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            dig.update(chunk)
    return dig.hexdigest()


def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def write_table(path, headings, data):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, headings, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(data)


def serializable_counter(c, maxn=30):
    return [{'unit':k, 'occurrences':v} for k,v in c.most_common(maxn)]


def audit(args):
    source = Path(args.input)
    assignment = Path(args.split)
    tokpath = Path(args.tokenizer) / 'tokenizer.json'
    if not all(p.is_file() for p in (source, assignment, tokpath)):
        raise FileNotFoundError(f'Required files: {source}, {assignment}, {tokpath}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    details = json.loads(tokpath.read_text(encoding='utf-8'))
    if details.get('model',{}).get('type') != 'WordLevel':
        raise ValueError('Expected WordLevel candidate_tokenizer tokenizer.json')
    vocab = details['model']['vocab']
    if not isinstance(vocab, dict):
        raise ValueError('Invalid tokenizer.model.vocab')

    split_by_id, dup_assign, invalid_parts = {}, [], []
    for r in rows(assignment):
        mid = (r.get('mat_id') or '').strip()
        part = (r.get('split') or '').strip()
        if not mid or mid in split_by_id:
            dup_assign.append(mid)
        if part not in PARTS:
            invalid_parts.append((mid, part))
        if mid and part in PARTS:
            split_by_id[mid] = part
    if dup_assign or invalid_parts:
        raise ValueError(f'duplicate assignments={len(dup_assign)}, invalid split={len(invalid_parts)}')

    counters = {p:Counter() for p in PARTS}
    global_counter = Counter()
    by_matid = {p:[] for p in PARTS}
    counts = Counter()
    valid_source_records = 0
    seen_ids = Counter()
    for r in rows(source):
        mid = (r.get('mat_id') or '').strip()
        s = (r.get(args.column) or '').strip()
        if not s:
            continue
        if args.require_valid and not (truth(r.get('slices_basic_valid')) and truth(r.get('slices_valid'))):
            if mid in split_by_id:
                raise ValueError('Split contains a row that fails validity flags: '+mid)
            continue
        valid_source_records += 1
        tokens = s.split()
        global_counter.update(tokens)
        if mid in split_by_id:
            seen_ids[mid] += 1
            part = split_by_id[mid]
            counters[part].update(tokens)
            by_matid[part].append((mid,tokens))
            counts[part] += 1
    missing = set(split_by_id) - set(seen_ids)
    repeats = {k:v for k,v in seen_ids.items() if v!=1}
    if missing or repeats:
        raise ValueError(f'Missing assigned mat_ids={len(missing)}, repeated source mat_ids={len(repeats)}')
    if any(counts[p]==0 for p in PARTS):
        raise ValueError('Every split must have at least one row')
    missing_vocab = sorted(set(global_counter) - set(vocab))
    if missing_vocab:
        raise ValueError(f'Tokenizer does not contain {len(missing_vocab)} global lexical units: {missing_vocab[:15]}')
    train_types = set(counters['train'])
    heldout = (set(counters['val'])|set(counters['test'])) - train_types
    global_only = set(global_counter) - train_types
    all_tokens = set(global_counter)
    lex_rows = []
    for t in sorted(all_tokens):
        fr = {p:counters[p][t] for p in PARTS}
        lex_rows.append({'unit':t, 'token_id':vocab[t],
                         'train_occurrences':fr['train'],
                         'val_occurrences':fr['val'],
                         'test_occurrences':fr['test'],
                         'all_source_occurrences':global_counter[t],
                         'not_seen_in_train':int(t not in train_types),
                         'only_in_excluded':int(t not in train_types and not fr['val'] and not fr['test'])})
    write_table(out/'lexical_provenance.csv',
                ['unit','token_id','train_occurrences','val_occurrences','test_occurrences',
                 'all_source_occurrences','not_seen_in_train','only_in_excluded'], lex_rows)

    sample_rows = []
    sizes = sorted(set(int(i) for i in args.subset_sizes.split(',') if i.strip()))
    if any(x <= 0 for x in sizes):
        raise ValueError('--subset-sizes must contain positive integers')
    ordered = sorted(by_matid['train'], key=lambda rec:hashlib.sha256(
        (str(args.seed)+'|'+rec[0]).encode()).hexdigest())
    total_train_tokens = sum(counters['train'].values())
    for n in sorted(set(sizes+[counts['train']])):
        if n > counts['train']:
            continue
        sub = Counter()
        for _, toks in ordered[:n]:
            sub.update(toks)
        covered = set(sub)
        rare = sum(1 for k in train_types if 0 < sub[k] < args.rare_threshold)
        fraction = sum(counters['train'][u] for u in covered)/total_train_tokens if total_train_tokens else 0
        sample_rows.append({'structures':n,'unique_units':len(covered),
                            'pct_train_unique_units_covered':round(100*len(covered)/len(train_types),4),
                            'pct_train_token_occurrences_covered_by_seen_types':round(100*fraction,4),
                            'units_missing_vs_full_train':len(train_types-covered),
                            'units_with_1_to_threshold_minus_1_occurrences':rare,
                            'units_with_at_least_threshold_occurrences':sum(sub[u]>=args.rare_threshold for u in covered),
                            'total_lexical_occurrences':sum(sub.values())})
    write_table(out/'subset_coverage.csv', list(sample_rows[0].keys()), sample_rows)

    if heldout or global_only:
        status = 'VOCAB_PROVENANCE_REVIEW_REQUIRED'
    else:
        status = 'TRAIN_LEXICON_MATCHES_GLOBAL_AUDIT_PASS'
    report = {
        'status':status,
        'scope':'Lexical inventory/provenance only. This does not approve crystallographic duplicate equivalence or labels.',
        'source':str(source), 'source_sha256':file_sha(source),
        'split':str(assignment), 'split_sha256':file_sha(assignment),
        'tokenizer_json':str(tokpath), 'tokenizer_sha256':file_sha(tokpath),
        'structures_per_split':dict(counts),
        'audited_source_structures':valid_source_records,
        'audited_source_lexical_units':len(global_counter),
        'train_lexical_units':len(train_types),
        'validation_lexical_units':len(counters['val']),
        'test_lexical_units':len(counters['test']),
        'heldout_units_absent_from_train':len(heldout),
        'heldout_only_examples':serializable_counter(Counter({u: counters['val'][u]+counters['test'][u] for u in heldout})),
        'global_source_units_absent_from_train':len(global_only),
        'only_in_excluded_units':len(global_only-heldout),
        'global_only_examples':serializable_counter(Counter({u:global_counter[u] for u in global_only})),
        'tokenizer_vocabulary_size':len(vocab),
        'tokenizer_missing_source_units':len(missing_vocab),
        'subset_comparison':sample_rows,
        'interpretation': ('Inventory was built from the entire CSV before split. If train lexical set equals full corpus, '
                           'candidate WordLevel lexical inventory is reproducible using train alone. '
                           'If they differ, rebuild train-only candidate or explicitly document transductive inventory; '
                           'do not use held-out text for fitting embeddings or selecting hyperparameters.'),
        'remaining_gates':['Confirm structural equivalence/label integrity before supervised final dataset',
                           'Audit/sanitize MLM collator, especially random replacements from forbidden IDs',
                           'Design embedding initialization and DAPT optimization']
    }
    (out/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('=== CLEARANCE 05B — VOCABULARY PROVENANCE ===')
    print('Estado:',status)
    print('Estructuras:',dict(counts))
    print('Unidades: global=',len(global_counter),'train=',len(train_types),
          'heldout no vistas train=',len(heldout), 'full corpus no vistas train=',len(global_only))
    for r in sample_rows:
        print('Subconjunto:',r['structures'],'unidades=',r['unique_units'],
              'cobertura tipos train=',r['pct_train_unique_units_covered'])
    print('Resultados:',out.resolve())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--column',default='slices')
    ap.add_argument('--require-valid',action='store_true',help='Require slices_basic_valid and slices_valid, reproducing the earlier audit')
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--subset-sizes',default='15000,45000')
    ap.add_argument('--rare-threshold',type=int,default=30)
    audit(ap.parse_args())

if __name__=='__main__':
    main()
