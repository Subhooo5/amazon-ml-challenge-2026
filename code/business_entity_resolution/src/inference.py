import os
import sys
import time
import json
import gc
import math
import collections
import numpy as np

import normalization as norm
from features import extract_features_for_pair, add_context, FEATURE_NAMES
from model import EntityMatcherModel
from blocking import get_blocking_keys, prune_index, rank_candidates
from thresholding import apply_gate_addon
from output import write_submission_tsv, write_final_report


def run_test_inference(test_dir, model_path, meta_path, output_dir, batch_size=5000, top_k=15):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print('=== Amazon ML Challenge 2026: Business Entity Resolution Inference ===')
    print(f'Test directory : {test_dir}')
    print(f'Model path     : {model_path}')
    print(f'Output directory: {output_dir}')

    print('\n[1/5] Loading production model and metadata...')
    model = EntityMatcherModel.load(model_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    if meta.get('feature_names') != FEATURE_NAMES or model.model.n_features_in_ != len(FEATURE_NAMES):
        sys.exit(f'Model/feature mismatch: model has {model.model.n_features_in_} features, metadata lists {len(meta.get("feature_names") or [])}, code expects {len(FEATURE_NAMES)}. Retrain with train.py.')
    gate = meta['gate_threshold']
    addon = meta['addon_threshold']
    params = model.model.get_params()
    print(f'Using decision rule: gate = {gate:.2f}, addon = {addon:.2f}')

    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')

    print('\n[2/5] Reading test Source 1 entities and partitioning by country...')
    ordered_s1_ids = []
    country_to_s1 = collections.defaultdict(list)

    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''

            ordered_s1_ids.append(eid)
            country_to_s1[country].append((eid, name, addr))

    total_s1 = len(ordered_s1_ids)
    print(f'Total test Source 1 entities: {total_s1:,}')
    for c, items in country_to_s1.items():
        print(f'  Country: {c:<10} -> {len(items):,} entities ({len(items)/total_s1*100:.1f}%)')

    all_matched_results = {}
    all_candidate_results = {}

    print('\n[3/5] Processing test candidates and predictions by country partition...')
    
    sorted_countries = sorted(country_to_s1.keys(), key=lambda c: len(country_to_s1[c]))

    for country_idx, country in enumerate(sorted_countries, 1):
        c_t0 = time.time()
        s1_list = country_to_s1[country]
        print(f'\n--- [{country_idx}/{len(sorted_countries)}] Processing Country: {country} ({len(s1_list):,} S1 entities) ---')

        print(f'  Loading target records from S2 and S3 for {country}...')
        targets = {}
        for src_path in [s2_path, s3_path]:
            if not os.path.isfile(src_path):
                continue
            with open(src_path, 'r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    line = line.rstrip('\r\n')
                    if not line:
                        continue
                    parts = line.split('\t')
                    c = parts[3] if len(parts) > 3 else ''
                    if c == country:
                        eid = parts[0]
                        name = parts[1] if len(parts) > 1 else ''
                        addr = parts[2] if len(parts) > 2 else ''
                        targets[eid] = (name, addr, c)

        print(f'  Loaded {len(targets):,} target records for {country}.')

        print('  Building inverted index for country...')
        target_preprocessed = {}
        index = collections.defaultdict(list)

        for tid, (rname, raddr, rcountry) in targets.items():
            cn, core_n, _, skel = norm.normalize_name(rname)
            ca, nums, _ = norm.normalize_address(raddr, rcountry)
            target_preprocessed[tid] = (cn, core_n, ca, nums, skel)
            tkeys = get_blocking_keys(rname, raddr, rcountry)
            for k in tkeys:
                index[k].append(tid)

        pruned = prune_index(index)
        log_n = math.log(max(len(targets), 1))
        df = collections.Counter(t for v in target_preprocessed.values() for t in set(v[1].split()))
        idf = collections.defaultdict(lambda: log_n, {t: log_n - math.log(1 + c) for t, c in df.items()})
        del df

        print(f'  Inverted index built with {len(index):,} active keys (pruned {pruned:,} keys).')

        print(f'  Scoring candidates for {len(s1_list):,} S1 records in batches of {batch_size}...')
        
        c_scores_dict = collections.defaultdict(list)
        n_batches = (len(s1_list) + batch_size - 1) // batch_size

        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, len(s1_list))
            batch = s1_list[b_start:b_end]

            batch_pairs = []
            for eid, rname, raddr in batch:
                skeys = get_blocking_keys(rname, raddr, country)
                cands = rank_candidates(skeys, index, len(targets), top_k)

                if cands:
                    cand_ids = [tid for tid, _ in cands]
                    all_candidate_results[eid] = cand_ids

                    cn, core_n, _, skel = norm.normalize_name(rname)
                    ca, nums, _ = norm.normalize_address(raddr, country)
                    s1_tup = (cn, core_n, ca, nums, skel)

                    rows = add_context([extract_features_for_pair(s1_tup, target_preprocessed[tid], tid, sh, idf) for tid, sh in cands])
                    batch_pairs.extend((eid, tid, feats) for (tid, _), feats in zip(cands, rows))
                else:
                    all_candidate_results[eid] = []
                    c_scores_dict[eid] = []

            if batch_pairs:
                X_batch = np.array([p[2] for p in batch_pairs], dtype=np.float32)
                probas = model.predict_proba(X_batch)
                for (eid, tid, _), p in zip(batch_pairs, probas):
                    c_scores_dict[eid].append((tid, float(p)))

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
                print(f'    Processed batch {b_idx + 1}/{n_batches} ({b_end:,}/{len(s1_list):,} records)...')

        print('  Applying gate/addon decision rule and 1-to-1 target consistency...')
        c_matches = apply_gate_addon(c_scores_dict, gate, addon)
        for eid, mids in c_matches.items():
            all_matched_results[eid] = mids

        del targets, target_preprocessed, index, idf, c_scores_dict, c_matches
        gc.collect()
        print(f'  Country {country} completed in {time.time()-c_t0:.1f}s.')

    print('\n[4/5] Writing output files in exact test Source 1 order...')
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    write_submission_tsv(matching_path, all_matched_results, 'matched_entity_ids', ordered_s1_ids)
    write_submission_tsv(candidate_path, all_candidate_results, 'candidate_entity_ids', ordered_s1_ids)

    print(f'Saved matching results : {matching_path}')
    print(f'Saved candidate pairs  : {candidate_path}')

    n_singletons = sum(1 for sid in ordered_s1_ids if len(all_matched_results.get(sid, [])) == 0)
    n_matched = total_s1 - n_singletons
    total_matched_links = sum(len(all_matched_results.get(sid, [])) for sid in ordered_s1_ids)
    total_candidate_links = sum(len(all_candidate_results.get(sid, [])) for sid in ordered_s1_ids)
    
    n_matched_s2 = sum(1 for sid in ordered_s1_ids for tid in all_matched_results.get(sid, []) if tid.startswith('S2-'))
    n_matched_s3 = sum(1 for sid in ordered_s1_ids for tid in all_matched_results.get(sid, []) if tid.startswith('S3-'))

    total_time = time.time() - t_start
    print('\n[5/5] Inference Summary Statistics:')
    print(f'  Total S1 Entities      : {total_s1:,}')
    print(f'  Predicted Singletons   : {n_singletons:,} ({n_singletons/total_s1*100:.2f}%)')
    print(f'  Entities with Matches  : {n_matched:,} ({n_matched/total_s1*100:.2f}%)')
    print(f'  Total Predicted Links  : {total_matched_links:,}')
    print(f'    - Matched S2 Links   : {n_matched_s2:,}')
    print(f'    - Matched S3 Links   : {n_matched_s3:,}')
    print(f'  Total Candidate Pairs  : {total_candidate_links:,}')
    print(f'  Total Processing Time  : {total_time:.1f}s ({total_time/60:.2f} min)')

    report_dict = {
        'Validation Performance': {
            label: f'{meta[key]:.6f}' if isinstance(meta[key], float) else meta[key]
            for label, key in [
                ('Validation F0.5', 'validation_f05'),
                ('Validation Precision', 'validation_precision'),
                ('Validation Recall', 'validation_recall'),
                ('Validation Macro F1', 'validation_f1'),
                ('Candidate Recall', 'candidate_recall'),
                ('Singleton Accuracy', 'singleton_accuracy'),
                ('False Positives', 'false_positives'),
                ('Total Predicted Links', 'total_pred_links'),
                ('False Negatives', 'false_negatives')
            ]
            if key in meta
        },
        'Model Configuration': {
            'Selected Model': 'LightGBM Gradient Boosted Decision Trees',
            'Tree Parameters': f"n_estimators={params['n_estimators']}, best_iteration_={model.model.best_iteration_}, learning_rate={params['learning_rate']}, num_leaves={params['num_leaves']}",
            'Gate Threshold': f'{gate:.2f}',
            'Addon Threshold': f'{addon:.2f}',
            'Global Consistency': 'Source-aware 1-to-1 target assignment (Greedy Highest-Probability)',
            'Feature Set Size': len(FEATURE_NAMES)
        },
        'Test Inference Results': {
            'Total Source 1 Entities': f'{total_s1:,}',
            'Predicted Singletons': f'{n_singletons:,} ({n_singletons/total_s1*100:.2f}%)',
            'Entities with Matches': f'{n_matched:,} ({n_matched/total_s1*100:.2f}%)',
            'Total Predicted Matches': f'{total_matched_links:,}',
            'Total S2 Matches': f'{n_matched_s2:,}',
            'Total S3 Matches': f'{n_matched_s3:,}',
            'Total Candidate Pairs': f'{total_candidate_links:,}',
            'Runtime': f'{total_time:.1f}s ({total_time/60:.2f} min)'
        },
        'Output Files': {
            'Matching Results': matching_path,
            'Candidate Pairs': candidate_path
        }
    }
    write_final_report(os.path.join(output_dir, 'final_report.txt'), report_dict)
    return report_dict
