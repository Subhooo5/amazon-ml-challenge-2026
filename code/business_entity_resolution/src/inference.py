import os
import sys
import time
import json
import gc
import math
import resource
import collections
import joblib
import numpy as np
from array import array

import normalization as norm
from features import extract_features_for_pair, add_context, FEATURE_NAMES
from model import EntityMatcherModel
from blocking import get_blocking_keys, prune_index, rank_candidates, build_views, rerank, reverse_add, TOP_K, NAME_K, POOL_K, REV_MAX, SINGLE_CAP, COMBO_CAP, VIEWS
from thresholding import apply_threshold_and_deduplication, apply_gate_addon, expected_f05_select
from output import write_submission_tsv, write_final_report


def run_test_inference(test_dir, model_path, meta_path, output_dir, batch_size=5000, top_k=TOP_K):
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
    blocking_keys = ('top_k', 'name_k', 'pool_k', 'rev_max', 'single_cap', 'combo_cap', 'views')
    if [meta.get(k) for k in blocking_keys] != [top_k, NAME_K, POOL_K, REV_MAX, SINGLE_CAP, COMBO_CAP, VIEWS]:
        sys.exit(f'Blocking mismatch: model trained with {[meta.get(k) for k in blocking_keys]}; inference uses {[top_k, NAME_K, POOL_K, REV_MAX, SINGLE_CAP, COMBO_CAP, VIEWS]} for {list(blocking_keys)}. Retrain with train.py.')
    rule = meta.get('decision_rule')
    rule_params = meta.get('decision_params') or {}
    rrf_weights = meta.get('rrf_weights')
    min_rev_sim = meta.get('min_rev_sim')
    calibrator_path = meta.get('calibrator_path') or ''
    if {'per_source': {'s2', 's3'}, 'gate_addon': {'gate', 'addon'}, 'expected_f05': {'floor'}}.get(rule) != set(rule_params) or not isinstance(rrf_weights, dict) or '*' not in rrf_weights or any(len(w) != len(VIEWS) + 1 for w in rrf_weights.values()) or not isinstance(min_rev_sim, dict) or '*' not in min_rev_sim or not os.path.isfile(calibrator_path):
        sys.exit(f'Metadata mismatch: decision_rule={rule} {rule_params}, rrf_weights={rrf_weights}, min_rev_sim={min_rev_sim}, calibrator_path={calibrator_path!r}. Retrain with train.py.')
    calibrator = joblib.load(calibrator_path)
    params = model.model.get_params()
    print(f'Using RRF weights {rrf_weights}, MIN_REV_SIM {min_rev_sim}, TOP_K {TOP_K}, REV_MAX {REV_MAX}, decision rule {rule} {rule_params} on calibrated probabilities')

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

        if not targets:
            for eid, _, _ in s1_list:
                all_candidate_results[eid] = []
                all_matched_results[eid] = set()
            continue

        print('  Building inverted index for country...')
        target_ids = sorted(targets)
        target_pre = []
        index = collections.defaultdict(list)
        for i, tid in enumerate(target_ids):
            rname, raddr, rcountry = targets[tid]
            cn, core_n, _, skel = norm.normalize_name(rname)
            ca, nums, _ = norm.normalize_address(raddr, rcountry)
            target_pre.append((cn, core_n, ca, nums, skel))
            for k in get_blocking_keys(rname, raddr, rcountry):
                index[k].append(i)
        del targets
        pruned = prune_index(index)
        log_n = math.log(max(len(target_ids), 1))
        df = collections.Counter(t for v in target_pre for t in set(v[1].split()))
        idf = collections.defaultdict(lambda: log_n, {t: log_n - math.log(1 + c) for t, c in df.items()})
        name_count = collections.Counter(v[1].replace(' ', '') for v in target_pre)
        name_freq = [name_count[v[1].replace(' ', '')] for v in target_pre]
        del df, name_count
        t_index = time.time()
        print(f'  Inverted index built with {len(index):,} active keys (pruned {len(pruned):,} keys); index {t_index - c_t0:.1f}s.')

        s1_pre = []
        for eid, rname, raddr in s1_list:
            cn, core_n, _, skel = norm.normalize_name(rname)
            ca, nums, _ = norm.normalize_address(raddr, country)
            s1_pre.append((cn, core_n, ca, nums, skel))
        vecs, T = build_views([v[1] for v in target_pre], [v[4] for v in target_pre], [v[2] for v in target_pre], [v[1].replace(' ', '') for v in target_pre])
        S = [v.transform(x) for v, x in zip(vecs, ([x[1] for x in s1_pre], [x[4] for x in s1_pre], [x[2] for x in s1_pre], [x[1].replace(' ', '') for x in s1_pre]))]
        c_weights = rrf_weights.get(country, rrf_weights['*'])
        c_min_sim = min_rev_sim.get(country, min_rev_sim['*'])
        t_views = time.time()

        store, n_pooled, t_pool = [array(tc) for tc in 'iihffffffh'], 0, 0.0
        for b0 in range(0, len(s1_list), batch_size):
            t0 = time.time()
            ps, pt, pc, pv = array('i'), array('i'), array('h'), array('f')
            for i in range(b0, min(b0 + batch_size, len(s1_list))):
                eid, rname, raddr = s1_list[i]
                pool = rank_candidates(get_blocking_keys(rname, raddr, country, query=True), index, len(target_ids), POOL_K)
                ps.extend([i] * len(pool))
                for arr, col in zip((pt, pc, pv), zip(*pool)):
                    arr.extend(col)
            ps, pt, pc, pv = (np.frombuffer(a, a.typecode) for a in (ps, pt, pc, pv))
            t_pool += time.time() - t0
            cn, cs, ca, cc, sim, rr = rerank(ps, pt, pv, S, T, [c_weights])
            keep = (rr[0] < TOP_K) | (sim >= c_min_sim)
            for col, x in zip(store, (ps, pt, pc, pv, cn, cs, ca, cc, sim, rr[0])):
                col.frombytes(x[keep].astype(col.typecode).tobytes())
            n_pooled += len(ps)
        del index, vecs, T, S
        ps, pt, pc, pv, cos_n, cos_s, cos_a, cos_c, sim, rrf_rank = (np.frombuffer(c, c.typecode) for c in store)
        t_rerank = time.time()
        rev_rank = reverse_add(ps, pt, sim, rrf_rank, c_min_sim)
        final = np.flatnonzero((rrf_rank < TOP_K) | (rev_rank >= 0))
        final = final[np.lexsort((np.where(rev_rank[final] >= 0, TOP_K + rev_rank[final], rrf_rank[final]), ps[final]))]
        bounds = np.searchsorted(ps[final], np.arange(len(s1_list) + 1))
        t_rev = time.time()
        print(f'  {n_pooled:,} pooled pairs, {len(ps):,} stored, {len(final):,} final pairs ({int((rev_rank >= 0).sum()):,} reverse additions); views {t_views - t_index:.1f}s, pass 1 {t_pool:.1f}s, rerank {t_rerank - t_views - t_pool:.1f}s, reverse {t_rev - t_rerank:.1f}s.')

        print(f'  Scoring candidates for {len(s1_list):,} S1 records in batches of {batch_size}...')
        c_scores_dict = collections.defaultdict(list)
        n_batches = (len(s1_list) + batch_size - 1) // batch_size

        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, len(s1_list))

            batch_pairs = []
            for i in range(b_start, b_end):
                eid = s1_list[i][0]
                f = final[bounds[i]:bounds[i + 1]]
                pairs = list(zip(*(a[f].tolist() for a in (pt, pc, pv, cos_n, cos_s, cos_a, cos_c, rrf_rank, rev_rank))))
                cand_ids = [target_ids[t] for t, *_ in pairs]
                all_candidate_results[eid] = cand_ids
                if pairs:
                    rows = add_context([extract_features_for_pair(s1_pre[i], target_pre[t], target_ids[t], c, idf, [n, k, a, cp, float(r), float(v >= 0), b], name_freq[t]) for t, c, b, n, k, a, cp, r, v in pairs])
                    batch_pairs.extend((eid, tid, feats) for tid, feats in zip(cand_ids, rows))
                else:
                    c_scores_dict[eid] = []

            if batch_pairs:
                X_batch = np.array([p[2] for p in batch_pairs], dtype=np.float32)
                probas = calibrator.predict(model.predict_proba(X_batch))
                for (eid, tid, _), p in zip(batch_pairs, probas):
                    c_scores_dict[eid].append((tid, float(p)))

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
                print(f'    Processed batch {b_idx + 1}/{n_batches} ({b_end:,}/{len(s1_list):,} records)...')

        print(f'  Applying {rule} decision rule and 1-to-1 target consistency...')
        if rule == 'per_source':
            c_matches = apply_threshold_and_deduplication(c_scores_dict, rule_params['s2'], rule_params['s3'])
        elif rule == 'gate_addon':
            c_matches = apply_gate_addon(c_scores_dict, rule_params['gate'], rule_params['addon'])
        else:
            c_matches = expected_f05_select(c_scores_dict, rule_params['floor'])
        for eid, mids in c_matches.items():
            all_matched_results[eid] = mids

        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == 'darwin' else 2 ** 20)
        print(f'  Country {country} completed in {time.time()-c_t0:.1f}s (pass 2 features+predict {time.time()-t_rev:.1f}s), peak RSS {peak_gb:.2f} GB.')
        del target_pre, idf, name_freq, pruned, s1_pre, store, ps, pt, pc, pv, cos_n, cos_s, cos_a, cos_c, sim, rrf_rank, rev_rank, final, c_scores_dict, c_matches
        gc.collect()

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
            'Decision Rule': f'{rule} {rule_params} on isotonic-calibrated probabilities',
            'Retrieval': f'POOL_K={POOL_K}, TOP_K={TOP_K}, views={VIEWS}, caps={SINGLE_CAP}/{COMBO_CAP}, RRF weights={rrf_weights}, REV_MAX={REV_MAX}, MIN_REV_SIM={min_rev_sim}',
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
