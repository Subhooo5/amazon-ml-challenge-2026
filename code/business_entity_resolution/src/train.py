import sys
import os
import gc
import time
import json
import math
import random
import argparse
import collections
import numpy as np

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import normalization as norm
from features import extract_features_for_pair, add_context, FEATURE_NAMES
from model import EntityMatcherModel
from evaluation import evaluate_predictions
from thresholding import optimize_source_specific_thresholds, apply_threshold_and_deduplication, optimize_gate_addon, apply_gate_addon
from blocking import get_blocking_keys, prune_index, rank_candidates, TOP_K, NAME_K


def auto_detect_train_dir():
    candidates = ['dataset/train', 'student_resource/dataset/train']
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description='Train Business Entity Resolution Model')
    parser.add_argument('--train-dir', default=auto_detect_train_dir(),
                        help='Path to train dataset directory containing train_source1/2/3.tsv and train_ground_truth.tsv')
    parser.add_argument('--val-ids', default='experiments/val_s1_ids.txt',
                        help='Path to file containing held-out validation Source 1 IDs')
    parser.add_argument('--n-train', type=int, default=200000,
                        help='Number of training Source 1 entities to sample')
    parser.add_argument('--pool-limit', type=int, default=0,
                        help='Per source file, keep all true targets plus the first N other targets of the country (0 = full pool)')
    parser.add_argument('--model-out', default='models/final_entity_matcher.joblib',
                        help='Output path for trained model')
    parser.add_argument('--meta-out', default='models/model_metadata.json',
                        help='Output path for model metadata')
    args = parser.parse_args()

    print('=== Training Business Entity Resolution Model ===')
    print(f'Train directory : {args.train_dir}')
    print(f'Validation split: {args.val_ids}')
    print(f'Training sample : {args.n_train:,} S1 entities')
    print(f'Pool limit      : {args.pool_limit:,} (0 = full per-country pool)')

    gt_file = os.path.join(args.train_dir, 'train_ground_truth.tsv')
    s1_file = os.path.join(args.train_dir, 'train_source1.tsv')
    s2_file = os.path.join(args.train_dir, 'train_source2.tsv')
    s3_file = os.path.join(args.train_dir, 'train_source3.tsv')

    if not os.path.isfile(gt_file):
        print(f'Error: Ground truth file not found at {gt_file}')
        sys.exit(1)

    gt_raw = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            gt_raw[p[0]] = p[1] if len(p) > 1 else ''
    gt_ids = sorted(gt_raw)

    if os.path.isfile(args.val_ids):
        with open(args.val_ids, 'r', encoding='utf-8') as f:
            val_s1_ids = [sid for sid in dict.fromkeys(line.strip() for line in f) if sid in gt_raw]
        print(f'Validation ids  : all ids of {args.val_ids} found in the ground truth')
    else:
        val_s1_ids = random.Random(42).sample(gt_ids, min(20000, len(gt_ids)))
        print(f'Validation ids  : random.Random(42).sample of 20,000 ground-truth ids ({args.val_ids} not found)')
    val_s1_set = set(val_s1_ids)
    print(f'Loaded {len(val_s1_set):,} validation entities.')
    if not val_s1_set:
        print(f'Error: no validation ids from {args.val_ids} are in the ground truth')
        sys.exit(1)

    pool_ids = [sid for sid in gt_ids if sid not in val_s1_set]
    train_s1_ids = random.Random(42).sample(pool_ids, min(args.n_train, len(pool_ids)))
    val_gt = {sid: set(gt_raw[sid].split(',')) if gt_raw[sid] else set() for sid in val_s1_ids}
    train_gt = {sid: set(gt_raw[sid].split(',')) if gt_raw[sid] else set() for sid in train_s1_ids}
    del gt_raw, gt_ids, pool_ids
    train_s1_set = set(train_s1_ids)
    es_s1_set = set(random.Random(42).sample(train_s1_ids, len(train_s1_ids) // 10))

    print(f'Training pool: {len(train_s1_ids):,} S1 records, {sum(len(v) for v in train_gt.values()):,} true positive links.')

    all_needed_s1 = val_s1_set | train_s1_set
    s1_raw = {}
    with open(s1_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in all_needed_s1:
                s1_raw[p[0]] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')

    s1_preprocessed = {}
    s1_blocking_keys = {}
    for sid, (rname, raddr, rcountry) in s1_raw.items():
        cn, core_n, _, skel = norm.normalize_name(rname)
        ca, nums, _ = norm.normalize_address(raddr, rcountry)
        s1_preprocessed[sid] = (cn, core_n, ca, nums, skel, rcountry)
        s1_blocking_keys[sid] = get_blocking_keys(rname, raddr, rcountry, query=True)

    needed_targets = collections.defaultdict(set)
    for gt in (train_gt, val_gt):
        for sid, mids in gt.items():
            needed_targets[s1_preprocessed[sid][5]].update(mids)

    print('Extracting features per country (all top-K candidates, full per-country target pool)...')
    t_feat_start = time.time()
    X_train = []
    y_train = []
    X_es = []
    y_es = []
    val_pair_list = []
    val_ranks = []
    val_ranks60 = []
    no_name_k_hits = 0
    miss_cat = collections.Counter()
    missed_examples = []
    missed_empty_addr = 0
    missed_non_ascii = 0

    for country in sorted({v[5] for v in s1_preprocessed.values()}):
        c_t0 = time.time()
        targets = {}
        for src_path in [s2_file, s3_file]:
            others = 0
            with open(src_path, 'r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    p = line.rstrip('\r\n').split('\t')
                    if (p[3] if len(p) > 3 else '') != country:
                        continue
                    if args.pool_limit > 0 and p[0] not in needed_targets[country]:
                        if others >= args.pool_limit:
                            continue
                        others += 1
                    targets[p[0]] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '')

        target_preprocessed = {}
        index = collections.defaultdict(list)
        for tid, (rname, raddr) in targets.items():
            cn, core_n, _, skel = norm.normalize_name(rname)
            ca, nums, _ = norm.normalize_address(raddr, country)
            target_preprocessed[tid] = (cn, core_n, ca, nums, skel)
            for k in get_blocking_keys(rname, raddr, country):
                index[k].append(tid)
        pruned = prune_index(index)
        log_n = math.log(max(len(targets), 1))
        df = collections.Counter(t for v in target_preprocessed.values() for t in set(v[1].split()))
        idf = collections.defaultdict(lambda: log_n, {t: log_n - math.log(1 + c) for t, c in df.items()})
        del df
        t_index = time.time() - c_t0

        for sid in [sid for sid in train_s1_ids if s1_preprocessed[sid][5] == country]:
            cands = rank_candidates(s1_blocking_keys[sid], index, len(targets), TOP_K)
            s1_tup = s1_preprocessed[sid][:5]
            rows = add_context([extract_features_for_pair(s1_tup, target_preprocessed[tid], tid, sh, idf) for tid, sh in cands])
            X_part, y_part = (X_es, y_es) if sid in es_s1_set else (X_train, y_train)
            X_part.extend(rows)
            y_part.extend(int(tid in train_gt[sid]) for tid, _ in cands)

        for sid in [sid for sid in val_s1_ids if s1_preprocessed[sid][5] == country]:
            skeys = s1_blocking_keys[sid]
            cands = rank_candidates(skeys, index, len(targets), TOP_K)
            rank_of = {tid: r for r, (tid, _) in enumerate(cands)}
            rank60 = {tid: r for r, (tid, _) in enumerate(rank_candidates(skeys, index, len(targets), 60))}
            no_name_k = {tid for tid, _ in rank_candidates(skeys, index, len(targets), TOP_K, 0)}
            s1_tup = s1_preprocessed[sid][:5]
            rows = add_context([extract_features_for_pair(s1_tup, target_preprocessed[tid], tid, sh, idf) for tid, sh in cands])
            val_pair_list.extend((sid, tid, feats) for (tid, _), feats in zip(cands, rows))
            for mid in sorted(val_gt[sid]):
                val_ranks.append(rank_of.get(mid))
                val_ranks60.append(rank60.get(mid))
                no_name_k_hits += mid in no_name_k
                if mid not in rank_of:
                    t_name, t_addr = targets.get(mid, ('', ''))
                    shared = skeys & get_blocking_keys(t_name, t_addr, country)
                    miss_cat['no_shared_key' if not shared else 'only_pruned' if shared <= pruned else 'ranked_out'] += 1
                    missed_empty_addr += not norm.clean_string(t_addr)
                    missed_non_ascii += not t_name.isascii()
                    if len(missed_examples) < 20:
                        missed_examples.append(f'{sid} {s1_raw[sid][0]} | {s1_raw[sid][1]}  <->  {mid} {t_name} | {t_addr}')

        print(f'  {country}: {len(targets):,} targets, {len(pruned):,} keys pruned, target load+normalize+index {t_index:.1f}s, S1 query+features {time.time() - c_t0 - t_index:.1f}s')
        del targets, target_preprocessed, index, idf, pruned
        gc.collect()

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_es = np.array(X_es, dtype=np.float32)
    y_es = np.array(y_es, dtype=np.int32)
    print(f'Training dataset extracted in {time.time()-t_feat_start:.1f}s: X_train shape = {X_train.shape}, Positives = {np.sum(y_train):,}, Negatives = {len(y_train)-np.sum(y_train):,}')
    print(f'Early-stopping set: {len(es_s1_set):,} S1, X_es shape = {X_es.shape}, Positives = {np.sum(y_es):,}')

    print('Training production LightGBM model...')
    t_train_start = time.time()
    final_model = EntityMatcherModel(
        'lightgbm',
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42
    )
    final_model.fit(X_train, y_train, eval_set=(X_es, y_es))
    print(f'Model trained in {time.time()-t_train_start:.1f}s, best_iteration_ = {final_model.model.best_iteration_}.')

    print('Evaluating on held-out validation set...')
    X_val = np.array([p[2] for p in val_pair_list], dtype=np.float32)
    val_probas = final_model.predict_proba(X_val)

    scores_dict = collections.defaultdict(list)
    for (sid, tid, _), p in zip(val_pair_list, val_probas):
        scores_dict[sid].append((tid, float(p)))

    for sid in val_s1_ids:
        if sid not in scores_dict:
            scores_dict[sid] = []
    retrieved_val_true = sum(r is not None for r in val_ranks)
    total_val_true = len(val_ranks)

    opt_s2, opt_s3, best_metrics = optimize_source_specific_thresholds(val_gt, scores_dict)
    source_metrics = evaluate_predictions(val_gt, apply_threshold_and_deduplication(scores_dict, opt_s2, opt_s3))
    gate, addon, _ = optimize_gate_addon(val_gt, scores_dict)
    metrics = evaluate_predictions(val_gt, apply_gate_addon(scores_dict, gate, addon))

    val_cand_recall = retrieved_val_true / total_val_true if total_val_true > 0 else 0.0

    print('\n' + '=' * 60)
    print('FINAL MODEL VALIDATION RESULTS')
    print('=' * 60)
    print(f"Per-source rule   : S2 = {opt_s2:.2f}, S3 = {opt_s3:.2f}, F0.5 = {source_metrics['macro_f05']:.6f}")
    print(f"Gate/addon rule   : gate = {gate:.2f}, addon = {addon:.2f}, F0.5 = {metrics['macro_f05']:.6f}")
    print(f"Best iteration    : {final_model.model.best_iteration_}")
    print(f"Validation F0.5   : {metrics['macro_f05']:.6f}")
    print(f"Precision         : {metrics['global_precision']:.6f}")
    print(f"Recall            : {metrics['global_recall']:.6f}")
    print(f"Macro F1          : {metrics['macro_f1']:.6f}")
    print(f"Candidate Recall  : {val_cand_recall:.6f}")
    print(f"False Positives   : {metrics['total_fp']}")
    print(f"False Negatives   : {metrics['total_fn']}")
    print(f"Predicted Links   : {metrics['total_pred_links']}")
    print(f"Singleton Accuracy: {metrics['singleton_accuracy']:.6f}")
    for k in (10, 15, 20, 25, 40, 60):
        print(f"Top-60 Recall@{k:<3}: {sum(r is not None and r < k for r in val_ranks60) / max(total_val_true, 1):.6f}")
    print(f"name_k=0 Recall@25: {no_name_k_hits / max(total_val_true, 1):.6f}")
    print(f"Missed at TOP_K   : no_shared_key {miss_cat['no_shared_key']:,}, only_pruned {miss_cat['only_pruned']:,}, ranked_out {miss_cat['ranked_out']:,}")
    print('=' * 60 + '\n')
    print(f'Missed validation links (first {len(missed_examples)}):')
    for ex in missed_examples:
        print(f'  {ex}')
    n_missed = total_val_true - retrieved_val_true
    print(f'Missed links with empty target address: {missed_empty_addr:,} of {n_missed:,}')
    print(f'Missed links with non-ASCII target name: {missed_non_ascii:,} of {n_missed:,}')

    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or '.', exist_ok=True)
    final_model.save(args.model_out)

    meta = {
        'optimal_s2_threshold': float(opt_s2),
        'optimal_s3_threshold': float(opt_s3),
        'source_rule_f05': float(source_metrics['macro_f05']),
        'gate_threshold': gate,
        'addon_threshold': addon,
        'best_iteration': int(final_model.model.best_iteration_),
        'validation_f05': float(metrics['macro_f05']),
        'validation_precision': float(metrics['global_precision']),
        'validation_recall': float(metrics['global_recall']),
        'validation_f1': float(metrics['macro_f1']),
        'candidate_recall': float(val_cand_recall),
        'false_positives': int(metrics['total_fp']),
        'false_negatives': int(metrics['total_fn']),
        'total_pred_links': int(metrics['total_pred_links']),
        'singleton_accuracy': float(metrics['singleton_accuracy']),
        'top_k': TOP_K,
        'name_k': NAME_K,
        'n_train': len(train_s1_ids),
        'n_val': len(val_s1_ids),
        'pool_limit': args.pool_limit,
        'feature_names': FEATURE_NAMES
    }
    with open(args.meta_out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Model saved to {args.model_out} and metadata to {args.meta_out}.')


if __name__ == '__main__':
    main()
