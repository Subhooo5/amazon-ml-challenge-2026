import sys
import os
import time
import json
import argparse
import collections
import numpy as np

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import normalization as norm
from features import extract_features_for_pair, FEATURE_NAMES
from model import EntityMatcherModel
from evaluation import evaluate_predictions
from thresholding import optimize_source_specific_thresholds, apply_threshold_and_deduplication
from blocking import get_blocking_keys


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
    parser.add_argument('--n-train', type=int, default=50000,
                        help='Number of training Source 1 entities to sample')
    parser.add_argument('--model-out', default='models/final_entity_matcher.joblib',
                        help='Output path for trained model')
    parser.add_argument('--meta-out', default='models/model_metadata.json',
                        help='Output path for model metadata')
    args = parser.parse_args()

    print('=== Training Business Entity Resolution Model ===')
    print(f'Train directory : {args.train_dir}')
    print(f'Validation split: {args.val_ids}')
    print(f'Training sample : {args.n_train:,} S1 entities')

    gt_file = os.path.join(args.train_dir, 'train_ground_truth.tsv')
    s1_file = os.path.join(args.train_dir, 'train_source1.tsv')
    s2_file = os.path.join(args.train_dir, 'train_source2.tsv')
    s3_file = os.path.join(args.train_dir, 'train_source3.tsv')

    if not os.path.isfile(gt_file):
        print(f'Error: Ground truth file not found at {gt_file}')
        sys.exit(1)

    val_s1_ids = []
    if os.path.isfile(args.val_ids):
        with open(args.val_ids, 'r', encoding='utf-8') as f:
            for line in f:
                sid = line.strip()
                if sid:
                    val_s1_ids.append(sid)
                if len(val_s1_ids) >= 5000:
                    break
    val_s1_set = set(val_s1_ids)
    print(f'Loaded {len(val_s1_set):,} validation entities.')

    val_gt = {}
    val_true_targets = set()
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in val_s1_set:
                mids = p[1].split(',') if len(p) > 1 and p[1] else []
                val_gt[p[0]] = set(mids)
                val_true_targets.update(mids)

    train_s1_ids = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            sid = p[0]
            if sid not in val_s1_set:
                train_s1_ids.append(sid)
                if len(train_s1_ids) >= args.n_train:
                    break

    train_s1_set = set(train_s1_ids)
    train_gt = {}
    train_true_targets = set()
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in train_s1_set:
                mids = p[1].split(',') if len(p) > 1 and p[1] else []
                train_gt[p[0]] = set(mids)
                train_true_targets.update(mids)

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
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, _ = norm.normalize_address(raddr)
        s1_preprocessed[sid] = (cn, core_n, ca, nums, rcountry)
        s1_blocking_keys[sid] = get_blocking_keys(rname, raddr, rcountry)

    all_needed_targets = val_true_targets | train_true_targets
    target_raw = {}
    distractors_s2 = 0
    with open(s2_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            tid = p[0]
            if tid in all_needed_targets or distractors_s2 < 50000:
                target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
                if tid not in all_needed_targets:
                    distractors_s2 += 1

    distractors_s3 = 0
    with open(s3_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            tid = p[0]
            if tid in all_needed_targets or distractors_s3 < 50000:
                target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
                if tid not in all_needed_targets:
                    distractors_s3 += 1

    print(f'Total target pool: {len(target_raw):,} records loaded.')

    target_preprocessed = {}
    target_keys = {}
    for tid, (rname, raddr, rcountry) in target_raw.items():
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, _ = norm.normalize_address(raddr)
        target_preprocessed[tid] = (cn, core_n, ca, nums, rcountry)
        target_keys[tid] = get_blocking_keys(rname, raddr, rcountry)

    print('Building country inverted indices...')
    indices = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, tkeys in target_keys.items():
        rcountry = target_preprocessed[tid][4]
        for k in tkeys:
            indices[rcountry][k].append(tid)

    for c in indices:
        for k in list(indices[c].keys()):
            limit = 300 if (k[0].startswith('n') or k[0].startswith('core') or k[0].startswith('compact')) else 150
            if len(indices[c][k]) > limit:
                del indices[c][k]

    def get_candidates(sid, top_k=20):
        s_keys = s1_blocking_keys[sid]
        country = s1_preprocessed[sid][4]
        c_index = indices[country]
        counts = collections.Counter()
        for k in s_keys:
            if k in c_index:
                counts.update(c_index[k])
        if counts:
            return counts.most_common(top_k)
        return []

    print('Extracting features for training pairs (positives + hard negatives)...')
    t_feat_start = time.time()
    X_train = []
    y_train = []
    train_target_set = set(target_preprocessed.keys())

    for sid in train_s1_ids:
        true_mids = train_gt.get(sid, set()) & train_target_set
        cands = get_candidates(sid, top_k=20)
        cand_mids = {tid: count for tid, count in cands}

        s1_tup = s1_preprocessed[sid][:4]
        for mid in true_mids:
            if mid in target_preprocessed:
                t_tup = target_preprocessed[mid][:4]
                sh = cand_mids.get(mid, 1)
                feats = extract_features_for_pair(s1_tup, t_tup, mid, sh)
                X_train.append(feats)
                y_train.append(1)

        neg_count = 0
        for tid, sh in cands:
            if tid not in true_mids and tid in target_preprocessed:
                t_tup = target_preprocessed[tid][:4]
                feats = extract_features_for_pair(s1_tup, t_tup, tid, sh)
                X_train.append(feats)
                y_train.append(0)
                neg_count += 1
                if neg_count >= max(2, len(true_mids) * 2):
                    break

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f'Training dataset extracted in {time.time()-t_feat_start:.1f}s: X_train shape = {X_train.shape}, Positives = {np.sum(y_train):,}, Negatives = {len(y_train)-np.sum(y_train):,}')

    print('Training production LightGBM model...')
    t_train_start = time.time()
    final_model = EntityMatcherModel(
        'lightgbm',
        n_estimators=400,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=30
    )
    final_model.fit(X_train, y_train)
    print(f'Model trained in {time.time()-t_train_start:.1f}s.')

    print('Evaluating on held-out validation set...')
    val_pair_list = []
    retrieved_val_true = 0
    total_val_true = sum(len(v) for v in val_gt.values())

    for sid in val_s1_ids:
        cands = get_candidates(sid, top_k=20)
        cand_ids = [tid for tid, _ in cands]
        retrieved_val_true += len(val_gt[sid] & set(cand_ids))

        s1_tup = s1_preprocessed[sid][:4]
        for tid, sh in cands:
            if tid in target_preprocessed:
                t_tup = target_preprocessed[tid][:4]
                feats = extract_features_for_pair(s1_tup, t_tup, tid, sh)
                val_pair_list.append((sid, tid, feats))

    X_val = np.array([p[2] for p in val_pair_list], dtype=np.float32)
    val_probas = final_model.predict_proba(X_val)

    scores_dict = collections.defaultdict(list)
    for (sid, tid, _), p in zip(val_pair_list, val_probas):
        scores_dict[sid].append((tid, float(p)))

    for sid in val_s1_ids:
        if sid not in scores_dict:
            scores_dict[sid] = []

    opt_s2, opt_s3, best_metrics = optimize_source_specific_thresholds(val_gt, scores_dict)
    preds = apply_threshold_and_deduplication(scores_dict, opt_s2, opt_s3)
    metrics = evaluate_predictions(val_gt, preds)

    val_cand_recall = retrieved_val_true / total_val_true if total_val_true > 0 else 0.0

    print('\n' + '=' * 60)
    print('FINAL MODEL VALIDATION RESULTS')
    print('=' * 60)
    print(f"Optimal Thresholds: S2 = {opt_s2:.2f}, S3 = {opt_s3:.2f}")
    print(f"Validation F0.5   : {metrics['macro_f05']:.6f}")
    print(f"Precision         : {metrics['global_precision']:.6f}")
    print(f"Recall            : {metrics['global_recall']:.6f}")
    print(f"Macro F1          : {metrics['macro_f1']:.6f}")
    print(f"Candidate Recall  : {val_cand_recall:.6f}")
    print(f"False Positives   : {metrics['total_fp']}")
    print(f"False Negatives   : {metrics['total_fn']}")
    print(f"Predicted Links   : {metrics['total_pred_links']}")
    print(f"Singleton Accuracy: {metrics['singleton_accuracy']:.6f}")
    print('=' * 60 + '\n')

    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or '.', exist_ok=True)
    final_model.save(args.model_out)

    meta = {
        'optimal_s2_threshold': float(opt_s2),
        'optimal_s3_threshold': float(opt_s3),
        'validation_f05': float(metrics['macro_f05']),
        'validation_precision': float(metrics['global_precision']),
        'validation_recall': float(metrics['global_recall']),
        'validation_f1': float(metrics['macro_f1']),
        'candidate_recall': float(val_cand_recall),
        'false_positives': int(metrics['total_fp']),
        'false_negatives': int(metrics['total_fn']),
        'total_pred_links': int(metrics['total_pred_links']),
        'singleton_accuracy': float(metrics['singleton_accuracy']),
        'feature_names': FEATURE_NAMES
    }
    with open(args.meta_out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Model saved to {args.model_out} and metadata to {args.meta_out}.')


if __name__ == '__main__':
    main()
