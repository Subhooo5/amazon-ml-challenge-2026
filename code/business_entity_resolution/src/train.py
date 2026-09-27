import sys
import os
import gc
import time
import json
import math
import random
import argparse
import resource
import itertools
import collections
import joblib
import numpy as np
from array import array
from sklearn.isotonic import IsotonicRegression

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import normalization as norm
from features import extract_features_for_pair, add_context, FEATURE_NAMES
from model import EntityMatcherModel
from evaluation import evaluate_predictions
from thresholding import optimize_source_specific_thresholds, apply_threshold_and_deduplication, optimize_gate_addon, apply_gate_addon, expected_f05_select
from blocking import get_blocking_keys, prune_index, rank_candidates, build_views, rerank, reverse_add, TOP_K, NAME_K, POOL_K, REV_MAX

RRF_GRID = [(1, 1, 1, 1)] + [w for w in itertools.product((1, 2), (1, 2), (0.5, 1), (1, 2)) if w != (1, 1, 1, 1)]


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

    print('Extracting features per country (sweep 1: key pool -> TF-IDF views -> RRF weight grid; sweep 2: reverse lookup -> features)...')
    print('Note: reverse lookup in training runs over the sampled S1 only, so its recall gain is optimistic versus test.')
    t_feat_start = time.time()
    X_train = []
    y_train = []
    X_es = []
    y_es = []
    val_pair_list = []
    stage_hits = collections.defaultdict(collections.Counter)
    rev_stats = collections.Counter()
    miss_cat = collections.Counter()
    missed_examples = []
    missed_empty_addr = 0
    missed_non_ascii = 0
    stores = {}
    countries = sorted({v[5] for v in s1_preprocessed.values()})

    for sweep in (1, 2):
        if sweep == 2:
            n_train_links = sum(len(train_gt[sid]) for sid in train_s1_ids)
            grid_hits = sum(((st['rr'] < TOP_K) & st['label'] & st['train']).sum(axis=1) for st in stores.values())
            best_w = int(np.argmax(grid_hits))
            rrf_weights = RRF_GRID[best_w]
            for w, h in zip(RRF_GRID, grid_hits):
                print(f'  RRF weights (blocking, name, skel, addr) {w}: training top-25 recall {h / max(n_train_links, 1):.6f}')
            true_sims = np.sort(np.concatenate([st['sim'][(reverse_add(st['ps'], st['pt'], st['sim'], st['rr'][best_w]) >= 0) & st['label'] & st['train']] for st in stores.values()]))[::-1]
            min_rev_sim = math.floor(100 * float(true_sims[math.ceil(0.95 * len(true_sims)) - 1])) / 100 if len(true_sims) else 2.0
            print(f'  Chosen RRF weights {rrf_weights}; MIN_REV_SIM = {min_rev_sim:.2f} (keeps 95% of {len(true_sims):,} true reverse additions among training S1)')
        for country in countries:
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

            target_ids = sorted(targets)
            val_raw = {mid: targets[mid] for sid in val_s1_ids if s1_preprocessed[sid][5] == country for mid in val_gt[sid] if mid in targets}
            target_pre = []
            index = collections.defaultdict(list)
            for i, tid in enumerate(target_ids):
                rname, raddr = targets[tid]
                cn, core_n, _, skel = norm.normalize_name(rname)
                ca, nums, _ = norm.normalize_address(raddr, country)
                target_pre.append((cn, core_n, ca, nums, skel))
                if sweep == 1:
                    for k in get_blocking_keys(rname, raddr, country):
                        index[k].append(i)
            del targets
            q_ids = [sid for sid in train_s1_ids + val_s1_ids if s1_preprocessed[sid][5] == country]
            t_index = time.time()

            if sweep == 1:
                pruned = prune_index(index)
                vecs, T = build_views([v[1] for v in target_pre], [v[4] for v in target_pre], [v[2] for v in target_pre])
                S = [v.transform([s1_preprocessed[sid][j] for sid in q_ids]) for v, j in zip(vecs, (1, 4, 2))]
                t_views = time.time()
                parts, old25, t_pool = [], {}, 0.0
                for b0 in range(0, len(q_ids), 5000):
                    t0 = time.time()
                    ps, pt, pc, pv = array('i'), array('i'), array('h'), array('f')
                    for i in range(b0, min(b0 + 5000, len(q_ids))):
                        pool = rank_candidates(s1_blocking_keys[q_ids[i]], index, len(target_ids), POOL_K)
                        ps.extend([i] * len(pool))
                        for arr, col in zip((pt, pc, pv), zip(*pool)):
                            arr.extend(col)
                        if q_ids[i] in val_gt:
                            old25[i] = {t for t, _, _ in rank_candidates(s1_blocking_keys[q_ids[i]], index, len(target_ids), TOP_K)}
                    ps, pt, pc, pv = (np.frombuffer(a, a.typecode) for a in (ps, pt, pc, pv))
                    t_pool += time.time() - t0
                    parts.append((ps, pt, pc, pv) + rerank(ps, pt, pv, S, T, RRF_GRID))
                del index, vecs, T, S
                st = {k: np.concatenate(c, axis=-1) for k, c in zip(('ps', 'pt', 'pc', 'pv', 'cn', 'cs', 'ca', 'sim', 'rr'), zip(*parts))}
                gts = [train_gt[sid] if sid in train_gt else val_gt[sid] for sid in q_ids]
                st['label'] = np.array([target_ids[t] in gts[i] for i, t in zip(st['ps'].tolist(), st['pt'].tolist())], dtype=bool)
                st['train'] = np.array([sid in train_gt for sid in q_ids], dtype=bool)[st['ps']]
                st.update(pruned=pruned, old25=old25)
                stores[country] = st
                del parts, target_pre
                peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == 'darwin' else 2 ** 20)
                print(f'  sweep 1 {country}: {len(target_ids):,} targets, {len(pruned):,} keys pruned, {len(q_ids):,} S1, {len(st["ps"]):,} pooled pairs; '
                      f'index {t_index - c_t0:.1f}s, views {t_views - t_index:.1f}s, pass 1 {t_pool:.1f}s, rerank {time.time() - t_views - t_pool:.1f}s, peak RSS {peak_gb:.2f} GB')
                gc.collect()
                continue

            log_n = math.log(max(len(target_ids), 1))
            df = collections.Counter(t for v in target_pre for t in set(v[1].split()))
            idf = collections.defaultdict(lambda: log_n, {t: log_n - math.log(1 + c) for t, c in df.items()})
            name_count = collections.Counter(v[1].replace(' ', '') for v in target_pre)
            name_freq = [name_count[v[1].replace(' ', '')] for v in target_pre]
            del df, name_count
            val_idx = {tid: i for i, tid in enumerate(target_ids) if tid in val_raw}
            st = stores.pop(country)
            ps, pt, pc, pv, cos_n, cos_s, cos_a, sim = (st[k] for k in ('ps', 'pt', 'pc', 'pv', 'cn', 'cs', 'ca', 'sim'))
            rrf_rank = st['rr'][best_w]
            rev0 = reverse_add(ps, pt, sim, rrf_rank)
            rev_rank = reverse_add(ps, pt, sim, rrf_rank, min_rev_sim)
            final = np.flatnonzero((rrf_rank < TOP_K) | (rev_rank >= 0))
            final = final[np.lexsort((np.where(rev_rank[final] >= 0, TOP_K + rev_rank[final], rrf_rank[final]), ps[final]))]
            bounds = np.searchsorted(ps[final], np.arange(len(q_ids) + 1))
            pool_off = np.searchsorted(ps, np.arange(len(q_ids) + 1))
            t_rev = time.time()
            t_feat = 0.0

            for i, sid in enumerate(q_ids):
                f = final[bounds[i]:bounds[i + 1]]
                pairs = list(zip(*(a[f].tolist() for a in (pt, pc, pv, cos_n, cos_s, cos_a, rrf_rank, rev_rank))))
                s1_tup = s1_preprocessed[sid][:5]
                t0 = time.time()
                rows = add_context([extract_features_for_pair(s1_tup, target_pre[t], target_ids[t], c, idf, [n, k, a, float(r), float(v >= 0), b], name_freq[t]) for t, c, b, n, k, a, r, v in pairs])
                t_feat += time.time() - t0
                cand = [target_ids[t] for t, *_ in pairs]
                if sid not in val_gt:
                    X_part, y_part = (X_es, y_es) if sid in es_s1_set else (X_train, y_train)
                    X_part.extend(rows)
                    y_part.extend(int(tid in train_gt[sid]) for tid in cand)
                    continue
                val_pair_list.extend((sid, tid, feats) for tid, feats in zip(cand, rows))
                skeys = s1_blocking_keys[sid]
                sl = slice(pool_off[i], pool_off[i + 1])
                pos = {t: j for j, t in enumerate(pt[sl].tolist())}
                kept = set(pt[sl][rrf_rank[sl] < TOP_K].tolist())
                add0 = set(pt[sl][rev0[sl] >= 0].tolist())
                add1 = set(pt[sl][rev_rank[sl] >= 0].tolist())
                truth = {val_idx[mid] for mid in val_gt[sid] if mid in val_idx}
                rev_stats.update({'s1': 1, 'size0': len(kept) + len(add0), 'add0': len(add0), 'true0': len(add0 & truth), 'size1': len(kept) + len(add1), 'add1': len(add1), 'true1': len(add1 & truth)})
                for mid in sorted(val_gt[sid]):
                    t = val_idx.get(mid)
                    t_name, t_addr = val_raw.get(mid, ('', ''))
                    hits = {'pool@50': pos.get(t, POOL_K) < 50, 'pool@100': pos.get(t, POOL_K) < 100, 'pool@200': t in pos, 'old@25': t in st['old25'][i], 'rrf@25': t in kept, 'unfiltered': t in kept or t in add0, 'final': t in kept or t in add1}
                    for g in ['all', country] + ([] if t_name.isascii() else ['non-ASCII']):
                        stage_hits[g].update(['n'] + [k for k, h in hits.items() if h])
                    if not hits['final']:
                        shared = skeys & get_blocking_keys(t_name, t_addr, country)
                        miss_cat['no_shared_key' if not shared else 'only_pruned' if shared <= st['pruned'] else 'ranked_out'] += 1
                        missed_empty_addr += not norm.clean_string(t_addr)
                        missed_non_ascii += not t_name.isascii()
                        if len(missed_examples) < 20:
                            missed_examples.append(f'{sid} {s1_raw[sid][0]} | {s1_raw[sid][1]}  <->  {mid} {t_name} | {t_addr}')

            peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == 'darwin' else 2 ** 20)
            print(f'  sweep 2 {country}: {len(final):,} final pairs ({int((rev0 >= 0).sum()):,} reverse additions before the filter, {int((rev_rank >= 0).sum()):,} after); '
                  f'reload+normalize {t_index - c_t0:.1f}s, reverse {t_rev - t_index:.1f}s, pass 2 {time.time() - t_rev:.1f}s, features {1e6 * t_feat / max(len(final), 1):.1f} us/pair, peak RSS {peak_gb:.2f} GB')
            del target_pre, idf, name_freq, st, ps, pt, pc, pv, cos_n, cos_s, cos_a, sim, rrf_rank, rev0, rev_rank, final, val_raw, val_idx
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
        n_estimators=6000,
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
    gain = final_model.model.booster_.feature_importance('gain', iteration=final_model.model.best_iteration_)
    print('Top 20 features by gain: ' + ', '.join(f'{FEATURE_NAMES[i]} {gain[i]:.0f}' for i in np.argsort(-gain, kind='stable')[:20]))
    new_features = FEATURE_NAMES[FEATURE_NAMES.index('blk_score'):FEATURE_NAMES.index('blk_rank')] + ['blk_score_gap', 'gap_cos_name']
    print('New feature gains: ' + ', '.join(f'{n} {gain[FEATURE_NAMES.index(n)]:.0f}' for n in new_features))

    calibrator = IsotonicRegression(out_of_bounds='clip').fit(final_model.predict_proba(X_es), y_es)
    calibrator_path = os.path.join(os.path.dirname(args.model_out) or '.', 'calibrator.joblib')

    print('Evaluating on held-out validation set...')
    X_val = np.array([p[2] for p in val_pair_list], dtype=np.float32)
    val_probas = calibrator.predict(final_model.predict_proba(X_val))

    scores_dict = collections.defaultdict(list)
    for (sid, tid, _), p in zip(val_pair_list, val_probas):
        scores_dict[sid].append((tid, float(p)))

    for sid in val_s1_ids:
        if sid not in scores_dict:
            scores_dict[sid] = []
    retrieved_val_true = stage_hits['all']['final']
    total_val_true = stage_hits['all']['n']

    opt_s2, opt_s3, _ = optimize_source_specific_thresholds(val_gt, scores_dict)
    gate, addon, _ = optimize_gate_addon(val_gt, scores_dict)
    floor = max((float(f) for f in np.round(np.arange(0.05, 0.601, 0.05), 2)), key=lambda f: evaluate_predictions(val_gt, expected_f05_select(scores_dict, f))['macro_f05'])
    rules = {
        'per_source': {'s2': float(opt_s2), 's3': float(opt_s3)},
        'gate_addon': {'gate': gate, 'addon': addon},
        'expected_f05': {'floor': floor}
    }
    rule_metrics = {
        'per_source': evaluate_predictions(val_gt, apply_threshold_and_deduplication(scores_dict, opt_s2, opt_s3)),
        'gate_addon': evaluate_predictions(val_gt, apply_gate_addon(scores_dict, gate, addon)),
        'expected_f05': evaluate_predictions(val_gt, expected_f05_select(scores_dict, floor))
    }
    decision_rule = max(rule_metrics, key=lambda r: rule_metrics[r]['macro_f05'])
    metrics = rule_metrics[decision_rule]

    val_cand_recall = retrieved_val_true / total_val_true if total_val_true > 0 else 0.0

    print('\n' + '=' * 60)
    print('FINAL MODEL VALIDATION RESULTS')
    print('=' * 60)
    for r, m in rule_metrics.items():
        print(f"Rule {r:<13}: {rules[r]}, F0.5 {m['macro_f05']:.6f}, P {m['global_precision']:.6f}, R {m['global_recall']:.6f}, singleton acc {m['singleton_accuracy']:.6f}")
    print(f"Decision rule     : {decision_rule} {rules[decision_rule]} (calibrated probabilities)")
    print(f"RRF weights       : {rrf_weights}, MIN_REV_SIM {min_rev_sim:.2f}, REV_MAX {REV_MAX}")
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
    for g, h in stage_hits.items():
        print(f"Recall {g:<9}: " + ', '.join(f'{k} {h[k] / h["n"]:.6f}' for k in ('pool@50', 'pool@100', 'pool@200', 'old@25', 'rrf@25', 'unfiltered', 'final')) + f' ({h["n"]:,} links)')
    for tag, sfx in (('before filter', '0'), ('after filter ', '1')):
        print(f"Final list {tag}: {rev_stats['size' + sfx] / max(rev_stats['s1'], 1):.2f} per S1, reverse additions {rev_stats['add' + sfx] / max(rev_stats['s1'], 1):.2f} per S1, {rev_stats['true' + sfx] / max(rev_stats['add' + sfx], 1):.6f} of them true")
    print(f"Missed in final   : no_shared_key {miss_cat['no_shared_key']:,}, only_pruned {miss_cat['only_pruned']:,}, ranked_out {miss_cat['ranked_out']:,}")
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
    joblib.dump(calibrator, calibrator_path)

    meta = {
        'decision_rule': decision_rule,
        'decision_params': rules[decision_rule],
        'rule_f05': {r: float(m['macro_f05']) for r, m in rule_metrics.items()},
        'rule_params': rules,
        'calibrator_path': calibrator_path,
        'rrf_weights': list(rrf_weights),
        'min_rev_sim': min_rev_sim,
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
        'pool_k': POOL_K,
        'rev_max': REV_MAX,
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
