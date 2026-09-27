import math
import numpy as np
from evaluation import evaluate_predictions


def optimize_global_threshold(ground_truth_dict, candidate_scores_dict, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.90, 31)

    best_threshold = 0.5
    best_metrics = None
    best_f05 = -1.0

    for thresh in thresholds:
        preds = {}
        for s1_id, scores in candidate_scores_dict.items():
            matched = {tid for tid, p in scores if p >= thresh}
            preds[s1_id] = matched

        metrics = evaluate_predictions(ground_truth_dict, preds)
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_threshold = float(thresh)
            best_metrics = metrics

    return best_threshold, best_metrics


def optimize_source_specific_thresholds(ground_truth_dict, candidate_scores_dict,
                                        s2_grid=None, s3_grid=None):
    if s2_grid is None:
        s2_grid = np.linspace(0.40, 0.85, 10)
    if s3_grid is None:
        s3_grid = np.linspace(0.40, 0.85, 10)

    best_s2 = 0.5
    best_s3 = 0.5
    best_metrics = None
    best_f05 = -1.0

    for t2 in s2_grid:
        for t3 in s3_grid:
            preds = {}
            for s1_id, scores in candidate_scores_dict.items():
                matched = set()
                for tid, p in scores:
                    if tid.startswith('S2-') and p >= t2:
                        matched.add(tid)
                    elif tid.startswith('S3-') and p >= t3:
                        matched.add(tid)
                preds[s1_id] = matched

            metrics = evaluate_predictions(ground_truth_dict, preds)
            if metrics['macro_f05'] > best_f05:
                best_f05 = metrics['macro_f05']
                best_s2 = float(t2)
                best_s3 = float(t3)
                best_metrics = metrics

    return best_s2, best_s3, best_metrics


def apply_threshold_and_deduplication(candidate_scores_dict, s2_threshold=0.5, s3_threshold=0.5):
    all_pairs = []
    for s1_id, scores in candidate_scores_dict.items():
        for tid, p in scores:
            t = s2_threshold if tid.startswith('S2-') else s3_threshold
            if p >= t:
                all_pairs.append((p, s1_id, tid))

    all_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_targets = set()
    result = {s1_id: set() for s1_id in candidate_scores_dict.keys()}

    for p, s1_id, tid in all_pairs:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            result[s1_id].add(tid)

    return result


def apply_gate_addon(scores_dict, gate, addon, margin=1.0):
    kept = {}
    for s1_id, scores in scores_dict.items():
        ranked = sorted(scores, key=lambda x: (-x[1], x[0]))
        kept[s1_id] = [(tid, p) for i, (tid, p) in enumerate(ranked) if ranked[0][1] >= gate and (i == 0 or (p >= addon and ranked[0][1] - p <= margin))]
    return apply_threshold_and_deduplication(kept, 0.0, 0.0)


def optimize_gate_addon(ground_truth_dict, candidate_scores_dict):
    best_gate, best_addon, best_metrics, best_f05 = 0.5, 0.5, None, -1.0
    for gate in np.round(np.arange(0.30, 0.901, 0.02), 2):
        for addon in np.round(np.arange(0.40, 0.951, 0.02), 2):
            metrics = evaluate_predictions(ground_truth_dict, apply_gate_addon(candidate_scores_dict, gate, addon))
            if metrics['macro_f05'] > best_f05:
                best_f05 = metrics['macro_f05']
                best_gate, best_addon, best_metrics = float(gate), float(addon), metrics
    best_margin = 1.0
    for margin in (0.05, 0.10, 0.20, 0.30):
        metrics = evaluate_predictions(ground_truth_dict, apply_gate_addon(candidate_scores_dict, best_gate, best_addon, margin))
        if metrics['macro_f05'] > best_f05:
            best_f05, best_margin, best_metrics = metrics['macro_f05'], margin, metrics
    g0, a0 = best_gate, best_addon
    for gate in np.round(g0 + np.arange(-0.04, 0.041, 0.02), 2):
        for addon in np.round(a0 + np.arange(-0.04, 0.041, 0.02), 2):
            metrics = evaluate_predictions(ground_truth_dict, apply_gate_addon(candidate_scores_dict, gate, addon, best_margin))
            if metrics['macro_f05'] > best_f05:
                best_f05 = metrics['macro_f05']
                best_gate, best_addon, best_metrics = float(gate), float(addon), metrics
    return best_gate, best_addon, best_margin, best_metrics


def expected_f05_select(scores_dict, floor):
    kept = {}
    for s1_id, scores in scores_dict.items():
        ranked = sorted(scores, key=lambda x: (-x[1], x[0]))
        total = sum(p for _, p in ranked)
        best_e, k, acc = math.prod(1.0 - p for _, p in ranked), 0, 0.0
        for j, (_, p) in enumerate(ranked):
            if p < floor:
                break
            acc += p
            if 1.25 * acc / (j + 1 + 0.25 * total) > best_e:
                best_e, k = 1.25 * acc / (j + 1 + 0.25 * total), j + 1
        kept[s1_id] = ranked[:k]
    return apply_threshold_and_deduplication(kept, 0.0, 0.0)
