import math

import numpy as np

BUCKETS = ("exact", "singleton_fp", "fully_missed", "all_wrong", "extra_only", "missing_only", "extra_and_missing")
COUNTS = ("n_true", "true_s2", "n_pred", "pred_s2", "tp", "tp_s2")


def f05(tp, n_pred, n_true):
    if n_pred == 0 and n_true == 0:
        return 1.0
    return 1.25 * tp / (n_pred + 0.25 * n_true)


def f05_vec(tp, n_pred, n_true):
    tp, n_pred, n_true = (np.asarray(a, dtype=np.float64) for a in (tp, n_pred, n_true))
    den = n_pred + 0.25 * n_true
    return np.divide(1.25 * tp, den, out=np.ones_like(den), where=den > 0)


def _ratio(a, b):
    return float(a) / float(b) if b else 1.0


def _per_s1(truth, pred):
    def gen():
        for s1, t in truth.items():
            p = pred.get(s1, ())
            p = p if isinstance(p, (set, frozenset)) else set(p)
            for ids in (t, p, t & p):
                yield len(ids)
                yield sum(i.startswith("S2-") for i in ids)

    a = np.fromiter(gen(), dtype=np.int64, count=len(COUNTS) * len(truth)).reshape(-1, len(COUNTS))
    return dict(zip(COUNTS, a.T))


def _groups(truth, country_of):
    if country_of is None:
        return {}
    cc = np.array([country_of.get(s, "") for s in truth], dtype=object)
    return {c: cc == c for c in sorted(set(cc))}


def _prf(tp, n_pred, n_true):
    tp, n_pred, n_true = int(tp), int(n_pred), int(n_true)
    return {"tp": tp, "fp": n_pred - tp, "fn": n_true - tp, "precision": _ratio(tp, n_pred), "recall": _ratio(tp, n_true)}


def _links(c):
    s = {k: int(v.sum()) for k, v in c.items()}
    out = _prf(s["tp"], s["n_pred"], s["n_true"])
    out["S2"] = _prf(s["tp_s2"], s["pred_s2"], s["true_s2"])
    out["S3"] = _prf(s["tp"] - s["tp_s2"], s["n_pred"] - s["pred_s2"], s["n_true"] - s["true_s2"])
    return out


def evaluate(truth, pred, country_of=None):
    c = _per_s1(truth, pred)
    tp, n_pred, n_true = c["tp"], c["n_pred"], c["n_true"]
    f = f05_vec(tp, n_pred, n_true)
    n = len(f)
    fp, fn = n_pred - tp, n_true - tp
    macro = math.fsum(f) / n
    single = n_true == 0
    label = np.select([(fp == 0) & (fn == 0), single, n_pred == 0, tp == 0, fn == 0, fp == 0], list(range(6)), 6)
    buckets = {b: {"count": int((label == k).sum()), "loss": math.fsum(1 - f[label == k]) / n} for k, b in enumerate(BUCKETS)}
    assert abs(math.fsum(v["loss"] for v in buckets.values()) - (1 - macro)) < 1e-9
    out = {
        "n": n,
        "macro_f05": macro,
        "extra_pred_keys": sum(1 for k in pred if k not in truth),
        "singletons": {"n": int(single.sum()), "correct": int((single & (n_pred == 0)).sum()), "false_merges": int((single & (n_pred > 0)).sum())},
        "non_singletons": {"n": int((~single).sum()), "macro_f05": _ratio(math.fsum(f[~single]), (~single).sum())},
        "links": _links(c),
        "buckets": buckets,
    }
    groups = _groups(truth, country_of)
    if groups:
        out["countries"] = {k: {"n": int(m.sum()), "macro_f05": _ratio(math.fsum(f[m]), m.sum())} for k, m in groups.items()}
    return out


def _coverage(c, m, space=None):
    g = {k: v[m] for k, v in c.items()}
    tp, n_true, n_pred = g["tp"], g["n_true"], g["n_pred"]
    ns = n_true > 0
    out = {
        "n_s1": int(m.sum()),
        "link_recall": _ratio(tp.sum(), n_true.sum()),
        "link_recall_s2": _ratio(g["tp_s2"].sum(), g["true_s2"].sum()),
        "link_recall_s3": _ratio((tp - g["tp_s2"]).sum(), (n_true - g["true_s2"]).sum()),
        "all_found": _ratio((tp == n_true)[ns].sum(), ns.sum()),
        "any_found": _ratio((tp > 0)[ns].sum(), ns.sum()),
        "avg_candidates": _ratio(n_pred.sum(), len(n_pred)),
        "max_candidates": int(n_pred.max(initial=0)),
        "zero_candidates": int((n_pred == 0).sum()),
        "candidate_pairs": int(n_pred.sum()),
        "oracle_macro_f05": _ratio(math.fsum(f05_vec(tp, tp, n_true)), len(tp)),
    }
    if space:
        out["reduction_ratio"] = 1 - out["candidate_pairs"] / space
    return out


def candidate_report(truth, candidates, country_of=None, pool_sizes=None):
    c = _per_s1(truth, candidates)
    pool_sizes = pool_sizes or {}
    out = _coverage(c, np.ones(len(truth), dtype=bool), sum(a * b for a, b in pool_sizes.values()))
    groups = _groups(truth, country_of)
    if groups:
        out["countries"] = {k: _coverage(c, m, math.prod(pool_sizes.get(k, (0, 0)))) for k, m in groups.items()}
    return out


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.6f}"
    return str(v)


def _lines(d, depth):
    for k, v in d.items():
        pad = "  " * depth
        if isinstance(v, dict):
            yield f"{pad}{k}:"
            yield from _lines(v, depth + 1)
        else:
            yield f"{pad}{k}: {_fmt(v)}"


def format_report(d):
    return "\n".join(_lines(d, 0))
