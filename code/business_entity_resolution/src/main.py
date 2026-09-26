import argparse
import gc
import resource
import sys
import time

import numpy as np

from config import CACHE_DIR, SEED, START, data_dir, log, source_path, truth_path
from io_utils import read_source, read_truth, run_validator, write_outputs
from metrics import candidate_report, evaluate, f05, f05_vec, format_report

ANOMALY_KEYS = ("blank", "short", "long", "bad_prefix")
SELFTEST_DIR = CACHE_DIR / "selftest"
SELFTEST_S1 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tA\tX\tUS\nS1-2\tB\tY\tUS\nS1-3\tC\tZ\tIndia\n"
SELFTEST_ORDER = ["S1-1", "S1-2", "S1-3"]
SELFTEST_CANDIDATES = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-2"]}
SELFTEST_MATCHES = {"S1-1": ["S3-1"]}


def _report(name, ok):
    print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)
    return ok


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def _reference_f05(tp, n_pred, n_true):
    if tp == 0:
        return 1.0 if n_pred == n_true == 0 else 0.0
    p, r = tp / n_pred, tp / n_true
    return 1.25 * p * r / (0.25 * p + r)


def _random_triples():
    rng = np.random.default_rng(SEED)
    n_pred = rng.integers(0, 41, 100_000)
    n_true = rng.integers(0, 12, 100_000)
    tp = rng.integers(0, np.minimum(n_pred, n_true) + 1)
    return list(zip(tp.tolist(), n_pred.tolist(), n_true.tolist()))


def _check_closed_form():
    return all(abs(f05(*t) - _reference_f05(*t)) <= 1e-12 for t in _random_triples())


def _check_vectorized():
    triples = _random_triples()
    scalar = np.array([f05(*t) for t in triples])
    return bool(np.abs(f05_vec(*np.array(triples).T) - scalar).max() <= 1e-12)


def _check_evaluate():
    truth = {
        "S1-A": set(), "S1-B": {"S2-1", "S3-1"}, "S1-C": set(), "S1-D": {"S2-2"},
        "S1-E": {"S2-3"}, "S1-F": {"S2-4"}, "S1-G": {"S2-5", "S3-5"}, "S1-H": {"S2-6", "S3-6"},
    }
    pred = {
        "S1-B": {"S2-1", "S3-1"}, "S1-C": {"S2-9"}, "S1-E": {"S3-3"}, "S1-F": {"S2-4", "S3-4"},
        "S1-G": {"S2-5"}, "S1-H": {"S2-6", "S3-7"}, "S1-X": {"S2-8"},
    }
    country_of = {k: "US" if k < "S1-E" else "India" for k in truth}
    r = evaluate(truth, pred, country_of)
    counts = {b: v["count"] for b, v in r["buckets"].items()}
    losses = sum(v["loss"] for v in r["buckets"].values())
    return (
        abs(r["macro_f05"] - 35 / 72) < 1e-12
        and counts == {"exact": 2, "singleton_fp": 1, "fully_missed": 1, "all_wrong": 1, "extra_only": 1, "missing_only": 1, "extra_and_missing": 1}
        and abs(losses - (1 - r["macro_f05"])) < 1e-9
        and r["extra_pred_keys"] == 1
        and r["singletons"] == {"n": 2, "correct": 1, "false_merges": 1}
        and (r["links"]["tp"], r["links"]["fp"], r["links"]["fn"]) == (5, 4, 4)
        and (r["links"]["S2"]["tp"], r["links"]["S3"]["tp"]) == (4, 1)
        and abs(r["countries"]["US"]["macro_f05"] - 0.5) < 1e-12
        and abs(r["countries"]["India"]["macro_f05"] - 17 / 36) < 1e-12
    )


def _check_validator():
    SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    (SELFTEST_DIR / "test_source1.tsv").write_text(SELFTEST_S1, encoding="utf-8")
    paths = write_outputs(SELFTEST_DIR / "out", SELFTEST_ORDER, SELFTEST_MATCHES, SELFTEST_CANDIDATES)
    code, out = run_validator(*paths, SELFTEST_DIR)
    if code != 0:
        print(out)
    return code == 0


def selftest(args):
    bad = SELFTEST_DIR / "bad"
    checks = [
        ("f05(2, 3, 2) rounds to 0.714", lambda: round(f05(2, 3, 2), 3) == 0.714),
        ("f05(0, 0, 0) == 1", lambda: f05(0, 0, 0) == 1),
        ("f05(0, 1, 0) == 0", lambda: f05(0, 1, 0) == 0),
        ("f05(0, 0, 1) == 0", lambda: f05(0, 0, 1) == 0),
        ("f05(1, 1, 1) == 1", lambda: f05(1, 1, 1) == 1),
        ("f05 == (1.25PR)/(0.25P+R) on 100,000 random triples", _check_closed_form),
        ("f05_vec == f05 on 100,000 random triples", _check_vectorized),
        ("evaluate buckets, macro, links, countries", _check_evaluate),
        ("write_outputs + validator PASS", _check_validator),
        ("write_outputs raises on match outside candidates", lambda: _raises(write_outputs, bad, SELFTEST_ORDER, {"S1-2": ["S2-1"]}, SELFTEST_CANDIDATES)),
        ("write_outputs raises on duplicate S1", lambda: _raises(write_outputs, bad, SELFTEST_ORDER + ["S1-1"], SELFTEST_MATCHES, SELFTEST_CANDIDATES)),
        ("write_outputs raises on bad prefix", lambda: _raises(write_outputs, bad, SELFTEST_ORDER, {}, {"S1-3": ["S1-9"]})),
    ]
    for name, fn in checks:
        if not _report(name, fn()):
            return 1
    return 0


def _read_split(root, split, anomalies):
    out = []
    for n in (1, 2, 3):
        path = source_path(root, split, n)
        order, by_country, stats = read_source(path)
        log(path.name)
        print(format_report(stats), flush=True)
        anomalies.update({f"{path.name} {k}": stats[k] for k in ANOMALY_KEYS if stats[k]})
        out.append((order, by_country))
    return out


def _check_train(root, anomalies):
    (_, s1), s2, s3 = _read_split(root, "train", anomalies)
    truth = read_truth(truth_path(root))
    log("train_ground_truth.tsv")
    country_of = {i: c for c, t in s1.items() for i in t.ids}
    assert truth.keys() == country_of.keys(), "ground-truth S1 ids differ from train_source1 ids"
    target_country = {i: c for _, tables in (s2, s3) for c, t in tables.items() for i in t.ids}
    del s1, s2, s3
    counts = {"missing": 0, "cross_country": 0, "multi_assigned": 0}
    seen, multi = set(), set()
    for s1_id, ids in truth.items():
        for i in ids:
            c = target_country.get(i)
            if c is None:
                counts["missing"] += 1
            elif c != country_of[s1_id]:
                counts["cross_country"] += 1
            if i in seen:
                multi.add(i)
            seen.add(i)
    counts["multi_assigned"] = len(multi)
    del target_country, seen, multi
    sizes = np.fromiter((len(v) for v in truth.values()), dtype=np.int64, count=len(truth))
    s2n = np.fromiter((sum(i.startswith("S2-") for i in v) for v in truth.values()), dtype=np.int64, count=len(truth))
    summary = {
        "s1": len(truth),
        "singletons": int((sizes == 0).sum()),
        "matched_ids": int(sizes.sum()),
        "max_per_s1": int(sizes.max()),
        "max_s2_per_s1": int(s2n.max()),
        "max_s3_per_s1": int((sizes - s2n).max()),
        **counts,
    }
    print(format_report({"ground_truth": summary}), flush=True)
    anomalies.update({k: v for k, v in counts.items() if v})
    ok = _report("evaluate(truth, truth) == 1.0", evaluate(truth, truth)["macro_f05"] == 1.0)
    empty = evaluate(truth, {}, country_of)
    ok &= _report("evaluate(truth, empty) == singletons / S1", empty["macro_f05"] == summary["singletons"] / summary["s1"])
    ok &= _report("candidate_report(truth, truth) oracle == 1.0", candidate_report(truth, truth)["oracle_macro_f05"] == 1.0)
    print(format_report({"empty_baseline": empty}), flush=True)
    return ok


def _check_test(root, anomalies):
    (s1_order, _), (s2_order, s2), (_, s3) = _read_split(root, "test", anomalies)
    valid = {i for tables in (s2, s3) for t in tables.values() for i in t.ids}
    pairs = {s: [s2_order[0]] for s in s1_order[:10]}
    ok = True
    for name, matches, candidates, valid_ids in (("empty", {}, {}, None), ("nonempty", pairs, pairs, valid)):
        paths = write_outputs(CACHE_DIR / "smoke" / name, s1_order, matches, candidates, valid_ids)
        code, out = run_validator(*paths, root / "test")
        print(out, flush=True)
        ok &= _report(f"validator smoke set {name}", code == 0)
    return ok


def check(args):
    root = data_dir(args.data_dir)
    log(f"data dir {root}")
    anomalies = {}
    ok = True
    if args.split in ("train", "all"):
        ok &= _check_train(root, anomalies)
        gc.collect()
    if args.split in ("test", "all"):
        ok &= _check_test(root, anomalies)
        gc.collect()
    print(format_report({"anomalies": anomalies}) if anomalies else "anomalies: none", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    log(f"peak RSS {rss / 2**30:.2f} GiB, total {time.perf_counter() - START:.1f}s")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(prog="main.py")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest").set_defaults(func=selftest)
    p = sub.add_parser("check")
    p.add_argument("--data-dir")
    p.add_argument("--split", choices=["train", "test", "all"], default="all")
    p.set_defaults(func=check)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
