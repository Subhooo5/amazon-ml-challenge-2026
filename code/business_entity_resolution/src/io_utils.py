import json
import os
import re
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from config import VALIDATOR, log

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")
TARGET_PREFIXES = ("S2-", "S3-")
ID = r"S[23]-[^\t,\n\r ]*"
ID_RE = re.compile(ID)
ID_LIST_RE = re.compile(f"{ID}(?:,{ID})*")


class Table(NamedTuple):
    ids: list[str]
    names: list[str]
    addrs: list[str]


def _examples(items):
    return sorted(items)[:5]


@contextmanager
def _atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_source(path, countries=None):
    path = Path(path)
    prefix = f"S{path.stem[-1]}-"
    keep = None if countries is None else set(countries)
    stats = {"rows": 0, "blank": 0, "short": 0, "long": 0, "bad_prefix": 0, "countries": {}}
    per_country = stats["countries"]
    order, by_country, seen, dups = [], {}, set(), []
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        missing = [c for c in SOURCE_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path}: header {header} lacks {missing}")
        i_id, i_name, i_addr, i_country = map(header.index, SOURCE_COLUMNS)
        width = len(header)
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                stats["blank"] += 1
                continue
            stats["rows"] += 1
            parts = line.split("\t")
            if len(parts) < width:
                stats["short"] += 1
                parts += [""] * (width - len(parts))
            if len(parts) > width:
                stats["long"] += 1
                name, addr, country = parts[i_name], " ".join(parts[i_name + 1:-1]), parts[-1]
            else:
                name, addr, country = parts[i_name], parts[i_addr], parts[i_country]
            eid, country = parts[i_id].strip(), country.strip()
            if eid in seen:
                dups.append(eid)
            seen.add(eid)
            if not eid.startswith(prefix):
                stats["bad_prefix"] += 1
            per_country[country] = per_country.get(country, 0) + 1
            if keep is None or country in keep:
                if country not in by_country:
                    by_country[country] = Table([], [], [])
                table = by_country[country]
                table.ids.append(eid)
                table.names.append(name.strip())
                table.addrs.append(addr.strip())
                order.append(eid)
    if dups:
        raise ValueError(f"{path}: {len(dups)} duplicate ids, e.g. {_examples(dups)}")
    return order, by_country, stats


def read_truth(path):
    truth = {}
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header != TRUTH_COLUMNS:
            raise ValueError(f"{path}: header {header} != {TRUTH_COLUMNS}")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0].strip()
            ids = [i for i in (x.strip() for x in (parts[1] if len(parts) > 1 else "").split(",")) if i]
            if s1 in truth:
                raise ValueError(f"{path}: duplicate row for {s1}")
            if not all(i.startswith(TARGET_PREFIXES) for i in ids):
                raise ValueError(f"{path}: {s1} has ids without an S2-/S3- prefix: {ids}")
            if len(set(ids)) != len(ids):
                raise ValueError(f"{path}: {s1} has duplicate ids: {ids}")
            truth[s1] = frozenset(ids)
    return truth


def write_id_lists(path, header_cols, ordered_s1, mapping):
    order = set(ordered_s1)
    if len(order) != len(ordered_s1):
        raise ValueError(f"duplicate S1 ids in order, e.g. {_examples(k for k, v in Counter(ordered_s1).items() if v > 1)}")
    extra = mapping.keys() - order
    if extra:
        raise ValueError(f"{len(extra)} S1 ids not in order, e.g. {_examples(extra)}")
    with _atomic(path) as f:
        f.write("\t".join(header_cols) + "\n")
        for s1 in ordered_s1:
            ids = sorted({i.strip() for i in mapping.get(s1, ())} - {""})
            joined = ",".join(ids)
            if ids and (joined.count(",") != len(ids) - 1 or not ID_LIST_RE.fullmatch(joined)):
                raise ValueError(f"{s1}: invalid ids {[i for i in ids if not ID_RE.fullmatch(i)][:5]}")
            f.write(f"{s1}\t{joined}\n")


def write_outputs(out_dir, ordered_s1, matches, candidates, valid_ids=None):
    outside = [s for s, m in matches.items() if not set(m) <= set(candidates.get(s, ()))]
    if outside:
        raise ValueError(f"{len(outside)} S1 have matches outside their candidates, e.g. {_examples(outside)}")
    if valid_ids is not None:
        unknown = {i for m in (matches, candidates) for ids in m.values() for i in ids if i not in valid_ids}
        if unknown:
            raise ValueError(f"{len(unknown)} ids not in the target set, e.g. {_examples(unknown)}")
    out_dir = Path(out_dir)
    matching, candidate = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    write_id_lists(candidate, CANDIDATE_COLUMNS, ordered_s1, candidates)
    write_id_lists(matching, MATCH_COLUMNS, ordered_s1, matches)
    return matching, candidate


def run_validator(matching, candidate, test_dir, check_ids=False):
    if not VALIDATOR.exists():
        log(f"WARNING: validator not found at {VALIDATOR}")
        return None, ""
    cmd = [sys.executable, str(VALIDATOR), "--matching", str(matching), "--candidate", str(candidate), "--test-dir", str(test_dir)]
    result = subprocess.run(cmd + (["--check-ids"] if check_ids else []), capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def write_json(path, obj):
    with _atomic(path) as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
