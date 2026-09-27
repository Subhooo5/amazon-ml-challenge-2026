import sys
import math
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
import normalization as norm

FEATURE_NAMES = [
    'name_exact_clean',
    'name_exact_core',
    'name_lev_sim',
    'name_jw_sim',
    'name_token_sort',
    'name_token_set',
    'name_token_jaccard',
    'name_token_overlap',
    'name_len_diff',
    'name_len_ratio',
    'name_prefix_match',
    'name_char_3gram',
    'name_first_token_match',
    'name_first_token_conflict',
    'addr_has_val',
    'addr_exact_clean',
    'addr_lev_sim',
    'addr_jw_sim',
    'addr_token_sort',
    'addr_token_set',
    'addr_token_jaccard',
    'num_common_count',
    'num_has_match',
    'num_has_mismatch',
    'exact_core_no_addr',
    'cross_prod',
    'cross_sum',
    'cross_max',
    'cross_min',
    'is_source2',
    'is_source3',
    'shared_keys_count',
    'skel_token_set',
    'skel_equal',
    'legal_equal',
    'legal_conflict',
    'desc_conflict',
    'idf_jaccard',
    'max_idf_unshared',
    'cos_name',
    'cos_skel',
    'cos_addr',
    'rrf_rank',
    'is_reverse',
    'blk_score',
    'name_partial',
    'compact_ratio',
    'acronym_match',
    's1_addr_missing',
    'num_jaccard',
    'name_pool_freq',
    'blk_rank',
    'n_cands',
    'gap_name_set',
    'gap_name_jw',
    'gap_addr_set',
    'gap_cross_prod',
    'gap_skel_set',
    'best_name_set',
    'best_cross_prod',
    'blk_score_gap',
    'gap_cos_name'
]

LEGAL_CANON = {
    'incorporated': 'inc', 'corporation': 'corp', 'company': 'co', 'cie': 'co',
    'limited': 'ltd', 'private': 'pvt'
}

GAP_COLS = [FEATURE_NAMES.index(n) for n in ('name_token_set', 'name_jw_sim', 'addr_token_set', 'cross_prod', 'skel_token_set')]
BEST_COLS = [FEATURE_NAMES.index(n) for n in ('name_token_set', 'cross_prod')]
BLK_COL = FEATURE_NAMES.index('blk_score')
COS_NAME_COL = FEATURE_NAMES.index('cos_name')


def char_ngrams(s, n=3):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def extract_features_for_pair(s1_tuple, target_tuple, target_id, shared_keys=1, idf=None, retrieval=(), name_freq=0):
    s1_name, s1_core, s1_addr, s1_nums, s1_skel = s1_tuple
    t_name, t_core, t_addr, t_nums, t_skel = target_tuple

    exact_clean = 1.0 if s1_name == t_name and s1_name else 0.0
    exact_core = 1.0 if s1_core == t_core and s1_core else 0.0
    
    n_lev = Levenshtein.normalized_similarity(s1_name, t_name) if s1_name and t_name else 0.0
    n_jw = JaroWinkler.similarity(s1_name, t_name) if s1_name and t_name else 0.0
    n_sort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0
    n_set = fuzz.token_set_ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0
    
    s1_toks = s1_core.split()
    t_toks = t_core.split()
    if s1_toks and t_toks:
        s1_s = set(s1_toks)
        t_s = set(t_toks)
        intersection = len(s1_s & t_s)
        union = len(s1_s | t_s)
        n_jaccard = intersection / union if union > 0 else 0.0
        n_overlap = intersection / min(len(s1_s), len(t_s))
        
        f1_sim = Levenshtein.normalized_similarity(s1_toks[0], t_toks[0])
        first_token_match = 1.0 if f1_sim >= 0.80 else 0.0
        first_token_conflict = 1.0 if f1_sim < 0.50 else 0.0
    else:
        n_jaccard = 0.0
        n_overlap = 0.0
        first_token_match = 0.0
        first_token_conflict = 0.0

    len1 = len(s1_name)
    len2 = len(t_name)
    n_len_diff = abs(len1 - len2)
    n_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
    prefix_match = 1.0 if len(s1_core) >= 4 and len(t_core) >= 4 and s1_core[:4] == t_core[:4] else 0.0

    ng1 = char_ngrams(s1_core, 3)
    ng2 = char_ngrams(t_core, 3)
    if ng1 and ng2:
        char_3gram_sim = len(ng1 & ng2) / len(ng1 | ng2)
    else:
        char_3gram_sim = 0.0

    has_addr = 1.0 if t_addr else 0.0
    exact_core_no_addr = 1.0 if (not has_addr and exact_core == 1.0) else 0.0

    if has_addr and s1_addr:
        a_exact = 1.0 if s1_addr == t_addr else 0.0
        a_lev = Levenshtein.normalized_similarity(s1_addr, t_addr)
        a_jw = JaroWinkler.similarity(s1_addr, t_addr)
        a_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
        a_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
        
        s1_a_toks = set(s1_addr.split())
        t_a_toks = set(t_addr.split())
        a_union = len(s1_a_toks | t_a_toks)
        a_jaccard = len(s1_a_toks & t_a_toks) / a_union if a_union > 0 else 0.0
        
        shared_nums = len(s1_nums & t_nums)
        has_num_match = 1.0 if shared_nums > 0 else 0.0
        has_num_mismatch = 1.0 if len(s1_nums) > 0 and len(t_nums) > 0 and shared_nums == 0 else 0.0
    else:
        a_exact = 0.0
        a_lev = 0.0
        a_jw = 0.0
        a_sort = 0.0
        a_set = 0.0
        a_jaccard = 0.0
        shared_nums = 0
        has_num_match = 0.0
        has_num_mismatch = 0.0

    effective_addr_sim = a_sort if has_addr else n_sort
    cross_prod = n_sort * effective_addr_sim
    cross_sum = n_sort + effective_addr_sim
    cross_max = max(n_sort, effective_addr_sim)
    cross_min = min(n_sort, effective_addr_sim)

    is_s2 = 1.0 if target_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if target_id.startswith('S3-') else 0.0

    skel_set = fuzz.token_set_ratio(s1_skel, t_skel) / 100.0 if s1_skel and t_skel else 0.0
    skel_equal = 1.0 if s1_skel and s1_skel == t_skel else 0.0

    s1_legal = {LEGAL_CANON.get(t, t) for t in s1_name.split() if t in norm.LEGAL_SUFFIXES and t != 'france'}
    t_legal = {LEGAL_CANON.get(t, t) for t in t_name.split() if t in norm.LEGAL_SUFFIXES and t != 'france'}
    legal_equal = 1.0 if s1_legal and s1_legal == t_legal else 0.0
    legal_conflict = 1.0 if s1_legal and t_legal and not s1_legal & t_legal else 0.0
    s1_desc = norm.DESCRIPTORS.intersection(s1_name.split())
    t_desc = norm.DESCRIPTORS.intersection(t_name.split())
    desc_conflict = 1.0 if s1_desc and t_desc and not s1_desc & t_desc else 0.0

    if idf is not None and s1_toks and t_toks:
        union_w = sum(idf[t] for t in s1_s | t_s)
        idf_jaccard = sum(idf[t] for t in s1_s & t_s) / union_w if union_w > 0 else 0.0
        max_idf_unshared = max((idf[t] for t in s1_s ^ t_s), default=0.0)
    else:
        idf_jaccard = 0.0
        max_idf_unshared = 0.0

    s1_compact = s1_core.replace(' ', '')
    t_compact = t_core.replace(' ', '')
    partial = fuzz.partial_ratio(s1_core, t_core) / 100.0 if s1_core and t_core else 0.0
    compact_ratio = Levenshtein.normalized_similarity(s1_compact, t_compact) if s1_compact and t_compact else 0.0
    acronym = 1.0 if any(len(a) >= 2 and len(b) == 1 and ''.join(w[0] for w in a) == b[0] for a, b in ((s1_toks, t_toks), (t_toks, s1_toks))) else 0.0
    num_jaccard = len(s1_nums & t_nums) / len(s1_nums | t_nums) if s1_nums and t_nums else 0.0

    return [
        exact_clean,
        exact_core,
        n_lev,
        n_jw,
        n_sort,
        n_set,
        n_jaccard,
        n_overlap,
        n_len_diff,
        n_len_ratio,
        prefix_match,
        char_3gram_sim,
        first_token_match,
        first_token_conflict,
        has_addr,
        a_exact,
        a_lev,
        a_jw,
        a_sort,
        a_set,
        a_jaccard,
        float(shared_nums),
        has_num_match,
        has_num_mismatch,
        exact_core_no_addr,
        cross_prod,
        cross_sum,
        cross_max,
        cross_min,
        is_s2,
        is_s3,
        float(shared_keys),
        skel_set,
        skel_equal,
        legal_equal,
        legal_conflict,
        desc_conflict,
        idf_jaccard,
        max_idf_unshared
    ] + list(retrieval) + [
        partial,
        compact_ratio,
        acronym,
        0.0 if s1_addr else 1.0,
        num_jaccard,
        math.log1p(name_freq)
    ]


def add_context(rows):
    best = {c: max((r[c] for r in rows), default=0.0) for c in GAP_COLS + [BLK_COL, COS_NAME_COL]}
    for rank, r in enumerate(rows):
        r.extend([float(rank), float(len(rows))])
        r.extend(r[c] - best[c] for c in GAP_COLS)
        r.extend(1.0 if r[c] == best[c] else 0.0 for c in BEST_COLS)
        r.extend([best[BLK_COL] - r[BLK_COL], r[COS_NAME_COL] - best[COS_NAME_COL]])
    return rows
