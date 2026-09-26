import collections
import heapq
import math
import normalization as norm

TOP_K = 25
SINGLE_CAP = 300
COMBO_CAP = 1000
NAME_KEYS = {'compact_n', 'compact_sort_n', 'core_n', 'n2', 'sort_n', 'n1', 'n_tok'}

COMMON_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'drive', 'court', 'lane', 'place',
    'circle', 'way', 'trail', 'parkway', 'highway', 'suite', 'floor', 'apartment',
    'building', 'room', 'number', 'near', 'opposite', 'behind', 'block', 'sector',
    'phase', 'plot', 'flat', 'door', 'fl', 'no', 'unit', 'north', 'south', 'east', 'west',
    'india', 'us', 'usa', 'france', 'state', 'district', 'city', 'nagar', 'colony',
    'bazaar', 'marg', 'gali', 'null', 'rd', 'st', 'ave', 'blvd', 'dr', 'ct', 'ln',
    'hwy', 'apt', 'ste', 'first', 'second', 'third', 'ground'
}

STATE_WORDS = set(' '.join(norm.STATE_MAP.values()).split())


def get_blocking_keys(name, addr, country):
    c_n, core_n, sort_n = norm.normalize_name(name)
    c_a, nums, sort_a = norm.normalize_address(addr)

    keys = set()
    n_tokens = core_n.split()
    a_tokens = c_a.split()

    compact_name = ''.join(n_tokens)
    if len(compact_name) >= 4:
        keys.add(('compact_n', compact_name))
        keys.add(('compact_sort_n', ''.join(sorted(n_tokens))))

    if len(core_n) >= 3:
        keys.add(('core_n', core_n))

    if len(n_tokens) >= 2:
        keys.add(('n2', f'{n_tokens[0]}_{n_tokens[1]}'))
        keys.add(('sort_n', ' '.join(sorted(n_tokens))))
    elif len(n_tokens) == 1 and len(n_tokens[0]) >= 3:
        keys.add(('n1', n_tokens[0]))

    for t in n_tokens:
        if len(t) >= 4:
            keys.add(('n_tok', t))

    places = sorted({t for t in a_tokens if t.isalpha() and len(t) >= 3 and t not in COMMON_ADDR_WORDS and t not in STATE_WORDS})
    nums = sorted(nums)[:3]
    nt = [t for t in n_tokens if len(t) >= 3][:2]

    keys.update(('np', t, p) for t in nt for p in places[:8])
    keys.update(('nump', n, p) for n in nums for p in places[:4])
    keys.update(('addr_pair', a, b) for i, a in enumerate(places[:4]) for b in places[i + 1:4])
    keys.update(('name_num', t, n) for t in nt for n in nums)

    return keys


def prune_index(index):
    pruned = [k for k, postings in index.items() if len(postings) > (SINGLE_CAP if k[0] in NAME_KEYS else COMBO_CAP)]
    for k in pruned:
        del index[k]
    return len(pruned)


def rank_candidates(keys, index, n_targets, top_k=TOP_K):
    score = collections.defaultdict(float)
    count = collections.Counter()
    for k in sorted(keys):
        postings = index.get(k)
        if postings:
            w = math.log(1 + n_targets / len(postings))
            for tid in postings:
                score[tid] += w
                count[tid] += 1
    return [(tid, count[tid]) for tid in heapq.nsmallest(top_k, score, key=lambda t: (-score[t], t))]
