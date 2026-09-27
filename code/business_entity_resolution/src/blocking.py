import collections
import heapq
import math
import normalization as norm

TOP_K = 25
NAME_K = 8
SINGLE_CAP = 500
COMBO_CAP = 2000
NAME_KEYS = {'compact_n', 'compact_sort_n', 'core_n', 'n2', 'sort_n', 'n1', 'n_tok', 'sk', 'compact_strict', 'skn'}

COMMON_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'drive', 'court', 'lane', 'place',
    'circle', 'way', 'trail', 'parkway', 'highway', 'suite', 'floor', 'apartment',
    'building', 'room', 'number', 'near', 'opposite', 'behind', 'block', 'sector',
    'phase', 'plot', 'flat', 'door', 'fl', 'no', 'unit', 'north', 'south', 'east', 'west',
    'india', 'us', 'usa', 'france', 'state', 'district', 'city', 'nagar', 'colony',
    'bazaar', 'marg', 'gali', 'null', 'rd', 'st', 'ave', 'blvd', 'dr', 'ct', 'ln',
    'hwy', 'apt', 'ste', 'first', 'second', 'third', 'ground', 'rue', 'des', 'les', 'aux',
    'sur', 'chemin', 'allee', 'impasse', 'route', 'quai', 'saint', 'sainte', 'bis', 'ter'
}

STATE_WORDS = {
    'US': set(' '.join(norm.US_STATES.values()).split()),
    'India': set(' '.join(norm.IN_STATES.values()).split()),
    'France': {
        'hauts', 'france', 'nouvelle', 'aquitaine', 'pays', 'loire', 'ile', 'occitanie', 'bretagne', 'normandie',
        'grand', 'est', 'auvergne', 'rhone', 'alpes', 'provence', 'cote', 'azur', 'bourgogne', 'franche', 'comte',
        'centre', 'val', 'corse'
    }
}


def get_blocking_keys(name, addr, country, query=False):
    c_n, core_n, sort_n, skel = norm.normalize_name(name)
    c_a, nums, sort_a = norm.normalize_address(addr, country)

    keys = set()
    n_tokens = core_n.split()
    a_tokens = c_a.split()

    compact_name = ''.join(n_tokens)
    if len(compact_name) >= 4:
        keys.add(('compact_n', compact_name))
        keys.add(('compact_sort_n', ''.join(sorted(n_tokens))))
    strict = ''.join(t for t in n_tokens if t not in norm.DESCRIPTORS)
    if len(strict) >= 4 and strict != compact_name:
        keys.add(('compact_strict', strict))

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

    skels = skel.split()
    if skels:
        keys.add(('sk', ' '.join(sorted(skels))))
    sk2 = [k for k in skels if len(k) >= 2][:2]
    if sk2:
        keys.add(('skn', ' '.join(sorted(k[:3] for k in sk2))))

    places = sorted({t for t in a_tokens if t.isalpha() and len(t) >= 3 and t not in COMMON_ADDR_WORDS})
    sp = [p for p in places if p not in STATE_WORDS.get(country, set())] or places
    nums = sorted(nums)
    if not query:
        places, sp, nums = places[:8], sp[:4], nums[:3]
    nt = [t for t in n_tokens if len(t) >= 3][:2]

    keys.update(('np', t, p) for t in nt for p in places)
    keys.update(('skp', k, p) for k in sk2 for p in places)
    keys.update(('nump', n, p) for n in nums for p in sp)
    keys.update(('addr_pair', a, b) for i, a in enumerate(sp[:12]) for b in sp[i + 1:12])
    keys.update(('name_num', t, n) for t in nt for n in nums)

    return keys


def prune_index(index):
    pruned = {k for k, postings in index.items() if len(postings) > (SINGLE_CAP if k[0] in NAME_KEYS else COMBO_CAP)}
    for k in pruned:
        del index[k]
    return pruned


def rank_candidates(keys, index, n_targets, top_k=TOP_K, name_k=NAME_K):
    score = collections.defaultdict(float)
    name_score = collections.defaultdict(float)
    count = collections.Counter()
    for k in sorted(keys):
        postings = index.get(k)
        if postings:
            w = math.log(1 + n_targets / len(postings))
            for tid in postings:
                score[tid] += w
                count[tid] += 1
            if k[0] in NAME_KEYS:
                for tid in postings:
                    name_score[tid] += w
    picked = set(heapq.nsmallest(min(name_k, top_k), name_score, key=lambda t: (-name_score[t], t)))
    picked.update(heapq.nsmallest(top_k - len(picked), (t for t in score if t not in picked), key=lambda t: (-score[t], t)))
    return [(tid, count[tid]) for tid in sorted(picked, key=lambda t: (-score[t], t))]
