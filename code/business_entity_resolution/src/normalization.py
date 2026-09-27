import re
import unicodedata
import text_unidecode

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'pvt', 'private', 'llc', 'llp', 'pllc', 'plc', 'lp',
    'sarl', 'sas', 'sasu', 'sa', 'sci', 'snc', 'eurl', 'ei', 'eirl', 'selarl', 'ets', 'cie',
    'gmbh', 'bv', 'france'
}

DESCRIPTORS = {
    'services', 'service', 'solutions', 'solution', 'technologies', 'technology',
    'enterprises', 'enterprise', 'industries', 'industry', 'group', 'holdings', 'ventures',
    'consulting', 'consultants', 'associates', 'trading', 'traders', 'systems', 'international'
}

SKEL_RULES = [
    ('tion', 'shn'), ('ph', 'f'), ('bh', 'b'), ('dh', 'd'), ('th', 't'), ('kh', 'k'), ('gh', 'g'),
    ('sh', 's'), ('ch', 'c'), ('ck', 'k'), ('c', 'k'), ('q', 'k'), ('z', 'j'), ('w', 'v')
]

SKEL_LEGAL = {'pr', 'l', 'prvt', 'pvt', 'lmtd', 'ltd', 'lld', 'llp', 'llc', 'ink', 'inkrprtd', 'krp', 'kmpn', 'k'}

ADDR_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ct': 'court', 'ln': 'lane', 'pl': 'place',
    'sq': 'square', 'terr': 'terrace', 'pkwy': 'parkway',
    'hwy': 'highway', 'fl': 'floor', 'ste': 'suite', 'apt': 'apartment',
    'bldg': 'building', 'no': 'number', 'nr': 'near', 'opp': 'opposite'
}

FR_ABBR = {**ADDR_ABBR, 'st': 'saint', 'ste': 'sainte'}

US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire',
    'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina',
    'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota', 'tn': 'tennessee',
    'tx': 'texas', 'ut': 'utah', 'vt': 'vermont', 'va': 'virginia', 'wa': 'washington',
    'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia'
}

IN_STATES = {
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam', 'br': 'bihar',
    'cg': 'chhattisgarh', 'dl': 'delhi', 'ga': 'goa', 'gj': 'gujarat', 'hr': 'haryana',
    'hp': 'himachal pradesh', 'jh': 'jharkhand', 'ka': 'karnataka', 'kl': 'kerala',
    'mp': 'madhya pradesh', 'mh': 'maharashtra', 'mn': 'manipur', 'ml': 'meghalaya',
    'mz': 'mizoram', 'nl': 'nagaland', 'od': 'odisha', 'pb': 'punjab', 'rj': 'rajasthan',
    'sk': 'sikkim', 'tn': 'tamil nadu', 'tg': 'telangana', 'ts': 'telangana',
    'tr': 'tripura', 'up': 'uttar pradesh', 'uk': 'uttarakhand', 'wb': 'west bengal'
}

PUNCT_RE = re.compile(r'[^a-zA-Z0-9\s]')
DOMAIN_RE = re.compile(r'\.(com|org|net|in|co|io|fr)\b')
URL_RE = re.compile(r'https?://(?:www\.)?')
MULTI_SPACE_RE = re.compile(r'\s+')
NUMBER_RE = re.compile(r'\b\d+\b')
ORDINAL_RE = re.compile(r'(\d+)(st|nd|rd|th)\b')
DIGIT_SPLIT_RE = re.compile(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])')
REPEAT_RE = re.compile(r'(.)\1+')


def clean_string(s):
    if not s or s == 'null':
        return ''
    s = text_unidecode.unidecode(s)
    s = s.lower()
    s = URL_RE.sub('', s)
    s = DOMAIN_RE.sub(' ', s)
    s = ORDINAL_RE.sub(r'\1', s)
    s = DIGIT_SPLIT_RE.sub(' ', s)
    s = PUNCT_RE.sub(' ', s)
    s = MULTI_SPACE_RE.sub(' ', s).strip()
    return s


def skeleton(token):
    for a, b in SKEL_RULES:
        token = token.replace(a, b)
    token = token[:1] + ''.join(ch for ch in token[1:] if ch not in 'aeiouy')
    return REPEAT_RE.sub(r'\1', token)


def normalize_name(name):
    cleaned = clean_string(name)
    if not cleaned:
        return '', '', '', ''
    tokens = cleaned.split()
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = tokens
    core_name = ' '.join(core_tokens)
    token_sorted = ' '.join(sorted(tokens))
    skel = ' '.join(k for k in map(skeleton, core_tokens) if k not in SKEL_LEGAL)
    return cleaned, core_name, token_sorted, skel


def normalize_address(addr, country=''):
    cleaned = clean_string(addr)
    if not cleaned:
        return '', set(), ''
    abbr = FR_ABBR if country == 'France' else ADDR_ABBR
    states = US_STATES if country == 'US' else IN_STATES if country == 'India' else {}
    tokens = cleaned.split()
    expanded_tokens = []
    for t in tokens:
        if t in abbr:
            expanded_tokens.append(abbr[t])
        elif t in states:
            expanded_tokens.append(states[t])
        else:
            expanded_tokens.append(t)
    expanded_addr = ' '.join(expanded_tokens)
    numbers = {n.lstrip('0') or '0' for n in NUMBER_RE.findall(cleaned)}
    token_sorted = ' '.join(sorted(expanded_tokens))
    return expanded_addr, numbers, token_sorted
