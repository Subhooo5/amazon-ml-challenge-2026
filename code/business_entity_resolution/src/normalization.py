import re
import unicodedata
import text_unidecode

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'pvt', 'private', 'llc', 'llp', 'pllc',
    'enterprises', 'enterprise', 'industries', 'industry',
    'technologies', 'technology', 'services', 'service', 'solutions', 'solution',
    'group', 'holdings', 'ventures', 'consulting', 'associates', 'trading',
    'sarl', 'sas', 'sci', 'sa', 'snc', 'eurl', 'ste', 'gmbh', 'bv'
}

ADDR_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ct': 'court', 'ln': 'lane', 'pl': 'place',
    'sq': 'square', 'terr': 'terrace', 'pkwy': 'parkway',
    'hwy': 'highway', 'fl': 'floor', 'ste': 'suite', 'apt': 'apartment',
    'bldg': 'building', 'no': 'number', 'nr': 'near', 'opp': 'opposite'
}

STATE_MAP = {
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
    'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
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


def normalize_name(name):
    cleaned = clean_string(name)
    if not cleaned:
        return '', '', ''
    tokens = cleaned.split()
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = tokens
    core_name = ' '.join(core_tokens)
    token_sorted = ' '.join(sorted(tokens))
    return cleaned, core_name, token_sorted


def normalize_address(addr):
    cleaned = clean_string(addr)
    if not cleaned:
        return '', set(), ''
    tokens = cleaned.split()
    expanded_tokens = []
    for t in tokens:
        if t in ADDR_ABBR:
            expanded_tokens.append(ADDR_ABBR[t])
        elif t in STATE_MAP:
            expanded_tokens.append(STATE_MAP[t])
        else:
            expanded_tokens.append(t)
    expanded_addr = ' '.join(expanded_tokens)
    numbers = set(NUMBER_RE.findall(cleaned))
    token_sorted = ' '.join(sorted(expanded_tokens))
    return expanded_addr, numbers, token_sorted
