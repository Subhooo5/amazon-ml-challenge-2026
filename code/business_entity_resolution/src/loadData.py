import os


def load_tsv_records(path, country_filter=None):
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''

            if country_filter is None or country == country_filter:
                records[eid] = (name, addr, country)
    return records


def load_ground_truth(path):
    gt = {}
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0]
            mids = parts[1].split(',') if len(parts) > 1 and parts[1] else []
            gt[s1_id] = set(mids)
    return gt


def stream_s1_records(path, country_filter=None):
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''

            if country_filter is None or country == country_filter:
                yield eid, name, addr, country


def get_distinct_countries(path):
    countries = []
    seen = set()
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            c = parts[3] if len(parts) > 3 else ''
            if c and c not in seen:
                seen.add(c)
                countries.append(c)
    return countries
