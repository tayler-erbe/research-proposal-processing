"""Generate synthetic proposal/award data from an aggregate profile.

    python -m synthetic.generate --profile synthetic/profiles/placeholder.yaml --seed 7 --scale 1

Writes canonical tables (storage/synthetic/<name>/canonical/*.csv) and the same
data in real Kuali shape (.../kuali/*.csv: real table and column names, real
key formats, freshly generated values). Deterministic for a given seed,
profile and scale. Uses the standard library only (plus PyYAML).
"""
import argparse
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

from adapters import mapping
from adapters.tableio import write_table

ROOT = Path(__file__).resolve().parents[1]
NARRATIVE_TYPES = ['535', 'N01', 'N02', 'N03']   # '535' is the type the pipeline reads; the others are invented
AWARD_ATTACHMENT_TYPES = ['1', '2', '3', '4']     # '3' is the internal type the pipeline excludes
EXT = {'pdf': ('.pdf', 'application/pdf'),
       'docx': ('.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
       'doc': ('.doc', 'application/msword'),
       'other': ('.bin', 'application/octet-stream')}
STATUS_FOR = {'PENDING': '1', 'FUNDED': '2', 'NOT_FUNDED': '3', 'WITHDRAWN': '5', 'VOID': '8'}


def pick(rng, shares):
    keys = list(shares)
    return rng.choices(keys, weights=[shares[k] for k in keys])[0]


def from_quantiles(rng, qs, integer=False, floor=1.0):
    """Sample a positive value by log-linear interpolation of p10..p90, extrapolating the tails."""
    pts = sorted((int(k[1:]) / 100, math.log(max(float(v), floor))) for k, v in qs.items())
    u = min(max(rng.random(), 0.01), 0.99)
    if u <= pts[0][0]:
        (u0, y0), (u1, y1) = pts[0], pts[1]
    elif u >= pts[-1][0]:
        (u0, y0), (u1, y1) = pts[-2], pts[-1]
    else:
        i = next(i for i in range(len(pts) - 1) if pts[i][0] <= u <= pts[i + 1][0])
        (u0, y0), (u1, y1) = pts[i], pts[i + 1]
    y = y0 + (y1 - y0) * (u - u0) / (u1 - u0)
    v = max(math.exp(y), floor)
    return int(round(v)) if integer else round(v, 2)


def guid(rng):
    h = '%032x' % rng.getrandbits(128)
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'.upper()


def generate(profile, seed=7, scale=1.0):
    rng = random.Random(seed)
    cat, cond, card = profile['categorical'], profile['conditional'], profile['cardinality']
    n = max(1, round(profile['meta']['n_dev_proposals'] * scale))

    # Reference entities: sponsors and units, with fresh codes.
    sponsors, sponsors_by_type = [], {}
    codes = rng.sample(range(100000, 999999), 30 * len(cat['dev_proposal.sponsor_type']))
    for i, stype in enumerate(sorted(cat['dev_proposal.sponsor_type'])):
        mine = [f'{c:06d}' for c in codes[i * 30:(i + 1) * 30]]
        sponsors_by_type[stype] = mine
        sponsors += [{'sponsor_code': c, 'sponsor_type_code': stype} for c in mine]
    units = {'UIUC': ['1' + '%05d' % u for u in rng.sample(range(100000), 80)],
             'UIC': ['2' + '%05d' % u for u in rng.sample(range(100000), 30)]}
    zipf = [1 / (k + 1) for k in range(30)]

    ids = {name: rng.randint(10_000, 90_000) * 10 for name in ('dev', 'inst_number', 'inst_version', 'award_number', 'award_version', 'link', 'file')}

    def next_id(name):
        ids[name] += rng.randint(1, 3)
        return ids[name]

    dev, inst, awards, links, docs = [], [], [], [], []
    p_binary = cat['spine.outcome'].get('FUNDED', 0) + cat['spine.outcome'].get('NOT_FUNDED', 0)
    other_outcomes = {k: v for k, v in cat['spine.outcome'].items() if k not in ('FUNDED', 'NOT_FUNDED')}
    conflict_rate = profile['rates']['outcome_link_review_required'] / p_binary if p_binary else 0
    funded = []  # (dev row, current inst version id, submitted date)

    for _ in range(n):
        campus = pick(rng, cat['dev_proposal.campus'])
        stype = pick(rng, cat['dev_proposal.sponsor_type'])
        fy = int(pick(rng, cat['dev_proposal.fiscal_year']))
        submitted = date(fy - 1, 7, 1) + timedelta(days=rng.randint(0, 364))
        start = submitted + timedelta(days=rng.randint(90, 300))
        row = {'dev_proposal_number': next_id('dev'),
               'lead_unit_number': rng.choice(units.get(campus, units['UIUC'])),
               'sponsor_code': rng.choices(sponsors_by_type[stype], weights=zipf)[0],
               'activity_type_code': pick(rng, cat['dev_proposal.activity_type_code']),
               'proposal_type_code': pick(rng, cat['dev_proposal.proposal_type_code']),
               'deadline_date': (submitted + timedelta(days=rng.randint(0, 3))).isoformat(),
               'requested_start_date': start.isoformat(),
               'requested_end_date': (start + timedelta(days=365 * rng.randint(1, 5))).isoformat(),
               'submitted_date': submitted.isoformat(), 'linked_inst_version_id': None}
        dev.append(row)

        if rng.random() < p_binary:
            outcome = 'FUNDED' if rng.random() < cond['funded_rate_by.sponsor_type'].get(stype, 0.3) else 'NOT_FUNDED'
        else:
            outcome = pick(rng, other_outcomes)
        if outcome == 'UNKNOWN':
            continue  # no institutional proposal reached from this development proposal

        number = '%08d' % next_id('inst_number')
        total = from_quantiles(rng, profile['quantiles_by']['requested_total_by.sponsor_type'][stype])
        direct = round(total * rng.uniform(0.6, 0.8), 2)
        versions = int(pick(rng, card['inst_versions_per_proposal']))
        version_ids = []
        for s in range(1, versions + 1):
            vid = next_id('inst_version')
            version_ids.append(vid)
            inst.append({'inst_version_id': vid, 'inst_proposal_number': number, 'sequence_number': s,
                         'is_current': s == versions, 'status_code': STATUS_FOR[outcome] if s == versions else '1',
                         'requested_direct': direct, 'requested_indirect': round(total - direct, 2)})
        row['linked_inst_version_id'] = version_ids[0]

        conflict = outcome in ('FUNDED', 'NOT_FUNDED') and rng.random() < conflict_rate
        n_awards = 0
        if outcome == 'FUNDED' and not conflict:
            n_awards = int(pick(rng, card['awards_per_funded_proposal']))
        elif outcome == 'NOT_FUNDED' and conflict:
            n_awards = 1
        for _a in range(n_awards):
            base = next_id('award_number')
            lag = from_quantiles(rng, profile['quantiles']['award_lag_days'], integer=True)
            begin = submitted + timedelta(days=lag)
            n_versions = int(pick(rng, card['award_versions_per_award']))
            first_vid = None
            for s in range(1, n_versions + 1):
                vid = next_id('award_version')
                first_vid = first_vid or vid
                awards.append({'award_version_id': vid, 'award_number': f'{base:06d}-00001', 'sequence_number': s,
                               'is_current': s == n_versions, 'status_code': '1',
                               'sponsor_code': row['sponsor_code'], 'lead_unit_number': row['lead_unit_number'],
                               'begin_date': begin.isoformat()})
            links.append({'funding_link_id': next_id('link'), 'inst_version_id': rng.choice(version_ids),
                          'award_version_id': first_vid, 'active': True})
        if outcome == 'FUNDED' and not conflict:
            funded.append(version_ids[-1])

    # Some awards are funded by more than one institutional proposal (continuations, supplements).
    extra = 1 - card['proposals_per_award'].get('1', 1.0)
    first_versions = {}
    for a in awards:
        first_versions.setdefault(a['award_number'], a['award_version_id'])
    for vid in first_versions.values():
        if funded and rng.random() < extra:
            links.append({'funding_link_id': next_id('link'), 'inst_version_id': rng.choice(funded),
                          'award_version_id': vid, 'active': rng.random() < 0.9})

    fmt = cat['document.detected_format']
    p_pdf = fmt.get('pdf', 0)
    rename_rate = profile['rates']['pdf_name_not_pdf_bytes'] * p_pdf / (1 - p_pdf) if p_pdf < 1 else 0

    def document(source, **keys):
        detected = pick(rng, fmt)
        location = pick(rng, cat['document.payload_location'])
        ext, ctype = EXT[detected]
        if detected != 'pdf' and rng.random() < rename_rate:
            ext = '.pdf'  # a Word file uploaded under a .pdf name
        missing = location == 'MISSING'
        doc = {'source': source, 'dev_proposal_number': None, 'module_number': None,
               'award_version_id': None, 'attachment_file_id': None,
               'file_data_id': guid(rng) if location == 'FILE_DATA' else None,
               'file_name': f'attachment_{rng.getrandbits(32):08x}{ext}', 'content_type': ctype,
               'payload_location': location, 'detected_format': detected,
               'byte_size': None if missing else from_quantiles(rng, profile['quantiles']['document.byte_size'], integer=True),
               'n_pages': None if missing else from_quantiles(rng, profile['quantiles']['document.n_pages'], integer=True)}
        doc.update(keys)
        return doc

    for p in dev:
        for m in range(1, int(pick(rng, card['documents_per_proposal'])) + 1):
            d = document('PROPOSAL_NARRATIVE', dev_proposal_number=p['dev_proposal_number'], module_number=m,
                         attachment_type_code=NARRATIVE_TYPES[0] if m == 1 else rng.choice(NARRATIVE_TYPES))
            d['attachment_key'] = f"{p['dev_proposal_number']}-{m}"
            docs.append(d)
    versions_by_number = {}
    for a in awards:
        versions_by_number.setdefault(a['award_number'], []).append(a['award_version_id'])
    for number, vids in versions_by_number.items():
        for _ in range(int(pick(rng, card['documents_per_award']))):
            fid = next_id('file')
            d = document('AWARD', award_version_id=rng.choice(vids), attachment_file_id=fid,
                         attachment_type_code=rng.choice(AWARD_ATTACHMENT_TYPES))
            d['attachment_key'] = str(fid)
            docs.append(d)

    tables = {'dev_proposal': dev, 'inst_proposal': inst, 'award': awards,
              'proposal_award_link': links, 'sponsor': sponsors, 'document': docs}
    return mapping.derive(mapping.cast_all(tables))


def write(tables, out_dir, manifest):
    out_dir = Path(out_dir)
    model = mapping.load_model()
    for entity, rows in tables.items():
        write_table(rows, out_dir / 'canonical' / (entity + '.csv'), list(mapping.field_types(entity, model)))
    for table, rows in mapping.to_source(tables, 'kuali').items():
        write_table(rows, out_dir / 'kuali' / (table + '.csv'))
    manifest['row_counts'] = {e: len(r) for e, r in tables.items()}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', type=Path, default=ROOT / 'synthetic/profiles/placeholder.yaml')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    profile = mapping.load_yaml(args.profile)
    tables = generate(profile, args.seed, args.scale)
    out = args.out or ROOT / 'storage/synthetic' / f'seed{args.seed}_scale{args.scale:g}'
    write(tables, out, {'profile': str(args.profile.name), 'profile_kind': profile['meta']['kind'],
                        'seed': args.seed, 'scale': args.scale})
    print(json.dumps({'out': str(out), 'rows': {e: len(r) for e, r in tables.items()}}))


if __name__ == '__main__':
    main()
