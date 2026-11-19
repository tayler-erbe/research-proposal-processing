"""Aggregate profiles of canonical tables, and the disclosure rules for sharing them.

profile_tables() is run on the real local sample (work laptop) and on
synthetic output, so the generator target and the fidelity check use one
definition. disclose() makes a real profile safe for the public repository
(handoff section 6.3): no cell under 10, rounded shares, quantiles only.
"""
from collections import Counter
from datetime import date

from adapters.views import spine as build_spine

QUANTILES = (10, 25, 50, 75, 90)
MIN_CELL = 10


def _shares(values):
    c = Counter(v for v in values if v is not None)
    n = sum(c.values())
    return ({str(k): v / n for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))} if n else {}), dict(c)


def _quantiles(values):
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return {}
    out = {}
    for q in QUANTILES:
        pos = (len(xs) - 1) * q / 100
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        out[f'p{q}'] = xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return out


def _days(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def profile_tables(tables, kind='measured'):
    """Return (profile, counts). counts holds the cell sizes behind every share."""
    sp = build_spine(tables)
    dev = tables['dev_proposal']
    docs = tables['document']
    profile = {'meta': {'kind': kind, 'n_dev_proposals': len(dev)},
               'categorical': {}, 'conditional': {}, 'quantiles': {}, 'quantiles_by': {},
               'cardinality': {}, 'rates': {}}
    counts = {'categorical': {}, 'conditional': {}, 'quantiles_by': {}, 'cardinality': {}}

    def cat(name, values, where='categorical'):
        profile[where][name], counts[where][name] = _shares(values)

    for f in ('campus', 'sponsor_type', 'activity_type_code', 'proposal_type_code', 'fiscal_year'):
        cat('dev_proposal.' + f, [r[f] for r in dev])
    cat('spine.outcome', [r['outcome'] for r in sp])
    cat('document.source', [r['source'] for r in docs])
    cat('document.detected_format', [r['detected_format'] for r in docs])
    cat('document.payload_location', [r['payload_location'] for r in docs])

    eligible = [r for r in sp if r['binary_outcome_eligible']]
    by_type = {}
    for r in eligible:
        by_type.setdefault(r['sponsor_type'], []).append(r['outcome'] == 'FUNDED')
    profile['conditional']['funded_rate_by.sponsor_type'] = {k: sum(v) / len(v) for k, v in sorted(by_type.items())}
    counts['conditional']['funded_rate_by.sponsor_type'] = {k: len(v) for k, v in by_type.items()}

    totals = {}
    for r in sp:
        if r['requested_total'] is not None:
            totals.setdefault(r['sponsor_type'], []).append(r['requested_total'])
    profile['quantiles_by']['requested_total_by.sponsor_type'] = {k: _quantiles(v) for k, v in sorted(totals.items())}
    counts['quantiles_by']['requested_total_by.sponsor_type'] = {k: len(v) for k, v in totals.items()}
    profile['quantiles']['award_lag_days'] = _quantiles(
        [_days(r['submitted_date'], r['first_award_begin_date']) for r in sp
         if r['first_award_begin_date'] and r['submitted_date']])
    profile['quantiles']['document.n_pages'] = _quantiles([r['n_pages'] for r in docs])
    profile['quantiles']['document.byte_size'] = _quantiles([r['byte_size'] for r in docs])

    inst_versions = Counter(r['inst_proposal_number'] for r in tables['inst_proposal'])
    award_versions = Counter(r['award_number'] for r in tables['award'])
    award_number = {a['award_version_id']: a['award_number'] for a in tables['award']}
    inst_number = {r['inst_version_id']: r['inst_proposal_number'] for r in tables['inst_proposal']}
    props_per_award = {}
    for link in tables['proposal_award_link']:
        props_per_award.setdefault(award_number.get(link['award_version_id']), set()).add(inst_number.get(link['inst_version_id']))
    docs_per_prop = Counter(r['dev_proposal_number'] for r in docs if r['source'] == 'PROPOSAL_NARRATIVE')
    docs_per_award = Counter(award_number.get(r['award_version_id']) for r in docs if r['source'] == 'AWARD')
    cat('inst_versions_per_proposal', list(inst_versions.values()), 'cardinality')
    cat('award_versions_per_award', list(award_versions.values()), 'cardinality')
    cat('proposals_per_award', [len(v) for k, v in props_per_award.items() if k], 'cardinality')
    cat('awards_per_funded_proposal', [r['n_awards'] for r in sp if r['outcome'] == 'FUNDED' and r['n_awards']], 'cardinality')
    cat('documents_per_proposal', [docs_per_prop.get(r['dev_proposal_number'], 0) for r in dev], 'cardinality')
    cat('documents_per_award', [docs_per_award.get(n, 0) for n in award_versions], 'cardinality')

    n = len(sp) or 1
    profile['rates']['outcome_link_review_required'] = sum(r['outcome_link_review_required'] for r in sp) / n
    named_pdf = [r for r in docs if (r['file_name'] or '').lower().endswith('.pdf')]
    profile['rates']['pdf_name_not_pdf_bytes'] = (sum(r['detected_format'] != 'pdf' for r in named_pdf) / len(named_pdf)) if named_pdf else 0.0
    return profile, counts


def _sig2(x):
    if not x:
        return 0.0
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + 1)


def disclose(profile, counts, min_cell=MIN_CELL):
    """Copy of a real profile that follows the public-repository disclosure rules."""
    out = {'meta': {'kind': 'real-disclosed', 'n_dev_proposals': int(round(profile['meta']['n_dev_proposals'], -1)),
                    'rules': f'cells under {min_cell} merged into SUPPRESSED; shares to 3 decimals; amounts and sizes as 2-significant-figure quantiles'},
           'categorical': {}, 'conditional': {}, 'quantiles': {}, 'quantiles_by': {}, 'cardinality': {}, 'rates': {}}
    for section in ('categorical', 'cardinality'):
        for name, shares in profile[section].items():
            cell = counts[section][name]
            n_of = {k: cell.get(_key(cell, k), 0) for k in shares}
            kept = {k: v for k, v in shares.items() if n_of[k] >= min_cell}
            # A merged cell must itself reach min_cell, or it could be recovered by subtraction.
            while kept and 0 < sum(n_of[k] for k in shares if k not in kept) < min_cell:
                kept.pop(min(kept, key=lambda k: n_of[k]))
            dropped = sum(v for k, v in shares.items() if k not in kept)
            if dropped:
                kept['SUPPRESSED'] = dropped
            out[section][name] = {k: round(v, 3) for k, v in kept.items()}
    for name, rates in profile['conditional'].items():
        cell = counts['conditional'][name]
        out['conditional'][name] = {k: round(v, 2) for k, v in rates.items() if cell.get(k, 0) >= min_cell}
    for name, groups in profile['quantiles_by'].items():
        cell = counts['quantiles_by'][name]
        out['quantiles_by'][name] = {k: {q: _sig2(v) for q, v in qs.items()} for k, qs in groups.items() if cell.get(k, 0) >= min_cell}
    for name, qs in profile['quantiles'].items():
        out['quantiles'][name] = {q: _sig2(v) for q, v in qs.items()}
    out['rates'] = {k: round(v, 3) for k, v in profile['rates'].items()}
    return out


def _key(cell, k):
    """Counter keys keep their type; profile keys are strings."""
    for ck in cell:
        if str(ck) == k:
            return ck
    return k
