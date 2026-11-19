"""Compare a profile of synthetic data with a target profile and report pass/fail.

    python -m synthetic.fidelity --data storage/synthetic/seed7_scale1 --profile synthetic/profiles/placeholder.yaml

Checks (thresholds in synthetic/fidelity_thresholds.yaml):
  categorical and cardinality shares: total variation distance
  conditional rates (funded rate by sponsor type): largest absolute difference
  quantiles: largest |log ratio| across p10..p90
  rates: absolute difference
  utility: the ranking of funded rate by sponsor type agrees (Kendall tau)
Writes report.json and report.md next to the data. Aggregates only.
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

from adapters import mapping
from synthetic.profile import profile_tables

ROOT = Path(__file__).resolve().parents[1]


def tvd(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def max_log_ratio(a, b):
    ratios = [abs(math.log(max(b[q], 1e-9) / max(a[q], 1e-9))) for q in a if q in b]
    return max(ratios) if ratios else None


def kendall_tau(a, b):
    keys = sorted(set(a) & set(b))
    pairs = list(combinations(keys, 2))
    if not pairs:
        return None
    score = sum(1 if (a[x] - a[y]) * (b[x] - b[y]) > 0 else -1 if (a[x] - a[y]) * (b[x] - b[y]) < 0 else 0 for x, y in pairs)
    return score / len(pairs)


def compare(target, measured, thresholds, counts=None):
    checks = []
    min_n = thresholds.get('min_group_n', 0)

    def add(group, name, value, limit, higher_is_better=False, note=None):
        ok = None if value is None or note else (value >= limit if higher_is_better else value <= limit)
        checks.append({'group': group, 'name': name, 'value': None if value is None else round(value, 4),
                       'limit': limit, 'passed': ok, 'note': note})

    for section in ('categorical', 'cardinality'):
        for name, shares in target[section].items():
            if 'SUPPRESSED' in shares:  # disclosed profiles merge small cells; compare on the kept categories
                kept = {k: v for k, v in shares.items() if k != 'SUPPRESSED'}
                got = measured[section].get(name, {})
                got = {**{k: got.get(k, 0.0) for k in kept}, 'SUPPRESSED': 1 - sum(got.get(k, 0.0) for k in kept)}
            else:
                got = measured[section].get(name, {})
            add(section, name, tvd(shares, got), thresholds['tvd'])
    for name, rates in target['conditional'].items():
        got = measured['conditional'].get(name, {})
        diffs = [abs(v - got[k]) for k, v in rates.items() if k in got]
        add('conditional', name, max(diffs) if diffs else None, thresholds['conditional_abs'])
        add('utility', name + ' ranking (Kendall tau)', kendall_tau(rates, got), thresholds['rank_tau_min'], True)
    for name, qs in target['quantiles'].items():
        add('quantiles', name, max_log_ratio(qs, measured['quantiles'].get(name, {})), thresholds['quantile_log_ratio'])
    for name, groups in target['quantiles_by'].items():
        for g, qs in groups.items():
            n = (counts or {}).get('quantiles_by', {}).get(name, {}).get(g)
            note = f'only {n} synthetic values; below min_group_n {min_n}' if n is not None and n < min_n else None
            add('quantiles', f'{name}[{g}]', max_log_ratio(qs, measured['quantiles_by'].get(name, {}).get(g, {})),
                thresholds['quantile_log_ratio'], note=note)
    for name, v in target['rates'].items():
        add('rates', name, abs(v - measured['rates'].get(name, 0.0)), thresholds['rate_abs'])
    return checks


def summary(checks):
    failed = [c for c in checks if c['passed'] is False]
    missing = [c for c in checks if c['passed'] is None]
    return {'checks': len(checks), 'passed': sum(c['passed'] is True for c in checks),
            'failed': len(failed), 'not_measurable': len(missing), 'ok': not failed}


def to_markdown(target_kind, checks):
    s = summary(checks)
    lines = [f"# Fidelity report (target profile: {target_kind})", '',
             f"{s['passed']} of {s['checks']} checks passed, {s['failed']} failed, {s['not_measurable']} not measurable.", '',
             '| Group | Check | Value | Limit | Result |', '|---|---|---|---|---|']
    for c in checks:
        result = 'pass' if c['passed'] else (c.get('note') or 'not measurable') if c['passed'] is None else 'FAIL'
        lines.append(f"| {c['group']} | {c['name']} | {c['value']} | {c['limit']} | {result} |")
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, required=True, help='generator output folder')
    ap.add_argument('--profile', type=Path, default=ROOT / 'synthetic/profiles/placeholder.yaml')
    ap.add_argument('--thresholds', type=Path, default=ROOT / 'synthetic/fidelity_thresholds.yaml')
    args = ap.parse_args()
    target = mapping.load_yaml(args.profile)
    cfg = {'data_source': 'synthetic', 'paths': {'synthetic': str(args.data.resolve())}}
    measured, counts = profile_tables(mapping.load_canonical(cfg), kind='synthetic')
    checks = compare(target, measured, mapping.load_yaml(args.thresholds), counts)
    (args.data / 'fidelity.json').write_text(json.dumps({'summary': summary(checks), 'checks': checks}, indent=2) + '\n', encoding='utf-8')
    (args.data / 'fidelity.md').write_text(to_markdown(target['meta']['kind'], checks), encoding='utf-8')
    print(json.dumps(summary(checks)))


if __name__ == '__main__':
    main()
