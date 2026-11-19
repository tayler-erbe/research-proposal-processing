"""Check synthetic tables for values carried over from the real local sample.

Run on the work laptop before committing any synthetic artifact:

    python -m synthetic.leakage --synthetic storage/synthetic/seed7_scale1 --real storage/samples/latest/kuali

Checks (handoff section 7.3):
  identifiers: no synthetic key value appears in the real sample
  near-copies: no synthetic proposal matches a real one on the quasi-identifier
               tuple (sponsor, unit, fiscal year, requested total to the dollar)
  text:        no 8-token sequence of a synthetic text appears in a real text
Prints counts only, never the matching values.
"""
import argparse
import json
import re
from pathlib import Path

from adapters import mapping
from adapters.views import spine

IDENTIFIERS = {
    'dev_proposal': ['dev_proposal_number'],
    'inst_proposal': ['inst_proposal_number', 'inst_version_id'],
    'award': ['award_number', 'award_version_id'],
    'document': ['file_data_id', 'attachment_file_id'],
}


def _quasi(tables):
    by_number = {p['dev_proposal_number']: p for p in tables['dev_proposal']}
    out = set()
    for r in spine(tables):
        p = by_number[r['dev_proposal_number']]
        if r['requested_total'] is not None:
            out.add((p['sponsor_code'], p['lead_unit_number'], r['fiscal_year'], round(r['requested_total'])))
    return out


def ngrams(text, n=8):
    tokens = re.findall(r'\w+', (text or '').lower())
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def check(synthetic, real, synthetic_texts=(), real_texts=(), n=8):
    report = {'identifier_overlaps': {}, 'quasi_identifier_matches': 0, 'text_ngram_overlaps': 0}
    for entity, fields in IDENTIFIERS.items():
        for f in fields:
            a = {str(r[f]) for r in synthetic.get(entity, []) if r.get(f) is not None}
            b = {str(r[f]) for r in real.get(entity, []) if r.get(f) is not None}
            report['identifier_overlaps'][f'{entity}.{f}'] = len(a & b)
    report['quasi_identifier_matches'] = len(_quasi(synthetic) & _quasi(real))
    real_grams = set()
    for t in real_texts:
        real_grams |= ngrams(t, n)
    report['text_ngram_overlaps'] = sum(len(ngrams(t, n) & real_grams) for t in synthetic_texts)
    report['clean'] = (not any(report['identifier_overlaps'].values())
                       and report['quasi_identifier_matches'] == 0 and report['text_ngram_overlaps'] == 0)
    return report


def _texts(directory):
    return [p.read_text(encoding='utf-8', errors='ignore') for p in Path(directory).rglob('*.txt')] if directory else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--synthetic', type=Path, required=True, help='generator output folder')
    ap.add_argument('--real', type=Path, required=True, help='real-shaped Kuali sample folder (local only)')
    ap.add_argument('--synthetic-text', type=Path, help='folder of synthetic .txt documents')
    ap.add_argument('--real-text', type=Path, help='folder of real extracted .txt documents (local only)')
    args = ap.parse_args()
    syn = mapping.load_canonical({'data_source': 'synthetic', 'paths': {'synthetic': str(args.synthetic.resolve())}})
    real = mapping.load_canonical({'data_source': 'kuali', 'paths': {'kuali': str(args.real.resolve())}})
    report = check(syn, real, _texts(args.synthetic_text), _texts(args.real_text))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['clean'] else 1)


if __name__ == '__main__':
    main()
