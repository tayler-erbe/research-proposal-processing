"""Local-only exports: long and wide CSVs, a case view and an offline review page.

Case view rule (point in time): only pre-decision proposal material feeds the
predictor columns. Award documents are reported in separate award__ columns and
must not be used to explain whether a proposal was funded.
"""
import csv
import html
import json
from pathlib import Path

from .pipeline import consolidated

EVIDENCE_CHARS = 600


def _safe(value):
    value = '' if value is None else str(value)
    # Neutralize spreadsheet formula injection from document text.
    return "'" + value if value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else value


def _write(path, columns, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: _safe(r.get(k)) for k in columns} for r in rows)


def _display(record):
    if record['type'] == 'enum':
        return record.get('value') or ('|'.join(record.get('competing_values') or []))
    if record['type'] == 'multi_enum':
        return ';'.join(v['value'] for v in record.get('values', []))
    if record['type'] == 'labels':
        return ';'.join(dict.fromkeys(l['label'] for l in record.get('labels', [])))
    return ' … '.join(e['text'] for e in record.get('evidence', []))[:EVIDENCE_CHARS]


def export(db, root):
    root = Path(root)
    out = root / 'tables'
    out.mkdir(exist_ok=True)
    classes = {r['source']: dict(r) for r in db.execute('SELECT * FROM classifications')}
    unread = {r['source']: r['n'] for r in db.execute("SELECT source,count(*) AS n FROM pages WHERE state='needs_ocr' GROUP BY source")}
    reviews = {}
    for r in db.execute('SELECT * FROM reviews ORDER BY created'):
        reviews[(r['job'], r['field'])] = dict(r)   # latest decision wins; history stays in the table

    documents = []
    for s in db.execute('SELECT s.*,(SELECT count(*) FROM spans WHERE source=s.id) AS spans, '
                        '(SELECT count(DISTINCT case_id) FROM associations WHERE source=s.id) AS cases FROM sources s'):
        c = classes.get(s['id'], {})
        documents.append(dict(s, family=c.get('family'), doc_type=c.get('doc_type'), class_method=c.get('method'),
                              unread_pages=unread.get(s['id'], 0)))
    _write(out / 'documents.csv', ['id', 'format', 'pages', 'state', 'error', 'extraction_scope', 'family', 'doc_type',
                                   'class_method', 'unread_pages', 'spans', 'cases', 'path'], documents)
    _write(out / 'associations.csv', ['source', 'case_id', 'role', 'family', 'source_table', 'attachment_key',
                                      'attachment_type', 'file_name', 'cohort', 'match_group', 'inst_proposal', 'award',
                                      'predictor_eligibility'], map(dict, db.execute('SELECT * FROM associations')))

    long_rows, wide = [], {}
    merged = consolidated(db)
    for m in merged:
        c = classes.get(m['source'], {})
        review = reviews.get((m['source'] + ':' + m['run_key'], m['field']), {})
        row = {'source': m['source'], 'family': c.get('family'), 'doc_type': c.get('doc_type'), 'field': m['field'],
               'type': m['type'], 'outcome': m['outcome'], 'value': _display(m),
               'evidence_pages': ','.join(sorted({str(e['page']) for e in m['evidence'] if e['page']})),
               'evidence_spans': ','.join(e['span'][:12] for e in m['evidence']),
               'chunks': f"{m['chunks_completed']}/{m['chunks_planned']}", 'unread_units': m['unread_units'],
               'backend': m['backend'], 'model': m['model'], 'schema_version': m['schema_version'],
               'run_key': m['run_key'], 'review_status': review.get('decision', 'unreviewed'),
               'reviewed_value': review.get('value')}
        long_rows.append(row)
        w = wide.setdefault((m['source'], m['run_key']), {'source': m['source'], 'family': c.get('family'),
                                                           'doc_type': c.get('doc_type'), 'backend': m['backend'],
                                                           'model': m['model'], 'run_key': m['run_key']})
        w[m['field']] = row['value']
        w[m['field'] + '__outcome'] = m['outcome']
    _write(out / 'field_values.csv', ['source', 'family', 'doc_type', 'field', 'type', 'outcome', 'value',
                                      'evidence_pages', 'evidence_spans', 'chunks', 'unread_units', 'backend', 'model',
                                      'schema_version', 'run_key', 'review_status', 'reviewed_value'], long_rows)
    for family in ('proposal', 'award'):
        rows = [w for w in wide.values() if w['family'] == family]
        extra = sorted({k for r in rows for k in r} - {'source', 'family', 'doc_type', 'backend', 'model', 'run_key'})
        _write(out / f'{family}_features.csv', ['source', 'doc_type', 'backend', 'model', 'run_key'] + extra, rows)

    # Case view: one row per development proposal case.
    cases = {}
    for a in db.execute('SELECT * FROM associations WHERE case_id IS NOT NULL'):
        case = cases.setdefault(a['case_id'], {'case_id': a['case_id'], 'cohort': a['cohort'],
                                               'match_group': a['match_group'], 'awards': set(),
                                               'n_predictor_docs': 0, 'n_review_required_docs': 0, 'n_award_docs': 0})
        if a['award']:
            case['awards'].add(a['award'])
        key = {'eligible_pre_decision': 'n_predictor_docs', 'review_required': 'n_review_required_docs',
               'excluded_post_decision': 'n_award_docs'}.get(a['predictor_eligibility'])
        if key:
            case[key] += 1
        for m in merged:
            if m['source'] != a['source'] or m['outcome'] not in ('found', 'ambiguous'):
                continue
            prefix = {'eligible_pre_decision': '', 'excluded_post_decision': 'award__'}.get(a['predictor_eligibility'])
            if prefix is None:
                continue   # review_required material is held out until a person classifies it
            values = case.setdefault(prefix + m['field'], set())
            values.update(x for x in _display(m).split(';') if x and m['type'] != 'evidence')
            if m['type'] == 'evidence':
                values.add('found')
    case_rows = [{k: (';'.join(sorted(v)) if isinstance(v, set) else v) for k, v in c.items()} for c in cases.values()]
    extra = sorted({k for r in case_rows for k in r} - {'case_id', 'cohort', 'match_group', 'awards', 'n_predictor_docs',
                                                         'n_review_required_docs', 'n_award_docs'})
    _write(out / 'case_features.csv', ['case_id', 'cohort', 'match_group', 'awards', 'n_predictor_docs',
                                       'n_review_required_docs', 'n_award_docs'] + extra, case_rows)
    _write(out / 'stage_queue.csv', ['stage', 'state', 'n'],
           map(dict, db.execute('SELECT stage,state,count(*) AS n FROM jobs GROUP BY stage,state')))
    _review_page(root, documents, merged, classes)
    return {'documents': len(documents), 'field_values': len(long_rows), 'cases': len(case_rows)}


def _review_page(root, documents, merged, classes):
    paths = {d['id']: d['path'] for d in documents}
    esc = lambda v: html.escape('' if v is None else str(v), quote=True)
    rows = []
    for m in merged:
        c = classes.get(m['source'], {})
        page = next((e['page'] for e in m['evidence'] if e['page']), None)
        link = Path(paths[m['source']]).as_uri() + (f'#page={page}' if page else '')
        evidence = '\n---\n'.join(e['text'] for e in m['evidence'])[:2000]
        rows.append('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in [
            Path(paths[m['source']]).name, c.get('family'), c.get('doc_type'), m['field'], m['outcome'],
            _display(m)[:300], evidence, m['backend']]) +
            f'<td><a href="{esc(link)}" target="_blank" rel="noopener">open{" p." + str(page) if page else ""}</a></td></tr>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Research intake review</title><style>body{font:15px system-ui;margin:24px;color:#1d2b33;background:#f6f7f4}'
            'table{border-collapse:collapse;width:100%;background:#fff}th,td{padding:8px 10px;border-bottom:1px solid #dde3e1;'
            'text-align:left;vertical-align:top}th{position:sticky;top:0;background:#1d2b33;color:#fff}td:nth-child(7){white-space:pre-wrap;'
            'max-width:520px}input,select{padding:8px;margin:0 8px 12px 0}</style>'
            '<h1>Research proposal and award extraction — local review</h1>'
            '<p>Private, on this computer only. These are unreviewed proposals. Evidence text is copied from the source by code; '
            'labels are model interpretation. Rows from backend "stub" are plumbing tests, not model output. '
            '"not_found_in_read_text" means unread pages remain; it is not absence.</p>'
            '<input id="q" type="search" placeholder="Search"><select id="o"><option value="">All outcomes</option>'
            + ''.join(f'<option>{o}</option>' for o in ['found', 'ambiguous', 'insufficient_evidence', 'not_found_in_full_text',
                                                        'not_found_in_read_text', 'incomplete', 'rejected']) +
            '</select><small id="n"></small><table><thead><tr><th>File</th><th>Family</th><th>Type</th><th>Field</th><th>Outcome</th>'
            '<th>Value</th><th>Evidence</th><th>Backend</th><th>Source</th></tr></thead><tbody>' + ''.join(rows) +
            '</tbody></table><script>const R=[...document.querySelectorAll("tbody tr")],q=document.getElementById("q"),o=document.getElementById("o");'
            'function f(){let n=0;for(const r of R){const ok=r.textContent.toLowerCase().includes(q.value.toLowerCase())&&(!o.value||r.cells[4].textContent===o.value);'
            'r.hidden=!ok;if(ok)n++}document.getElementById("n").textContent=n+" of "+R.length+" rows"}q.oninput=f;o.onchange=f;f()</script></html>')
    (root / 'review.html').write_text(page, encoding='utf8')
