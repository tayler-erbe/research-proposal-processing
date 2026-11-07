"""Document-type classification by metadata cascade.

Order: (1) a local attachment-type code map supplied by the owner (Kuali codes
are institution-specific and are not committed), (2) rules on the attachment
file name, (3) rules on the opening text. Unmatched documents are 'other' with
method 'unclassified', never guessed. The family (proposal/award) always comes
from the Kuali role, not from text.
"""
import json
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / 'config' / 'doc_types.json'


def load_rules():
    return json.loads(CONFIG.read_text(encoding='utf8'))


def classify(db, code_map=None):
    rules = load_rules()
    code_map = code_map or {}
    count = 0
    for src in db.execute("SELECT id FROM sources WHERE state='parsed'").fetchall():
        source = src['id']
        assoc = db.execute('SELECT family,attachment_type,file_name FROM associations WHERE source=?', (source,)).fetchall()
        families = sorted({a['family'] for a in assoc})
        # A byte-identical file attached to both a proposal and an award is kept in
        # both families through its associations; classification uses the earliest role.
        family = 'proposal' if 'proposal' in families else (families[0] if families else 'unknown')
        opening = ' '.join(r['text'] for r in db.execute(
            'SELECT text FROM spans WHERE source=? ORDER BY COALESCE(page,0),seq LIMIT 8', (source,)))[:2000]
        doc_type, method, basis = 'other', 'unclassified', ''
        for a in assoc:
            code = a['attachment_type']
            if code and f'{a["family"]}:{code}' in code_map:
                doc_type, method, basis = code_map[f'{a["family"]}:{code}'], 'attachment_type_code', code
                break
        if method == 'unclassified':
            for target, label in [([a['file_name'] or '' for a in assoc], 'file_name'), ([opening], 'opening_text')]:
                for rule in rules.get(family, []):
                    if any(re.search(rule['pattern'], t, re.I) for t in target if t):
                        doc_type, method, basis = rule['doc_type'], label, rule['pattern']
                        break
                if method != 'unclassified':
                    break
        with db:
            db.execute('INSERT OR REPLACE INTO classifications VALUES(?,?,?,?,?)',
                       (source, family, doc_type, method, basis))
        count += 1
    return count
