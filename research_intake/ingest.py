"""Register sources and their case associations, then parse them.

Two adapters:
  * ingest_paired: reads the paired sample produced on the work laptop by
    scripts/verify_paired_sample.py (storage/linkage/paired_documents_complete.parquet)
    and the files under storage/linkage/files/. Relationships come from Kuali
    IDs in that table, never from filenames.
  * ingest_manifest: a JSON list for synthetic or other local documents.
"""
import json
from pathlib import Path

from .parse import detect, parse
from .store import digest

ROLE_FAMILY = {
    'PROPOSAL_MATERIAL': 'proposal',
    'INSTITUTIONAL_PROPOSAL_MATERIAL': 'proposal',
    'AWARD_OUTCOME_MATERIAL': 'award',
}
# Point-in-time rule: only pre-decision proposal material may be used as a predictor
# of funding. Institutional-proposal attachments can include post-decision items.
ROLE_ELIGIBILITY = {
    'PROPOSAL_MATERIAL': 'eligible_pre_decision',
    'INSTITUTIONAL_PROPOSAL_MATERIAL': 'review_required',
    'AWARD_OUTCOME_MATERIAL': 'excluded_post_decision',
}


def _clean(value):
    if value is None:
        return None
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    return str(value)


def register(db, path, associations):
    """Hash, parse once per unique byte sequence, and attach every association."""
    path = Path(path).resolve()
    data = path.read_bytes()
    source = digest(data)
    if not db.execute('SELECT 1 FROM sources WHERE id=?', (source,)).fetchone():
        kind = detect(data)
        with db:
            db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?)', (source, str(path), kind, None, 'registered', None, None))
            state, scope, pages = parse(db, source, path, kind)
            db.execute('UPDATE sources SET state=?,pages=?,error=?,extraction_scope=? WHERE id=?',
                       (state, pages, scope if state != 'parsed' else None, scope if state == 'parsed' else None, source))
    with db:
        for a in associations:
            role = a.get('role')
            db.execute('INSERT OR IGNORE INTO associations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                source, _clean(a.get('case_id')), role, a.get('family') or ROLE_FAMILY.get(role, 'unknown'),
                _clean(a.get('source_table')), _clean(a.get('attachment_key')) or '', _clean(a.get('attachment_type')),
                _clean(a.get('file_name')), _clean(a.get('cohort')), _clean(a.get('match_group')),
                _clean(a.get('inst_proposal')), _clean(a.get('award')),
                a.get('predictor_eligibility') or ROLE_ELIGIBILITY.get(role, 'review_required')))
    return source


def ingest_manifest(db, manifest):
    """manifest: [{"path": ..., "associations": [{case_id, role, ...}]}]"""
    return [register(db, item['path'], item.get('associations', [])) for item in manifest]


def ingest_paired(db, linkage_root):
    import pandas as pd
    root = Path(linkage_root)
    table = pd.read_parquet(root / 'paired_documents_complete.parquet')
    table = table[table['STATUS'].eq('DOWNLOADED')]
    first = lambda row, *names: next((row[n] for n in names if n in row and _clean(row[n]) is not None), None)
    groups = {}
    for row in table.to_dict('records'):
        groups.setdefault(row['SAVED_NAME'], []).append(row)
    registered = []
    for name, rows in groups.items():
        if Path(name).name != name:
            raise ValueError('Manifest file name escapes the files directory')
        path = root / 'files' / name
        if digest(path.read_bytes()) != rows[0]['SHA256']:
            raise ValueError('Stored file does not match its recorded SHA-256')
        associations = [{
            'case_id': row.get('CASE_DEV_PROPOSAL_NUMBER'), 'role': row.get('DOCUMENT_ROLE'),
            'source_table': row.get('SOURCE_TABLE'),
            'attachment_key': first(row, 'PAYLOAD_KEY', 'ATTACHMENT_ROWID'),
            'attachment_type': first(row, 'NARRATIVE_TYPE_CODE', 'ATTACHMENT_TYPE_CODE', 'TYPE_CODE', 'DOCUMENT_TYPE_CODE'),
            'file_name': first(row, 'FILE_NAME'), 'cohort': row.get('COHORT'), 'match_group': row.get('MATCH_GROUP'),
            'inst_proposal': row.get('INST_PROPOSAL_NUMBER'), 'award': row.get('AWARD_NUMBER'),
        } for row in rows]
        registered.append(register(db, path, associations))
    return registered


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding='utf8'))
