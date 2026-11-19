"""Canonical <-> source mapping driven by contracts/*.yaml.

    tables = load_canonical()                 # reads configs/data_source.yaml
    real = to_source(tables, 'kuali')          # real-shaped tables (real names, real key formats)
    back = from_source(real, 'kuali')          # canonical again

Switching between synthetic and real data is the data_source setting in
configs/data_source.yaml (or the DATA_SOURCE environment variable). Nothing
here connects to a database; build_select() only writes read-only SQL text.
"""
import os
from pathlib import Path

import yaml

from adapters.tableio import find_table, read_table

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / 'contracts'
MEASURES_TABLE = '_DOCUMENT_MEASURES'  # not a Kuali table: byte/inventory measurements keyed by canonical names

TRANSFORMS = {
    # name: (canonical -> source, source -> canonical)
    'zero_pad_8': (lambda v: None if v is None else str(v).zfill(8),
                   lambda v: None if v is None else str(v).strip().zfill(8)),
    'active_archived': (lambda v: None if v is None else ('ACTIVE' if v else 'ARCHIVED'),
                        lambda v: None if v is None else str(v).upper() == 'ACTIVE'),
    'y_n': (lambda v: None if v is None else ('Y' if v else 'N'),
            lambda v: None if v is None else str(v).upper() in ('Y', '1', 'TRUE')),
}


def load_yaml(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model():
    return load_yaml(CONTRACTS / 'canonical_model.yaml')


def load_source_map(name):
    return load_yaml(CONTRACTS / 'sources' / (name + '.yaml'))


def field_types(entity, model=None):
    model = model or load_model()
    spec = model['entities'][entity]['fields'] if entity in model['entities'] else model['views'][entity]['fields']
    return {k: (v['type'] if isinstance(v, dict) else v) for k, v in spec.items()}


def base_fields(entity, model=None):
    model = model or load_model()
    return [k for k, v in model['entities'][entity]['fields'].items() if not (isinstance(v, dict) and v.get('derived'))]


def _cast_value(v, kind):
    if v is None or v == '':
        return None
    if kind == 'int':
        return int(float(v))
    if kind == 'float':
        return float(v)
    if kind == 'bool':
        return v if isinstance(v, bool) else str(v).strip().lower() in ('true', '1', 'y', 'yes')
    if kind == 'date':
        return v.isoformat()[:10] if hasattr(v, 'isoformat') else str(v)[:10]
    return str(v)


def cast(entity, rows, model=None):
    types = field_types(entity, model)
    return [{k: _cast_value(r.get(k), t) for k, t in types.items()} for r in rows]


def _column_spec(spec):
    return (spec, None) if isinstance(spec, str) else (spec['field'], spec.get('transform'))


def to_source(tables, source='kuali'):
    """Canonical entity tables -> {real table name: rows}."""
    smap = load_source_map(source)
    out = {}
    for entity, espec in smap['entities'].items():
        rows = tables.get(entity, [])
        for part in espec['parts']:
            consts = part.get('constants', {})
            keys = part['merge_on']
            selected = [r for r in rows if all(r.get(k) == v for k, v in consts.items())]
            for i, (table, cols) in enumerate(part['tables'].items()):
                emitted = out.setdefault(table, [])
                for r in selected:
                    real = {}
                    for col, spec in cols.items():
                        field, tr = _column_spec(spec)
                        real[col] = TRANSFORMS[tr][0](r.get(field)) if tr else r.get(field)
                    # A secondary table only has a row when it carries something besides the keys.
                    nonkey = [_column_spec(s)[0] for s in cols.values() if _column_spec(s)[0] not in keys]
                    if i == 0 or any(r.get(f) is not None for f in nonkey):
                        emitted.append(real)
        if espec.get('measured'):
            m = out.setdefault(MEASURES_TABLE, [])
            for r in rows:
                m.append({k: r.get(k) for k in ['source', 'attachment_key'] + espec['measured']})
    return out


def from_source(real, source='kuali'):
    """{real table name: rows} -> canonical entity tables (base fields only, uncast)."""
    smap = load_source_map(source)
    tables = {}
    for entity, espec in smap['entities'].items():
        result = []
        for part in espec['parts']:
            keys = part['merge_on']
            merged = {}
            for table, cols in part['tables'].items():
                for row in real.get(table, []):
                    canon = {}
                    for col, spec in cols.items():
                        field, tr = _column_spec(spec)
                        v = row.get(col)
                        canon[field] = TRANSFORMS[tr][1](v) if tr else v
                    k = tuple(str(canon.get(x)) for x in keys)
                    merged.setdefault(k, {}).update({f: v for f, v in canon.items() if v is not None or f not in merged.get(k, {})})
            for r in merged.values():
                r.update(part.get('constants', {}))
                result.append(r)
        if entity == 'document':
            for r in result:
                r['attachment_key'] = (f"{r.get('dev_proposal_number')}-{r.get('module_number')}"
                                       if r['source'] == 'PROPOSAL_NARRATIVE' else str(r.get('attachment_file_id')))
            measures = {(m['source'], str(m['attachment_key'])): m for m in real.get(MEASURES_TABLE, [])}
            for r in result:
                m = measures.get((r['source'], r['attachment_key']), {})
                for f in espec.get('measured', []):
                    r[f] = m.get(f)
        tables[entity] = result
    return tables


def fiscal_year(d):
    """University fiscal year runs July to June: 2025-07-01 is FY2026."""
    if not d:
        return None
    y, m = int(d[:4]), int(d[5:7])
    return y + 1 if m >= 7 else y


def derive(tables, source='kuali'):
    """Fill derived canonical fields identically for every source."""
    lookups = load_source_map('kuali')['lookups']  # the canonical lookups live with the real source
    prefixes = lookups['campus_unit_prefix']
    labels = lookups.get('sponsor_type_code_to_label') or {}
    sponsor_type = {s['sponsor_code']: labels.get(s['sponsor_type_code'], s['sponsor_type_code'])
                    for s in tables.get('sponsor', [])}
    for r in tables.get('dev_proposal', []):
        unit = r.get('lead_unit_number') or ''
        r['campus'] = next((c for c, ps in prefixes.items() if any(unit.startswith(p) for p in ps)), 'OTHER')
        r['fiscal_year'] = fiscal_year(r.get('submitted_date') or r.get('deadline_date'))
        r['sponsor_type'] = sponsor_type.get(r.get('sponsor_code'), 'UNKNOWN')
    return tables


def cast_all(tables, model=None):
    model = model or load_model()
    return {e: cast(e, rows, model) for e, rows in tables.items() if e in model['entities']}


def load_config(path=None):
    path = Path(path or ROOT / 'configs' / 'data_source.yaml')
    cfg = load_yaml(path)
    cfg['data_source'] = os.environ.get('DATA_SOURCE', cfg['data_source'])
    return cfg


def load_canonical(config=None):
    """Return canonical, typed, derived tables for the configured source."""
    cfg = config if isinstance(config, dict) else load_config(config)
    source = cfg['data_source']
    directory = ROOT / cfg['paths'][source]
    model = load_model()
    if source == 'synthetic':
        raw = {}
        for entity in model['entities']:
            p = find_table(directory / 'canonical', entity)
            raw[entity] = read_table(p) if p else []
    else:
        real = {}
        for p in list(directory.glob('*.csv')) + list(directory.glob('*.parquet')):
            real[p.stem] = read_table(p)
        raw = from_source(real, source)
    return derive(cast_all(raw, model), source)


def build_select(table, source='kuali'):
    """Read-only SELECT text for one mapped table (for the laptop sample pull)."""
    smap = load_source_map(source)
    for espec in smap['entities'].values():
        for part in espec['parts']:
            if table in part['tables']:
                cols = ', '.join(part['tables'][table])
                return f"SELECT {cols} FROM {smap['owner']}.{table}"
    raise KeyError(table)

