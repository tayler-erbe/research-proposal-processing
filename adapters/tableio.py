"""Read and write row tables as CSV (always) or parquet (when pyarrow loads).

Rows are lists of dicts. CSV loses types, so readers return strings and
adapters.mapping.cast() restores canonical types.
"""
import csv
from pathlib import Path


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except Exception:  # missing, or blocked by the machine's application policy
        return None


def write_table(rows, path, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or sorted({k for r in rows for k in r})
    if path.suffix == '.parquet':
        lib = _pyarrow()
        if lib is None:
            raise RuntimeError('pyarrow is not available; write .csv instead')
        pa, pq = lib
        pq.write_table(pa.Table.from_pylist([{c: r.get(c) for c in columns} for r in rows]), path)
        return path
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: '' if r.get(c) is None else r.get(c) for c in columns})
    return path


def read_table(path):
    path = Path(path)
    if path.suffix == '.parquet':
        lib = _pyarrow()
        if lib is None:
            raise RuntimeError('pyarrow is not available to read ' + path.name)
        return lib[1].read_table(path).to_pylist()
    with path.open(newline='', encoding='utf-8') as f:
        return [{k: (None if v == '' else v) for k, v in r.items()} for r in csv.DictReader(f)]


def find_table(directory, name):
    """Return the parquet or CSV file for a table name, preferring parquet."""
    for ext in ('.parquet', '.csv'):
        p = Path(directory) / (name + ext)
        if p.exists():
            return p
    return None
