"""Durable private state: sources, case associations, spans, jobs and reviews.

Job identity = stage + subject + configuration hash (model digest, prompt hash,
schema version, input snapshot hash). Changing any of them creates a new job and
never overwrites an earlier result.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from . import VERSION

REPO = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 900


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def connect(root):
    root = Path(root).resolve()
    if root == REPO or REPO in root.parents:
        raise ValueError('Private runtime must be outside the repository')
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / 'research_intake.sqlite3', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS sources(
        id TEXT PRIMARY KEY,          -- SHA-256 of the bytes
        path TEXT NOT NULL, format TEXT, pages INTEGER,
        state TEXT, error TEXT, extraction_scope TEXT);
    -- One source can belong to several attachments, proposals and awards.
    CREATE TABLE IF NOT EXISTS associations(
        source TEXT NOT NULL REFERENCES sources(id), case_id TEXT, role TEXT,
        family TEXT, source_table TEXT, attachment_key TEXT, attachment_type TEXT,
        file_name TEXT, cohort TEXT, match_group TEXT, inst_proposal TEXT, award TEXT,
        predictor_eligibility TEXT,
        PRIMARY KEY(source, case_id, role, attachment_key));
    CREATE TABLE IF NOT EXISTS pages(
        source TEXT, page INTEGER, state TEXT, chars INTEGER, PRIMARY KEY(source, page));
    CREATE TABLE IF NOT EXISTS spans(
        id TEXT PRIMARY KEY, source TEXT, page INTEGER, seq INTEGER, kind TEXT,
        locator TEXT, text TEXT, method TEXT);
    CREATE TABLE IF NOT EXISTS classifications(
        source TEXT PRIMARY KEY, family TEXT, doc_type TEXT, method TEXT, basis TEXT);
    CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, stage TEXT, subject TEXT, config TEXT, state TEXT,
        attempts INTEGER DEFAULT 0, lease REAL, error TEXT);
    CREATE TABLE IF NOT EXISTS attempts(
        id INTEGER PRIMARY KEY, job TEXT, started REAL, finished REAL, state TEXT, error TEXT);
    CREATE TABLE IF NOT EXISTS outputs(job TEXT PRIMARY KEY, payload TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS reviews(
        job TEXT, field TEXT, reviewer TEXT, decision TEXT, value TEXT, reason TEXT,
        created REAL, PRIMARY KEY(job, field, created));
    CREATE TABLE IF NOT EXISTS controls(key TEXT PRIMARY KEY, value TEXT);
    CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
        span UNINDEXED, source UNINDEXED, page UNINDEXED, text);
    ''')
    return db


def put_span(db, source, page, seq, kind, locator, text, method):
    sid = digest(canonical([source, page, seq, kind, locator, method]))
    if db.execute('INSERT OR IGNORE INTO spans VALUES(?,?,?,?,?,?,?,?)',
                  (sid, source, page, seq, kind, canonical(locator), text, method)).rowcount and text.strip():
        db.execute('INSERT INTO search VALUES(?,?,?,?)', (sid, source, page, text))
    return sid


def enqueue(db, stage, subject, config):
    config = dict(config, protocol=VERSION)
    key = digest(canonical([stage, subject, config]))
    db.execute('INSERT OR IGNORE INTO jobs(id,stage,subject,config,state) VALUES(?,?,?,?,?)',
               (key, stage, subject, canonical(config), 'queued'))
    return key


def claim(db, stage):
    db.execute('BEGIN IMMEDIATE')
    try:
        # An expired lease means the outcome is uncertain: never replay it automatically.
        db.execute("UPDATE jobs SET state='interrupted',error='Lease expired; explicit retry required' "
                   "WHERE state='running' AND lease<?", (time.time(),))
        if db.execute("SELECT 1 FROM controls WHERE key='pause' AND value='1'").fetchone():
            db.commit()
            return None
        row = db.execute("SELECT * FROM jobs WHERE stage=? AND state='queued' ORDER BY rowid LIMIT 1",
                         (stage,)).fetchone()
        if row:
            db.execute("UPDATE jobs SET state='running',attempts=attempts+1,lease=? WHERE id=?",
                       (time.time() + LEASE_SECONDS, row['id']))
            db.execute("INSERT INTO attempts(job,started,state) VALUES(?,?,'running')", (row['id'], time.time()))
        db.commit()
        return row
    except BaseException:
        db.rollback()
        raise


def finish(db, job, payload=None, error=None):
    with db:
        state = 'failed' if error else 'complete'
        if payload is not None:
            db.execute('INSERT OR REPLACE INTO outputs VALUES(?,?,?)', (job, canonical(payload), time.time()))
        db.execute('UPDATE jobs SET state=?,lease=NULL,error=? WHERE id=?', (state, error, job))
        db.execute("UPDATE attempts SET finished=?,state=?,error=? WHERE job=? AND state='running'",
                   (time.time(), state, error, job))


def status(db):
    q = lambda sql: [dict(r) for r in db.execute(sql)]
    return {'sources': q('SELECT state,format,count(*) AS n FROM sources GROUP BY state,format'),
            'pages': q('SELECT state,count(*) AS n FROM pages GROUP BY state'),
            'classes': q('SELECT family,doc_type,count(*) AS n FROM classifications GROUP BY family,doc_type'),
            'jobs': q('SELECT stage,state,count(*) AS n FROM jobs GROUP BY stage,state')}
