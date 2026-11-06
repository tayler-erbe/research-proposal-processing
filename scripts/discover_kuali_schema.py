#!/usr/bin/env python3
# Read-only schema discovery for the proposal -> award linkage work.
#
# Purpose: figure out how development proposals (the documents this
# pipeline already extracts), institutional proposals and awards join
# together in *this* Kuali instance, without guessing from vendor docs.
#
# What it does (SELECT-only, against the Kuali connection in
# configs/database.yaml):
#   1. Inventories tables in the Kuali schema whose names look relevant
#      (proposal, award, sponsor, unit, person, budget, narrative...)
#      with their columns, types and optimizer row-count estimates.
#   2. Lists foreign keys between those tables.
#   3. Prints small code -> description lookup tables (status codes,
#      attachment types) so outcome codes can be interpreted.
#   4. Runs a join "funnel": how many distinct proposals survive each
#      hop from narrative attachment to funded award. COUNTs only.
#
# What it does NOT do: it never prints row-level record contents (no
# titles, names, amounts or file data), and it never writes to any
# database. The report is metadata + aggregate counts, intended to be
# safe to share for schema design. Review it before sharing anyway.
#
# Usage (from the repo root, on a machine that can reach Kuali):
#   python scripts/discover_kuali_schema.py
#   python scripts/discover_kuali_schema.py --owner KUALI --out storage/discovery/report.md
#
# Every query is wrapped individually: a missing table or permission
# error is recorded in the report and the script moves on.

import argparse
from contextlib import contextmanager
import json
import re
import socket
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_$#]{0,127}$")


def schema_owner(value):
    value = value.upper()
    if not IDENTIFIER.fullmatch(value):
        raise argparse.ArgumentTypeError("Owner must be an unquoted Oracle schema identifier")
    return value


def load_database(path):
    import yaml
    with Path(path).open(encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)["oracle"]


@contextmanager
def discovery_connection(db, args):
    # Independent of production modules: no import-time client initialization.
    import oracledb
    if args.oracle_client_lib:
        oracledb.init_oracle_client(lib_dir=args.oracle_client_lib)
    source, credentials = db["kuali"], db["credentials"]
    params = oracledb.ConnectParams(host=source["host"], port=int(source["port"]),
        service_name=source["service"], tcp_connect_timeout=10, retry_count=0)
    with oracledb.connect(user=credentials["username"], password=credentials["password"], params=params) as conn:
        conn.call_timeout = args.call_timeout_ms
        conn.autocommit = False
        yield conn


FILE_BEARING_SQL = """
    SELECT table_name, column_name, data_type, nullable
    FROM all_tab_columns
    WHERE owner = :owner
      AND (column_name IN ('FILE_DATA_ID', 'FILE_ID') OR data_type = 'BLOB')
    ORDER BY table_name, column_id
"""

# Table-name fragments worth inventorying. Deliberately broad: the
# point is to find the local equivalents, not to assume vendor names.
NAME_PATTERNS = [
    "%PROPOSAL%", "EPS_%", "%AWARD%", "%SPONSOR%", "%NARRATIVE%",
    "%PROP_PERSON%", "%PERSON%ROLE%", "UNIT", "UNIT_%", "%BUDGET%",
    "%ACTIVITY_TYPE%", "%FILE_DATA%", "%ATTACHMENT%", "%NEGOTIATION%",
    "%SUBAWARD%", "%CFDA%", "%STATUS%",
]

# Tables we'd skip from the column dump because they are huge and
# irrelevant (audit/history/doc-handler noise). Still counted.
SKIP_COLUMN_DUMP_SUFFIXES = ("_HIST", "_AUD", "_BAK", "_TMP", "_OLD")

# Small lookup tables worth printing as code -> description.
# (table, code column, description column). Missing ones are skipped.
LOOKUPS = [
    ("PROPOSAL_STATUS",        "PROPOSAL_STATUS_CODE", "DESCRIPTION"),
    ("EPS_PROPOSAL_STATUS",    "STATUS_CODE",          "DESCRIPTION"),
    ("AWARD_STATUS",           "STATUS_CODE",          "DESCRIPTION"),
    ("PROPOSAL_TYPE",          "PROPOSAL_TYPE_CODE",   "DESCRIPTION"),
    ("ACTIVITY_TYPE",          "ACTIVITY_TYPE_CODE",   "DESCRIPTION"),
    ("NARRATIVE_TYPE",         "NARRATIVE_TYPE_CODE",  "DESCRIPTION"),
    ("AWARD_ATTACHMENT_TYPE",  "TYPE_CODE",            "DESCRIPTION"),
    ("SPONSOR_TYPE",           "SPONSOR_TYPE_CODE",    "DESCRIPTION"),
    ("PROPOSAL_PERSON_ROLE",   "PROP_PERSON_ROLE_ID",  "DESCRIPTION"),
]
LOOKUP_MAX_ROWS = 200


def build_funnel(o):
    """Join funnel from the documents we already extract to awards.

    Each step is a standalone COUNT query so one bad guess about a
    table/column doesn't hide the steps that do work. Names follow
    standard Kuali Coeus; failures tell us where local naming differs.
    """
    base = f"""
        FROM {o}.NARRATIVE_ATTACHMENT na
        JOIN {o}.NARRATIVE n
          ON n.PROPOSAL_NUMBER = na.PROPOSAL_NUMBER
         AND n.MODULE_NUMBER   = na.MODULE_NUMBER
        WHERE n.NARRATIVE_TYPE_CODE = '535'
    """
    return [
        ("1. Dev proposals with a type-535 narrative (what the pipeline extracts)",
         f"SELECT COUNT(DISTINCT na.PROPOSAL_NUMBER) {base}"),

        ("2. ...that exist in EPS_PROPOSAL (dev proposal header)",
         f"""SELECT COUNT(DISTINCT na.PROPOSAL_NUMBER) {base.replace('WHERE', f'''
            JOIN {o}.EPS_PROPOSAL ep ON ep.PROPOSAL_NUMBER = na.PROPOSAL_NUMBER
            WHERE''', 1)}"""),

        ("3. ...that link to an institutional proposal via PROPOSAL_ADMIN_DETAILS",
         f"""SELECT COUNT(DISTINCT na.PROPOSAL_NUMBER) {base.replace('WHERE', f'''
            JOIN {o}.PROPOSAL_ADMIN_DETAILS pad
              ON pad.DEV_PROPOSAL_NUMBER = na.PROPOSAL_NUMBER
             AND pad.INST_PROPOSAL_ID IS NOT NULL
            WHERE''', 1)}"""),

        ("4. ...whose institutional proposal row exists in PROPOSAL",
         f"""SELECT COUNT(DISTINCT na.PROPOSAL_NUMBER) {base.replace('WHERE', f'''
            JOIN {o}.PROPOSAL_ADMIN_DETAILS pad ON pad.DEV_PROPOSAL_NUMBER = na.PROPOSAL_NUMBER
            JOIN {o}.PROPOSAL ip ON ip.PROPOSAL_ID = pad.INST_PROPOSAL_ID
            WHERE''', 1)}"""),

        ("5. ...whose institutional proposal (any version) is linked to an award",
         f"""SELECT COUNT(DISTINCT na.PROPOSAL_NUMBER) {base.replace('WHERE', f'''
            JOIN {o}.PROPOSAL_ADMIN_DETAILS pad ON pad.DEV_PROPOSAL_NUMBER = na.PROPOSAL_NUMBER
            JOIN {o}.PROPOSAL ip  ON ip.PROPOSAL_ID = pad.INST_PROPOSAL_ID
            JOIN {o}.PROPOSAL ipv ON ipv.PROPOSAL_NUMBER = ip.PROPOSAL_NUMBER
            JOIN {o}.AWARD_FUNDING_PROPOSALS afp ON afp.PROPOSAL_ID = ipv.PROPOSAL_ID
            WHERE''', 1)}"""),

        ("6. ...where that award has an ACTIVE sequence",
         f"""SELECT COUNT(DISTINCT na.PROPOSAL_NUMBER) {base.replace('WHERE', f'''
            JOIN {o}.PROPOSAL_ADMIN_DETAILS pad ON pad.DEV_PROPOSAL_NUMBER = na.PROPOSAL_NUMBER
            JOIN {o}.PROPOSAL ip  ON ip.PROPOSAL_ID = pad.INST_PROPOSAL_ID
            JOIN {o}.PROPOSAL ipv ON ipv.PROPOSAL_NUMBER = ip.PROPOSAL_NUMBER
            JOIN {o}.AWARD_FUNDING_PROPOSALS afp ON afp.PROPOSAL_ID = ipv.PROPOSAL_ID
            JOIN {o}.AWARD a  ON a.AWARD_ID = afp.AWARD_ID
            JOIN {o}.AWARD aa ON aa.AWARD_NUMBER = a.AWARD_NUMBER
                             AND aa.AWARD_SEQUENCE_STATUS = 'ACTIVE'
            WHERE''', 1)}"""),
    ]


def build_shape_checks(o):
    """Aggregate checks on versioning and outcome codes. Counts only."""
    return [
        ("PROPOSAL: rows vs distinct PROPOSAL_NUMBER (versioning)",
         f"SELECT COUNT(*) AS ROWS_, COUNT(DISTINCT PROPOSAL_NUMBER) AS DISTINCT_NUMBERS FROM {o}.PROPOSAL"),
        ("PROPOSAL: rows by PROPOSAL_SEQUENCE_STATUS",
         f"SELECT PROPOSAL_SEQUENCE_STATUS, COUNT(*) FROM {o}.PROPOSAL GROUP BY PROPOSAL_SEQUENCE_STATUS ORDER BY 2 DESC"),
        ("PROPOSAL: current-version rows by STATUS_CODE (outcome distribution)",
         f"""SELECT ip.STATUS_CODE, COUNT(*)
             FROM {o}.PROPOSAL ip
             WHERE ip.PROPOSAL_SEQUENCE_STATUS = 'ACTIVE'
             GROUP BY ip.STATUS_CODE ORDER BY 2 DESC"""),
        ("AWARD: rows vs distinct AWARD_NUMBER (versioning)",
         f"SELECT COUNT(*) AS ROWS_, COUNT(DISTINCT AWARD_NUMBER) AS DISTINCT_NUMBERS FROM {o}.AWARD"),
        ("AWARD: rows by AWARD_SEQUENCE_STATUS",
         f"SELECT AWARD_SEQUENCE_STATUS, COUNT(*) FROM {o}.AWARD GROUP BY AWARD_SEQUENCE_STATUS ORDER BY 2 DESC"),
        ("PROPOSAL_ADMIN_DETAILS: rows / distinct dev / distinct inst",
         f"""SELECT COUNT(*), COUNT(DISTINCT DEV_PROPOSAL_NUMBER), COUNT(DISTINCT INST_PROPOSAL_ID)
             FROM {o}.PROPOSAL_ADMIN_DETAILS"""),
        ("AWARD_FUNDING_PROPOSALS: rows / distinct awards / distinct proposals",
         f"""SELECT COUNT(*), COUNT(DISTINCT AWARD_ID), COUNT(DISTINCT PROPOSAL_ID)
             FROM {o}.AWARD_FUNDING_PROPOSALS"""),
        ("NARRATIVE: rows by NARRATIVE_TYPE_CODE (top 25)",
         f"""SELECT * FROM (
               SELECT NARRATIVE_TYPE_CODE, COUNT(*) AS N FROM {o}.NARRATIVE
               GROUP BY NARRATIVE_TYPE_CODE ORDER BY N DESC
             ) WHERE ROWNUM <= 25"""),
        ("EPS_PROPOSAL: dev proposals by update year (not creation/submission year)",
         f"""SELECT EXTRACT(YEAR FROM UPDATE_TIMESTAMP) AS YR, COUNT(*)
             FROM {o}.EPS_PROPOSAL GROUP BY EXTRACT(YEAR FROM UPDATE_TIMESTAMP) ORDER BY 1"""),
    ]


class Report:
    def __init__(self):
        self.lines = []

    def h(self, text, level=2):
        self.add("")
        self.add("#" * level + " " + text)
        self.add("")

    def add(self, text=""):
        print(text, flush=True)
        self.lines.append(text)

    def table(self, headers, rows):
        self.add("| " + " | ".join(str(h) for h in headers) + " |")
        self.add("|" + "---|" * len(headers))
        for r in rows:
            self.add("| " + " | ".join("" if v is None else str(v) for v in r) + " |")

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def run(cur, sql, binds=None):
    if not sql.lstrip().upper().startswith("SELECT"):
        raise ValueError("Discovery accepts only its SELECT statements")
    cur.execute(sql, binds or {})
    headers = [d[0] for d in cur.description]
    rows = cur.fetchmany(10001)
    if len(rows) > 10000:
        raise ValueError("Result exceeds discovery row limit; narrow the metadata scope")
    return headers, rows


def safe(report, label, fn):
    try:
        return fn()
    except Exception as e:  # missing table, privilege, bad column name
        # Driver errors can include endpoints or connection configuration.
        # Report codes and exception type, never the raw exception text.
        codes = re.findall(r"(?:ORA|DPY|DPI)-\d+", str(e))
        msg = ", ".join(sorted(set(codes))) or type(e).__name__
        report.add(f"- **{label}**: FAILED — `{msg}`")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", type=schema_owner, default="KUALI", help="Kuali schema owner (default KUALI)")
    ap.add_argument("--out", default="storage/discovery/kuali_schema_report.md")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/database.yaml")
    ap.add_argument("--oracle-client-lib", help="Approved local Instant Client directory when thick mode is required")
    ap.add_argument("--call-timeout-ms", type=int, default=30000)
    ap.add_argument("--with-counts", action="store_true", help="Also run source-table aggregate scans after metadata review")
    ap.add_argument("--preflight", action="store_true", help="Check config shape and TCP reachability only; no DB login")
    args = ap.parse_args()
    if not 1000 <= args.call_timeout_ms <= 600000:
        ap.error("call timeout must be between 1000 and 600000 ms")
    output = (PROJECT_ROOT / args.out).resolve()
    if not output.is_relative_to((PROJECT_ROOT / "storage").resolve()):
        ap.error("Reports must stay under the gitignored storage directory")
    o = args.owner.upper()
    db = load_database(args.config)
    if args.preflight:
        source = db["kuali"]
        status = {"configuration_present": True, "target": "kuali", "tcp_reachable": False, "database_login_attempted": False}
        try:
            with socket.create_connection((source["host"], int(source["port"])), timeout=5):
                status["tcp_reachable"] = True
        except OSError as exc:
            status["error_type"] = type(exc).__name__
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".preflight.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status))
        return 0 if status["tcp_reachable"] else 2

    rpt = Report()
    rpt.add(f"# Kuali schema discovery — owner {o}")
    rpt.add(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Metadata and aggregate counts only.")

    t0 = time.time()
    with discovery_connection(db, args) as conn:
        cur = conn.cursor()

        # ── 0. Who/where am I ─────────────────────────────────────────
        rpt.h("0. Connection")
        safe(rpt, "context", lambda: rpt.table(*run(cur, """
            SELECT USER, SYS_CONTEXT('USERENV','DB_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA')
            FROM dual""")))

        # ── 1. Table inventory ────────────────────────────────────────
        rpt.h("1. Relevant tables and views")
        like = " OR ".join(f"object_name LIKE :p{i}" for i in range(len(NAME_PATTERNS)))
        binds = {f"p{i}": p for i, p in enumerate(NAME_PATTERNS)}
        binds["owner"] = o
        res = safe(rpt, "inventory", lambda: run(cur, f"""
            SELECT o.object_name, o.object_type, t.num_rows, t.last_analyzed
            FROM all_objects o
            LEFT JOIN all_tables t ON t.owner = o.owner AND t.table_name = o.object_name
            WHERE o.owner = :owner AND o.object_type IN ('TABLE','VIEW') AND ({like})
            ORDER BY o.object_name""", binds))
        tables = []
        if res:
            headers, rows = res
            rpt.add(f"{len(rows)} objects matched. NUM_ROWS is the optimizer estimate, not an exact count.")
            rpt.add("")
            rpt.table(["NAME", "TYPE", "NUM_ROWS (est.)", "LAST_ANALYZED"], rows)
            tables = [r[0] for r in rows]
            if not rows:
                rpt.add("")
                rpt.add("**No objects found.** Either the owner name is wrong (try --owner) "
                        "or this account lacks visibility; check the grants.")

        rpt.h("1b. All file-bearing columns, independent of table-name guesses")
        res = safe(rpt, "file-bearing columns", lambda: run(cur, FILE_BEARING_SQL, {"owner": o}))
        if res:
            rpt.table(*res)
            tables = sorted(set(tables) | {row[0] for row in res[1]})
            rpt.add("Candidates only: this does not establish attachment ownership, scope or join keys.")

        # ── 2. Columns ────────────────────────────────────────────────
        rpt.h("2. Columns")
        dump = [t for t in tables if not t.endswith(SKIP_COLUMN_DUMP_SUFFIXES)]
        for tname in dump:
            res = safe(rpt, tname, lambda: run(cur, """
                SELECT column_name, data_type, data_length, nullable
                FROM all_tab_columns WHERE owner = :o AND table_name = :t
                ORDER BY column_id""", {"o": o, "t": tname}))
            if res:
                cols = ", ".join(
                    f"{c}:{dt}{'(' + str(dl) + ')' if dt in ('VARCHAR2', 'CHAR') else ''}{'' if n == 'Y' else '!'}"
                    for c, dt, dl, n in res[1])
                rpt.add(f"- **{tname}** — {cols}")
        rpt.add("")
        rpt.add("_`!` = NOT NULL._")

        # ── 3. Foreign keys ───────────────────────────────────────────
        rpt.h("3. Declared foreign keys (among relevant tables)")
        res = safe(rpt, "foreign keys", lambda: run(cur, """
            SELECT c.table_name, LISTAGG(cc.column_name, ',') WITHIN GROUP (ORDER BY cc.position),
                   r.table_name, LISTAGG(rc.column_name, ',') WITHIN GROUP (ORDER BY rc.position)
            FROM all_constraints c
            JOIN all_cons_columns cc ON cc.owner = c.owner AND cc.constraint_name = c.constraint_name
            JOIN all_constraints r   ON r.owner = c.r_owner AND r.constraint_name = c.r_constraint_name
            JOIN all_cons_columns rc ON rc.owner = r.owner AND rc.constraint_name = r.constraint_name
                                    AND rc.position = cc.position
            WHERE c.owner = :o AND c.constraint_type = 'R'
            GROUP BY c.constraint_name, c.table_name, r.table_name
            ORDER BY c.table_name""", {"o": o}))
        if res:
            keep = set(tables)
            fk_rows = [r for r in res[1] if r[0] in keep or r[2] in keep]
            if fk_rows:
                rpt.table(["CHILD", "CHILD COLS", "PARENT", "PARENT COLS"], fk_rows)
            else:
                rpt.add("None declared (common in Kuali — joins are enforced in the app layer). "
                        "The funnel in section 5 tests joins empirically instead.")

        # ── 4. Lookups ────────────────────────────────────────────────
        rpt.h("4. Code lookups")
        for tname, code, desc in LOOKUPS:
            res = safe(rpt, tname, lambda: run(cur, f"""
                SELECT * FROM (SELECT {code}, {desc} FROM {o}.{tname} ORDER BY {code})
                WHERE ROWNUM <= {LOOKUP_MAX_ROWS}"""))
            if res:
                rpt.add(f"**{tname}**")
                rpt.add("")
                rpt.table(res[0], res[1])
                rpt.add("")

        # ── 5. Join funnel ────────────────────────────────────────────
        rpt.h("5. Join funnel: extracted narratives -> awards")
        rpt.add("Distinct dev proposals surviving each hop. A big drop shows where the "
                "join assumption is wrong or where proposals were never funded.")
        rpt.add("")
        if not args.with_counts:
            rpt.add("Source-table aggregate scans skipped. Review metadata first, then use --with-counts.")
        for label, sql in build_funnel(o) if args.with_counts else []:
            s = time.time()
            res = safe(rpt, label, lambda: run(cur, sql))
            if res:
                rpt.add(f"- **{label}**: {res[1][0][0]:,}  _({time.time() - s:.1f}s)_")

        # ── 6. Shape checks ───────────────────────────────────────────
        rpt.h("6. Versioning and outcome distributions")
        for label, sql in build_shape_checks(o) if args.with_counts else []:
            res = safe(rpt, label, lambda: run(cur, sql))
            if res:
                rpt.add(f"**{label}**")
                rpt.add("")
                rpt.table(res[0], res[1])
                rpt.add("")

    rpt.add("")
    rpt.add(f"_Finished in {time.time() - t0:.0f}s._")
    out = output
    rpt.save(out)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        codes = re.findall(r"(?:ORA|DPY|DPI)-\d+", str(exc))
        print("Discovery stopped: " + (", ".join(sorted(set(codes))) or type(exc).__name__) +
              ". No production entry point was run. Check network, config and client prerequisites.", file=sys.stderr)
        sys.exit(1)
