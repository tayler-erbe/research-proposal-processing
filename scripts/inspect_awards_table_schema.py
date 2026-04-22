#!/usr/bin/env python3
# One-off schema inspector for the awards staging table.
#
# Prints column definitions, constraints, row count, and up to three
# sample rows (truncated). Used to confirm the exact column shapes the
# DB writer targets match the live table before a live write.
#
# Run from the repo root:
#   python scripts/inspect_awards_table_schema.py

import os
import sys
from pathlib import Path

# Make the repo root importable whether or not it's already on PYTHONPATH.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from ingestion.oracle_connection import oracle_conn
from configs.config_loader       import load_config


SCHEMA = "ANALYTICS_SCHEMA"
TABLE  = "T_FILE_DATA_BLOB2TEXT_AWARDS"


SCHEMA_QUERY = f"""
SELECT
    column_id,
    column_name,
    data_type,
    data_length,
    data_precision,
    data_scale,
    nullable,
    data_default
FROM all_tab_columns
WHERE owner      = '{SCHEMA}'
  AND table_name = '{TABLE}'
ORDER BY column_id
"""

CONSTRAINTS_QUERY = f"""
SELECT
    acc.constraint_name,
    acc.column_name,
    ac.constraint_type,
    ac.status
FROM all_cons_columns acc
JOIN all_constraints  ac
  ON acc.owner           = ac.owner
 AND acc.constraint_name = ac.constraint_name
WHERE acc.owner      = '{SCHEMA}'
  AND acc.table_name = '{TABLE}'
ORDER BY ac.constraint_type, acc.position
"""

SAMPLE_QUERY = f"""
SELECT *
FROM {SCHEMA}.{TABLE}
WHERE ROWNUM <= 3
"""


def main():
    db_cfg = load_config("database.yaml")
    dsstag = db_cfg["oracle"]["dsstag"]
    creds  = db_cfg["oracle"]["credentials"]

    print(f"\nConnecting to analytics DB ({dsstag['host']})...\n")

    with oracle_conn(
        dsstag["host"], dsstag["port"], dsstag["service"],
        creds["username"], creds["password"]
    ) as conn:
        cursor = conn.cursor()

        # ── Column definitions ───────────────────────────────────────
        print("=" * 70)
        print(f"TABLE: {SCHEMA}.{TABLE}")
        print("=" * 70)
        print(f"{'COL_ID':<7} {'COLUMN_NAME':<30} {'DATA_TYPE':<15} "
              f"{'LENGTH':<8} {'PREC':<6} {'SCALE':<6} {'NULL':<5} DEFAULT")
        print("-" * 100)

        cursor.execute(SCHEMA_QUERY)
        rows = cursor.fetchall()

        if not rows:
            print(f"  !! Table {SCHEMA}.{TABLE} not found or no SELECT privilege.")
        else:
            for r in rows:
                col_id, col_name, dtype, dlength, dprec, dscale, nullable, default = r
                print(
                    f"{str(col_id):<7} {str(col_name):<30} {str(dtype):<15} "
                    f"{str(dlength or ''):<8} {str(dprec or ''):<6} "
                    f"{str(dscale or ''):<6} {str(nullable):<5} {default or ''}"
                )

        # ── Constraints ──────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("CONSTRAINTS")
        print("=" * 70)
        cursor.execute(CONSTRAINTS_QUERY)
        for r in cursor.fetchall():
            cname, col, ctype, status = r
            type_label = {
                "P": "PRIMARY KEY",
                "U": "UNIQUE",
                "C": "CHECK",
                "R": "FOREIGN KEY",
            }.get(ctype, ctype)
            print(f"  [{type_label}]  {col}  ({cname})  — {status}")

        # ── Row count ────────────────────────────────────────────────
        cursor.execute(f"SELECT COUNT(*) FROM {SCHEMA}.{TABLE}")
        count = cursor.fetchone()[0]
        print(f"\nCurrent row count: {count:,}")

        # ── Sample rows ──────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("SAMPLE (up to 3 rows) — text columns truncated to 200 chars")
        print("=" * 70)
        cursor.execute(SAMPLE_QUERY)
        col_names   = [d[0] for d in cursor.description]
        sample_rows = cursor.fetchall()
        if not sample_rows:
            print("  (table is empty)")
        else:
            for row in sample_rows:
                print()
                for col, val in zip(col_names, row):
                    display = str(val)[:200] if val is not None else "NULL"
                    print(f"  {col:<30}: {display}")

        cursor.close()

    print("\nSchema inspection complete.\n")


if __name__ == "__main__":
    main()
