# Awards staging table writer.
#
# Uses executemany with a WHERE NOT EXISTS clause so re-runs are
# naturally idempotent — if a FILE_DATA_ID is already present, the
# INSERT is a no-op. This is belt-and-suspenders alongside the
# change_detection anti-join; in practice change_detection filters
# out the vast majority of duplicates, but if two pipeline runs
# race each other (shouldn't happen in production, does happen
# during development), the NOT EXISTS prevents duplicate rows.
#
# The oracledb.CLOB setinputsizes call is the non-obvious part:
# without it, large CONTENT strings trigger ORA-01461 ("can bind
# a LONG value only for insert into a LONG column") because oracledb
# tries to bind the string as VARCHAR2 rather than CLOB. Telling it
# up front that CONTENT is a CLOB fixes it.

import math
import pandas as pd
from pathlib import Path

import oracledb
from ingestion.oracle_connection import oracle_conn
from configs.config_loader       import load_config

TARGET_SCHEMA = "ANALYTICS_SCHEMA"
TARGET_TABLE  = "T_FILE_DATA_BLOB2TEXT_AWARDS"
FULL_TABLE    = f"{TARGET_SCHEMA}.{TARGET_TABLE}"

INSERT_SQL = f"""
INSERT INTO {FULL_TABLE} (
    AWARD_ATTACHMENT_ID, AWARD_ID, AWARD_NUMBER, SEQUENCE_NUMBER, DESCRIPTION,
    UPDATE_TIMESTAMP, UPDATE_USER, LAST_UPDATE_TIMESTAMP, LAST_UPDATE_USER,
    FILE_NAME, CONTENT_TYPE, FILE_DATA_ID, TYPE_CODE, ID, LEAD_UNIT_NUMBER,
    ACTIVITY_TYPE_CODE, AWARD_TYPE_CODE, AWARD_EFFECTIVE_DATE,
    AWARD_EXECUTION_DATE, BEGIN_DATE, SPONSOR_AWARD_NUMBER, OBJ_ID,
    LENGTH, SUCCESSFULLY_PARSED, CONTENT
)
SELECT
    :AWARD_ATTACHMENT_ID, :AWARD_ID, :AWARD_NUMBER, :SEQUENCE_NUMBER, :DESCRIPTION,
    :UPDATE_TIMESTAMP, :UPDATE_USER, :LAST_UPDATE_TIMESTAMP, :LAST_UPDATE_USER,
    :FILE_NAME, :CONTENT_TYPE, :FILE_DATA_ID, :TYPE_CODE, :ID, :LEAD_UNIT_NUMBER,
    :ACTIVITY_TYPE_CODE, :AWARD_TYPE_CODE, :AWARD_EFFECTIVE_DATE,
    :AWARD_EXECUTION_DATE, :BEGIN_DATE, :SPONSOR_AWARD_NUMBER, :OBJ_ID,
    :LENGTH, :SUCCESSFULLY_PARSED, :CONTENT
FROM DUAL
WHERE NOT EXISTS (
    SELECT 1 FROM {FULL_TABLE}
    WHERE FILE_DATA_ID = :FILE_DATA_ID
)
"""

# 50 rows per commit is a pragmatic balance: large enough to amortize
# the commit overhead, small enough that a mid-batch failure only
# strands 50 rows of work rather than the whole run.
BATCH_INSERT_SIZE = 50


def _clean_val(val):
    """Unwrap NaN/NaT/numpy scalars into native Python types that
    oracledb can bind without complaining."""
    try:
        if val is None:
            return None
        if isinstance(val, float) and math.isnan(val):
            return None
        if hasattr(val, "item"):
            return val.item()
        if pd.isna(val):
            return None
        return val
    except (TypeError, ValueError):
        return val


def _df_to_records(df: pd.DataFrame) -> list:
    """DataFrame → list of dicts, one per row, with all values cleaned.
    Also force CONTENT to a real string since Oracle's CLOB bind won't
    accept bytes or None-masquerading-as-empty-string."""
    records = []
    for _, row in df.iterrows():
        rec = {col: _clean_val(row[col]) for col in df.columns}
        if rec.get("CONTENT") is not None:
            rec["CONTENT"] = str(rec["CONTENT"])
        records.append(rec)
    return records


def run_awards_db_writer(
    intermediate_dir: Path,
    dry_run: bool = False,
) -> None:
    """Load the awards parquet and insert each row into the staging
    table. Use dry_run=True to preview row counts without writing."""
    intermediate_dir = Path(intermediate_dir)
    parquet_path     = intermediate_dir / "awards_full_table.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"[awards_db_writer] Cannot find {parquet_path}. "
            "Run the awards ingestion pipeline first."
        )

    df = pd.read_parquet(parquet_path)
    print(f"[awards_db_writer] Loaded {len(df)} rows from {parquet_path.name}")

    if df.empty:
        print("[awards_db_writer] DataFrame is empty — nothing to upload.")
        return

    db_cfg = load_config("database.yaml")
    dsstag = db_cfg["oracle"]["dsstag"]
    creds  = db_cfg["oracle"]["credentials"]

    records = _df_to_records(df)
    total   = len(records)
    batches = [records[i:i + BATCH_INSERT_SIZE] for i in range(0, total, BATCH_INSERT_SIZE)]

    if dry_run:
        print(f"[awards_db_writer] DRY RUN — would insert up to {total} rows into {FULL_TABLE}")
        return

    print(f"[awards_db_writer] Connecting to analytics DB ({dsstag['host']})...")
    with oracle_conn(
        dsstag["host"], dsstag["port"], dsstag["service"],
        creds["username"], creds["password"]
    ) as conn:
        cursor = conn.cursor()

        # Tell oracledb that CONTENT is a CLOB before binding, or
        # long strings will trigger ORA-01461.
        cursor.setinputsizes(CONTENT=oracledb.CLOB)

        inserted = 0
        for batch_num, batch in enumerate(batches, 1):
            try:
                cursor.executemany(INSERT_SQL, batch)
                conn.commit()
                inserted += len(batch)
                print(
                    f"  Batch {batch_num}/{len(batches)} committed "
                    f"({inserted}/{total} rows processed)"
                )
            except Exception as e:
                # Rollback the failed batch. Prior batches are already
                # committed and stay. The re-raise bubbles the error
                # so the orchestrator can decide whether to retry.
                conn.rollback()
                print(f"  [awards_db_writer] ERROR in batch {batch_num}: {e}")
                raise

        cursor.close()

    print(f"[awards_db_writer] Complete — {inserted} rows processed into {FULL_TABLE}")
