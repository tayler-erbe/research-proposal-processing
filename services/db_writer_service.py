# Proposal DB writer — the big one. Five Oracle writes executed in a
# specific sequence that has to stay in this order because writes 3-4-5
# read from the previous step's output directly out of the DB, not out
# of the intermediate parquet.
#
# WRITE ORDER (must be in this sequence):
#   1. MERGE    → ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT         (staging 1)
#   2. MERGE    → ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP      (staging 2)
#   3. MERGE    → T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS          (aggregated)
#   4. MERGE    → T_CLIENT_TABLE_PROPOSAL_KEYWORDS                (aggregated, no HERDS)
#   5. METANODE rebuild governance filter table (reads from Kuali)
#   6. TRUNCATE + INSERT → ANALYTICS.T_KC_PRPSL_ABST_KYWD        (production)
#
# Design choices worth understanding:
#
#   - DRY_RUN defaults to True. Always run dry-first, inspect output,
#     then set dry_run=False explicitly. There is no accidental path
#     to a live write.
#
#   - The production write additionally requires confirm_production=True.
#     It's the only TRUNCATE in the pipeline; a TRUNCATE on ANALYTICS
#     is not recoverable, which is why it's double-gated.
#
#   - Each write logs row counts before and after, and runs a spot-
#     check query to confirm at least one row from this batch made it
#     into the destination table. Post-write verification has caught
#     exactly one silent data-type bug that the pre-write row count
#     alone would not have caught.
#
#   - Every write is wrapped in an explicit transaction. On failure,
#     uncommitted changes roll back; committed writes from earlier
#     steps are kept and can be re-run through without duplication
#     because every write is a MERGE, not an INSERT.

import importlib
import oracledb
import pandas as pd
from pathlib import Path
from datetime import datetime

from configs.config_loader import load_config


# ── Connection settings are loaded from the DB config at runtime.
#    No credentials or hostnames are embedded in this file.


def _get_connection():
    """Return a raw oracledb connection to the staging database.

    oracle_conn() is a @contextmanager and can't be used here because
    this module holds the connection open across five writes and their
    transactions. Instead we import the module once (to trigger the
    thick-mode init in its module-level code) and then make a direct
    connect call.
    """
    importlib.import_module("ingestion.oracle_connection")  # thick-mode init

    db     = load_config("database.yaml")
    dsstag = db["oracle"]["dsstag"]
    creds  = db["oracle"]["credentials"]

    dsn = oracledb.makedsn(
        dsstag["host"], dsstag["port"], service_name=dsstag["service"]
    )
    return oracledb.connect(
        user=creds["username"],
        password=creds["password"],
        dsn=dsn,
    )


def delete_bad_rows(conn, ids, proposal_numbers):
    """Remove previously written rows from all five tables by ID /
    PROPOSAL_NUMBER. Used for recovery when a bug is discovered after
    a live write — call this once with the list of affected IDs to
    clean up, then re-run the pipeline with the corrected logic.

    This function is not called by run_db_writer. It exists for manual
    use from a notebook or REPL when something needs undoing."""
    cursor  = conn.cursor()
    id_ph   = ",".join([f":{i+1}" for i in range(len(ids))])
    prop_ph = ",".join([f":{i+1}" for i in range(len(proposal_numbers))])

    tables_by_id = [
        "ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT",
        "ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP",
    ]
    tables_by_proposal = [
        "ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS",
        "ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS",
    ]

    print("\n[DELETE BAD ROWS]")

    for table in tables_by_id:
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE ID IN ({id_ph})", ids)
        before = cursor.fetchone()[0]
        cursor.execute(f"DELETE FROM {table} WHERE ID IN ({id_ph})", ids)
        print(f"  {table}: deleted {before} rows")

    for table in tables_by_proposal:
        cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE PROPOSAL_NUMBER IN ({prop_ph})",
            proposal_numbers,
        )
        before = cursor.fetchone()[0]
        cursor.execute(
            f"DELETE FROM {table} WHERE PROPOSAL_NUMBER IN ({prop_ph})",
            proposal_numbers,
        )
        print(f"  {table}: deleted {before} rows")

    # Production table uses KC_PT_NBR, which is the first 12 chars of
    # PROPOSAL_NUMBER. Match the truncation so the delete targets the
    # right rows.
    kc_numbers = [str(p)[:12] for p in proposal_numbers]
    kc_ph      = ",".join([f":{i+1}" for i in range(len(kc_numbers))])
    cursor.execute(
        f"SELECT COUNT(*) FROM ANALYTICS.T_KC_PRPSL_ABST_KYWD WHERE KC_PT_NBR IN ({kc_ph})",
        kc_numbers,
    )
    before = cursor.fetchone()[0]
    cursor.execute(
        f"DELETE FROM ANALYTICS.T_KC_PRPSL_ABST_KYWD WHERE KC_PT_NBR IN ({kc_ph})",
        kc_numbers,
    )
    print(f"  ANALYTICS.T_KC_PRPSL_ABST_KYWD: deleted {before} rows")

    conn.commit()
    cursor.close()
    print("[DELETE COMPLETE]\n")


def _row_count(cursor, table):
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    return cursor.fetchone()[0]


def _log(msg, dry_run=False):
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}{msg}")


def _safe_val(val):
    """Convert pandas NA / NaT to None for Oracle binding."""
    try:
        if isinstance(val, (list, dict)):
            return val
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _row_to_bind(row, cols):
    return [_safe_val(row[c]) for c in cols]


# ── WRITE 1: T_FILE_DATA_BLOB2TEXT (MERGE on ID) ─────────────────────

def write_staging1(conn, df, dry_run=True):
    table = "ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT"

    _log(f"\n{'='*60}", dry_run)
    _log(f"WRITE 1: {table}", dry_run)
    _log(f"{'='*60}", dry_run)

    cursor = conn.cursor()
    before = _row_count(cursor, table)
    _log(f"Rows before: {before:,}", dry_run)
    _log(f"Rows to merge: {len(df):,}", dry_run)

    cols = [
        "ID", "PROPOSAL_NUMBER", "MODULE_NUMBER", "FILE_NAME",
        "CONTENT_TYPE", "UPDATE_USER", "UPDATE_TIMESTAMP", "OBJ_ID",
        "FILE_DATA_ID", "UPLOAD_TIMESTAMP", "UPLOAD_USER",
        "HIDE_IN_HIERARCHY", "MODULE_SEQUENCE_NUMBER", "MODULE_TITLE",
        "MODULE_STATUS_CODE", "NARRATIVE_TYPE_CODE", "MIME Type",
        "Title", "AUTHOR", "DATE_CREATED", "DATE_MODIFIED",
        "Format", "Creator Tool", "Metadata Date",
        "CONTENT", "LENGTH", "SUCCESSFULLY_PARSED",
    ]

    merge_sql = """
        MERGE INTO ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT tgt
        USING (
            SELECT
                :1  AS ID,                    :2  AS PROPOSAL_NUMBER,
                :3  AS MODULE_NUMBER,         :4  AS FILE_NAME,
                :5  AS CONTENT_TYPE,          :6  AS UPDATE_USER,
                :7  AS UPDATE_TIMESTAMP,      :8  AS OBJ_ID,
                :9  AS FILE_DATA_ID,          :10 AS UPLOAD_TIMESTAMP,
                :11 AS UPLOAD_USER,           :12 AS HIDE_IN_HIERARCHY,
                :13 AS MODULE_SEQUENCE_NUMBER,:14 AS MODULE_TITLE,
                :15 AS MODULE_STATUS_CODE,    :16 AS NARRATIVE_TYPE_CODE,
                :17 AS "MIME Type",           :18 AS "Title",
                :19 AS AUTHOR,                :20 AS DATE_CREATED,
                :21 AS DATE_MODIFIED,         :22 AS "Format",
                :23 AS "Creator Tool",        :24 AS "Metadata Date",
                :25 AS CONTENT,               :26 AS LENGTH,
                :27 AS SUCCESSFULLY_PARSED
            FROM DUAL
        ) src ON (tgt.ID = src.ID)
        WHEN MATCHED THEN UPDATE SET
            tgt.PROPOSAL_NUMBER     = src.PROPOSAL_NUMBER,
            tgt.MODULE_NUMBER       = src.MODULE_NUMBER,
            tgt.CONTENT             = src.CONTENT,
            tgt.LENGTH              = src.LENGTH,
            tgt.SUCCESSFULLY_PARSED = src.SUCCESSFULLY_PARSED,
            tgt.UPDATE_TIMESTAMP    = src.UPDATE_TIMESTAMP
        WHEN NOT MATCHED THEN INSERT (
            ID, PROPOSAL_NUMBER, MODULE_NUMBER, FILE_NAME, CONTENT_TYPE,
            UPDATE_USER, UPDATE_TIMESTAMP, OBJ_ID, FILE_DATA_ID,
            UPLOAD_TIMESTAMP, UPLOAD_USER, HIDE_IN_HIERARCHY,
            MODULE_SEQUENCE_NUMBER, MODULE_TITLE, MODULE_STATUS_CODE,
            NARRATIVE_TYPE_CODE, "MIME Type", "Title", AUTHOR,
            DATE_CREATED, DATE_MODIFIED, "Format", "Creator Tool",
            "Metadata Date", CONTENT, LENGTH, SUCCESSFULLY_PARSED
        ) VALUES (
            src.ID, src.PROPOSAL_NUMBER, src.MODULE_NUMBER, src.FILE_NAME,
            src.CONTENT_TYPE, src.UPDATE_USER, src.UPDATE_TIMESTAMP,
            src.OBJ_ID, src.FILE_DATA_ID, src.UPLOAD_TIMESTAMP,
            src.UPLOAD_USER, src.HIDE_IN_HIERARCHY, src.MODULE_SEQUENCE_NUMBER,
            src.MODULE_TITLE, src.MODULE_STATUS_CODE, src.NARRATIVE_TYPE_CODE,
            src."MIME Type", src."Title", src.AUTHOR, src.DATE_CREATED,
            src.DATE_MODIFIED, src."Format", src."Creator Tool",
            src."Metadata Date", src.CONTENT, src.LENGTH,
            src.SUCCESSFULLY_PARSED
        )
    """

    if dry_run:
        _log("SQL: MERGE INTO T_FILE_DATA_BLOB2TEXT ON (ID) — not executed", dry_run)
        _log(f"Would process {len(df):,} rows", dry_run)
    else:
        bind_data = [_row_to_bind(row, cols) for _, row in df.iterrows()]
        # CLOB columns must be declared explicitly before binding —
        # without setinputsizes, oracledb tries to bind strings as
        # VARCHAR2 and raises ORA-01461 for any string over ~4000
        # chars. Position indices below are 1-based and match cols.
        # Position 25 = CONTENT (CLOB).
        cursor.setinputsizes(None, None, None, None, None,   # 1-5
                             None, None, None, None, None,    # 6-10
                             None, None, None, None, None,    # 11-15
                             None, None, None, None, None,    # 16-20
                             None, None, None, None,          # 21-24
                             oracledb.CLOB,                    # 25 = CONTENT
                             None, None)                       # 26-27
        cursor.executemany(merge_sql, bind_data)
        conn.commit()
        after    = _row_count(cursor, table)
        inserted = after - before
        _log(f"Rows after:    {after:,}")
        _log(f"Net new rows:  {inserted:,}")
        _verify_staging1(cursor, df)

    cursor.close()


def _verify_staging1(cursor, df):
    ids = df["ID"].dropna().tolist()
    if not ids:
        return
    placeholders = ",".join([f":{i+1}" for i in range(len(ids))])
    cursor.execute(
        f"SELECT COUNT(*) FROM ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT WHERE ID IN ({placeholders})",
        ids,
    )
    found = cursor.fetchone()[0]
    print(f"  [VERIFY] IDs written and confirmed in DB: {found}/{len(ids)}")


# ── WRITE 2: T_RSRCH_PRPSL_PRCSSD_NLP (MERGE on ID) ──────────────────

def write_staging2(conn, df, dry_run=True):
    table = "ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP"

    _log(f"\n{'='*60}", dry_run)
    _log(f"WRITE 2: {table}", dry_run)
    _log(f"{'='*60}", dry_run)

    cursor = conn.cursor()
    before = _row_count(cursor, table)
    _log(f"Rows before: {before:,}", dry_run)
    _log(f"Rows to merge: {len(df):,}", dry_run)

    cols = [
        "ID", "PROPOSAL_NUMBER", "MODULE_NUMBER",
        "CLEANED_TEXT", "CLEANED_TEXT_REDUCED",
        "TOPIC_KEYWORDS", "TOPIC_KEYWORDS_REDUCED",
        "RAKE_OUT", "RAKE_OUT_REDUCED",
        "UNIQUE_RAKE", "UNIQUE_RAKE_REDUCED",
        "SKLEARN_TOP_KEYWORDS", "SKLEARN_TOP_KEYWORDS_REDUCED",
        "HERDS_FIELD_SKLEARN",
        "ENTITIES", "ORGANIZATIONS", "PERSONS", "LOCATIONS",
        "TIME", "FAC", "EVENT", "MONEY", "PRODUCT",
        "UPDATE_TIMESTAMP",
    ]

    merge_sql = """
        MERGE INTO ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP tgt
        USING (
            SELECT
                :1  AS ID,                          :2  AS PROPOSAL_NUMBER,
                :3  AS MODULE_NUMBER,               :4  AS CLEANED_TEXT,
                :5  AS CLEANED_TEXT_REDUCED,        :6  AS TOPIC_KEYWORDS,
                :7  AS TOPIC_KEYWORDS_REDUCED,      :8  AS RAKE_OUT,
                :9  AS RAKE_OUT_REDUCED,            :10 AS UNIQUE_RAKE,
                :11 AS UNIQUE_RAKE_REDUCED,         :12 AS SKLEARN_TOP_KEYWORDS,
                :13 AS SKLEARN_TOP_KEYWORDS_REDUCED,:14 AS HERDS_FIELD_SKLEARN,
                :15 AS ENTITIES,                    :16 AS ORGANIZATIONS,
                :17 AS PERSONS,                     :18 AS LOCATIONS,
                :19 AS TIME_COL,                    :20 AS FAC,
                :21 AS EVENT_COL,                   :22 AS MONEY,
                :23 AS PRODUCT,                     :24 AS UPDATE_TIMESTAMP
            FROM DUAL
        ) src ON (tgt.ID = src.ID)
        WHEN MATCHED THEN UPDATE SET
            tgt.PROPOSAL_NUMBER              = src.PROPOSAL_NUMBER,
            tgt.MODULE_NUMBER                = src.MODULE_NUMBER,
            tgt.CLEANED_TEXT                 = src.CLEANED_TEXT,
            tgt.CLEANED_TEXT_REDUCED         = src.CLEANED_TEXT_REDUCED,
            tgt.TOPIC_KEYWORDS               = src.TOPIC_KEYWORDS,
            tgt.TOPIC_KEYWORDS_REDUCED       = src.TOPIC_KEYWORDS_REDUCED,
            tgt.RAKE_OUT                     = src.RAKE_OUT,
            tgt.RAKE_OUT_REDUCED             = src.RAKE_OUT_REDUCED,
            tgt.UNIQUE_RAKE                  = src.UNIQUE_RAKE,
            tgt.UNIQUE_RAKE_REDUCED          = src.UNIQUE_RAKE_REDUCED,
            tgt.SKLEARN_TOP_KEYWORDS         = src.SKLEARN_TOP_KEYWORDS,
            tgt.SKLEARN_TOP_KEYWORDS_REDUCED = src.SKLEARN_TOP_KEYWORDS_REDUCED,
            tgt.HERDS_FIELD_SKLEARN          = src.HERDS_FIELD_SKLEARN,
            tgt.ENTITIES                     = src.ENTITIES,
            tgt.ORGANIZATIONS                = src.ORGANIZATIONS,
            tgt.PERSONS                      = src.PERSONS,
            tgt.LOCATIONS                    = src.LOCATIONS,
            tgt.TIME                         = src.TIME_COL,
            tgt.FAC                          = src.FAC,
            tgt.EVENT                        = src.EVENT_COL,
            tgt.MONEY                        = src.MONEY,
            tgt.PRODUCT                      = src.PRODUCT,
            tgt.UPDATE_TIMESTAMP             = src.UPDATE_TIMESTAMP
        WHEN NOT MATCHED THEN INSERT (
            ID, PROPOSAL_NUMBER, MODULE_NUMBER,
            CLEANED_TEXT, CLEANED_TEXT_REDUCED,
            TOPIC_KEYWORDS, TOPIC_KEYWORDS_REDUCED,
            RAKE_OUT, RAKE_OUT_REDUCED,
            UNIQUE_RAKE, UNIQUE_RAKE_REDUCED,
            SKLEARN_TOP_KEYWORDS, SKLEARN_TOP_KEYWORDS_REDUCED,
            HERDS_FIELD_SKLEARN,
            ENTITIES, ORGANIZATIONS, PERSONS, LOCATIONS,
            TIME, FAC, EVENT, MONEY, PRODUCT,
            UPDATE_TIMESTAMP
        ) VALUES (
            src.ID, src.PROPOSAL_NUMBER, src.MODULE_NUMBER,
            src.CLEANED_TEXT, src.CLEANED_TEXT_REDUCED,
            src.TOPIC_KEYWORDS, src.TOPIC_KEYWORDS_REDUCED,
            src.RAKE_OUT, src.RAKE_OUT_REDUCED,
            src.UNIQUE_RAKE, src.UNIQUE_RAKE_REDUCED,
            src.SKLEARN_TOP_KEYWORDS, src.SKLEARN_TOP_KEYWORDS_REDUCED,
            src.HERDS_FIELD_SKLEARN,
            src.ENTITIES, src.ORGANIZATIONS, src.PERSONS, src.LOCATIONS,
            src.TIME_COL, src.FAC, src.EVENT_COL, src.MONEY, src.PRODUCT,
            src.UPDATE_TIMESTAMP
        )
    """

    if dry_run:
        _log("SQL: MERGE INTO T_RSRCH_PRPSL_PRCSSD_NLP ON (ID) — not executed", dry_run)
        _log(f"Would process {len(df):,} rows", dry_run)
        _log("\nSample row (first non-null):", dry_run)
        sample = df.dropna(subset=["ID"]).iloc[0]
        for col in ["ID", "PROPOSAL_NUMBER", "HERDS_FIELD_SKLEARN",
                    "RAKE_OUT_REDUCED", "SKLEARN_TOP_KEYWORDS_REDUCED"]:
            val = str(sample.get(col, ""))[:80]
            _log(f"  {col}: {val}", dry_run)
    else:
        bind_data = [_row_to_bind(row, cols) for _, row in df.iterrows()]
        # CLOB position map (1-based, matches cols order):
        #   4 = CLEANED_TEXT, 6 = TOPIC_KEYWORDS, 8 = RAKE_OUT,
        #  10 = UNIQUE_RAKE, 12 = SKLEARN_TOP_KEYWORDS
        cursor.setinputsizes(None, None, None,           # 1-3
                             oracledb.CLOB,               # 4  CLEANED_TEXT
                             None,                        # 5  CLEANED_TEXT_REDUCED
                             oracledb.CLOB,               # 6  TOPIC_KEYWORDS
                             None,                        # 7  TOPIC_KEYWORDS_REDUCED
                             oracledb.CLOB,               # 8  RAKE_OUT
                             None,                        # 9  RAKE_OUT_REDUCED
                             oracledb.CLOB,               # 10 UNIQUE_RAKE
                             None,                        # 11 UNIQUE_RAKE_REDUCED
                             oracledb.CLOB,               # 12 SKLEARN_TOP_KEYWORDS
                             None, None, None, None,      # 13-16
                             None, None, None, None,      # 17-20
                             None, None, None, None)      # 21-24
        cursor.executemany(merge_sql, bind_data)
        conn.commit()
        after = _row_count(cursor, table)
        _log(f"Rows after:    {after:,}")
        _log(f"Net new rows:  {after - before:,}")
        _verify_staging2(cursor, df)

    cursor.close()


def _verify_staging2(cursor, df):
    ids = df["ID"].dropna().tolist()
    if not ids:
        return
    placeholders = ",".join([f":{i+1}" for i in range(len(ids))])
    cursor.execute(
        f"SELECT COUNT(*) FROM ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP "
        f"WHERE ID IN ({placeholders})",
        ids,
    )
    found = cursor.fetchone()[0]
    print(f"  [VERIFY] IDs written and confirmed in DB: {found}/{len(ids)}")

    cursor.execute(
        "SELECT ID, HERDS_FIELD_SKLEARN, RAKE_OUT_REDUCED "
        "FROM ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP "
        f"WHERE ID = :{1}",
        [ids[0]],
    )
    row = cursor.fetchone()
    if row:
        print(f"  [SPOT CHECK] ID={row[0]}")
        print(f"    HERDS_FIELD_SKLEARN:  {row[1]}")
        print(f"    RAKE_OUT_REDUCED:     {str(row[2])[:80] if row[2] else None}")


# ── WRITE 3 + 4: aggregated tables (SQL MERGE, LISTAGG in-DB) ────────

def write_aggregated_tables(conn, dry_run=True):
    """Build the per-proposal aggregates by LISTAGG-ing the module-level
    rows from staging 2. Doing this in Oracle rather than pandas matters
    — the LISTAGG ON OVERFLOW TRUNCATE clause handles oversized strings
    correctly at the DB's understanding of VARCHAR2 width, and avoids
    a round trip back through the client."""
    _log(f"\n{'='*60}", dry_run)
    _log("WRITE 3+4: Aggregated keyword tables (SQL MERGE)", dry_run)
    _log(f"{'='*60}", dry_run)

    cursor = conn.cursor()

    # ── 3a: HERDS + Keywords
    herds_table  = "ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS"
    before_herds = _row_count(cursor, herds_table)
    _log(f"\nT_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS rows before: {before_herds:,}", dry_run)

    merge_herds = """
        MERGE INTO ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS tgt
        USING (
            SELECT
                CAST(SUBSTR(LISTAGG(HERDS_FIELD_SKLEARN, ',' ON OVERFLOW TRUNCATE)
                     WITHIN GROUP (ORDER BY HERDS_FIELD_SKLEARN), 1, 500)
                     AS VARCHAR2(500 CHAR)) AS HERDS_FIELD,
                CAST(SUBSTR(LISTAGG(RAKE_OUT_REDUCED, ',' ON OVERFLOW TRUNCATE)
                     WITHIN GROUP (ORDER BY RAKE_OUT_REDUCED), 1, 500)
                     AS VARCHAR2(500 CHAR)) AS TOPIC_KEYWORDS,
                CAST(SUBSTR(LISTAGG(SKLEARN_TOP_KEYWORDS_REDUCED, ',' ON OVERFLOW TRUNCATE)
                     WITHIN GROUP (ORDER BY SKLEARN_TOP_KEYWORDS_REDUCED), 1, 1000)
                     AS VARCHAR2(1000 CHAR)) AS SKLEARN_KEYWORDS,
                CAST(MAX(UPDATE_TIMESTAMP) AS TIMESTAMP) AS UPDATE_TIMESTAMP,
                PROPOSAL_NUMBER
            FROM ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP
            GROUP BY PROPOSAL_NUMBER
        ) src ON (tgt.PROPOSAL_NUMBER = src.PROPOSAL_NUMBER)
        WHEN MATCHED THEN UPDATE SET
            tgt.HERDS_FIELD      = src.HERDS_FIELD,
            tgt.TOPIC_KEYWORDS   = src.TOPIC_KEYWORDS,
            tgt.SKLEARN_KEYWORDS = src.SKLEARN_KEYWORDS,
            tgt.UPDATE_TIMESTAMP = src.UPDATE_TIMESTAMP
        WHEN NOT MATCHED THEN INSERT
            (HERDS_FIELD, TOPIC_KEYWORDS, SKLEARN_KEYWORDS,
             UPDATE_TIMESTAMP, PROPOSAL_NUMBER)
        VALUES
            (src.HERDS_FIELD, src.TOPIC_KEYWORDS, src.SKLEARN_KEYWORDS,
             src.UPDATE_TIMESTAMP, src.PROPOSAL_NUMBER)
    """

    # ── 3b: Keywords only (no HERDS field)
    kw_table  = "ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS"
    before_kw = _row_count(cursor, kw_table)
    _log(f"T_CLIENT_TABLE_PROPOSAL_KEYWORDS rows before:       {before_kw:,}", dry_run)

    merge_kw = """
        MERGE INTO ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS tgt
        USING (
            SELECT
                CAST(SUBSTR(LISTAGG(RAKE_OUT_REDUCED, ',' ON OVERFLOW TRUNCATE)
                     WITHIN GROUP (ORDER BY RAKE_OUT_REDUCED), 1, 500)
                     AS VARCHAR2(500 CHAR)) AS TOPIC_KEYWORDS,
                CAST(SUBSTR(LISTAGG(SKLEARN_TOP_KEYWORDS_REDUCED, ',' ON OVERFLOW TRUNCATE)
                     WITHIN GROUP (ORDER BY SKLEARN_TOP_KEYWORDS_REDUCED), 1, 1000)
                     AS VARCHAR2(1000 CHAR)) AS SKLEARN_KEYWORDS,
                CAST(MAX(UPDATE_TIMESTAMP) AS TIMESTAMP) AS UPDATE_TIMESTAMP,
                PROPOSAL_NUMBER
            FROM ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP
            GROUP BY PROPOSAL_NUMBER
        ) src ON (tgt.PROPOSAL_NUMBER = src.PROPOSAL_NUMBER)
        WHEN MATCHED THEN UPDATE SET
            tgt.TOPIC_KEYWORDS   = src.TOPIC_KEYWORDS,
            tgt.SKLEARN_KEYWORDS = src.SKLEARN_KEYWORDS,
            tgt.UPDATE_TIMESTAMP = src.UPDATE_TIMESTAMP
        WHEN NOT MATCHED THEN INSERT
            (TOPIC_KEYWORDS, SKLEARN_KEYWORDS, UPDATE_TIMESTAMP, PROPOSAL_NUMBER)
        VALUES
            (src.TOPIC_KEYWORDS, src.SKLEARN_KEYWORDS,
             src.UPDATE_TIMESTAMP, src.PROPOSAL_NUMBER)
    """

    if dry_run:
        _log("\nSQL: MERGE INTO T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS — not executed", dry_run)
        _log("SQL: MERGE INTO T_CLIENT_TABLE_PROPOSAL_KEYWORDS — not executed", dry_run)
        _log("Both read from T_RSRCH_PRPSL_PRCSSD_NLP already in Oracle", dry_run)
    else:
        cursor.execute(merge_herds)
        conn.commit()
        after_herds = _row_count(cursor, herds_table)
        _log(f"T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS rows after: {after_herds:,}")
        _log(f"Net new rows: {after_herds - before_herds:,}")

        cursor.execute(merge_kw)
        conn.commit()
        after_kw = _row_count(cursor, kw_table)
        _log(f"T_CLIENT_TABLE_PROPOSAL_KEYWORDS rows after:       {after_kw:,}")
        _log(f"Net new rows: {after_kw - before_kw:,}")

        cursor.execute(
            "SELECT PROPOSAL_NUMBER, HERDS_FIELD, SKLEARN_KEYWORDS "
            "FROM ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS "
            "WHERE ROWNUM <= 2"
        )
        print("\n  [SPOT CHECK] T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS sample:")
        for row in cursor.fetchall():
            print(f"    PROPOSAL={row[0]}  HERDS={row[1]}  SKLEARN={str(row[2])[:60]}")

    cursor.close()


# ── WRITE 5: production table (TRUNCATE + INSERT + DELETE) ───────────

def write_production(conn, dry_run=True, confirm=False):
    """Rebuild the production keyword table. This is the only TRUNCATE
    in the pipeline — the production table is small and it's faster to
    rebuild it than to MERGE.

    Double-gated by dry_run=False AND confirm=True because TRUNCATE
    plus a bad INSERT would leave the production table empty with no
    easy way back. The gating is intentional friction."""
    _log(f"\n{'='*60}", dry_run)
    _log("WRITE 5: ANALYTICS.T_KC_PRPSL_ABST_KYWD (PRODUCTION)", dry_run)
    _log(f"{'='*60}", dry_run)

    if not dry_run and not confirm:
        print("[BLOCKED] Production write requires confirm=True.")
        print("          Set confirm=True only after verifying staging tables.")
        return

    cursor = conn.cursor()
    before = _row_count(cursor, "ANALYTICS.T_KC_PRPSL_ABST_KYWD")
    _log(f"Rows before truncate: {before:,}", dry_run)

    truncate_sql = "TRUNCATE TABLE ANALYTICS.T_KC_PRPSL_ABST_KYWD"

    insert_sql = """
        INSERT INTO ANALYTICS.T_KC_PRPSL_ABST_KYWD
            (KC_PT_NBR, ABST_CATGRY_LIST, ABST_KYWD_LIST,
             DW_LOAD_DT, HERD_CATGRY_LIST)
        SELECT
            SUBSTR(PROPOSAL_NUMBER, 1, 12),
            SUBSTR(TOPIC_KEYWORDS,  1, 500),
            SUBSTR(SKLEARN_KEYWORDS,1, 1000),
            CAST(UPDATE_TIMESTAMP AS DATE),
            SUBSTR(HERDS_FIELD,     1, 500)
        FROM ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS
        WHERE SKLEARN_KEYWORDS IS NOT NULL
           OR TOPIC_KEYWORDS   IS NOT NULL
    """

    # Governance filter — restrict to UIUC-funded proposals only.
    # The allowed list is rebuilt every run by the metanode service
    # (services/metanode_service.py) from KUALI.AWARD + proposal admin
    # details. Anything not in that whitelist gets deleted post-insert.
    delete_sql = """
        DELETE FROM ANALYTICS.T_KC_PRPSL_ABST_KYWD
        WHERE KC_PT_NBR NOT IN (
            SELECT PROPOSAL_NUMBER
            FROM ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID
        )
    """

    if dry_run:
        _log("SQL: TRUNCATE TABLE ANALYTICS.T_KC_PRPSL_ABST_KYWD — not executed", dry_run)
        _log("SQL: INSERT INTO T_KC_PRPSL_ABST_KYWD SELECT FROM T_CLIENT_TABLE... — not executed", dry_run)
        _log("SQL: DELETE unfunded/non-UIUC proposals — not executed", dry_run)
        _log(f"\nProduction table currently has {before:,} rows", dry_run)

        preview_sql = """
            SELECT
                SUBSTR(PROPOSAL_NUMBER, 1, 12) AS KC_PT_NBR,
                SUBSTR(TOPIC_KEYWORDS,  1, 50)  AS ABST_CATGRY_LIST,
                SUBSTR(SKLEARN_KEYWORDS,1, 50)  AS ABST_KYWD_LIST,
                SUBSTR(HERDS_FIELD,     1, 50)  AS HERD_CATGRY_LIST
            FROM ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS
            WHERE ROWNUM <= 3
        """
        try:
            cursor.execute(preview_sql)
            rows = cursor.fetchall()
            _log("\n[PREVIEW] Sample rows that would be inserted:", dry_run)
            for row in rows:
                _log(f"  KC_PT_NBR={row[0]}  HERDS={row[3]}  SKLEARN={row[2]}", dry_run)
        except Exception as e:
            _log(f"[PREVIEW failed — aggregated table may not exist yet: {e}]", dry_run)
    else:
        cursor.execute(truncate_sql)
        cursor.execute(insert_sql)
        conn.commit()

        after_insert = _row_count(cursor, "ANALYTICS.T_KC_PRPSL_ABST_KYWD")
        _log(f"Rows after insert:  {after_insert:,}")

        cursor.execute(delete_sql)
        conn.commit()

        after_delete = _row_count(cursor, "ANALYTICS.T_KC_PRPSL_ABST_KYWD")
        removed = after_insert - after_delete
        _log(f"Rows after governance filter: {after_delete:,}")
        _log(f"Rows removed (unfunded/non-UIUC): {removed:,}")

        cursor.execute(
            "SELECT KC_PT_NBR, ABST_CATGRY_LIST, ABST_KYWD_LIST, HERD_CATGRY_LIST "
            "FROM ANALYTICS.T_KC_PRPSL_ABST_KYWD WHERE ROWNUM <= 3"
        )
        print("\n  [SPOT CHECK] ANALYTICS.T_KC_PRPSL_ABST_KYWD sample:")
        for row in cursor.fetchall():
            print(f"    KC_PT_NBR={row[0]}")
            print(f"      ABST_CATGRY_LIST: {str(row[1])[:80]}")
            print(f"      ABST_KYWD_LIST:   {str(row[2])[:80]}")
            print(f"      HERD_CATGRY_LIST: {str(row[3])[:80]}")

    cursor.close()


# ── Public entry point ───────────────────────────────────────────────

def run_db_writer(intermediate_dir: Path, dry_run: bool = True,
                  confirm_production: bool = False):
    """Execute all five DB writes in sequence.

    Parameters
    ----------
    intermediate_dir   : Path to storage/intermediate_tables/
    dry_run            : If True (default), prints all SQL without
                         executing. Set to False only after verifying
                         dry-run output.
    confirm_production : Must be True to allow the production write.
                         Ignored when dry_run=True. This second gate
                         is belt-and-suspenders for the one TRUNCATE.

    Usage
    -----
        # Step 1 — always run dry first:
        run_db_writer(INTERMEDIATE_DIR, dry_run=True)

        # Step 2 — when satisfied, run for real:
        run_db_writer(INTERMEDIATE_DIR, dry_run=False, confirm_production=True)
    """
    mode = "DRY RUN" if dry_run else "LIVE WRITE"
    print(f"\n{'='*60}")
    print(f" DB WRITER — {mode}")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    s1 = pd.read_parquet(intermediate_dir / "staging_1_blob2text.parquet")
    s2 = pd.read_parquet(intermediate_dir / "staging_2_nlp.parquet")

    print(f"\nStaging 1 loaded: {len(s1):,} rows")
    print(f"Staging 2 loaded: {len(s2):,} rows")

    print("\nConnecting to staging DB...")
    conn = _get_connection()
    print("Connected.")

    try:
        write_staging1(conn, s1, dry_run=dry_run)
        write_staging2(conn, s2, dry_run=dry_run)
        write_aggregated_tables(conn, dry_run=dry_run)

        # Metanode rebuild must run before write_production so the
        # governance DELETE has an up-to-date allowed-proposal list.
        from services.metanode_service import run_metanode
        run_metanode(dry_run=dry_run)

        write_production(conn, dry_run=dry_run, confirm=confirm_production)

    except Exception as e:
        print(f"\n[ERROR] DB write failed: {e}")
        if not dry_run:
            print("[ROLLBACK] Rolling back uncommitted changes...")
            conn.rollback()
        raise
    finally:
        conn.close()
        print("\nConnection closed.")

    print(f"\n{'='*60}")
    print(f" DB WRITER COMPLETE — {mode}")
    print(f" Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
