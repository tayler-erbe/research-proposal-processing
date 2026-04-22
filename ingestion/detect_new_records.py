# Diagnostic + anti-join helper for the proposal pipeline.
#
# The diagnostic queries at the top of get_processed_ids() look
# gratuitous but earn their keep — they print the connected user,
# database, current schema, and table visibility before the real
# query runs. After getting bitten once by a schema-switch silently
# pulling the wrong table, these three-lines-of-output diagnostics
# make "wrong environment" bugs obvious within the first second of
# a pipeline run.

import csv


def get_min_max_blob_dates(conn):
    """Unused in the current pipeline but kept for ad hoc audits."""
    sql = """
        SELECT MIN(UPDATE_TIMESTAMP), MAX(UPDATE_TIMESTAMP)
        FROM ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()


def get_processed_ids(conn):
    """Return the set of FILE_DATA_IDs already processed into the
    staging table, so the caller can anti-join against the Kuali
    source and extract only new work."""
    sql = """
        SELECT FILE_DATA_ID
        FROM ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT
    """

    ids = set()

    with conn.cursor() as cur:
        # ── Environment sanity checks ───────────────────────────────
        # These print-and-move-on queries are the fastest early-warning
        # signal that the pipeline is pointed at the wrong database.
        print("\n--- Oracle Connection Debug Info ---", flush=True)

        cur.execute("SELECT USER FROM dual")
        print("Connected Oracle user:", cur.fetchone(), flush=True)

        cur.execute("SELECT SYS_CONTEXT('USERENV','DB_NAME') FROM dual")
        print("Connected DB:", cur.fetchone(), flush=True)

        cur.execute("SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM dual")
        print("Current schema:", cur.fetchone(), flush=True)

        cur.execute("""
            SELECT owner, table_name
            FROM all_tables
            WHERE table_name = 'T_FILE_DATA_BLOB2TEXT'
        """)
        print("Table visibility check:", cur.fetchall(), flush=True)

        # ── Real work ───────────────────────────────────────────────
        print("\n--- Running Processed ID Query ---", flush=True)
        cur.execute(sql)

        for (id_val,) in cur:
            if id_val is not None:
                ids.add(str(id_val))

    print(f"\nTotal processed IDs loaded: {len(ids)}", flush=True)
    return ids


def write_ids_to_csv(ids, path):
    """Debugging helper — dump a set of IDs to a single-column CSV.
    Not called by the pipeline itself, useful for manual reconciliation."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID"])
        for id_val in sorted(ids):
            writer.writerow([id_val])
