# Proposal change detection — the anti-join that defines "new work."
#
# A proposal is "new" to this pipeline if its FILE_DATA_ID exists in
# Kuali's narrative attachments but not in our analytics staging table.
# We query both sides, subtract in Python, and return only the delta.
#
# Why the anti-join is in Python instead of a single DB link: the two
# Oracle databases here are in different security zones, and setting
# up a dblink between them would have required a security review we
# didn't want to block on. Pulling both ID sets into memory and
# diffing in pandas is fast enough (~1 second for millions of IDs)
# and needs no cross-environment infrastructure.

from ingestion.detect_new_records import get_processed_ids
from ingestion.oracle_connection   import oracle_conn
from configs.config_loader         import load_config
import pandas as pd


def get_new_document_ids():
    """Return (new_ids, kuali_metadata_df) for unprocessed proposals.

    Steps:
      1. Pull the set of already-processed IDs from the analytics DB.
      2. Pull candidate proposal metadata from Kuali, filtered to
         narrative type '535' (the SPA-designated "proposal narrative"
         type code — other type codes are budgets, justifications,
         routing sheets, etc. that we don't want in the NLP corpus).
      3. Subtract the processed set from the Kuali candidate set.
    """
    db     = load_config("database.yaml")
    creds  = db["oracle"]["credentials"]
    dsstag = db["oracle"]["dsstag"]
    kuali  = db["oracle"]["kuali"]

    # ── STEP 1: processed IDs from the analytics DB ─────────────────
    print("\nFetching processed IDs from DSSTAG...")

    with oracle_conn(
        dsstag["host"], dsstag["port"], dsstag["service"],
        creds["username"], creds["password"]
    ) as conn:
        processed_ids = get_processed_ids(conn)

    print(f"Processed IDs found: {len(processed_ids)}")

    # ── STEP 2: candidate proposals from Kuali ──────────────────────
    print("\nFetching proposal metadata from KUALI...")
    rows = []

    with oracle_conn(
        kuali["host"], kuali["port"], kuali["service"],
        creds["username"], creds["password"]
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    a.ID,
                    b.FILE_DATA_ID,
                    b.PROPOSAL_NUMBER,
                    b.MODULE_NUMBER,
                    b.FILE_NAME,
                    b.CONTENT_TYPE,
                    b.UPDATE_USER,
                    b.UPDATE_TIMESTAMP,
                    b.OBJ_ID,
                    b.UPLOAD_TIMESTAMP,
                    b.UPLOAD_USER,
                    c.MODULE_SEQUENCE_NUMBER,
                    c.MODULE_TITLE,
                    c.MODULE_STATUS_CODE,
                    c.NARRATIVE_TYPE_CODE
                FROM KUALI.FILE_DATA a
                INNER JOIN KUALI.NARRATIVE_ATTACHMENT b
                    ON a.ID = b.FILE_DATA_ID
                LEFT JOIN KUALI.NARRATIVE c
                    ON b.PROPOSAL_NUMBER = c.PROPOSAL_NUMBER
                   AND b.MODULE_NUMBER   = c.MODULE_NUMBER
                WHERE c.NARRATIVE_TYPE_CODE = '535'
            """)

            columns = [col[0] for col in cur.description]
            for row in cur:
                rows.append(dict(zip(columns, row)))

    kuali_metadata_df = pd.DataFrame(rows)
    print(f"Candidate proposal rows found: {len(kuali_metadata_df)}")

    # ── STEP 3: the delta ───────────────────────────────────────────
    kuali_metadata_df["ID"] = kuali_metadata_df["ID"].astype(str)
    kuali_ids = set(kuali_metadata_df["ID"])
    new_ids   = list(kuali_ids - processed_ids)

    print("\nNew proposals detected:", len(new_ids))
    if new_ids:
        print("\nSample new IDs:")
        print(new_ids[:10])

    # Filter the metadata down to just the new rows so downstream
    # stages don't have to re-filter.
    kuali_metadata_df = kuali_metadata_df[
        kuali_metadata_df["ID"].isin(new_ids)
    ]

    return new_ids, kuali_metadata_df
