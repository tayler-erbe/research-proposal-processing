# Awards counterpart to change_detection_service.py.
#
# Same two-step anti-join pattern, different tables. Awards on the
# Kuali side require a four-way join (see awards_fetch_documents for
# the full explanation of why TYPE_CODE != '3' and AWARD_SEQUENCE_STATUS
# = 'ACTIVE' are the right filters).
#
# The credentials block is shared between the two Oracle databases
# here — the same service account has read access to both. If your
# deployment splits these accounts, move the credentials under each
# database's block in database.yaml and adjust the two oracle_conn
# calls below accordingly.

import pandas as pd
from ingestion.oracle_connection import oracle_conn
from configs.config_loader       import load_config


KUALI_AWARDS_QUERY = """
SELECT
    aa.AWARD_ATTACHMENT_ID,
    aa.AWARD_ID,
    aa.AWARD_NUMBER,
    aa.SEQUENCE_NUMBER,
    aa.DESCRIPTION,
    aa.UPDATE_TIMESTAMP,
    aa.UPDATE_USER,
    aa.LAST_UPDATE_TIMESTAMP,
    aa.LAST_UPDATE_USER,
    af.FILE_NAME,
    af.CONTENT_TYPE,
    af.FILE_DATA_ID,
    ty.TYPE_CODE,
    ty.DESCRIPTION         AS TYPE_DESCRIPTION,
    a.AWARD_SEQUENCE_STATUS
FROM KUALI.AWARD_ATTACHMENT aa
INNER JOIN KUALI.ATTACHMENT_FILE      af ON aa.FILE_ID     = af.FILE_ID
INNER JOIN KUALI.AWARD_ATTACHMENT_TYPE ty ON ty.TYPE_CODE   = aa.TYPE_CODE
INNER JOIN KUALI.FILE_DATA            fd ON fd.ID           = af.FILE_DATA_ID
INNER JOIN KUALI.AWARD                 a ON a.AWARD_ID      = aa.AWARD_ID
WHERE ty.TYPE_CODE != '3'
  AND a.AWARD_SEQUENCE_STATUS = 'ACTIVE'
"""

STAGING_PROCESSED_QUERY = """
SELECT FILE_DATA_ID
FROM ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT_AWARDS
"""


def get_new_awards_document_ids():
    """Return (new_ids, kuali_metadata_df) for unprocessed awards."""
    db     = load_config("database.yaml")
    creds  = db["oracle"]["credentials"]
    kuali  = db["oracle"]["kuali"]
    dsstag = db["oracle"]["dsstag"]

    # Kuali — all active, non-type-3 award attachments.
    print("[change_detection] Connecting to Kuali to fetch award attachment IDs...")
    with oracle_conn(
        kuali["host"], kuali["port"], kuali["service"],
        creds["username"], creds["password"]
    ) as conn:
        kuali_df = pd.read_sql(KUALI_AWARDS_QUERY, conn)

    kuali_df["FILE_DATA_ID"] = kuali_df["FILE_DATA_ID"].astype(str).str.strip()
    print(f"[change_detection] Kuali → {len(kuali_df)} active award attachments found")

    # Analytics staging — everything already processed.
    print("[change_detection] Connecting to analytics DB to fetch already-processed IDs...")
    with oracle_conn(
        dsstag["host"], dsstag["port"], dsstag["service"],
        creds["username"], creds["password"]
    ) as conn:
        staging_df = pd.read_sql(STAGING_PROCESSED_QUERY, conn)

    staging_df["FILE_DATA_ID"] = staging_df["FILE_DATA_ID"].astype(str).str.strip()
    already_processed = set(staging_df["FILE_DATA_ID"].tolist())
    print(f"[change_detection] Analytics DB → {len(already_processed)} already-processed IDs")

    # The delta.
    new_df  = kuali_df[~kuali_df["FILE_DATA_ID"].isin(already_processed)].copy()
    new_ids = new_df["FILE_DATA_ID"].tolist()
    print(f"[change_detection] New unprocessed awards: {len(new_ids)}")

    return new_ids, new_df
