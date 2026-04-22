# Upsert writer for the NLP output table.
#
# Uses Oracle MERGE rather than INSERT + UPDATE separately. MERGE is a
# single atomic statement per row: match by ID, UPDATE if matched,
# INSERT if not. This is the right primitive for a pipeline where a
# proposal might be reprocessed and its keyword output refreshed.
#
# Column alias notes:
#   - RAKE_OUT in the DB is populated with KeyBERT output. The column
#     name predates the switch from RAKE to KeyBERT; renaming the
#     column would break every downstream consumer, so we kept the
#     name and changed what goes into it.

import oracledb
import pandas as pd


def upsert_nlp_results(df, conn):
    cursor = conn.cursor()

    sql = """
    MERGE INTO ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP target
    USING (
        SELECT
            :1  AS ID,
            :2  AS PROPOSAL_NUMBER,
            :3  AS MODULE_NUMBER,
            :4  AS UPDATE_TIMESTAMP,
            :5  AS CLEANED_TEXT,
            :6  AS CLEANED_TEXT_REDUCED,
            :7  AS ENTITIES,
            :8  AS SKLEARN_TOP_KEYWORDS,
            :9  AS RAKE_OUT,
            :10 AS TOPIC_KEYWORDS,
            :11 AS HERDS_FIELD_SKLEARN
        FROM dual
    ) source
    ON (target.ID = source.ID)

    WHEN MATCHED THEN UPDATE SET
        target.CLEANED_TEXT         = source.CLEANED_TEXT,
        target.CLEANED_TEXT_REDUCED = source.CLEANED_TEXT_REDUCED,
        target.ENTITIES             = source.ENTITIES,
        target.SKLEARN_TOP_KEYWORDS = source.SKLEARN_TOP_KEYWORDS,
        target.RAKE_OUT             = source.RAKE_OUT,
        target.TOPIC_KEYWORDS       = source.TOPIC_KEYWORDS,
        target.HERDS_FIELD_SKLEARN  = source.HERDS_FIELD_SKLEARN,
        target.UPDATE_TIMESTAMP     = source.UPDATE_TIMESTAMP

    WHEN NOT MATCHED THEN INSERT (
        ID, PROPOSAL_NUMBER, MODULE_NUMBER, UPDATE_TIMESTAMP,
        CLEANED_TEXT, CLEANED_TEXT_REDUCED, ENTITIES,
        SKLEARN_TOP_KEYWORDS, RAKE_OUT, TOPIC_KEYWORDS, HERDS_FIELD_SKLEARN
    )
    VALUES (
        source.ID, source.PROPOSAL_NUMBER, source.MODULE_NUMBER, source.UPDATE_TIMESTAMP,
        source.CLEANED_TEXT, source.CLEANED_TEXT_REDUCED, source.ENTITIES,
        source.SKLEARN_TOP_KEYWORDS, source.RAKE_OUT, source.TOPIC_KEYWORDS, source.HERDS_FIELD_SKLEARN
    )
    """

    data = [
        (
            row["ID"],
            row["PROPOSAL_NUMBER"],
            row["MODULE_NUMBER"],
            row["UPDATE_TIMESTAMP"],
            row["CLEANED_TEXT"],
            row["CLEANED_TEXT_REDUCED"],
            row["ENTITIES"],
            row["SKLEARN_TOP_KEYWORDS"],
            row["RAKE_OUT"],
            row["TOPIC_KEYWORDS"],
            row["HERDS_FIELD_SKLEARN"],
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany(sql, data)
    conn.commit()

    print(f"Upserted {len(data)} rows into NLP table")
