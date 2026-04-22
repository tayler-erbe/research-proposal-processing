# Proposal ingestion orchestrator.
#
# Reads new proposal narrative attachments out of Kuali, extracts text,
# merges Kuali metadata with PDF metadata, and writes one parquet file
# per run into the intermediate_tables folder. Downstream stages
# (nlp_pipeline, db writers) consume the parquet — this script doesn't
# touch the analytics DB directly.
#
# Each run is idempotent by design: STEP 1 anti-joins against the
# staging table to find only FILE_DATA_IDs that haven't been processed
# yet. If nothing is new, the function returns early without creating
# intermediate files — downstream stages interpret the missing parquet
# as "nothing to do" rather than as a failure.

import time
from pathlib import Path

from ingestion.oracle_connection    import oracle_conn
from ingestion.fetch_documents      import fetch_pdf_batch
from services.change_detection_service import get_new_document_ids
from utils.batching                 import create_batches
from configs.config_loader          import load_config
from services.text_extraction_service  import process_pdf_folder
from services.table_build_service   import build_text_dataframe, save_intermediate_table
from services.table_merge_service   import build_final_table

# 20 is a deliberate choice — large enough to amortize the connection
# round trip, small enough that a failed batch doesn't lose much work.
BATCH_SIZE   = 20

INTERMEDIATE = Path("storage/intermediate_tables")
TEMP_PDFS    = Path("storage/temp_pdfs")


def run_pipeline():
    print("Starting ingestion pipeline")
    pipeline_start = time.time()

    db    = load_config("database.yaml")
    creds = db["oracle"]["credentials"]
    kuali = db["oracle"]["kuali"]

    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    TEMP_PDFS.mkdir(parents=True, exist_ok=True)

    # STEP 1 — What's new since last run?
    # This is a two-query anti-join: all candidate IDs from Kuali,
    # minus every ID already in the staging table. Doing this in
    # Python rather than SQL avoids needing a db link between the
    # source and analytics environments.
    ids, kuali_metadata_df = get_new_document_ids()
    print(f"New documents detected: {len(ids)}")
    if not ids:
        # Graceful no-op rather than an error; the monitor treats this
        # as SUCCESS because it's the normal state between real changes.
        print("No new documents found. Pipeline complete.")
        return

    kuali_metadata_df.to_parquet(INTERMEDIATE / "kuali_metadata.parquet", index=False)

    # STEP 2 — Split IDs into Oracle-bind-safe batches.
    batches       = list(create_batches(ids, batch_size=BATCH_SIZE))
    total_batches = len(batches)
    print(f"Processing {len(ids)} documents in {total_batches} batch(es) of {BATCH_SIZE}")

    # STEP 3 — Pull BLOBs and write each to disk as a PDF.
    # One connection is held across all batches; opening/closing a
    # connection per batch was the original design and cost ~2 seconds
    # of overhead per batch against the university's Oracle infra.
    all_pdf_files = []
    with oracle_conn(kuali["host"], kuali["port"], kuali["service"],
                     creds["username"], creds["password"]) as conn:
        for batch_num, batch in enumerate(batches, 1):
            print(f"  Downloading batch {batch_num}/{total_batches} ({len(batch)} docs)...")
            pdf_files = fetch_pdf_batch(conn, batch)
            all_pdf_files.extend(pdf_files)
    print(f"Downloaded {len(all_pdf_files)} PDFs")

    # STEP 4 — OCR/text extraction. Invalid PDFs are skipped silently
    # (see process_pdf_folder for the magic-byte check).
    text_records, pdf_metadata_df = process_pdf_folder()
    pdf_metadata_df.to_parquet(INTERMEDIATE / "pdf_metadata.parquet", index=False)
    print(f"Extracted text from {len(text_records)} documents")

    # STEP 5 — Assemble the raw text table, score quality, flag parseable rows.
    df_text = build_text_dataframe(text_records)

    if df_text.empty:
        # This means every PDF downloaded this run failed extraction or
        # had a corrupt header. Rare but possible when Kuali stores
        # non-PDF bytes under a PDF filename.
        print("No valid text extracted from any document this run — pipeline complete.")
        print("(All downloaded PDFs had bad headers or failed text extraction.)")
        return

    save_intermediate_table(df_text)

    # STEP 6 — Left-merge Kuali metadata + PDF metadata + extracted text
    # on FILE_DATA_ID to produce the full table the NLP pipeline consumes.
    final_df = build_final_table()
    print(f"Final table: {final_df.shape[0]} rows, {final_df.shape[1]} columns")

    elapsed = round(time.time() - pipeline_start, 1)
    print(f"\nIngestion complete — {len(ids)} documents in {elapsed}s")

    # Every pipeline run prints rollback SQL. If something downstream
    # goes sideways and you need to undo this run, the commands below
    # are already formatted — paste them into an Oracle session and go.
    # Note: TRUNCATE on T_KC_PRPSL_ABST_KYWD is safe because it's a
    # derived aggregate, rebuilt in full on every run.
    print("\n" + "=" * 60)
    print("ROLLBACK GUIDE (if this run needs to be undone):")
    print("=" * 60)
    print("""
  DELETE FROM ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT
  WHERE UPDATE_TIMESTAMP >= TRUNC(SYSDATE);
  DELETE FROM ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP
  WHERE UPDATE_TIMESTAMP >= TRUNC(SYSDATE);
  TRUNCATE TABLE ANALYTICS.T_KC_PRPSL_ABST_KYWD;
""")
    print("=" * 60)
