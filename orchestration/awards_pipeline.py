# Awards text-extraction orchestrator.
#
# Structurally this mirrors pipeline.py (proposals) but diverges
# deliberately in a few places:
#
#   - Pulls from AWARD_ATTACHMENT-family tables instead of
#     NARRATIVE_ATTACHMENT, which requires its own change-detection
#     and fetch helpers.
#   - Writes to its own sibling storage folders so a proposal run and
#     an awards run can be in flight simultaneously without colliding.
#   - Does NOT run NLP. Awards text is useful for search and
#     downstream LLM workflows, but the HERDS classification and
#     keyword extraction that proposals go through aren't meaningful
#     for award-level documents (which are contracts, modifications,
#     and notices rather than scientific narratives).

import time
from pathlib import Path

from ingestion.oracle_connection           import oracle_conn
from ingestion.awards_fetch_documents      import fetch_awards_pdf_batch
from services.awards_change_detection_service import get_new_awards_document_ids
from services.text_extraction_service      import process_pdf_folder
from services.awards_table_build_service   import (
    build_awards_text_dataframe,
    save_awards_intermediate_table,
)
from utils.batching           import create_batches
from configs.config_loader    import load_config

BATCH_SIZE            = 20
AWARDS_INTERMEDIATE   = Path("storage/awards_intermediate_tables")
AWARDS_TEMP_PDFS      = Path("storage/awards_temp_pdfs")
AWARDS_EXTRACTED_TEXT = Path("storage/awards_extracted_text")


def run_awards_pipeline():
    """
    Six-stage awards text extraction:

      1. Anti-join against the staging table to find new FILE_DATA_IDs
      2. Batch IDs for Oracle bind-variable limits
      3. Download BLOBs from Kuali and write each to disk as a PDF
      4. Extract text, write one .txt per document
      5. Build the typed DataFrame and save intermediate parquet
      6. Upload to the awards staging table via the db writer

    Nothing in this flow runs NLP. Text stays as extracted text.
    """
    print("=" * 60)
    print(" AWARDS INGESTION PIPELINE STARTING")
    print("=" * 60)
    pipeline_start = time.time()

    db    = load_config("database.yaml")
    creds = db["oracle"]["credentials"]
    kuali = db["oracle"]["kuali"]

    for d in (AWARDS_INTERMEDIATE, AWARDS_TEMP_PDFS, AWARDS_EXTRACTED_TEXT):
        d.mkdir(parents=True, exist_ok=True)

    # STEP 1 — Detect new FILE_DATA_IDs via anti-join against the awards
    # staging table. Same shape as the proposal pipeline's change detection
    # but joins a different set of source tables on the Kuali side.
    ids, awards_metadata_df = get_new_awards_document_ids()
    print(f"[1/6] New awards documents detected: {len(ids)}")

    if not ids:
        print("No new awards documents found. Awards pipeline complete.")
        return

    awards_metadata_df.to_parquet(
        AWARDS_INTERMEDIATE / "awards_kuali_metadata.parquet", index=False
    )
    print(f"      Metadata saved → awards_kuali_metadata.parquet")

    # STEP 2 — Chunk IDs.
    batches       = list(create_batches(ids, batch_size=BATCH_SIZE))
    total_batches = len(batches)
    print(f"[2/6] {len(ids)} IDs split into {total_batches} batch(es) of {BATCH_SIZE}")

    # STEP 3 — One persistent Kuali connection, batched BLOB pulls.
    all_pdf_files = []
    with oracle_conn(
        kuali["host"], kuali["port"], kuali["service"],
        creds["username"], creds["password"]
    ) as conn:
        for batch_num, batch in enumerate(batches, 1):
            print(f"  Downloading awards batch {batch_num}/{total_batches} ({len(batch)} docs)...")
            pdf_files = fetch_awards_pdf_batch(
                conn, batch,
                output_dir=AWARDS_TEMP_PDFS
            )
            all_pdf_files.extend(pdf_files)

    print(f"[3/6] Downloaded and converted {len(all_pdf_files)} award PDFs")

    # STEP 4 — Reuse the proposal pipeline's text extractor, just
    # pointed at the awards folders. The extractor's signature accepts
    # optional pdf_dir/text_dir arguments specifically to enable this.
    text_records, pdf_metadata_df = process_pdf_folder(
        pdf_dir=AWARDS_TEMP_PDFS,
        text_dir=AWARDS_EXTRACTED_TEXT,
    )
    pdf_metadata_df.to_parquet(
        AWARDS_INTERMEDIATE / "awards_pdf_metadata.parquet", index=False
    )
    print(f"[4/6] Extracted text from {len(text_records)} award documents")

    # STEP 5 — Shape the DataFrame to match the awards staging table
    # exactly: same column order, types coerced to Oracle-friendly values,
    # strings truncated to VARCHAR2 limits.
    df_awards = build_awards_text_dataframe(text_records, awards_metadata_df)
    save_awards_intermediate_table(df_awards, AWARDS_INTERMEDIATE)
    print(f"[5/6] Awards intermediate table saved → awards_full_table.parquet")

    # STEP 6 — Commit to the analytics DB. Import is kept local to avoid
    # any chance of import-time side effects when the writer is not used.
    from services.awards_db_writer_service import run_awards_db_writer
    run_awards_db_writer(AWARDS_INTERMEDIATE, dry_run=False)
    print(f"[6/6] Awards data uploaded to ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT_AWARDS")

    elapsed = round(time.time() - pipeline_start, 1)
    print(f"\nAwards ingestion complete — {len(ids)} documents in {elapsed}s")

    # See pipeline.py for the rationale behind printing rollback SQL
    # on every successful run.
    print("=" * 60)
    print("ROLLBACK GUIDE (if this run needs to be undone):")
    print("=" * 60)
    print("""
  DELETE FROM ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT_AWARDS
  WHERE UPDATE_TIMESTAMP >= TRUNC(SYSDATE);
""")
    print("=" * 60)
