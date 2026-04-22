#!/usr/bin/env python3
# Full pipeline entry point — .py companion to notebooks/main.ipynb.
#
# Runs proposal ingestion, NLP, DB write, and awards ingestion in
# sequence, each wrapped in the PipelineLogger so a separate job
# execution monitor can read the log parquet and surface failures.
#
# Prerequisites
# -------------
#   - configs/database.yaml populated (see database.yaml.example)
#   - data/real_word_vocab.pkl built via: python scripts/build_vocab_cache.py
#   - data/models/all-MiniLM-L6-v2/ present
#   - NLTK corpora: punkt, punkt_tab, wordnet, words, brown, stopwords
#
# Usage
# -----
#   python main.py
#
# The pipeline is idempotent. Re-running does no harm if nothing
# has changed upstream — each stage checks for its own "new work"
# condition and skips cleanly with a logged SUCCESS + skip_reason.

import sys
import os
import shutil
import pandas as pd
from pathlib import Path


# ── Setup ────────────────────────────────────────────────────────────

PIPELINE_ROOT = Path(__file__).resolve().parent
os.chdir(PIPELINE_ROOT)
sys.path.insert(0, str(PIPELINE_ROOT))

import nltk
for pkg in ["punkt", "punkt_tab", "wordnet", "words", "brown", "stopwords"]:
    nltk.download(pkg, quiet=True)

INTERMEDIATE_DIR        = Path("storage/intermediate_tables")
AWARDS_INTERMEDIATE_DIR = Path("storage/awards_intermediate_tables")
LOG_DIR                 = Path("storage/logs")

LOG_DIR.mkdir(parents=True, exist_ok=True)

PATHS_TO_CLEAR = [
    Path("storage/temp_pdfs"),
    INTERMEDIATE_DIR,
    Path("storage/extracted_text"),
    Path("storage/awards_temp_pdfs"),
    AWARDS_INTERMEDIATE_DIR,
    Path("storage/awards_extracted_text"),
]
for path in PATHS_TO_CLEAR:
    if path.exists():
        for item in path.iterdir():
            item.unlink() if item.is_file() else shutil.rmtree(item)

Path("storage/awards_temp_pdfs").mkdir(parents=True, exist_ok=True)
AWARDS_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
Path("storage/awards_extracted_text").mkdir(parents=True, exist_ok=True)


# ── Imports ──────────────────────────────────────────────────────────

from orchestration.pipeline                    import run_pipeline
from orchestration.nlp_pipeline                import run_nlp_pipeline
from services.db_writer_service                import run_db_writer
from services.awards_change_detection_service  import get_new_awards_document_ids
from services.awards_table_build_service       import (
    build_awards_text_dataframe,
    save_awards_intermediate_table,
)
from services.awards_db_writer_service         import run_awards_db_writer
from services.text_extraction_service          import process_pdf_folder
from ingestion.oracle_connection               import oracle_conn
from ingestion.awards_fetch_documents          import fetch_awards_pdf_batch
from utils.batching                            import create_batches
from utils.pipeline_logger                     import PipelineLogger
from configs.config_loader                     import load_config

AWARDS_TEMP_PDFS      = Path("storage/awards_temp_pdfs")
AWARDS_EXTRACTED_TEXT = Path("storage/awards_extracted_text")
AWARDS_INTERMEDIATE   = Path("storage/awards_intermediate_tables")


def main():
    db     = load_config("database.yaml")
    creds  = db["oracle"]["credentials"]
    kuali  = db["oracle"]["kuali"]
    dsstag = db["oracle"]["dsstag"]

    logger = PipelineLogger(
        workflow_name    = "proposal_processing",
        log_dir          = LOG_DIR,
        pipeline_version = "2026-04-17",
    )

    with logger.pipeline_run():

        # ── STAGE 1: Proposal ingestion ──────────────────────────────
        print("=" * 60)
        print("STAGE 1: PROPOSAL INGESTION")
        print("=" * 60)

        with logger.stage("ingestion", order=1):
            run_pipeline()

        proposal_parquet  = INTERMEDIATE_DIR / "proposal_full_table.parquet"
        has_new_proposals = proposal_parquet.exists()

        if has_new_proposals:
            _df = pd.read_parquet(proposal_parquet)
            print(f"\n[HEALTH] proposal_full_table.parquet — {len(_df)} rows, {len(_df.columns)} cols")
            logger.set_stage_metadata(input_count=len(_df), output_count=len(_df))
        else:
            print("\n[HEALTH] No new proposals this run — skipping NLP and DB write")

        # ── STAGE 2: Proposal NLP ────────────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 2: PROPOSAL NLP")
        print("=" * 60)

        if has_new_proposals:
            with logger.stage("nlp", order=2):
                run_nlp_pipeline()
            nlp_parquet = INTERMEDIATE_DIR / "proposal_full_table_nlp.parquet"
            if nlp_parquet.exists():
                _df2 = pd.read_parquet(nlp_parquet)
                print(f"[HEALTH] proposal_full_table_nlp.parquet — {len(_df2)} rows")
                logger.set_stage_metadata(output_count=len(_df2))
        else:
            with logger.stage("nlp", order=2):
                logger.set_stage_metadata(skip_reason="no_new_proposals")

        # ── STAGE 3: Proposal DB write ───────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 3: PROPOSAL DB WRITE")
        print("=" * 60)

        if has_new_proposals and (INTERMEDIATE_DIR / "proposal_full_table_nlp.parquet").exists():
            with logger.stage("db_write", order=3):
                run_db_writer(INTERMEDIATE_DIR, dry_run=False, confirm_production=True)
            print("\n[HEALTH] Proposal DB write complete")
        else:
            with logger.stage("db_write", order=3):
                logger.set_stage_metadata(skip_reason="no_new_proposals")

        # ── STAGE 4: Awards ingestion ────────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 4: AWARDS INGESTION")
        print("=" * 60)

        with logger.stage("awards_ingestion", order=4):
            print("\n[HEALTH] Step 1/6 — Detecting new award FILE_DATA_IDs...")
            ids, awards_metadata_df = get_new_awards_document_ids()
            print(f"[HEALTH] Total new unprocessed award IDs found: {len(ids)}")

            if not ids:
                print("[HEALTH] No new awards to process — skipping awards pipeline")
                logger.set_stage_metadata(
                    skip_reason="no_new_awards",
                    input_count=0, output_count=0,
                )
                return

            awards_metadata_df.to_parquet(
                AWARDS_INTERMEDIATE / "awards_kuali_metadata.parquet", index=False
            )
            print(f"[HEALTH] Metadata saved — {len(awards_metadata_df)} rows")

            BATCH_SIZE = 20
            batches    = list(create_batches(ids, batch_size=BATCH_SIZE))
            print(f"\n[HEALTH] Step 2/6 — Batching: {len(ids)} IDs to {len(batches)} batch(es)")

            print(f"\n[HEALTH] Step 3/6 — Downloading BLOBs from Kuali...")
            all_pdf_files = []
            with oracle_conn(
                kuali["host"], kuali["port"], kuali["service"],
                creds["username"], creds["password"]
            ) as conn:
                for batch_num, batch in enumerate(batches, 1):
                    print(f"  Downloading batch {batch_num}/{len(batches)} ({len(batch)} docs)...")
                    pdf_files = fetch_awards_pdf_batch(
                        conn, batch, output_dir=AWARDS_TEMP_PDFS
                    )
                    all_pdf_files.extend(pdf_files)

            print(f"[HEALTH] PDFs saved: {len(all_pdf_files)} / {len(ids)} files")
            if len(all_pdf_files) != len(ids):
                print(f"[HEALTH] Mismatch: {len(ids) - len(all_pdf_files)} BLOBs were null or unconvertible")

            print(f"\n[HEALTH] Step 4/6 — Extracting text from PDFs...")
            text_records, pdf_metadata_df = process_pdf_folder(
                pdf_dir  = AWARDS_TEMP_PDFS,
                text_dir = AWARDS_EXTRACTED_TEXT,
            )
            pdf_metadata_df.to_parquet(
                AWARDS_INTERMEDIATE / "awards_pdf_metadata.parquet", index=False
            )
            print(f"[HEALTH] Text extracted: {len(text_records)} records")

            print(f"\n[HEALTH] Step 5/6 — Building awards DataFrame...")
            df_awards = build_awards_text_dataframe(text_records, awards_metadata_df)
            print(f"[HEALTH] DataFrame shape: {df_awards.shape}")
            print(f"[HEALTH]   SUCCESSFULLY_PARSED: {df_awards['SUCCESSFULLY_PARSED'].value_counts(dropna=False).to_dict()}")
            print(f"[HEALTH]   LENGTH stats: min={df_awards['LENGTH'].min()}, max={df_awards['LENGTH'].max()}, mean={df_awards['LENGTH'].mean():.0f}")

            save_awards_intermediate_table(df_awards, AWARDS_INTERMEDIATE)

            # Rollback SQL — always printed before writing so the
            # remediation path is already formatted if this run needs to
            # be reversed.
            processed_file_data_ids = df_awards["FILE_DATA_ID"].dropna().tolist()
            id_list = "', '".join(processed_file_data_ids)
            print(f"\n[ROLLBACK] {len(processed_file_data_ids)} rows about to be written. To undo:")
            print(f"""
  DELETE FROM ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT_AWARDS
  WHERE FILE_DATA_ID IN ('{id_list}');
  COMMIT;
""")

            print(f"[HEALTH] Step 6/6 — Writing to DB...")
            run_awards_db_writer(AWARDS_INTERMEDIATE, dry_run=False)

            # Post-write verification — the last-chance sanity check
            # that the batch inserts succeeded as expected.
            print(f"\n[HEALTH] Verifying rows landed in DB...")
            id_bind = "', '".join(processed_file_data_ids)
            verify_sql = f"""
                SELECT COUNT(*) AS CONFIRMED,
                       COUNT(CASE WHEN SUCCESSFULLY_PARSED = 'Y' THEN 1 END) AS PARSED_Y,
                       COUNT(CASE WHEN SUCCESSFULLY_PARSED = 'N' THEN 1 END) AS PARSED_N
                FROM ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT_AWARDS
                WHERE FILE_DATA_ID IN ('{id_bind}')
            """
            with oracle_conn(
                dsstag["host"], dsstag["port"], dsstag["service"],
                creds["username"], creds["password"]
            ) as conn:
                verify_df = pd.read_sql(verify_sql, conn)

            confirmed = verify_df["CONFIRMED"].iloc[0]
            print(f"[HEALTH] Rows confirmed in DB: {confirmed} / {len(processed_file_data_ids)}")
            print(f"[HEALTH]   Parsed Y: {verify_df['PARSED_Y'].iloc[0]}  |  Parsed N: {verify_df['PARSED_N'].iloc[0]}")

            logger.set_stage_metadata(
                input_count  = len(ids),
                output_count = len(text_records),
                rows_written = int(confirmed),
            )

            if confirmed < len(processed_file_data_ids):
                raise RuntimeError(
                    f"Awards DB write incomplete: only {confirmed}/{len(processed_file_data_ids)} "
                    f"rows confirmed in DB. Check batch logs above."
                )

    print("\n" + "=" * 60)
    print("ALL PIPELINES COMPLETE")
    print("=" * 60)
    print(f'Logs at: {LOG_DIR / "pipeline_log.parquet"}')


if __name__ == "__main__":
    main()
