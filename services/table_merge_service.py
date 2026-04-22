# Final-table builder for the proposal pipeline.
#
# Reads the three intermediate artifacts — extracted text files, Kuali
# metadata parquet, PDF metadata parquet — and left-joins them into a
# single wide DataFrame keyed by FILE_DATA_ID. That wide DataFrame is
# the input to nlp_pipeline.py.

from pathlib import Path
import pandas as pd
from services.table_build_service import assess_text_quality

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
TEXT_DIR         = PROJECT_ROOT / "storage" / "extracted_text"
INTERMEDIATE_DIR = PROJECT_ROOT / "storage" / "intermediate_tables"

KUALI_META_FILE  = INTERMEDIATE_DIR / "kuali_metadata.parquet"
PDF_META_FILE    = INTERMEDIATE_DIR / "pdf_metadata.parquet"
FINAL_TABLE_FILE = INTERMEDIATE_DIR / "proposal_full_table.parquet"


def load_text_files():
    """Glob the extracted .txt files into a DataFrame. Each txt's
    filename (without extension) is its FILE_DATA_ID.

    Note the errors='ignore' on the file read — some PDFs extract to
    text that's mostly UTF-8 but has a few bad bytes sprinkled in from
    binary leakage. Ignoring those is preferable to failing the whole
    load."""
    rows = []
    for txt_file in TEXT_DIR.glob("*.txt"):
        file_data_id = txt_file.stem
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        rows.append({"FILE_DATA_ID": file_data_id, "CONTENT": content})

    df = pd.DataFrame(rows)

    # Re-run quality assessment here. The text files on disk are the
    # source of truth for CONTENT going into NLP — re-scoring means
    # we're flagging whatever actually made it through extraction,
    # not whatever was in a stale intermediate step.
    df["SUCCESSFULLY_PARSED"], df["ALPHA_RATIO"] = zip(
        *df["CONTENT"].apply(assess_text_quality)
    )

    return df


def normalize_columns(df):
    """Rename PDF metadata columns to match the DB schema conventions
    and ensure columns that the DB has but the pipeline doesn't
    populate exist as nulls."""
    rename_map = {
        "Author":       "AUTHOR",
        "CreationDate": "DATE_CREATED",
        "ModDate":      "DATE_MODIFIED",
    }
    df = df.rename(columns=rename_map)

    if "HIDE_IN_HIERARCHY" not in df.columns:
        df["HIDE_IN_HIERARCHY"] = None
    if "Metadata Date" not in df.columns:
        df["Metadata Date"] = None

    return df


def build_final_table():
    print("\nLoading text data...")
    text_df = load_text_files()

    print("Loading Kuali metadata...")
    kuali_df = pd.read_parquet(KUALI_META_FILE)

    print("Loading PDF metadata...")
    pdf_meta_df = pd.read_parquet(PDF_META_FILE)

    # Left-join everything on FILE_DATA_ID. Text is the driver — we
    # keep every row that made it through text extraction, even if a
    # Kuali metadata row is missing (rare but possible).
    print("\nMerging tables...")
    merged = text_df.merge(pdf_meta_df, how="left", on="FILE_DATA_ID")
    merged = merged.merge(kuali_df,     how="left", on="FILE_DATA_ID")
    print("Rows in merged table:", len(merged))

    # Document-type bucketing — see table_build_service.classify_document_type
    # for the heuristic and its known limitations.
    from services.table_build_service import classify_document_type

    if "MODULE_TITLE" not in merged.columns:
        raise ValueError("MODULE_TITLE missing — cannot classify document type")

    merged["DOCUMENT_TYPE"] = merged["MODULE_TITLE"].apply(classify_document_type)

    print("\nDOCUMENT_TYPE distribution:")
    print(merged["DOCUMENT_TYPE"].value_counts(dropna=False))

    print("\nMODULE_TITLE sample:")
    print(merged["MODULE_TITLE"].value_counts(dropna=False).head(10))

    merged.to_parquet(FINAL_TABLE_FILE, index=False)
    print("\nSaved final intermediate table:")
    print(FINAL_TABLE_FILE)

    return merged
