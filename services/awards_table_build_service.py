# Awards DataFrame builder.
#
# Produces a DataFrame shaped exactly like the awards staging table
# ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT_AWARDS — same 25 columns, in
# the same order, with types coerced to values Oracle will accept.
# The DB writer downstream does a straight executemany against this
# parquet, so any mismatch here shows up as an ORA error at write time.
#
# Schema confirmation: column definitions were captured by running
# scripts/inspect_awards_table_schema.py against the live DB and
# reconciled against this OUTPUT_COLUMNS list.

import pandas as pd
from pathlib import Path


# Exact column order from T_FILE_DATA_BLOB2TEXT_AWARDS by col_id.
OUTPUT_COLUMNS = [
    "AWARD_ATTACHMENT_ID",    # FLOAT(126)
    "AWARD_ID",               # FLOAT(126)
    "AWARD_NUMBER",           # VARCHAR2(255)
    "SEQUENCE_NUMBER",        # FLOAT(126)
    "DESCRIPTION",            # VARCHAR2(255)
    "UPDATE_TIMESTAMP",       # TIMESTAMP(6)
    "UPDATE_USER",            # VARCHAR2(255)
    "LAST_UPDATE_TIMESTAMP",  # TIMESTAMP(6)
    "LAST_UPDATE_USER",       # VARCHAR2(255)
    "FILE_NAME",              # VARCHAR2(255)
    "CONTENT_TYPE",           # VARCHAR2(255)
    "FILE_DATA_ID",           # VARCHAR2(255)
    "TYPE_CODE",              # VARCHAR2(255)
    "ID",                     # VARCHAR2(255) — same value as FILE_DATA_ID in source data
    "LEAD_UNIT_NUMBER",       # VARCHAR2(255)
    "ACTIVITY_TYPE_CODE",     # FLOAT(126)
    "AWARD_TYPE_CODE",        # FLOAT(126)
    "AWARD_EFFECTIVE_DATE",   # TIMESTAMP(6)
    "AWARD_EXECUTION_DATE",   # TIMESTAMP(6)
    "BEGIN_DATE",             # TIMESTAMP(6)
    "SPONSOR_AWARD_NUMBER",   # VARCHAR2(255)
    "OBJ_ID",                 # VARCHAR2(255)
    "LENGTH",                 # NUMBER
    "SUCCESSFULLY_PARSED",    # VARCHAR2(255)
    "CONTENT",                # CLOB — extracted text
]

MIN_CHAR_THRESHOLD = 50

FLOAT_COLS = [
    "AWARD_ATTACHMENT_ID", "AWARD_ID", "SEQUENCE_NUMBER",
    "ACTIVITY_TYPE_CODE", "AWARD_TYPE_CODE",
]

TIMESTAMP_COLS = [
    "UPDATE_TIMESTAMP", "LAST_UPDATE_TIMESTAMP",
    "AWARD_EFFECTIVE_DATE", "AWARD_EXECUTION_DATE", "BEGIN_DATE",
]

VARCHAR_COLS = [
    "AWARD_NUMBER", "DESCRIPTION", "UPDATE_USER", "LAST_UPDATE_USER",
    "FILE_NAME", "CONTENT_TYPE", "FILE_DATA_ID", "TYPE_CODE", "ID",
    "LEAD_UNIT_NUMBER", "SPONSOR_AWARD_NUMBER", "OBJ_ID", "SUCCESSFULLY_PARSED",
]


def _safe_str(val, max_len: int = 255):
    """String coercion with VARCHAR2 truncation. Returns None for NaN
    so Oracle stores a real NULL rather than the string 'nan'."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s[:max_len] if s else None


def build_awards_text_dataframe(
    text_records: list,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge extracted text onto award metadata and coerce types.

    Parameters
    ----------
    text_records : list of dicts, each {"FILE_DATA_ID": str, "CONTENT": str}
    metadata_df  : DataFrame from awards_change_detection_service

    Returns
    -------
    pd.DataFrame shaped to OUTPUT_COLUMNS, ready for DB upload.
    """
    if not text_records:
        print("[awards_table_build] No text records — returning empty DataFrame")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    text_df = pd.DataFrame(text_records)
    text_df["FILE_DATA_ID"] = text_df["FILE_DATA_ID"].astype(str).str.strip()

    meta_df = metadata_df.copy()
    meta_df["FILE_DATA_ID"] = meta_df["FILE_DATA_ID"].astype(str).str.strip()

    # Text-driven join: every extracted text record gets a row, even
    # if metadata is somehow missing.
    df = text_df.merge(meta_df, on="FILE_DATA_ID", how="left")

    # CONTENT column — the extractor uses CONTENT directly, but there
    # are some older code paths that wrote EXTRACTED_TEXT instead. Handle
    # both for safety.
    if "CONTENT" not in df.columns and "EXTRACTED_TEXT" in df.columns:
        df["CONTENT"] = df["EXTRACTED_TEXT"]
    df["CONTENT"] = df["CONTENT"].fillna("").apply(lambda x: x.strip() or None)

    # LENGTH / SUCCESSFULLY_PARSED. Threshold is lower here than in
    # the proposal pipeline because award documents (notices,
    # modifications) are often legitimately short.
    df["LENGTH"] = df["CONTENT"].apply(lambda x: len(x) if x else 0)
    df["SUCCESSFULLY_PARSED"] = df["LENGTH"].apply(
        lambda n: "Y" if n >= MIN_CHAR_THRESHOLD else "N"
    )

    # ID column mirrors FILE_DATA_ID in the source data — confirmed
    # via schema inspection of the live table.
    if "ID" not in df.columns:
        df["ID"] = df["FILE_DATA_ID"]

    # Type coercions. Invalid values become NaN/NaT and land as NULL
    # in Oracle rather than failing the insert.
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in TIMESTAMP_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in VARCHAR_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_safe_str)

    # Ensure every expected column exists — makes the downstream
    # executemany predictable regardless of what the metadata df had.
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    result = df[OUTPUT_COLUMNS].copy()
    result = result.loc[:, ~result.columns.duplicated()]

    parsed  = (result["SUCCESSFULLY_PARSED"] == "Y").sum()
    skipped = (result["SUCCESSFULLY_PARSED"] == "N").sum()
    print(
        f"[awards_table_build] Built {len(result)} rows — "
        f"{parsed} parsed, {skipped} skipped (< {MIN_CHAR_THRESHOLD} chars)"
    )
    return result


def save_awards_intermediate_table(df: pd.DataFrame, intermediate_dir: Path):
    """Write the shaped DataFrame to awards_full_table.parquet."""
    intermediate_dir = Path(intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    out_path = intermediate_dir / "awards_full_table.parquet"

    # Normalize pandas NAs to Python None for parquet consistency.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].where(df[col].notna(), other=None)

    df.to_parquet(out_path, index=False)
    print(f"[awards_table_build] Saved → {out_path.name}")
    return out_path
