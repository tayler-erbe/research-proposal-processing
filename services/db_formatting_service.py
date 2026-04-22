# Proposal DataFrame → analytics-DB shape.
#
# Pairs with df_write_service.py. This module handles type coercion
# and column ordering; df_write_service does the actual SQL.
#
# The GUID normalizer here is the richer one — it handles unhyphenated
# 32-hex-char variants by reinserting the dashes, on top of the simple
# lowercase+strip behavior. Use this one whenever writing to Oracle;
# the lighter version in utils/data_handling.py is for ad-hoc reconciliation.

import pandas as pd
import numpy as np
import re


# Target column order and set for T_FILE_DATA_BLOB2TEXT.
TARGET_COLUMNS = [
    'ID','PROPOSAL_NUMBER','MODULE_NUMBER','FILE_NAME','CONTENT_TYPE',
    'UPDATE_USER','UPDATE_TIMESTAMP','OBJ_ID','FILE_DATA_ID',
    'UPLOAD_TIMESTAMP','UPLOAD_USER','HIDE_IN_HIERARCHY',
    'MODULE_SEQUENCE_NUMBER','MODULE_TITLE','MODULE_STATUS_CODE',
    'NARRATIVE_TYPE_CODE','MIME Type','Title','AUTHOR','DATE_CREATED',
    'DATE_MODIFIED','Format','Creator Tool','Metadata Date','CONTENT',
    'LENGTH','SUCCESSFULLY_PARSED'
]


def normalize_guid(x):
    """Normalize GUID formats into canonical 8-4-4-4-12 lowercase hex.
    Handles inputs with or without braces, uppercase or lowercase, and
    the unhyphenated 32-hex-char form."""
    if pd.isna(x):
        return None

    s = str(x).strip().strip("{}()").lower()
    hex_chars = re.sub(r"[^0-9a-f]", "", s)

    if len(hex_chars) == 32:
        s = f"{hex_chars[:8]}-{hex_chars[8:12]}-{hex_chars[12:16]}-{hex_chars[16:20]}-{hex_chars[20:32]}"

    return s


def clean_dataframe(df):
    """Coerce a DataFrame into a DB-shaped copy: correct column set,
    GUIDs normalized, numerics as numerics, timestamps as timezone-naive
    UTC, strings trimmed and empty-string-to-None."""
    df = df.copy()

    # Ensure every target column is present (as None if new).
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[TARGET_COLUMNS]

    # GUIDs.
    for col in ["ID", "OBJ_ID", "FILE_DATA_ID"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_guid)

    # Numerics.
    numeric_cols = ["MODULE_NUMBER", "MODULE_SEQUENCE_NUMBER", "LENGTH"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamps — normalize to UTC-naive. Oracle stores without TZ
    # information, so we strip it here rather than at bind time.
    ts_cols = ["UPDATE_TIMESTAMP", "UPLOAD_TIMESTAMP"]
    for col in ts_cols:
        ts = pd.to_datetime(df[col], errors="coerce", utc=True)
        df[col] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    # Strings — trim whitespace, empty-string-to-None.
    for col in df.columns:
        if col not in numeric_cols + ts_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("", None)

    # CONTENT_TYPE defaults to application/pdf if missing. Every PDF
    # we process is a PDF, so this is a safe default vs. storing nulls.
    if "CONTENT_TYPE" in df.columns:
        df["CONTENT_TYPE"] = df["CONTENT_TYPE"].fillna("application/pdf")

    return df
