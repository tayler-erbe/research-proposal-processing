# Staging-table builder. One of the stranger modules in this pipeline
# because it exists to bridge three distinct audiences for the same
# underlying data:
#
#   staging_1_blob2text  — raw blob-to-text output for compliance and audit
#   staging_2_nlp        — NLP-annotated version with keywords and HERDS
#   staging_3_production — per-proposal aggregated view used by the
#                          production analytics table
#
# Each corresponds to a specific live Oracle table, and the column
# lists below are the source of truth for matching the DB schemas
# exactly. These schemas predate the pipeline; renaming columns was
# not an option because every downstream consumer is wired to them.
#
# Column mapping notes you'll need if you ever debug this:
#
#   KEYBERT_OUT            → RAKE_OUT             (DB keeps old name)
#   KEYBERT_OUT_REDUCED    → RAKE_OUT_REDUCED
#   UNIQUE_KEYBERT         → UNIQUE_RAKE
#   UNIQUE_KEYBERT_REDUCED → UNIQUE_RAKE_REDUCED
#   HERDS_FIELDS_SKLEARN   → HERDS_FIELD_SKLEARN  (DB column is singular)
#   HIDE_IN_HIERARCHY      → None                 (not populated by us)
#   DW_LOAD_DT             → date only            (DB column is DATE)

import pandas as pd
from pathlib import Path


# ─── Helpers ─────────────────────────────────────────────────────────

def _trunc(series, max_len):
    """Hard-truncate strings to max_len. None values stay None."""
    return series.apply(
        lambda x: x[:max_len] if isinstance(x, str) and len(x) > max_len else x
    )


def _safe_str(series, max_len=None):
    """Coerce a series to clean strings, optionally truncating."""
    result = series.where(series.notna(), other=None)
    result = result.apply(lambda x: str(x) if x is not None else None)
    if max_len:
        result = _trunc(result, max_len)
    return result


def _warn_truncations(df, constraints):
    """Print a one-line warning for each column where we had to
    truncate. Not a failure — truncations are expected — but useful
    signal if they start happening more often than normal."""
    for col, limit in constraints.items():
        if col in df.columns:
            too_long = df[col].dropna().apply(lambda x: len(str(x)) > limit).sum()
            if too_long > 0:
                print(f"  [WARN] {col}: {too_long} value(s) truncated to {limit} chars")


# ─── Staging 1: ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT ───────────────

def _build_staging1(full_df):
    """Raw blob-to-text shape. Keyword columns are not populated here —
    this table is for compliance/audit consumers who want the extracted
    text without the NLP annotations."""
    print("\n[STAGING 1] Building T_FILE_DATA_BLOB2TEXT...")

    df = full_df.copy()
    s = pd.DataFrame()

    s["ID"]                     = _safe_str(df["ID"], 255)
    s["PROPOSAL_NUMBER"]        = _safe_str(df["PROPOSAL_NUMBER"], 255)
    s["MODULE_NUMBER"]          = pd.to_numeric(df["MODULE_NUMBER"], errors="coerce")
    s["FILE_NAME"]              = _safe_str(df["FILE_NAME"], 255)
    s["CONTENT_TYPE"]           = _safe_str(df["CONTENT_TYPE"], 255)
    s["UPDATE_USER"]            = _safe_str(df["UPDATE_USER"], 255)
    s["UPDATE_TIMESTAMP"]       = pd.to_datetime(df["UPDATE_TIMESTAMP"], errors="coerce")
    s["OBJ_ID"]                 = _safe_str(df["OBJ_ID"], 255)
    s["FILE_DATA_ID"]           = _safe_str(df["FILE_DATA_ID"], 255)
    s["UPLOAD_TIMESTAMP"]       = pd.to_datetime(df["UPLOAD_TIMESTAMP"], errors="coerce")
    s["UPLOAD_USER"]            = _safe_str(df["UPLOAD_USER"], 255)
    s["HIDE_IN_HIERARCHY"]      = None
    s["MODULE_SEQUENCE_NUMBER"] = pd.to_numeric(df["MODULE_SEQUENCE_NUMBER"], errors="coerce")
    s["MODULE_TITLE"]           = _safe_str(df["MODULE_TITLE"], 255)
    s["MODULE_STATUS_CODE"]     = _safe_str(df["MODULE_STATUS_CODE"], 255)
    s["NARRATIVE_TYPE_CODE"]    = _safe_str(df["NARRATIVE_TYPE_CODE"], 255)
    s["MIME Type"]              = _safe_str(df["MIME Type"], 255)
    s["Title"]                  = _safe_str(df["Title"], 1000)
    s["AUTHOR"]                 = _safe_str(df["Author"], 255)       # source column is lowercase
    s["DATE_CREATED"]           = _safe_str(df["CreationDate"], 255) # rename from PDF metadata
    s["DATE_MODIFIED"]          = _safe_str(df["ModDate"], 255)      # rename from PDF metadata
    s["Format"]                 = _safe_str(df["Format"], 255)
    s["Creator Tool"]           = _safe_str(df["Creator Tool"], 255)
    s["Metadata Date"]          = None
    s["CONTENT"]                = df["CONTENT"].where(df["CONTENT"].notna(), other=None)
    s["LENGTH"]                 = pd.to_numeric(df.get("LENGTH"), errors="coerce")
    s["SUCCESSFULLY_PARSED"]    = _safe_str(df["SUCCESSFULLY_PARSED"], 255)

    print(f"  Rows: {len(s)}")
    return s


# ─── Staging 2: ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP ────────────

def _build_staging2(nlp_df):
    """NLP-annotated shape. This is where the RAKE→KeyBERT rename
    happens — we read from our KEYBERT_OUT columns and write into the
    DB's RAKE_OUT columns."""
    print("\n[STAGING 2] Building T_RSRCH_PRPSL_PRCSSD_NLP...")

    df = nlp_df.copy()
    s = pd.DataFrame()

    s["ID"]                           = _safe_str(df["ID"], 255)
    s["PROPOSAL_NUMBER"]              = _safe_str(df["PROPOSAL_NUMBER"], 255)
    s["MODULE_NUMBER"]                = pd.to_numeric(df["MODULE_NUMBER"], errors="coerce").astype("Int64")

    # CLOB columns — no width constraint.
    s["CLEANED_TEXT"]         = df["CLEANED_TEXT"].where(df["CLEANED_TEXT"].notna(), other=None)
    s["CLEANED_TEXT_REDUCED"] = _safe_str(df["CLEANED_TEXT_REDUCED"], 3000)

    s["TOPIC_KEYWORDS"]         = df["TOPIC_KEYWORDS"].where(df["TOPIC_KEYWORDS"].notna(), other=None)
    s["TOPIC_KEYWORDS_REDUCED"] = _safe_str(df["TOPIC_KEYWORDS_REDUCED"], 1000)

    # The KeyBERT → RAKE rename. KEYBERT_OUT on our side, RAKE_OUT in DB.
    s["RAKE_OUT"]            = df["KEYBERT_OUT"].where(df["KEYBERT_OUT"].notna(), other=None)
    s["RAKE_OUT_REDUCED"]    = _safe_str(df["KEYBERT_OUT_REDUCED"], 1000)
    s["UNIQUE_RAKE"]         = df["UNIQUE_KEYBERT"].where(df["UNIQUE_KEYBERT"].notna(), other=None)
    s["UNIQUE_RAKE_REDUCED"] = _safe_str(df["UNIQUE_KEYBERT_REDUCED"], 1000)

    s["SKLEARN_TOP_KEYWORDS"]         = df["SKLEARN_TOP_KEYWORDS"].where(df["SKLEARN_TOP_KEYWORDS"].notna(), other=None)
    s["SKLEARN_TOP_KEYWORDS_REDUCED"] = _safe_str(df["SKLEARN_TOP_KEYWORDS_REDUCED"], 1000)

    # HERDS_FIELD_SKLEARN — prefer the comma-joined top-3 format if
    # available (that's what the downstream LISTAGG aggregation expects);
    # fall back to the single-category format for legacy rows.
    if "HERDS_TOP3" in df.columns:
        s["HERDS_FIELD_SKLEARN"] = _safe_str(df["HERDS_TOP3"], 1000)
    else:
        s["HERDS_FIELD_SKLEARN"] = _safe_str(df["HERDS_FIELDS_SKLEARN"], 1000)

    # Entity columns — all VARCHAR2(1000) in DB, null by default here.
    s["ENTITIES"]      = _safe_str(df["ENTITIES"], 1000)
    s["ORGANIZATIONS"] = _safe_str(df["ORGANIZATIONS"], 1000)
    s["PERSONS"]       = _safe_str(df["PERSONS"], 1000)
    s["LOCATIONS"]     = _safe_str(df["LOCATIONS"], 1000)
    s["TIME"]          = _safe_str(df["TIME"], 1000)
    s["FAC"]           = _safe_str(df["FAC"], 1000)
    s["EVENT"]         = _safe_str(df["EVENT"], 1000)
    s["MONEY"]         = _safe_str(df["MONEY"], 1000)
    s["PRODUCT"]       = _safe_str(df["PRODUCT"], 1000)

    s["UPDATE_TIMESTAMP"] = pd.to_datetime(df["UPDATE_TIMESTAMP"], errors="coerce")

    _warn_truncations(s, {
        "CLEANED_TEXT_REDUCED":         3000,
        "TOPIC_KEYWORDS_REDUCED":       1000,
        "RAKE_OUT_REDUCED":             1000,
        "UNIQUE_RAKE_REDUCED":          1000,
        "SKLEARN_TOP_KEYWORDS_REDUCED": 1000,
        "HERDS_FIELD_SKLEARN":          1000,
        "ENTITIES":                     1000,
        "ORGANIZATIONS":                1000,
        "PERSONS":                      1000,
        "LOCATIONS":                    1000,
    })

    print(f"  Rows: {len(s)}")
    return s


# ─── Staging 3: ANALYTICS.T_KC_PRPSL_ABST_KYWD (production) ──────────

def _build_staging3(nlp_df):
    """Aggregated per-proposal production view.

    Replicates the original KNIME LISTAGG aggregation step:
      - Group by PROPOSAL_NUMBER
      - Concatenate keyword strings across modules of the same proposal
      - Map to the 5-column production schema

    Only rows with SUCCESSFULLY_PARSED = 'Y' are included. We don't
    want to pollute the production keyword table with NULL-keyword rows
    from documents that failed extraction."""
    print("\n[STAGING 3] Building ANALYTICS.T_KC_PRPSL_ABST_KYWD...")

    df = nlp_df[nlp_df["SUCCESSFULLY_PARSED"] == "Y"].copy()
    skipped = len(nlp_df) - len(df)
    if skipped > 0:
        print(f"  Skipping {skipped} rows with SUCCESSFULLY_PARSED != Y")

    def listagg(series):
        vals = series.dropna().astype(str).tolist()
        return ",".join(vals) if vals else None

    herds_col = "HERDS_TOP3" if "HERDS_TOP3" in df.columns else "HERDS_FIELDS_SKLEARN"
    df["_HERDS_FORMATTED"] = df[herds_col]

    agg = df.groupby("PROPOSAL_NUMBER", as_index=False).agg(
        HERDS_FIELD      = ("_HERDS_FORMATTED",              listagg),
        SKLEARN_KEYWORDS = ("SKLEARN_TOP_KEYWORDS_REDUCED",  listagg),
        KEYBERT_KEYWORDS = ("KEYBERT_OUT_REDUCED",           listagg),
        UPDATE_TIMESTAMP = ("UPDATE_TIMESTAMP",              "max"),
    )

    s = pd.DataFrame()

    # KC_PT_NBR is VARCHAR2(12) — truncate to avoid ORA errors on
    # proposal numbers that have been extended over the years.
    s["KC_PT_NBR"] = agg["PROPOSAL_NUMBER"].apply(
        lambda x: str(x)[:12] if pd.notna(x) else None
    )

    # ABST_KYWD_LIST: TF-IDF keywords (called "SKLEARN" for historical
    # reasons — the sklearn implementation predates the column name).
    # Max 1000 chars.
    s["ABST_KYWD_LIST"] = _trunc(
        agg["SKLEARN_KEYWORDS"].fillna("").astype(str),
        1000
    ).where(agg["SKLEARN_KEYWORDS"].notna(), other=None)

    # ABST_CATGRY_LIST: KeyBERT keyphrases. Max 500 chars. These are
    # the semantically-rich multi-word phrases — the category-like
    # signal, as opposed to the individual-token signal in ABST_KYWD_LIST.
    s["ABST_CATGRY_LIST"] = _trunc(
        agg["KEYBERT_KEYWORDS"].fillna("").astype(str),
        500
    ).where(agg["KEYBERT_KEYWORDS"].notna(), other=None)

    # DW_LOAD_DT is Oracle DATE, not TIMESTAMP — the column stores
    # date without time. pandas will store datetime but the Oracle
    # driver will coerce on insert.
    s["DW_LOAD_DT"] = agg["UPDATE_TIMESTAMP"]

    # HERD_CATGRY_LIST: top-3 HERDS categories, underscore-joined,
    # comma-separated across modules of the same proposal. Example:
    # "Psychology, Social_Work, Civil_Engineering"
    s["HERD_CATGRY_LIST"] = _trunc(
        agg["HERDS_FIELD"].fillna("").astype(str),
        500
    ).where(agg["HERDS_FIELD"].notna(), other=None)

    # Final gate — drop rows where all keyword fields are null. This
    # shouldn't happen given the SUCCESSFULLY_PARSED='Y' filter, but
    # the defensive check here has caught at least one bug historically.
    before = len(s)
    s = s[s["ABST_KYWD_LIST"].notna() | s["ABST_CATGRY_LIST"].notna()].copy()
    dropped = before - len(s)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with no keywords")

    _warn_truncations(s, {
        "ABST_KYWD_LIST":   1000,
        "ABST_CATGRY_LIST": 500,
        "HERD_CATGRY_LIST": 500,
    })

    print(f"  Rows: {len(s)} (parsed rows only, one per PROPOSAL_NUMBER)")
    return s


# ─── Public entry point ──────────────────────────────────────────────

def build_staging_tables(intermediate_dir: Path):
    """Build and save all three staging parquet files. Returns a dict
    of {name: DataFrame} for programmatic access by callers."""
    print("\n===================================")
    print(" BUILDING STAGING TABLES ")
    print("===================================")

    nlp_path  = intermediate_dir / "proposal_full_table_nlp.parquet"
    full_path = intermediate_dir / "proposal_full_table.parquet"

    print(f"\n[LOAD] Reading NLP parquet...")
    nlp_df = pd.read_parquet(nlp_path)
    print(f"  Rows: {len(nlp_df)}")

    print(f"[LOAD] Reading full table parquet...")
    full_df = pd.read_parquet(full_path)
    print(f"  Rows: {len(full_df)}")

    staging1 = _build_staging1(full_df)
    staging2 = _build_staging2(nlp_df)
    staging3 = _build_staging3(nlp_df)

    out1 = intermediate_dir / "staging_1_blob2text.parquet"
    out2 = intermediate_dir / "staging_2_nlp.parquet"
    out3 = intermediate_dir / "staging_3_production.parquet"

    staging1.to_parquet(out1, index=False)
    staging2.to_parquet(out2, index=False)
    staging3.to_parquet(out3, index=False)

    print(f"\n[SAVED] staging_1_blob2text.parquet   → {out1}")
    print(f"[SAVED] staging_2_nlp.parquet          → {out2}")
    print(f"[SAVED] staging_3_production.parquet   → {out3}")

    print("\n===================================")
    print(" STAGING TABLES COMPLETE ")
    print("===================================\n")

    return {
        "staging1": staging1,
        "staging2": staging2,
        "staging3": staging3,
    }
