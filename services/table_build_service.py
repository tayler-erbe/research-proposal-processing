# Proposal text table builder + quality flagging + document type heuristic.
#
# Called by pipeline.py after PDF text extraction. Adds LENGTH,
# SUCCESSFULLY_PARSED, and ALPHA_RATIO columns that downstream stages
# use to decide what to process and what to flag as corrupt.

from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "storage" / "intermediate_tables"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)


def assess_text_quality(text):
    """Return (SUCCESSFULLY_PARSED, ALPHA_RATIO) for one text string.

    The ALPHA_RATIO check catches PDFs whose text extracted but came
    out as mostly special characters or numbers — usually a sign of
    encoding issues or image-only PDFs where only the copyright footer
    extracted cleanly. Threshold of 0.6 is empirically around where
    documents start being genuinely useful vs. noise."""
    if not text or len(text) < 150:
        return "N", 0.0

    total_chars = len(text)
    alpha_chars = len(re.findall(r'[A-Za-z]', text))
    alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0

    if alpha_ratio < 0.6:
        return "N", alpha_ratio
    return "Y", alpha_ratio


def build_text_dataframe(text_records):
    """Wrap the list of {FILE_DATA_ID, CONTENT} dicts in a DataFrame
    and annotate with quality flags."""
    print("\nBuilding dataframe from extracted records...")

    if not text_records:
        print("No valid text records to build — returning empty DataFrame.")
        return pd.DataFrame(columns=["CONTENT", "LENGTH", "SUCCESSFULLY_PARSED", "ALPHA_RATIO"])

    df = pd.DataFrame(text_records)
    df["CONTENT"] = df["CONTENT"].fillna("")
    df["LENGTH"]  = df["CONTENT"].str.len()
    df["SUCCESSFULLY_PARSED"], df["ALPHA_RATIO"] = zip(
        *df["CONTENT"].apply(assess_text_quality)
    )
    return df


def save_intermediate_table(df):
    parquet_path = INTERMEDIATE_DIR / "proposal_text_extracted.parquet"
    print("\nSaving intermediate parquet table...")
    print(parquet_path)
    df.to_parquet(parquet_path, index=False)
    print("Parquet saved successfully")
    return parquet_path


def classify_document_type(module_title):
    """Coarse-grain bucket for what kind of document this is, based on
    the module title string Kuali attaches.

    TODO(voice): the "SPA" keyword in the BUDGET bucket matches the
    acronym for Sponsored Programs Administration. That's a false-
    positive risk — any document titled "SPA Review" or similar gets
    bucketed as BUDGET regardless of content. This is kept for now to
    match the production behavior downstream stages expect; if/when
    this is revisited, either replace "SPA" with a narrower token like
    "SPA BUDGET" or split the detection into SPA-specific and budget-
    specific logic."""
    if not module_title:
        return "UNKNOWN"

    title = module_title.upper()
    if any(x in title for x in ["BUDGET", "JUSTIFICATION", "SPA", "PAF"]):
        return "BUDGET"
    if any(x in title for x in ["LOI", "LETTER OF INTENT"]):
        return "LOI"
    if any(x in title for x in ["PROPOSAL", "FINAL", "SUBMITTED", "REVIEWED"]):
        return "PROPOSAL"
    return "OTHER"
