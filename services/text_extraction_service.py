# PDF text extraction. Called by both the proposal and awards pipelines.
#
# The default pdf_dir/text_dir arguments point at the proposal folders;
# the awards pipeline passes in its own folders via the kwargs. This
# kept one function as the source of truth for extraction while letting
# two pipelines share it without a fork.
#
# is_valid_pdf() exists because Kuali occasionally stores non-PDF bytes
# under a PDF filename — usually DOCX documents that were renamed
# rather than exported. Without the magic-byte check up front, pypdf
# spends seconds trying and failing to parse these and sometimes crashes
# hard enough to take down the whole run.

from pathlib import Path
from pypdf import PdfReader
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Defaults — used by the proposal pipeline.
PDF_DIR  = PROJECT_ROOT / "storage" / "temp_pdfs"
TEXT_DIR = PROJECT_ROOT / "storage" / "extracted_text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_pdf(path):
    """Magic-byte check — real PDFs start with the ASCII bytes '%PDF'."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def extract_text_from_pdf(pdf_path):
    """Return best-effort extracted text for a single PDF.

    Encrypted PDFs with empty passwords (surprisingly common — it's a
    lazy "encrypted" marker rather than real DRM) are decrypted on the
    fly. Real encrypted PDFs return empty strings rather than throwing,
    because a single bad doc shouldn't stop the batch."""
    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return ""
        text_chunks = []
        for page in reader.pages:
            try:
                text_chunks.append(page.extract_text() or "")
            except Exception:
                # Per-page failures — PDFs can have a corrupted page
                # in an otherwise readable document. Keep the good pages.
                text_chunks.append("")
        return "\n".join(text_chunks)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def extract_pdf_metadata(reader):
    """Pull the standard XMP/DocInfo fields. Returns None-valued keys
    for any field the document doesn't set, which is most of them for
    scanned-to-PDF documents."""
    meta = reader.metadata or {}
    return {
        "Title":        meta.get("/Title"),
        "Author":       meta.get("/Author"),
        "Creator Tool": meta.get("/Creator"),
        "Producer":     meta.get("/Producer"),
        "CreationDate": meta.get("/CreationDate"),
        "ModDate":      meta.get("/ModDate"),
    }


def process_pdf_folder(pdf_dir=None, text_dir=None):
    """Extract text from every .pdf in pdf_dir.

    Returns (text_records, pdf_metadata_df). Text records are dicts
    shaped {"FILE_DATA_ID": <guid>, "CONTENT": <text>}. Invalid PDFs
    are counted and skipped rather than processed."""
    _pdf_dir  = Path(pdf_dir)  if pdf_dir  else PDF_DIR
    _text_dir = Path(text_dir) if text_dir else TEXT_DIR
    _text_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(_pdf_dir.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDFs for extraction in {_pdf_dir}")

    text_records  = []
    metadata_rows = []
    skipped       = 0

    for pdf_path in pdf_files:
        file_data_id = pdf_path.stem

        if not is_valid_pdf(pdf_path):
            print(f"  [SKIP] Not a valid PDF (bad header): {pdf_path.name}")
            skipped += 1
            continue

        try:
            reader                    = PdfReader(str(pdf_path))
            metadata                  = extract_pdf_metadata(reader)
            metadata["FILE_DATA_ID"]  = file_data_id
            metadata["MIME Type"]     = "application/pdf"
            metadata["Format"]        = "application/pdf"
            metadata_rows.append(metadata)

            text = extract_text_from_pdf(pdf_path)
            text_records.append({"FILE_DATA_ID": file_data_id, "CONTENT": text})

            text_file = _text_dir / f"{file_data_id}.txt"
            with open(text_file, "w", encoding="utf-8", errors="ignore") as f:
                f.write(text)

        except Exception as e:
            print(f"  Failed processing {pdf_path}: {e}")

    if skipped:
        print(f"  Skipped {skipped} invalid files (not PDF)")

    pdf_metadata_df = pd.DataFrame(metadata_rows)
    return text_records, pdf_metadata_df
