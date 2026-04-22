# Awards counterpart to fetch_documents.py.
#
# Same mechanics — BLOB → PDF on disk — but the SQL is more involved
# because an "active award attachment" is a four-table join in Kuali:
#
#   AWARD_ATTACHMENT      — the attachment metadata
#   ATTACHMENT_FILE       — the join between attachment and file data
#   AWARD_ATTACHMENT_TYPE — used to filter out type code 3 (internal)
#   AWARD                 — used to filter to ACTIVE award sequences
#   FILE_DATA             — the BLOB itself
#
# Awards are versioned through the AWARD_SEQUENCE_STATUS column; only
# the ACTIVE version of any given award matters for downstream analysis.
# TYPE_CODE 3 is excluded because it represents internal-only documents
# (routing sheets and similar) that don't belong in the analytical corpus.

from pathlib import Path
from configs.config_loader import load_config

config         = load_config("config.yaml")
PDF_DIR        = Path(config["storage"]["pdf_dir"])
AWARDS_PDF_DIR = PDF_DIR.parent / "awards_temp_pdfs"
AWARDS_PDF_DIR.mkdir(parents=True, exist_ok=True)


def save_blob_as_pdf(blob_value, file_id, output_dir=None):
    """Write one award BLOB to {file_id}.pdf in the awards staging folder."""
    out_dir = Path(output_dir) if output_dir else AWARDS_PDF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data = blob_value.read() if hasattr(blob_value, "read") else blob_value
    path = out_dir / f"{file_id}.pdf"
    with open(path, "wb") as f:
        f.write(data)
    return path


def fetch_awards_pdf_batch(conn, ids, output_dir=None):
    """Fetch a batch of active award attachment BLOBs by FILE_DATA_ID."""
    if not ids:
        print("No IDs provided to fetch.")
        return []

    placeholders = ",".join([f":{i+1}" for i in range(len(ids))])
    sql = f"""
    SELECT
        af.FILE_DATA_ID,
        fd.DATA
    FROM KUALI.AWARD_ATTACHMENT aa
    INNER JOIN KUALI.ATTACHMENT_FILE      af ON aa.FILE_ID     = af.FILE_ID
    INNER JOIN KUALI.AWARD_ATTACHMENT_TYPE ty ON ty.TYPE_CODE   = aa.TYPE_CODE
    INNER JOIN KUALI.FILE_DATA            fd ON fd.ID           = af.FILE_DATA_ID
    INNER JOIN KUALI.AWARD                 a ON a.AWARD_ID      = aa.AWARD_ID
    WHERE ty.TYPE_CODE != '3'
      AND a.AWARD_SEQUENCE_STATUS = 'ACTIVE'
      AND af.FILE_DATA_ID IN ({placeholders})
    """

    saved_files = []
    with conn.cursor() as cur:
        print("\nExecuting awards BLOB fetch query...")
        print(f"IDs requested: {len(ids)}")
        cur.execute(sql, ids)

        for file_data_id, blob in cur:
            if blob is None:
                print(f"  Skipping {file_data_id} (no blob)")
                continue
            pdf_path = save_blob_as_pdf(blob, file_data_id, output_dir=output_dir)
            saved_files.append(pdf_path)
            print(f"  Saved PDF: {pdf_path}")

    print(f"\nDownloaded {len(saved_files)} award PDFs")
    return saved_files
