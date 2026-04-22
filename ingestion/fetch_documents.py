# Pulls proposal narrative PDFs out of Kuali's FILE_DATA table as BLOBs
# and writes them to the local PDF staging directory.
#
# The BLOB is read straight to disk without transformation — Kuali stores
# the original uploaded file bytes, so for PDFs there's nothing to convert.
# The FILE_DATA_ID (a GUID) is used as the filename so downstream stages
# can match extracted text back to the source record by filename alone.

import os
from utils.file_helpers import ensure_dir
from configs.config_loader import load_config

config  = load_config("config.yaml")
PDF_DIR = config["storage"]["pdf_dir"]


def save_blob_as_pdf(blob_value, file_id):
    """Write a single Oracle BLOB out as {file_id}.pdf."""
    ensure_dir(PDF_DIR)

    # oracledb returns LOBs as objects with a .read() method until you
    # actually read them, at which point you get bytes. Handle both so
    # the function works whether the caller has pre-read the LOB or not.
    data = blob_value.read() if hasattr(blob_value, "read") else blob_value

    path = os.path.join(PDF_DIR, f"{file_id}.pdf")
    with open(path, "wb") as f:
        f.write(data)
    return path


def fetch_pdf_batch(conn, ids):
    """Fetch a batch of BLOBs by ID and write each as a PDF."""
    if not ids:
        print("No IDs provided to fetch.")
        return []

    # Bind variables rather than string-formatting the IDs into the query.
    # Oracle caps bind variables at 1000 per statement, which is why the
    # caller batches ahead of this function.
    placeholders = ",".join([f":{i+1}" for i in range(len(ids))])
    sql = f"""
    SELECT
        a.ID,
        a.DATA
    FROM KUALI.FILE_DATA a
    WHERE a.ID IN ({placeholders})
    """

    saved_files = []
    with conn.cursor() as cur:
        print("\nExecuting BLOB fetch query...")
        print(f"IDs requested: {len(ids)}")
        cur.execute(sql, ids)

        for id_val, blob in cur:
            if blob is None:
                # Occasionally seen on old records where the attachment
                # row exists but the BLOB itself was never populated.
                print(f"Skipping {id_val} (no blob)")
                continue

            pdf_path = save_blob_as_pdf(blob, id_val)
            saved_files.append(pdf_path)
            print(f"Saved PDF: {pdf_path}")

    print(f"\nDownloaded {len(saved_files)} PDFs to {PDF_DIR}")
    return saved_files
