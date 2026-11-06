# Agent handoff: proposal → outcome data pull

> **Current task:** [`docs/handoffs/codex-sample-pull.md`](docs/handoffs/codex-sample-pull.md):
> pull representative samples and profiles from Kuali + the EDW, build a data
> contract and mapping layer, and generate realistic synthetic data. Read that
> first. Everything below is the full-pull reference plan, and its rules
> (read-only, no data or credentials in git) apply to all work in this repo.

This file briefs a coding agent (Codex or similar) running **on the work
laptop**, which can reach the Kuali Research Oracle DB. It is the full
context for the task; read it all before running anything.

---

## 1. Goal

Build a **complete, correctly joined, document-level dataset** of research
proposals and awards from Kuali, so we can later study what distinguishes
funded from unfunded proposals.

Concretely, produce local parquet files that:

1. **Pull every relevant file**, not just the subset the current pipeline
   takes (see §4 for what it currently misses).
2. **Union** proposal-side and award-side documents into one document table
   with a common schema.
3. **Join** each document to its proposal / award context: development
   proposal → institutional proposal → award, plus sponsor, unit
   (department/college), PI, requested and awarded amounts, dates, and
   outcome status.
4. **Prove completeness** through reconciliation experiments (§6): every
   count in the output is explained against a count in the source.

This is a **research extraction**, not a production change. Nothing gets
written back to any database.

---

## 2. Hard rules

- **Read-only.** SELECT only. No INSERT/UPDATE/MERGE/DELETE/TRUNCATE/DDL on
  any database, ever.
- **Do not run** `main.py`, `notebooks/main.ipynb`, `orchestration/*.py`,
  `services/db_writer_service.py`, or `services/awards_db_writer_service.py`.
  They clear local working folders and **write to production analytics
  tables**. Import helper functions from them if useful; don't execute them.
- **No data in git.** All outputs go under `storage/` (gitignored). Never
  commit `configs/database.yaml`, parquet, PDFs, extracted text, or reports
  that contain record-level values. Commit **code and docs only**.
- **Sensitive data.** Kuali holds personnel, budget and unpublished research
  content. Keep it on this machine. Logs and summaries you print should be
  counts and structure, not record contents, unless the user asks to see
  specific records.
- **Be gentle with the DB.** It's a shared production system. Batch BLOB
  reads (the existing code uses 20 IDs per batch), use bind variables, and
  avoid full scans in tight loops. Time out and checkpoint long pulls so a
  rerun resumes instead of starting over.
- Work on branch `claude/happy-clarke-plqt0c` (or a branch off it). Commit
  code as you go.

---

## 3. Environment

- Python 3.10+. Install with `pip install -r requirements.txt` (the heavy NLP
  packages aren't needed for this task; `oracledb`, `pandas`, `pyarrow`,
  `pyyaml`, `pypdf` are the essentials).
- DB credentials: `configs/database.yaml` (see `configs/database.yaml.example`).
  Two connections:
  - `kuali`: source system. **This task should only need this one.**
  - `dsstag`: analytics/staging DB. Read it only to compare against what the
    existing pipeline already processed (§6, experiment E7).
- Connect with `ingestion/oracle_connection.py:oracle_conn(...)`. It tries
  Oracle thick mode with a **hardcoded Linux path**
  (`/usr/lib/oracle/19.6/client64/lib`); on Windows or macOS this fails
  silently and falls back to thin mode. If thin mode can't connect (e.g.
  `DPY-3001` native network encryption error), find the local Instant Client
  and pass its path to `oracledb.init_oracle_client(lib_dir=...)` in your
  own script. Don't edit the shared module just for this.
- Schema owner is assumed to be `KUALI` (that's what all existing queries
  use). Confirm in step 1.

---

## 4. What exists today (read these files)

| File | What it tells you |
|---|---|
| `services/change_detection_service.py` | Proposal document query: `FILE_DATA ⋈ NARRATIVE_ATTACHMENT ⋈ NARRATIVE`, **only `NARRATIVE_TYPE_CODE = '535'`** |
| `ingestion/fetch_documents.py` | BLOB fetch: `SELECT ID, DATA FROM KUALI.FILE_DATA WHERE ID IN (...)`, written to `{FILE_DATA_ID}.pdf` |
| `services/awards_change_detection_service.py` | Award document query: `AWARD_ATTACHMENT ⋈ ATTACHMENT_FILE ⋈ AWARD_ATTACHMENT_TYPE ⋈ FILE_DATA ⋈ AWARD`, `TYPE_CODE != '3'`, `AWARD_SEQUENCE_STATUS = 'ACTIVE'` |
| `ingestion/awards_fetch_documents.py` | Award BLOB fetch (same joins) |
| `services/text_extraction_service.py` | pypdf extraction; **non-PDF bytes are skipped** via a `%PDF` magic-byte check |
| `services/table_build_service.py` | Quality flag: parsed only if text is ≥150 chars and ≥60% letters |
| `services/table_merge_service.py` | Left-joins text + PDF metadata + Kuali metadata on `FILE_DATA_ID` |
| `scripts/discover_kuali_schema.py` | **Run this first.** Read-only schema discovery + COUNT-only join funnel. Written without DB access, so not yet validated; fix it as needed |
| `docs/architecture.md`, `docs/roadmap.md` | Background and intended direction |

### Known gaps in the current pull (why "all the files" isn't happening yet)

1. **Proposal narratives limited to type `535`.** Other narrative types
   (abstract, project summary, budget justification, facilities, data
   management plan, biosketch, etc.) are excluded.
2. **Non-PDF files are dropped.** DOCX/DOC/RTF/TXT attachments fail the
   `%PDF` check and are silently skipped. Many proposal narratives are
   uploaded as Word files.
3. **Award attachments only from the ACTIVE sequence.** Attachments tied
   only to older award versions are excluded, and the same file can appear
   across versions (dedupe by `FILE_DATA_ID`).
4. **Other attachment tables are never queried.** Candidates in standard
   Kuali Coeus (confirm names locally): personnel attachments (e.g.
   `EPS_PROP_PERSON_BIO` / `EPS_PROP_PERSON_BIO_ATTACHMENT`), institutional
   proposal attachments, subaward attachments, negotiation attachments,
   budget subaward attachments (`BUDGET_SUB_AWARDS`/`..._ATT`).
5. **Possible legacy inline BLOBs.** Older Kuali versions stored file bytes
   directly on attachment rows (e.g. a `NARRATIVE_DATA` / `DATA` BLOB column)
   before the move to `FILE_DATA`. Rows with a NULL or dangling
   `FILE_DATA_ID` may still have content inline. Check for this.
6. **No outcome linkage.** Proposals and awards are pulled separately and
   never connected; sponsor/unit/PI/amounts are never pulled.
7. Minor: in the proposal query, `WHERE c.NARRATIVE_TYPE_CODE = '535'` turns
   the `LEFT JOIN NARRATIVE` into an inner join.

---

## 5. Plan

### Step 1: Discover the schema (no assumptions)

Run `python scripts/discover_kuali_schema.py` and read
`storage/discovery/kuali_schema_report.md`. Fix the script wherever local
names differ. Then, also read-only:

- Find **every column that references `FILE_DATA`**: search
  `all_tab_columns` for `FILE_DATA_ID`, `FILE_ID`, and any `BLOB` column in
  the KUALI schema. Each one is a potential file source. List them all.
- Confirm the join path below against real data. Expected standard Kuali
  Coeus path (verify; don't trust it):

```
EPS_PROPOSAL.PROPOSAL_NUMBER                       (development proposal; what NARRATIVE.PROPOSAL_NUMBER points to)
  └─ PROPOSAL_ADMIN_DETAILS.DEV_PROPOSAL_NUMBER → INST_PROPOSAL_ID
       └─ PROPOSAL.PROPOSAL_ID                      (institutional proposal, versioned by SEQUENCE_NUMBER;
                                                    use PROPOSAL_SEQUENCE_STATUS = 'ACTIVE' for current)
            └─ AWARD_FUNDING_PROPOSALS.PROPOSAL_ID → AWARD_ID
                 └─ AWARD.AWARD_ID                  (versioned; AWARD_NUMBER + AWARD_SEQUENCE_STATUS = 'ACTIVE')
                      └─ AWARD_AMOUNT_INFO          (obligated / anticipated amounts)
Lookups: SPONSOR, UNIT, PROPOSAL_STATUS, AWARD_STATUS, ACTIVITY_TYPE, PROPOSAL_TYPE, NARRATIVE_TYPE
People:  EPS_PROP_PERSON (dev), PROPOSAL_PERSONS (inst), AWARD_PERSONS; PI role usually 'PI'
```

Watch for: `AWARD_FUNDING_PROPOSALS` may point at a **non-active**
institutional proposal version, so join through `PROPOSAL_NUMBER` to reach
the current version. One award can be funded by several proposals and one
proposal can fund several awards (continuations, supplements). Data types
of join keys can differ (VARCHAR vs NUMBER); cast explicitly.

### Step 2: Inventory every file (metadata only, no BLOBs yet)

Write `scripts/build_file_inventory.py` → `storage/linkage/file_inventory.parquet`.
One row per (source table, attachment row) with at least:

`SOURCE` (e.g. `PROPOSAL_NARRATIVE`, `PROPOSAL_PERSONNEL`, `INST_PROPOSAL`,
`AWARD`, `SUBAWARD`, …), `SOURCE_TABLE`, `ATTACHMENT_PK`, `FILE_DATA_ID`,
`FILE_NAME`, `CONTENT_TYPE`, `FILE_EXT` (from name), `ATTACHMENT_TYPE_CODE`,
`ATTACHMENT_TYPE_DESC`, `DEV_PROPOSAL_NUMBER`, `INST_PROPOSAL_NUMBER`,
`AWARD_NUMBER`, `SEQUENCE_NUMBER`, `IS_ACTIVE_SEQUENCE`, `BLOB_LENGTH`
(`DBMS_LOB.GETLENGTH`, so empty BLOBs show up without being downloaded),
`HAS_INLINE_BLOB`, `UPLOAD_TIMESTAMP`, `UPDATE_TIMESTAMP`.

**Include all attachment types and all sequences.** Filtering happens later
as an explicit, documented choice, not silently in SQL. This is the
**union**: one table, one schema, all sources.

### Step 3: Build the proposal-outcome spine

Write `scripts/build_proposal_spine.py` → `storage/linkage/proposal_spine.parquet`.
**One row per development proposal**, with:

- Dev proposal: number, title, sponsor code + name + sponsor type, prime
  sponsor, activity type, proposal type, lead unit (+ parent/college via the
  `UNIT` hierarchy), requested start/end, deadline, dev status, submission
  date.
- Institutional proposal (current version): number, status code +
  description, requested direct / indirect / total.
- Award (if any): award number(s), current status, obligated and anticipated
  totals, start/end dates. Multiple awards → keep a separate
  `proposal_award_bridge.parquet` (many-to-many) and put summary columns on
  the spine (`N_AWARDS`, `TOTAL_OBLIGATED`, first award date).
- PI: person id and home unit only. **Don't add names or emails unless the
  user asks.**
- `OUTCOME`: derived label (`FUNDED` / `NOT_FUNDED` / `PENDING` /
  `WITHDRAWN` / `UNKNOWN`). Derive from the actual status lookup values you
  find, and **write the mapping out explicitly** in a dict in code and in the
  findings doc. Don't guess codes; get the user to confirm the mapping before
  anything depends on it.

Also make `award_spine.parquet` (one row per award number, current version)
so award documents with no matching dev proposal (pre-Kuali, migrated, or
created directly as awards) are still covered.

### Step 4: Pull the files

Write `scripts/pull_files.py`: reads `file_inventory.parquet`, downloads
BLOBs in batches of ≤20 by `FILE_DATA_ID` (dedupe first) to
`storage/linkage/files/{FILE_DATA_ID}{.ext}`, **using the real extension
from magic bytes**, not always `.pdf`:

- `%PDF` → `.pdf`
- `PK\x03\x04` → zip container: open it and check for `word/` (`.docx`),
  `xl/` (`.xlsx`), `ppt/` (`.pptx`)
- `\xD0\xCF\x11\xE0` → OLE (legacy `.doc`/`.xls`)
- `{\rtf` → `.rtf`
- else → `.bin`, and record it

Resumable: skip IDs already on disk with a matching size. Write a
`pull_manifest.parquet` (`FILE_DATA_ID`, bytes, detected type, sha256,
status, error).

Start with a **pilot of ~200 files** stratified across sources/types, check
everything, then run the full pull.

### Step 5: Extract text from all formats

Extend extraction (new module; don't change behaviour of
`services/text_extraction_service.py`, which production uses):
PDF via pypdf (existing logic), DOCX via `python-docx`, DOC/RTF via whatever
is available locally (e.g. `antiword`, `textract`, LibreOffice headless; note
which). Keep the existing quality flag (`assess_text_quality`). Mark
image-only PDFs (no text, many pages) as `NEEDS_OCR`; don't run OCR unless
the user approves it.

Output `storage/linkage/document_text.parquet`: `FILE_DATA_ID`, `FORMAT`,
`N_PAGES`, `TEXT`, `LENGTH`, `ALPHA_RATIO`, `SUCCESSFULLY_PARSED`,
`EXTRACT_STATUS`.

### Step 6: Assemble

`storage/linkage/documents.parquet` = inventory ⋈ text ⋈ spine (on dev
proposal number, falling back to award number). This is the final unioned,
linked, document-level table.

---

## 6. Completeness experiments (must do; report results)

Each experiment is a count in the source vs a count in the output, with
every difference explained. Put the results in
`docs/linkage_findings.md` (**counts and structure only, no record
contents**; this file can be committed).

- **E1 Source coverage:** for each file-bearing table found in step 1, source
  row count vs inventory row count. Should match exactly.
- **E2 Orphans:** attachment rows whose `FILE_DATA_ID` is NULL or not in
  `FILE_DATA`; `FILE_DATA` rows referenced by no attachment table. Check
  inline-BLOB columns on the orphans.
- **E3 Empty and duplicate files:** `BLOB_LENGTH` = 0 or NULL; the same
  `FILE_DATA_ID` under several attachment rows/sequences; the same sha256
  under different IDs.
- **E4 Format mix:** counts by detected type vs `CONTENT_TYPE` vs file
  extension. Quantify how many files the current pipeline drops for not
  being PDFs.
- **E5 Extraction yield:** by format and source: extracted OK / low quality /
  needs OCR / failed.
- **E6 Join funnel:** dev proposals → with inst proposal → with award, by
  fiscal year, sponsor type and college. Look for years or units where
  linkage collapses (e.g. data migrations, pre-Kuali records).
- **E7 Reconcile with the existing pipeline:** the IDs in
  `ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT` and `..._AWARDS` (read-only, via
  `dsstag`) should be a subset of the new inventory. List any that aren't,
  and give the overall size of what the old pipeline missed.
- **E8 Spot check:** randomly pick about 20 proposals across outcomes. Confirm
  their documents, institutional proposal and award match what the Kuali UI
  shows (the user can do the UI side; give them the proposal numbers).
- **E9 Outcome sanity:** funded rate overall and by year/sponsor type.
  Anything near 0% or 100% probably means the status mapping is wrong.

---

## 7. Deliverables checklist

- [ ] `scripts/discover_kuali_schema.py` fixed and run; report reviewed
- [ ] `scripts/build_file_inventory.py`, `build_proposal_spine.py`,
      `pull_files.py`, plus the multi-format extractor and assembly script
- [ ] Parquet outputs under `storage/linkage/` (not committed)
- [ ] `docs/linkage_findings.md`: confirmed join path with real table and
      column names, outcome-code mapping, E1–E9 results, open questions
- [ ] All SQL kept in code (or `sql/*.sql`) so the pull is reproducible
- [ ] Code committed and pushed; no data, no credentials

## 8. Ask the user before

- Confirming the status-to-`OUTCOME` mapping.
- Running OCR, or pulling PI names or other personal fields.
- Any query that looks like it will run for more than ~10 minutes, or any
  full BLOB pull above ~10 GB.
- Anything that would modify production code paths or tables.
