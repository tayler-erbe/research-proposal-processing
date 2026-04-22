# Architecture

Deeper walkthrough of the pipeline internals. Complements the high-level diagram in the README.

---

## Data flow end-to-end

```
  Kuali Research DB                        Analytics DB
  ─────────────────                        ────────────
  NARRATIVE_ATTACHMENT  ─┐
  AWARD_ATTACHMENT      ─┤
  FILE_DATA (BLOBs)     ─┤
  EPS_PROPOSAL          ─┤
  PROPOSAL              ─┤
  PROPOSAL_ADMIN_DETAILS ┘
           │
           │     ingestion/
           │     (change detection via anti-join,
           │      batched BLOB fetch, PDF write)
           ▼
  ┌──────────────────────────────┐
  │  storage/temp_pdfs/          │
  │  storage/extracted_text/     │   ← text_extraction_service
  │  storage/intermediate_tables │   ← table_build_service, table_merge_service
  └──────────────────────────────┘
           │
           │     orchestration/nlp_pipeline.py
           │     (TF-IDF + KeyBERT + LDA + HERDS classifier)
           ▼
  ┌──────────────────────────────┐
  │ proposal_full_table_nlp      │   ← one parquet, NLP-annotated
  │ staging_1_blob2text          │   ← raw text (compliance/audit consumers)
  │ staging_2_nlp                │   ← NLP output (data science consumers)
  │ staging_3_production         │   ← aggregated (analytics consumers)
  └──────────────────────────────┘
           │
           │     services/db_writer_service.py
           │     (5 Oracle writes + metanode governance rebuild)
           ▼
  ┌──────────────────────────────┐
  │ T_FILE_DATA_BLOB2TEXT        │
  │ T_RSRCH_PRPSL_PRCSSD_NLP     │
  │ T_CLIENT_TABLE_..._HERDS     │
  │ T_CLIENT_TABLE_..._KEYWORDS  │
  │ T_RSRCH_..._FND_ID           │   ← governance filter (rebuilt per run)
  │ T_KC_PRPSL_ABST_KYWD         │   ← production (TRUNCATE+INSERT+DELETE)
  └──────────────────────────────┘
```

---

## Change detection (the anti-join)

The two Oracle databases sit in different security zones. Instead of a dblink (multi-month security review), the pipeline performs a Python-side anti-join:

1. Query analytics staging table for every `FILE_DATA_ID` already processed
2. Query Kuali for every candidate `FILE_DATA_ID` (filtered to narrative type 535 for proposals, non-type-3 active awards)
3. Subtract the first set from the second; the remainder is the delta

For millions of IDs, this takes roughly a second and requires no cross-environment infrastructure. The diagnostic queries at the top of `get_processed_ids` (connected user, DB name, current schema, table visibility) add three lines of output and have caught more than one "wrong environment" bug in the first second of a run.

---

## PDF extraction

The extractor does three jobs:

1. **Magic-byte validation.** Real PDFs start with `%PDF`. Kuali occasionally stores non-PDF bytes under a PDF filename (usually DOCX files renamed rather than exported). Without the check, pypdf spends seconds on these and sometimes crashes hard enough to take down the run.
2. **Text extraction.** `pypdf.PdfReader` with per-page fault tolerance — a corrupted page in an otherwise-readable document returns an empty string for that page, not a failure.
3. **Metadata capture.** XMP/DocInfo fields (Title, Author, CreationDate, Producer, etc.) are captured alongside text for downstream merge.

Encrypted PDFs with empty passwords (a surprisingly common lazy "encrypted" marker rather than real DRM) are decrypted on the fly. Real encrypted PDFs return empty text rather than throwing.

---

## NLP pipeline — 11-step sequence

Step numbering matters because each step assumes the columns added by prior steps exist and are clean. If a run dies, the console output tells you exactly which step.

| Step  | What it does |
|-------|-------------|
| 1     | Load merged parquet from ingestion |
| 2     | Load stopwords, remove terms, real-word vocabulary (~500K words) |
| 3     | Coarse text cleaning (lowercase, strip URLs/punctuation, token-length window) |
| 4     | Lemmatization + name-like token stripping |
| 4D    | Hard vocabulary filter via combined NLTK/WordNet set + wordfreq zipf threshold |
| 4E    | Partition into parseable (>= 40 clean tokens) vs. skipped |
| 5A    | TF-IDF with ngram_range=(1,3), max_features=1000 |
| 5B    | KeyBERT with MMR diversity=0.5, shared sentence transformer |
| 5C    | LDA at the corpus level, same topic keywords for every row in the batch |
| 5D    | Per-keyword-list cleaning — dedupe, drop weak single words, drop same-word-repeated phrases |
| 5E–5G | Merge the three keyword sources; build the HERDS classifier input |
| 6     | HERDS classification — batched encode + matrix-multiplied cosine, blended scoring |
| 7     | Final DB-safe string columns; entity columns initialized as nulls |
| 8     | Reduced-length variants for VARCHAR2-constrained DB columns |
| 9     | Column renames to match the DB schema (CLEAN_CONTENT → CLEANED_TEXT, TFIDF_KEYWORDS_STR → SKLEARN_TOP_KEYWORDS) |
| 10    | Parquet-safe type normalization + append skipped rows with nulls |
| 11    | Build three staging tables via `building_staging_tables_service` |

---

## HERDS classifier

Input: a per-document bag-of-text composed of:

1. The TF-IDF keyword list (amplifies document-specific vocabulary)
2. First 2500 characters of cleaned text (general topic signal)
3. First 5 sentences (the abstract opener)

Output: best-match taxonomy node + top-3 alternates, scored via:

```
final_score = 0.75 * cosine(doc_embedding, taxonomy_embedding)
            + 0.25 * keyword_overlap(doc_text, taxonomy_keywords)
```

Performance mechanics:

- **Shared sentence transformer.** One `SentenceTransformer` instance loaded at import time, reused by KeyBERT. Saves ~500MB RAM + ~30s startup.
- **Taxonomy embeddings pre-computed.** 26 taxonomy nodes → 26 embeddings, cached to disk as a pickle. First run builds the cache; subsequent runs just load it.
- **Matrix-multiplied cosine.** The 26 taxonomy embeddings are stacked and pre-normalized. Per-document scoring is a single `(26, dim) @ (dim,) = (26,)` matmul.
- **Batched document encoding.** `assign_herds_batch` encodes all documents in one `model.encode(batch_size=32)` call rather than one per document. 5-10x speedup on realistic batches.

The `MIN_SCORE_THRESHOLD` of 0.12 is empirically where scores stop corresponding to meaningful topical matches. Below-threshold results return the score for visibility but nullify the label — these show up as "HERDS_SUB is NULL" rows in the output, which downstream consumers treat as "no confident classification."

---

## DB write sequencing

Five writes in a specific order. The order matters because writes 3-4-5 read from the previous step's output directly out of the DB, not from the intermediate parquet.

1. **MERGE** → `T_FILE_DATA_BLOB2TEXT` (staging 1, raw text)
2. **MERGE** → `T_RSRCH_PRPSL_PRCSSD_NLP` (staging 2, NLP-annotated)
3. **MERGE** → `T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS` (aggregated, reads from step 2 in DB)
4. **MERGE** → `T_CLIENT_TABLE_PROPOSAL_KEYWORDS` (aggregated, no HERDS field)
5. **Metanode rebuild** → `T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID` (governance filter, reads from Kuali)
6. **TRUNCATE + INSERT + DELETE** → `T_KC_PRPSL_ABST_KYWD` (production, filtered by governance table)

The governance rebuild has to happen before the production write so the DELETE clause in step 6 operates on an up-to-date allowed list.

Every write logs row counts before and after, and runs a spot-check query to confirm at least one row from this batch made it into the destination table. Post-write verification has caught exactly one silent data-type bug that pre-write row counts alone would not have caught.

---

## Governance filter logic

The metanode service rebuilds the allowed-proposal whitelist from Kuali's source-of-truth tables:

```
ALLOWED =  UIUC proposals (all, regardless of funding status)
             ∪
           UIC proposals that are funded (STATUS_CODE = 2)
```

UIUC identifier is `owned_by_unit` starting with `'1'` or `'9U'`. UIC identifier is `LEAD_UNIT_NUMBER` starting with `'2'`. The UIUC/UIC asymmetry is a deliberate scope decision — Urbana-Champaign contributes all proposal activity; UIC's footprint is large enough that only the funded subset is included in downstream analytics.

One join quirk worth noting: `KUALI.PROPOSAL.PROPOSAL_NUMBER` is a zero-padded string (e.g. `'00000003'`) while `PROPOSAL_ADMIN_DETAILS.DEV_PROPOSAL_NUMBER` is a plain integer (e.g. `3`). The SQL uses `TO_NUMBER()` on the left side to normalize. Without this, the join silently produces zero rows and the governance build looks "successful" while writing an empty allowlist.

---

## Logger schema

The canonical `PipelineLogger` writes one parquet row per stage invocation, with these critical fields:

- Identity: `workflow_name`, `pipeline_version`, `run_id`, `script_name`, `script_order`
- Status: `status`, `is_success`, `failure_stage`
- Timing: `event_time`, `start_time`, `end_time`, `duration_seconds`, `logged_at`
- Volumes: `input_count`, `output_count`, `rows_written`
- Skip tracking: `skip_reason` (for intentional no-ops)
- File tracking: `input_file`, `output_file`, `input_file_exists`, `output_file_written`
- Context: `hostname`, `user`, `python_version`, `working_directory`
- Errors: full stack trace on failure

The `skip_reason` field matters — it lets stages log SUCCESS rather than being absent from the run, so the monitor can distinguish "the stage intentionally skipped" from "the stage crashed silently."
