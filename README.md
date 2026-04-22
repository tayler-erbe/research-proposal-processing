# Research Proposal Processing Pipeline

End-to-end document processing pipeline that converts research proposal and award PDFs stored in the university's Kuali Research system into a structured, NLP-annotated analytical layer. Built at the University of Illinois System in partnership with Sponsored Programs Administration (SPA) to support research-funding decision intelligence across the System's ~$517M annual research portfolio.

This repository contains the production implementation — scrubbed of credentials and infrastructure specifics — plus a narrative case study in `docs/portfolio.html`.

---

## What it does

Every run, the pipeline:

1. **Detects new work** by anti-joining Kuali's proposal and award attachment tables against the analytics staging tables (no dblink between security zones — the delta is computed in Python)
2. **Downloads BLOB attachments** from Kuali in batched Oracle queries
3. **Extracts text** from PDFs with a magic-byte validator that catches DOCX files misnamed as PDFs
4. **Cleans and validates** text against a pre-cached ~500K-word vocabulary (NLTK words ∪ WordNet lemmas) blended with `wordfreq` zipf-frequency thresholds for modern terminology
5. **Extracts keywords** via a three-way ensemble (TF-IDF + KeyBERT + LDA), each method chosen for the distinct signal it captures
6. **Classifies HERDS fields** (NSF's Higher Education Research and Development Survey taxonomy) via cosine similarity against pre-computed taxonomy embeddings, blended 0.75/0.25 with keyword-overlap scoring
7. **Writes three staging tables** (raw text, NLP-annotated, production-aggregated) plus rebuilds a governance filter table that gates which proposals enter the production analytics view
8. **Logs everything** through a canonical `PipelineLogger` whose parquet output is consumed by a separate job execution monitor

The pipeline is idempotent: re-running does no harm. Change detection filters the delta, and every DB write is a MERGE (except the single production TRUNCATE, which is double-gated behind `dry_run=False` AND `confirm_production=True`).

---

## Architecture

```
┌─────────────────────┐        ┌────────────────────────┐
│  Kuali Research DB  │        │  Analytics DB          │
│  (source of truth)  │        │  (staging + prod)      │
│                     │        │                        │
│  NARRATIVE_ATTACH   │        │  T_FILE_DATA_BLOB2TEXT │
│  AWARD_ATTACHMENT   │        │  T_RSRCH_..._NLP       │
│  FILE_DATA (BLOBs)  │        │  T_CLIENT_...KEYWORDS  │
│  EPS_PROPOSAL       │        │  T_KC_PRPSL_ABST_KYWD  │
│  PROPOSAL           │        │  (production view)     │
└─────────┬───────────┘        └───────────┬────────────┘
          │                                │
          │   ingestion/   change_detection/   metanode/
          │                                │
          └────────────┬───────────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │   orchestration/         │
          │   pipeline.py            │   (proposal ingest)
          │   awards_pipeline.py     │   (awards ingest)
          │   nlp_pipeline.py        │   (keywords + HERDS)
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │   services/              │   (text extraction,
          │   (15 focused modules)   │   cleaning, KeyBERT,
          │                          │   LDA, classifier,
          │                          │   DB writers)
          └──────────────────────────┘
```

See `docs/architecture.md` for the deeper walkthrough.

---

## Repo layout

```
research-proposal-pipeline/
├── configs/               # YAML configs + loader
│   ├── database.yaml.example
│   ├── config.yaml
│   └── nlp.yaml
│
├── data/                  # Taxonomy + data artifacts (some gitignored)
│   └── herds_taxonomy.py  # 26 HERDS nodes, 184 unique keywords
│
├── ingestion/             # Oracle connection, BLOB fetch, change detection
│   ├── oracle_connection.py
│   ├── fetch_documents.py
│   ├── awards_fetch_documents.py
│   └── detect_new_records.py
│
├── orchestration/         # Top-level pipeline drivers
│   ├── pipeline.py
│   ├── awards_pipeline.py
│   └── nlp_pipeline.py
│
├── services/              # Focused single-responsibility modules
│   ├── text_extraction_service.py
│   ├── text_cleaning_service.py
│   ├── text_filter_service.py
│   ├── keyword_service.py       # TF-IDF + KeyBERT + LDA
│   ├── herds_classification_service.py
│   ├── change_detection_service.py
│   ├── awards_change_detection_service.py
│   ├── table_build_service.py
│   ├── table_merge_service.py
│   ├── awards_table_build_service.py
│   ├── building_staging_tables_service.py
│   ├── db_formatting_service.py
│   ├── db_writer_service.py
│   ├── awards_db_writer_service.py
│   ├── df_write_service.py
│   ├── entity_extraction_service.py
│   └── metanode_service.py      # governance filter rebuild
│
├── utils/                 # Batching, file helpers, logger
│   └── pipeline_logger.py       # canonical structured logger
│
├── scripts/               # One-off setup + inspection
│   ├── build_vocab_cache.py
│   └── inspect_awards_table_schema.py
│
├── notebooks/
│   └── main.ipynb         # entry point (cell-split, outputs cleared)
│
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   └── portfolio.html     # narrative case study
│
├── main.py                # .py companion to the notebook
├── requirements.txt
└── README.md
```

---

## Setup

1. **Clone and install**

   ```bash
   git clone https://github.com/<you>/research-proposal-pipeline.git
   cd research-proposal-pipeline
   pip install -r requirements.txt
   ```

2. **Configure database credentials**

   Copy the template and fill in real values:

   ```bash
   cp configs/database.yaml.example configs/database.yaml
   # edit configs/database.yaml with your host, port, service, user, password
   ```

   `configs/database.yaml` is gitignored. Never commit real credentials.

3. **Download the sentence transformer model**

   The HERDS classifier uses a local copy of `all-MiniLM-L6-v2` at `data/models/all-MiniLM-L6-v2/`. Download from HuggingFace or let `sentence-transformers` pull it on first run and move the cached copy into place.

4. **Build the vocabulary cache** (one-time, ~30 seconds)

   ```bash
   python scripts/build_vocab_cache.py
   ```

5. **Install NLTK corpora** (done automatically on first pipeline run, or manually):

   ```bash
   python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','wordnet','words','brown','stopwords']]"
   ```

---

## Running the pipeline

**Via notebook** (interactive, recommended for first run):

```bash
jupyter notebook notebooks/main.ipynb
```

**Via Python** (for scheduled runs):

```bash
python main.py
```

Both entry points do the same thing: clear storage, run proposal ingestion, run NLP, write to the DB, run awards ingestion, write to the DB. Every stage is idempotent and logs through the `PipelineLogger`.

### Dry-run the DB writer

The DB writer always defaults to `dry_run=True`. To inspect what would be written without touching the DB:

```python
from services.db_writer_service import run_db_writer
run_db_writer(INTERMEDIATE_DIR, dry_run=True)  # prints SQL, counts, preview
```

Set `dry_run=False, confirm_production=True` only after verifying the dry-run output. The double gate is intentional friction around the single TRUNCATE in the pipeline.

---

## Key design choices

**Three-way keyword ensemble.** TF-IDF, KeyBERT, and LDA each capture a different signal — frequency, semantic similarity, and corpus-level topics. Running all three produces richer keyword output than any single method, with minimal per-document overhead because KeyBERT and the HERDS classifier share one sentence-transformer model (saves ~500MB of RAM and ~30s of startup).

**HERDS classification as a 0.75/0.25 blend.** Pure semantic similarity tended to over-classify documents toward whichever broad category the grant boilerplate leaned (Education shows up often because of training plans). Blending in a 25% keyword-overlap score re-anchors the classification on terms the taxonomy considers authoritative.

**Matrix-multiplied cosine scoring.** Instead of 26 separate cosine calls per document, taxonomy embeddings are pre-computed and normalized once at module load; per-document scoring is a single matrix multiply. 5-10x speedup on real-sized batches.

**Change detection in Python, not a dblink.** The Kuali DB and the analytics DB sit in different security zones. Setting up a dblink would have required a multi-month security review. Pulling both ID sets into memory and diffing in pandas takes ~1 second for millions of IDs and requires no cross-environment infrastructure.

**Idempotent writes.** Every DB write is a MERGE keyed on a natural primary key. Re-running the pipeline over the same batch produces identical output — no duplicate rows, no partial-state surprises. The one TRUNCATE (production table) is double-gated and prints rollback SQL on every successful run.

**Defensive Oracle diagnostics.** The change-detection step prints the connected user, DB name, current schema, and table visibility before running the real query. After getting bitten once by a schema-switch silently pulling the wrong table, these three-lines-of-output diagnostics make "wrong environment" bugs obvious within the first second of a run.

**Staging tables for three audiences.** The pipeline writes three shaped views of its output: raw blob-to-text (for compliance and audit), NLP-annotated (for data-science consumers), and production-aggregated with governance filter applied (for analytical reporting). Each consumer reads exactly what they need without coupling.

---

## Scrubbing notes

This repo is a public scrub of the production pipeline. The following have been replaced with placeholders:

- Oracle hostnames → `<kuali-oracle-host>` / `<dsstag-oracle-host>`
- Service account → `<readonly_service_account>`
- Production schema name → `ANALYTICS_SCHEMA`
- All credentials → removed; `configs/database.yaml` is gitignored and sourced from `database.yaml.example`

Vendor-side table names (e.g. `KUALI.PROPOSAL`, `KUALI.FILE_DATA`) are retained because they're not sensitive — they're the standard Kuali Research schema.

---

## Roadmap

See `docs/roadmap.md` for the current backlog. Highlights:

- **Research Proposal Intelligence platform.** Extending this foundation into LLM-driven feature extraction on proposal narratives, supporting SPA's strategic questions about funding outcomes and proposal strategy optimization across the University System's research portfolio.
- **Semantic pipeline.** Replacing the three-way keyword ensemble with a single embedding-based pipeline for more consistent downstream analytics.
- **Entity extraction.** Re-enabling the spaCy entity extraction module (currently gated behind a perf flag) with a smarter model selection per document type.

---

## Context

Built for the University of Illinois System in partnership with Sponsored Programs Administration. First production pipeline in a broader decision-intelligence initiative extending from Legislation LLM Feature Extraction methodology to research-funding analytics. See `docs/portfolio.html` for the narrative case study with business context and impact framing.
