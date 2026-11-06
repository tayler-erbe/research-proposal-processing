# Codex handoff: real-data samples → data contract → realistic synthetic data

**Audience:** a coding agent (Codex) running on the work laptop, which has
approved access to the Kuali Research and EDW/analytics Oracle databases.
**Owner:** the repo owner (ask them anything this doc leaves open).
**Branch:** `claude/happy-clarke-plqt0c` (or a branch off it).

> This repository is **public on GitHub**. Read §2 before you write or
> commit anything.

---

## 1. Why we're doing this

We want to build out the research-proposal analytics (proposal → outcome,
document NLP, funded vs not-funded comparisons) **on realistic synthetic
data** that can be developed, shared and demoed openly. Real data stays on
the work machine.

To make the synthetic data trustworthy, we need a **ground-truth reference**
from the real systems:

1. **Pull small, representative samples** of the real tables we already
   use (Kuali source + the EDW/analytics tables this pipeline and others have
   built), plus **population-level aggregate profiles**.
2. **Turn them into a data contract:** real table and column names, types,
   keys, join paths, code lookups, cardinalities and distributions.
3. **Build a synthetic generator** that produces data matching that
   contract closely enough that analyses run on it give realistic comparisons
   (funded rates, amount distributions, timelines, document lengths, keyword
   and HERDS mixes).
4. **Keep a mapping layer** so the same downstream code runs on synthetic or
   real data by changing one config file. Recreating this project on real
   data should mean swapping a YAML, not rewriting code.
5. **Validate augmentations:** any augmentation or noise we apply to the
   synthetic data must keep it inside what the real profiles say is
   plausible, with a fidelity report to prove it.

The samples are a **reference**, not the product. The product is the
contract, the mapping, the generator and the fidelity report.

---

## 2. Hard rules (non-negotiable)

**Credentials**
- Credentials are **never** written into this repo: not in code, docs, YAML,
  notebooks, commit messages, logs or printed output.
- They live in `configs/database.yaml`, which is **gitignored**. The template
  is `configs/database.yaml.example`. If the file doesn't exist on this
  machine, **ask the user** to create it or provide the values interactively.
  Don't go looking for passwords elsewhere.
- If you need an additional connection (e.g. a separate EDW service), add a
  stanza to the local `database.yaml` and a **placeholder-only** stanza to
  `database.yaml.example`. Optionally support env-var overrides
  (`ORACLE_USER`, `ORACLE_PASSWORD`, …). Never hardcode defaults.

**Databases**
- **Read-only.** SELECT only. No INSERT/UPDATE/MERGE/DELETE/TRUNCATE/DDL, no
  temp tables, no `CREATE ... AS SELECT`.
- **Do not run** `main.py`, `notebooks/main.ipynb`, `orchestration/*`,
  `services/db_writer_service.py`, `services/awards_db_writer_service.py`, or
  `services/metanode_service.run_metanode()`. They wipe local folders and
  **write to production tables**. Copying their SQL is fine.
- Shared production DB: use bind variables, sample in the database rather
  than pulling full tables, batch BLOB reads (≤20 per query), and set
  `cursor.arraysize` sensibly. Ask before any query you expect to run
  longer than about 10 minutes.

**Data handling**
- **Real rows never get committed.** All raw samples, BLOBs, extracted text
  and row-level profiles go under `storage/` (gitignored). Check with
  `git status` before every commit.
- **What can be committed** (it's a public repo): code; schema metadata
  (table/column names, types, nullability, keys; the repo already discloses
  these); code lookup tables for non-person codes (status, type, activity,
  sponsor type); and **aggregate** statistics that pass §6.3's disclosure
  rules. **The user must review the first commit of anything under
  `contracts/profiles/` before you push it.**
- **Never** commit or print to logs: person names, emails, net IDs, UINs,
  proposal/award titles, abstracts, document text, budget line items,
  exact dollar amounts tied to a record, or record identifiers
  (proposal/award numbers, FILE_DATA_IDs).
- **People:** sample person IDs only to measure structure (e.g. persons per
  proposal), and hash them locally (salted SHA-256, salt kept in
  `storage/`). Don't pull names unless the user explicitly asks.
- **Documents:** pulling a small set of real documents for local reference is
  fine. Synthetic documents must be **generated, not copied**: no real
  sentences, titles or distinctive phrases carried over (see the leakage
  checks in §7.3).

---

## 3. Where the data is

### 3.1 Connections (from `configs/database.yaml`)

| Key | What it is | Use for |
|---|---|---|
| `oracle.kuali` | Kuali Research DB, the source system (schema `KUALI`) | Proposals, awards, attachments, lookups |
| `oracle.dsstag` | Analytics/EDW staging DB (schemas `ANALYTICS_SCHEMA`, `ANALYTICS`) | Tables this pipeline already builds, plus other EDW research marts |
| `oracle.credentials` | Shared service account used for both (per the example file) | — |

Connect with `ingestion/oracle_connection.py:oracle_conn(host, port,
service, user, password)`. It tries thick mode with a **Linux-only** Instant
Client path; on Windows or macOS it falls back to thin mode. If thin mode
can't connect (e.g. `DPY-3001` native network encryption), call
`oracledb.init_oracle_client(lib_dir=<local Instant Client>)` in your own
script before connecting.

### 3.2 Tables this repo already uses (known to exist)

**Kuali (`KUALI.`)**

| Table | Role | Known facts |
|---|---|---|
| `EPS_PROPOSAL` | Development proposal header | `PROPOSAL_NUMBER`, `OWNED_BY_UNIT`. UIUC = unit starting `'1'` or `'9U'` |
| `NARRATIVE` | Proposal narrative modules | `(PROPOSAL_NUMBER, MODULE_NUMBER)`, `NARRATIVE_TYPE_CODE`, `MODULE_TITLE`; current pipeline uses type `'535'` only |
| `NARRATIVE_ATTACHMENT` | Narrative ↔ file | `FILE_DATA_ID`, `FILE_NAME`, `CONTENT_TYPE`, upload/update user + timestamp |
| `FILE_DATA` | **All file BLOBs** | `ID`, `DATA` (BLOB) |
| `PROPOSAL` | Institutional proposal (versioned) | `PROPOSAL_NUMBER` is a **zero-padded string**, `STATUS_CODE` (`2` = funded per current code), `LEAD_UNIT_NUMBER` (UIC = starts `'2'`) |
| `PROPOSAL_ADMIN_DETAILS` | Dev ↔ institutional link | `DEV_PROPOSAL_NUMBER` is an **integer**, `INST_PROPOSAL_ID` |
| `AWARD` | Award (versioned) | `AWARD_ID`, `AWARD_NUMBER`, `AWARD_SEQUENCE_STATUS` (`'ACTIVE'` = current) |
| `AWARD_ATTACHMENT` | Award attachment metadata | `AWARD_ID`, `FILE_ID`, `TYPE_CODE` (`'3'` = internal, excluded today), `SEQUENCE_NUMBER` |
| `ATTACHMENT_FILE` | Attachment ↔ file | `FILE_ID`, `FILE_DATA_ID`, `FILE_NAME`, `CONTENT_TYPE` |
| `AWARD_ATTACHMENT_TYPE` | Lookup | `TYPE_CODE`, `DESCRIPTION` |

**Analytics/EDW (`dsstag`)**

| Table | Grain | Built by |
|---|---|---|
| `ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT` | 1 row per proposal file: Kuali metadata + extracted `CONTENT` | this pipeline |
| `ANALYTICS_SCHEMA.T_FILE_DATA_BLOB2TEXT_AWARDS` | 1 row per award file + text | this pipeline |
| `ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP` | 1 row per file: cleaned text, TF-IDF/KeyBERT(`RAKE_*`)/LDA keywords, HERDS field, entity columns | this pipeline |
| `ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS_HERDS` | 1 row per proposal, aggregated keywords + HERDS | this pipeline |
| `ANALYTICS_SCHEMA.T_CLIENT_TABLE_PROPOSAL_KEYWORDS` | Same without HERDS | this pipeline |
| `ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID` | Allowed proposal list (UIUC all ∪ UIC funded) | `metanode_service` |
| `ANALYTICS.T_KC_PRPSL_ABST_KYWD` | Production view; key `KC_PT_NBR` = first 12 chars of `PROPOSAL_NUMBER` | this pipeline |

Column lists for the staging tables are in
`services/building_staging_tables_service.py`. Note the renames there:
`KEYBERT_*` is stored as `RAKE_*`, and the DB uses `HERDS_FIELD_SKLEARN`
(singular).

### 3.3 Tables to discover (likely exist; confirm names)

- **Kuali:** `PROPOSAL_STATUS`, `AWARD_STATUS`, `AWARD_FUNDING_PROPOSALS`,
  `AWARD_AMOUNT_INFO`, `SPONSOR`, `SPONSOR_TYPE`, `UNIT` (+ hierarchy via
  `PARENT_UNIT_NUMBER`), `ACTIVITY_TYPE`, `PROPOSAL_TYPE`, `NARRATIVE_TYPE`,
  `EPS_PROP_PERSON`, `PROPOSAL_PERSONS`, `AWARD_PERSONS`, budget tables
  (`EPS_PROPOSAL_BUDGET_EXT`/`BUDGET`/`BUDGET_PERIODS`), and other attachment
  tables (personnel bio, institutional proposal, subaward, negotiation).
- **EDW research marts the user's team has already built.** The user says a
  lot of this already exists in the EDW. **Ask the user for the schema and
  table names** of the existing proposal/award/sponsor/expenditure marts. In
  parallel, search `all_tables`/`all_views` on `dsstag` for names matching
  `%PRPSL%`, `%PROPOSAL%`, `%AWARD%`, `%AWD%`, `%SPONSOR%`, `%SPNSR%`,
  `%GRANT%`, `%RSRCH%`, `%KC_%`, `%HERD%`, `%EXPND%`, and list what you find
  (names, row estimates, last analyzed) for the user to confirm.
  **Where an EDW mart already gives proposal → award → outcome, prefer it**
  over rebuilding the joins from raw Kuali, and record both paths in the
  contract.

Start with `python scripts/discover_kuali_schema.py` (already in this repo,
written without DB access, so fix it as needed). Add an equivalent pass for
`dsstag`.

### 3.4 Join facts and traps already known

- `TO_NUMBER(PROPOSAL.PROPOSAL_NUMBER) = PROPOSAL_ADMIN_DETAILS.DEV_PROPOSAL_NUMBER`
  is how `metanode_service.py` links today. Without `TO_NUMBER` it silently
  returns zero rows. **Open question:** that compares an *institutional*
  proposal number to a *development* proposal number. Verify whether that's a
  local convention (inst number = dev number) or a bug, against the
  `INST_PROPOSAL_ID → PROPOSAL.PROPOSAL_ID` path. Report how often they
  agree.
- Institutional proposals and awards are **versioned**. Pick the current
  version explicitly (`PROPOSAL_SEQUENCE_STATUS` / `AWARD_SEQUENCE_STATUS =
  'ACTIVE'`), and record version counts as a profile (the generator must
  reproduce versioning).
- Proposal ↔ award is **many-to-many** (continuations, supplements,
  multiple funding proposals).
- `KC_PT_NBR` in the production view is a 12-character truncation of
  `PROPOSAL_NUMBER`.
- Files: proposal files reach `FILE_DATA` via `NARRATIVE_ATTACHMENT.FILE_DATA_ID`;
  award files via `AWARD_ATTACHMENT.FILE_ID → ATTACHMENT_FILE.FILE_DATA_ID`.
  Not every file is a PDF (DOCX misnamed `.pdf` exists).
- "Funded" today means `PROPOSAL.STATUS_CODE = 2`. Confirm against the
  `PROPOSAL_STATUS` lookup and the award linkage; don't assume.

---

## 4. What data we need

Think in **canonical entities**. Each gets a real source, a sample and a
profile.

| Canonical entity | Grain | Key fields to capture (structure/distribution) |
|---|---|---|
| `dev_proposal` | 1 per development proposal | sponsor code/type, prime sponsor, activity type, proposal type, lead unit → college/campus, deadline, requested start/end, dev status, created/submitted dates |
| `inst_proposal` | 1 per inst proposal version | status code, sequence number, requested direct/indirect/total, dates |
| `award` | 1 per award version | status, sequence, obligated/anticipated amounts, start/end, sponsor |
| `proposal_award_link` | bridge | cardinalities both ways, lag from submission to award |
| `person_role` | person × proposal/award | role (PI/Co-I/KP), home unit, hashed person id, persons per proposal |
| `document` | 1 per file | source (narrative/award/personnel/…), attachment type, content type, detected format, byte size, page count, text length, alpha ratio, parse success |
| `document_text` | 1 per file | **local only**: real text for a small reference set; profiles of length, section headings, vocabulary stats |
| `nlp_output` | 1 per file / per proposal | keyword-list lengths, keyword frequency (top-N with §6.3 suppression), HERDS field distribution and confidence |
| `lookups` | code tables | status, type, activity, sponsor type, narrative type, attachment type, HERDS taxonomy |
| `unit` | org hierarchy | depth, college/campus rollup (UIUC/UIC/UIS prefixes); unit names are public org info but keep them in local profiles unless the user OKs committing them |
| `sponsor` | sponsor | sponsor type mix; sponsor names are public org info but have the same rule as units |

Time window: default to the **last 10 fiscal years**, and confirm with the
user. The UI fiscal year runs July–June.

---

## 5. Sampling design

**Sample keys first, then pull every related row for those keys**, so joins
in the sample are internally consistent.

1. **Population profiles (aggregate, in-database):** counts, null rates,
   distinct counts, min/max/percentiles, top-K categorical frequencies,
   monthly volumes and funded rates by year × sponsor type × campus.
   Compute these with SQL `GROUP BY` over the **full** population; they're
   cheap and are the main input to the generator.
2. **Row sample (local only):** a stratified sample of **~1,000 dev
   proposals** (confirm size with the user), stratified by fiscal year ×
   campus × outcome, with a minimum per stratum. Use deterministic sampling
   (`MOD(ORA_HASH(key, 4294967295, <seed>), N) < k`) so it's reproducible.
   Then pull the linked inst proposals, awards, links, person roles,
   attachments and NLP rows **for exactly those keys**.
3. **Document reference set (local only):** about 50–100 files across sources
   and formats (PDF/DOCX/DOC; narrative, award, personnel), BLOBs plus
   extracted text, to profile document structure.
4. **Edge-case pulls (local only, small):** proposals with many versions,
   awards with many funding proposals, files with bad headers, empty BLOBs,
   orphans. The generator must reproduce these at realistic rates.

Store under `storage/samples/<run_date>/` as parquet, with a
`sample_manifest.json` (queries, seed, row counts, timestamp, no
credentials).

---

## 6. What to build (repo layout)

```
contracts/                         # COMMITTED (after user review): the data contract
  canonical_model.yaml             # canonical entities + fields + types + keys + relationships
  sources/
    kuali.yaml                     # real table/column per canonical field, join SQL fragments
    edw.yaml                       # same for EDW marts
  lookups/                         # non-person code tables (status/type codes + descriptions)
  profiles/                        # aggregate stats that pass §6.3 (user review first)
sql/
  samples/*.sql                    # every extraction query, parameterized (binds), read-only
  profiles/*.sql                   # every aggregate profile query
scripts/
  discover_kuali_schema.py         # exists; fix + extend
  discover_edw_schema.py
  pull_samples.py                  # runs sql/samples → storage/samples/
  build_profiles.py                # runs sql/profiles + local sample → storage/profiles/, then
                                   #   writes the disclosure-safe subset to contracts/profiles/
synthetic/
  generate.py                      # contract + profiles → synthetic parquet in canonical schema
  augment.py                       # documented, parameterized augmentations
  fidelity.py                      # synthetic vs real-profile comparison report
  leakage.py                       # checks synthetic output for leaked real values
adapters/
  mapping.py                       # canonical ↔ source rename/cast layer, driven by contracts/sources/*.yaml
```

### 6.1 Canonical model and mapping (the "easy to swap" requirement)

- Downstream code (analysis, NLP, dashboards) **only reads canonical
  names**.
- `contracts/sources/kuali.yaml` and `edw.yaml` map each canonical field to
  `{schema, table, column, type, transform}` (e.g. `TO_NUMBER`, `SUBSTR(...,1,12)`,
  code → label).
- `adapters/mapping.py` loads either a real source (via SQL built from the
  mapping) or synthetic parquet, and returns **identical canonical
  DataFrames**. Switching is one setting: `DATA_SOURCE=synthetic|kuali|edw`.
- Synthetic output should also be producible in **real-shaped form**
  (real table and column names, real code values, real key formats like
  zero-padded proposal numbers). That way existing pipeline code such as
  `services/*` runs unmodified against synthetic tables.

### 6.2 Generator requirements

Reproduce, from profiles:
- **Marginals:** categorical frequencies, numeric distributions (fit
  log-normal or a mixture for amounts and keep the empirical quantiles), null
  rates.
- **Conditionals that matter for the analysis:** funded rate by year ×
  sponsor type × campus × activity type; amount by sponsor type; lag from
  submission to award; document count and type mix by proposal type.
- **Structure:** version counts, proposal ↔ award cardinalities, persons per
  proposal, files per proposal, orphan, empty-BLOB and bad-header rates.
- **Keys and formats:** same formats as the real ones (zero-padded strings vs
  integers, GUID-like FILE_DATA_IDs), but **values freshly generated**.
- **Documents:** synthetic narrative text that matches the length
  distribution, section structure and HERDS field mix, generated from
  templates or an LLM prompted with **field + structure only, never with real
  text**. Keywords and HERDS outputs should be consistent with the
  generated text, ideally by running the repo's own NLP on it.
- Deterministic by seed; scale is a parameter (1×, 0.1×, 10×).

### 6.3 Disclosure rules for anything committed

- Small-cell suppression: no committed cell with **n < 10**; collapse or
  suppress it.
- Round counts to the nearest 5 or 10, and amounts to 2 significant figures
  or quantiles only.
- No record identifiers, no free text, no person-level values, no
  combination of quasi-identifiers (unit + sponsor + year + amount) that
  narrows to a handful of records.
- Free-text fields get profiled as **lengths, formats and token stats only**.

### 6.4 Augmentation

`synthetic/augment.py` holds each augmentation (e.g. oversampling rare
sponsor types, injecting OCR noise, simulating late awards, scaling volume)
as a named, parameterized function with a short rationale. Each
augmentation is checked by `fidelity.py`. An augmentation that pushes a
profiled statistic outside the real range must be flagged (it may be
intentional, but it must be visible).

---

## 7. Validation

### 7.1 Sample integrity
For each sampled key set: every linked row pulled (no dangling FKs inside the
sample); row counts per entity recorded; strata minimums met.

### 7.2 Fidelity (`synthetic/fidelity.py` → `storage/reports/fidelity.html` + a committed summary)
- Per column: KS or Wasserstein distance (numeric), total variation
  distance (categorical), null-rate difference.
- Per key relationship: cardinality distribution difference.
- Key conditionals: funded rate by stratum, synthetic vs real.
- **Utility test:** run the same analysis (e.g. funded-rate model or
  comparison tables) on real sample vs synthetic and compare the
  conclusions. They should agree in direction and rough magnitude.
- Thresholds go in config; report pass/fail per check.

### 7.3 Leakage (`synthetic/leakage.py`)
- No synthetic identifier, title or text n-gram (≥8 tokens) matches the
  real local sample.
- Nearest-neighbour distance from synthetic rows to real rows isn't
  suspiciously small (no near-copies).
- Run before every commit of synthetic artifacts.

---

## 8. Steps and checkpoints

1. **Connect and discover** (Kuali + dsstag). Output local discovery reports.
   **Checkpoint:** send the user the list of candidate EDW tables and the
   confirmed join path, and ask which EDW marts to treat as canonical sources.
2. **Write `contracts/canonical_model.yaml` and `contracts/sources/*.yaml`.**
   Checkpoint: user reviews the mapping.
3. **Population profiles** (aggregate SQL). Draft the funded/not-funded
   `OUTCOME` mapping from real status codes. **Checkpoint:** user confirms
   the mapping and the time window.
4. **Stratified sample pull + document reference set** (local only).
   Sample-integrity checks.
5. **Generator v1** for structured entities. Fidelity + leakage reports.
6. **Documents + NLP outputs** in the generator. Run the repo NLP on the
   synthetic text; compare keyword/HERDS profiles.
7. **Adapters:** prove `DATA_SOURCE=synthetic` and `DATA_SOURCE=kuali` (on
   the sample) return identical canonical schemas; the same analysis notebook
   runs on both.
8. **Augmentations** with fidelity checks.
9. Write `docs/synthetic_data.md`: what was profiled, how the generator
   works, the fidelity results, how to switch to real data, and open
   questions.

Commit code at every step. Before each commit run `git status` and confirm
nothing under `storage/`, no `database.yaml`, and no real values are staged.

## 9. Ask the user before

- Committing anything under `contracts/profiles/` or `contracts/lookups/`
  for the first time (public repo).
- Pulling names or other personal fields, running OCR, or pulling more than
  ~1,000 proposals or ~100 documents.
- Queries expected to run longer than about 10 minutes.
- Treating any EDW table as the source of truth over Kuali.
- Anything that touches production code paths or writes to any database.

## 10. Related docs

- `AGENTS.md`: the full document-pull plan (complete file inventory,
  multi-format extraction, completeness experiments E1–E9). Its rules apply
  here too; this handoff narrows the current task to **samples + contract +
  synthetic generator**.
- `docs/architecture.md`, `services/building_staging_tables_service.py`,
  `services/metanode_service.py`, `services/db_writer_service.py`: real
  schemas and join logic for the analytics tables.
