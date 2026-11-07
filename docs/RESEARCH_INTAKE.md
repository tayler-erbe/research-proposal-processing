# Research intake prototype (`research_intake/`)

Review-assisted extraction from research **proposal** and **award** documents,
built on what the procurement intake prototype (private repo
`intake-pipeline-lab`) established. Status: **working prototype, tested end to
end on synthetic documents** with the stub and, on 2026-11-07, with local
Ollama (`qwen2.5vl:32b` page OCR, `qwen3:32b` extract). Nothing here has been
run on real documents yet, and no accuracy is claimed.

Synthetic run with real models (6 cases, 14 sources, 17 jobs, all complete):
doc types 14/14; the 3 scanned pages transcribed exactly; every field had
precision and recall 1.0 except `award_terms_flags` (precision 0.833: an F&A
rate cap tagged `payment_restriction`). The synthetic text is simple and
keyword-aligned, so this shows the model path works, not that it is accurate.
The evaluator scores only positive truth: boolean fields expected `False` and
documents with no truth fields (biosketch, DMP, modification) are not scored
for false positives.

## What we kept from the procurement prototype

| Lesson / component | How it appears here |
|---|---|
| Content-hash source identity; changed bytes are a new source | `store.sources.id` = SHA-256; paired-sample ingest re-verifies every hash |
| Durable SQLite job queue, leases, pause/resume, no automatic replay of uncertain requests | `store.claim/finish`; `pause`, `resume`, `retry` commands |
| Job identity = stage + subject + model digest + prompt hash + schema + input hash | Changing any of them creates new jobs; old results are never overwritten |
| One model per stage across the whole corpus, then unload (fits one L4/L40S) | `pipeline.run`; refuses to start if another model is resident |
| Evidence by ID: model picks excerpt IDs, code copies the text | `evidence.py`; unknown IDs, out-of-vocabulary values, `found` without evidence are rejected |
| Full-text coverage in bounded chunks; never claim absence from a partial read | Every readable span is scheduled; `not_found_in_read_text` when pages are unread |
| Truncated or malformed output is a failure, not absence | Stored raw, job marked failed, field shows `incomplete` |
| Local-only inference; refuse a missing model rather than substitute | `OllamaBackend.identity()` |
| Runtime directory must be outside the repo | `store.connect` refuses paths inside the repo |
| CSV export with formula neutralization; offline review page; literal FTS search | `export.py`, `review.html`, `search` |
| Model routes from the completed evaluation | `config/routes.json`: Qwen3 32B thinking-off for fields (GPT-OSS 20B low as alternative); Nemotron Parse and PaddleOCR-VL as the target page/table routes once provisioned |

## What is different for research documents

- **Source is Kuali, via the paired sample.** `ingest-paired` reads
  `storage/linkage/paired_documents_complete.parquet` and `files/` produced by
  `scripts/verify_paired_sample.py`. Proposal/award relationships come from
  Kuali IDs in that table, never from file names.
- **Many-to-many associations.** One file can belong to several attachments,
  proposals and awards (boilerplate, reused plans). It is parsed once and keeps
  every association.
- **Mixed formats.** PDF (page + bounding box), DOCX (paragraph/table-row
  identity; Word has no stable pages), UTF-8 text. Legacy `.doc`, RTF,
  spreadsheets and archives are quarantined with a visible reason until an
  approved converter is added.
- **Family from Kuali role, type by metadata cascade.** Family (proposal/award)
  comes from the document's Kuali role. Document type comes from a local
  attachment-type code map (`--type-map`, not committed), then file-name and
  opening-text rules (`config/doc_types.json`), otherwise `other`.
- **Narrative-only fields.** Sponsor, amounts, dates, PI, unit, campus and
  outcome come from Kuali/EDW columns, not the model. The model is asked only
  for what lives in text (`config/fields.json`, **provisional**): research
  themes, methods, specific aims, translational stage, population focus,
  external collaborators, solicitation reference, compliance flags, facilities
  needs and topic watchlist for proposals; award document type, award action,
  reporting requirements, deliverables, terms flags, F&A terms, period of
  performance and closeout requirements for awards.
- **Typed values.** `evidence` (copied text), `enum`, `multi_enum` (controlled
  vocabulary, evidence per value) and `labels` (short interpretations, marked
  as interpretation, each with evidence).
- **Point-in-time rule.** `case_features.csv` has one row per development
  proposal. Only pre-decision proposal material feeds the plain columns.
  Award documents appear only in `award__` columns, and institutional-proposal
  attachments are held out (`review_required`) until someone classifies them,
  because they can contain post-decision material.
- **Synthetic generator with ground truth.** Fake proposals, biosketches,
  shared boilerplate, notices of award and scanned modifications.
  `evaluate-synthetic` scores the run. With the stub this is a plumbing check:
  the stub uses the same keywords the generator writes, so perfect scores mean
  nothing about accuracy.

## Outputs (all under the private `--root`)

`tables/documents.csv`, `associations.csv`, `field_values.csv` (long: one row
per document × field with outcome, value, evidence pages, model, review state),
`proposal_features.csv` and `award_features.csv` (wide), `case_features.csv`,
`stage_queue.csv`; `review.html` (offline, filterable, links to source pages);
`search-results.json`.

Field outcomes: `found`, `ambiguous` (conflicting enum values kept),
`insufficient_evidence`, `not_found_in_full_text` (all text read, nothing
found; still not a legal "absent"), `not_found_in_read_text` (unread pages
remain), `no_readable_text`, `incomplete` (a chunk failed or was rejected),
`rejected`.

## Run it

Synthetic, anywhere (no GPU):

```sh
pip install -r requirements-intake.txt
python -m research_intake synthetic --out ../ri-synth
python -m research_intake --root ../ri-run ingest --manifest ../ri-synth/manifest.json
python -m research_intake --root ../ri-run classify
python -m research_intake --root ../ri-run plan --stage extract --backend stub
python -m research_intake --root ../ri-run run --stage extract --backend stub
python -m research_intake --root ../ri-run export
python -m research_intake --root ../ri-run evaluate-synthetic --truth ../ri-synth/ground_truth.json
```

On the work laptop, with the paired sample and local Ollama (private root
outside the repo; content never leaves the machine):

```powershell
python -m research_intake --root $private ingest-paired --linkage storage/linkage
python -m research_intake --root $private classify --type-map $private\type_map.json   # optional
python -m research_intake --root $private plan --stage page_ocr --model <installed vision tag>
python -m research_intake --root $private run  --stage page_ocr --model <installed vision tag>
python -m research_intake --root $private plan --stage extract  --model <installed text tag>
python -m research_intake --root $private run  --stage extract  --model <installed text tag> --limit 20
python -m research_intake --root $private export
```

`type_map.json` maps `"<family>:<Kuali attachment type code>"` to a doc type,
e.g. `{"proposal:535": "narrative"}`. Run `page_ocr` before `extract`;
extraction plans pick up OCR text as new input. Start with `--limit` and look
at `review.html` before running the full sample.

## Not done yet

1. **Field schema sign-off.** `config/fields.json` is provisional. Lock it with
   the SPA owner; then version it.
2. **Real model runs** on the laptop, then a labelled set (the procurement
   evaluation suggests ~100–300 documents, a double-labelled subset and a
   locked holdout) before any accuracy statement.
3. **Selected parser routes.** PaddleOCR-VL (tables) and Nemotron Parse (pages)
   need on-prem provisioning and adapters behind `generate()`. Budgets in PDFs
   are not table-parsed yet. The Ollama vision OCR is a development fallback.
4. **Converters** for legacy `.doc`/RTF/spreadsheets, and DOCX headers,
   footnotes and images.
5. **Review workflow.** `review` records accept/reject/correct per field. A
   proper reviewer UI is the next build, following the procurement workbench.
6. **Join to structured Kuali/EDW fields** (sponsor, amounts, outcome) on case
   ID for analysis, and the analysis views themselves.
7. **Natural-language search** over approved fields: a restricted query API,
   never model-written SQL.
