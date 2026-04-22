# Roadmap

Forward-looking backlog. Items are grouped roughly by scope of change, not by priority.

---

## Research Proposal Intelligence platform (emerging)

The natural extension of this pipeline. The University System manages roughly $517M in annual research funding, and SPA together with other stakeholders has been exploring what an LLM-driven decision-intelligence platform on top of this data would look like. Questions the extended platform aims to answer:

- What characteristics are associated with successful funding outcomes?
- Where is the System underperforming relative to peer institutions or to its own historical baselines?
- How can proposal strategy be optimized at the PI, department, and college level?

This pipeline is the foundation — structured, annotated proposal and award text at the document level. The next layer adds LLM-driven feature extraction (methodology, collaboration structure, budget narrative themes) and predictive modeling against funding outcomes. Builds directly on Legislation LLM Feature Extraction methodology used elsewhere in the organization.

Marginal improvements in proposal strategy translate to millions in additional research funding plus indirect cost recovery; the value case is well-bounded.

---

## Semantic pipeline consolidation

The current three-way ensemble (TF-IDF + KeyBERT + LDA) was the right call for the initial production pipeline — each method captures distinct signal, the cost of running all three is acceptable, and downstream consumers already expect the three-column output. For the next iteration, a unified embedding-based pipeline would:

- Simplify the staging schema (one keyword column instead of three)
- Produce more consistent per-document signal (embeddings handle OOV terms gracefully where LDA does not)
- Reduce maintenance surface area (one model to upgrade, not three libraries to keep aligned)

Open question: whether to retire TF-IDF entirely or keep it as a cheap frequency-based fallback when the embedding model is unavailable.

---

## Entity extraction re-enablement

`services/entity_extraction_service.py` is fully wired but gated. The default pipeline populates the entity columns as nulls because per-document spaCy runs add ~1-2s each, dominating wall-clock on larger runs.

Options for re-enabling:

- Run entity extraction asynchronously, writing to the entity columns as a follow-up pass rather than inline with the main pipeline
- Use a smaller transformer-based NER model (e.g. DistilBERT-based) for better speed/accuracy tradeoff than spaCy's `en_core_web_sm`
- Extract entities only for documents flagged as high-value (funded awards, flagship programs) rather than the full corpus

---

## False-positive classification risk in BUDGET bucket

`services/table_build_service.classify_document_type` flags any document title containing the string "SPA" as BUDGET. That's a false-positive risk — documents titled "SPA Review" or similar get bucketed as BUDGET regardless of content. Currently kept as-is to match the production behavior downstream stages expect.

Safe remediations (in order of invasiveness):

1. Replace `"SPA"` with `"SPA BUDGET"` — narrower match, same column semantics
2. Split the detection into SPA-specific and budget-specific logic, emitting a new `DOCUMENT_SUBTYPE` column downstream consumers can opt into
3. Drop the heuristic entirely and let downstream consumers classify from `MODULE_TITLE` directly

---

## Cross-environment resilience

Current weak points on the "different environments" axis:

- The Oracle Instant Client path in `oracle_connection.py` is Linux-default; Windows/macOS developers need to override. A small environment-variable-driven lookup would make this painless.
- The `data/models/all-MiniLM-L6-v2/` directory is expected to be present locally. A small "download on first run" helper script with integrity check would reduce setup friction.
- The sentence-transformers model is currently version-pinned via its local copy. A more explicit version check at import time would surface silent upgrades faster.

---

## Log reader tooling

The `PipelineLogger` writes a well-shaped parquet log. A simple CLI tool that reads the log directory and produces a last-N-runs summary (success/failure, duration percentiles, rows-written trends) would make ops review of pipeline health substantially faster. Currently this is done ad-hoc through the job execution monitor dashboard; a local CLI would also serve as a dev-time debugging aid.

---

## KeyBERT-to-production schema rename

The DB column `RAKE_OUT` stores KeyBERT output. The column name predates the switch from RAKE to KeyBERT, and renaming it would break every downstream consumer — acceptable short-term, but worth coordinating a migration alongside the next schema revision that touches this table for other reasons. Same applies to `UNIQUE_RAKE` / `UNIQUE_RAKE_REDUCED`.

---

## Awards-side NLP

Awards currently get text extraction but not NLP. The HERDS classifier and keyword extraction aren't meaningful for award-level documents (which are contracts, modifications, and notices rather than scientific narratives). But there's a specific subclass of award documents — the scope-of-work narrative and the budget justification attachment — that are semantically similar to proposal narratives and could usefully flow through the same NLP pipeline. Worth a scope study once the proposal pipeline's LLM extension lands.
