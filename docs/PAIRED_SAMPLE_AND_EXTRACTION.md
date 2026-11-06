# Paired research-document sample and extraction boundary

## What exists

The read-only source path is development proposal → proposal administration link → linked institutional proposal version → institutional proposal business number across versions → award funding link → linked award version → award business number. Current institutional/award records are resolved separately. The bridge preserves funding-link and historical version IDs instead of equating one document ID with one proposal or one award.

`FILE_DATA_ID` identifies a payload, not a research case. The same payload may have several attachments or appear in several versions. Attachment provenance and proposal–award relationships remain separate from binary deduplication. A claim that files are paired requires a completed download manifest and case coverage check, not merely matching metadata IDs.

## Local run sequence

1. `discover_kuali_schema.py`: bounded metadata and code lookup discovery.
2. `validate_kuali_links.py`: aggregate checks against the actual schema.
3. `build_file_inventory.py`: metadata union, missing payloads and per-source reconciliation; no BLOB content.
4. `build_proposal_spine.py`: development/current-award context, proposal–award bridge and raw statuses. The owner confirmed institutional statuses 1 Pending, 2 Funded, 3 Not Funded, 5 Withdrawn, 8 Void. Missing/unmapped stay Unknown. Status/link discrepancies are excluded from binary comparisons pending review.
5. `size_linkage_pull.py`: deduplicate by payload identity and measure storage before a download. Content deduplication requires later SHA256 hashes.
6. `pull_files.py`: a 200-file, 500 MiB format pilot. This deliberately spans source/MIME strata and is not representative of population format prevalence.
7. `plan_paired_sample.py`: up to 200 funded paired cases plus same-development-year/sponsor-type not-funded controls, sized as complete available case bundles under an 8 GiB ceiling.
8. `pull_paired_sample.py`: fetch the sized case sample, in at most 20 IDs per batch, with byte-bounded batches, size checks, hashes and per-file resume journal. It never follows an external file location.
9. `verify_paired_sample.py`: verify files on both sides and reconcile missing/empty and downloaded references for every case. All types and historical versions in the narrative/institutional/award sources are retained; other source families remain separate.
10. `assemble_documents.py`: retain the full metadata union and an explicit many-to-many document/proposal context bridge. Text from the format pilot is sparse by design; this is not a fully extracted corpus.

All scripts use the existing private YAML through `--config`; no credential values belong in arguments or Git. Outputs are in gitignored `storage/linkage/`. Install `requirements-linkage.txt` in the isolated environment. Preserve the canonical reports there; none of these scripts runs existing orchestration or database writers.

## Reading contract for reuse of an existing OCR implementation

The reusable boundary is a document parser, not a second ingestion pipeline or an assumption that a particular model wins on this corpus. The procurement implementation should accept a local immutable payload and return the following versioned artifacts:

- Document ID, SHA256, detected MIME/type, parse status and parser/model/config versions.
- One page record per rendered PDF page: page number, dimensions/rotation, page text, stable block IDs and bounding boxes. Native Word/Office needs a traceable render/conversion before it receives page-based citations; native paragraph anchors must not masquerade as page numbers.
- Tables with cells, row/column spans, source page/boxes, raw strings and normalized numeric values kept separately. Financial totals, currencies, units and percentages require explicit checks and review.
- Quality states such as readable, low-confidence, encrypted, unsupported, failed and needs-OCR. An unreadable page is not evidence that a clause or compliance issue is absent.
- A source-linked evidence span for every extraction: document hash, page, exact quote, position and parser run. A citation must resolve to the actual source, not a generated summary.

The current local format pilot extracts PDF text layers, DOCX body paragraphs/tables and UTF-8 text. It does not run OCR or convert legacy Office files. OLE containers are not assumed to be Word; ZIPs are inspected for Office members. Unsupported formats are retained with explicit statuses. Current DOCX extraction does not claim coverage of headers, footnotes or images. Production integration must evaluate these gaps rather than treating successful parsing as complete reading.

The procurement repository, reusable entry point, model evaluation artifacts and endpoint details must be supplied/located before integration. No remote inference service or new OCR model is invoked by these sampling scripts.

## Research vs operational extraction

Preserve a full document graph for authorized operational questions. Separately create an analysis view with point-in-time restrictions: award documents, reviewer outcomes, later modifications and post-decision annotations cannot become predictors of whether a proposal was funded. Development creation year used for sampling is not a submission/fiscal-year success-rate definition. A matched exploratory sample is not a population success-rate estimate or a causal comparison.

Quantities and business facts already in source columns come from those columns first. Model extraction supplies evidence-backed narrative facts and document-stated discrepancies, not replacement sponsor IDs, inferred funding outcomes or guessed missing links. A single award amount must not be summed once per attached document or once per linked proposal. Financial version/allocation rules and no-dollar award coverage require separate validation.

## Remaining validation

Stable business attachment primary keys must supplement run-scoped ROWID locators before long-term incremental operation. The read-committed source reads are not a single-SCN snapshot. Full shared-payload orphan coverage, additional source-family context adapters, staging reconciliation and user-assisted UI checks remain open. Review corpus scope for personnel, special-review, negotiation and workflow sources before downloading content. A full pull above the handoff threshold remains a separate staged-capacity decision.

See python-oracledb's [LOB guidance](https://python-oracledb.readthedocs.io/en/stable/user_guide/lob_data.html) for direct byte fetch versus streamed LOBs. The paired downloader uses expected-length predicates and byte-bounded batches; it does not attempt an unbounded population fetch.
