# Proposal / award linkage findings and validation state

Read-only discovery and aggregate validation have connected successfully to the approved source. The development proposal -> administration link -> institutional proposal business number across versions -> funding link -> award business number path was exercised against the actual schema. Current-version uniqueness is checked independently, and historical link IDs are preserved.

The owner confirmed institutional statuses 1 Pending, 2 Funded, 3 Not Funded, 5 Withdrawn and 8 Void. Missing/unmapped statuses remain Unknown. Records whose status and award links disagree are flagged and excluded from binary comparisons pending review. A missing link never automatically means Not Funded.

## Implemented and exercised locally

The prototype includes an attachment metadata union, source reconciliations, proposal/current-award spines, a many-to-many bridge, source/payload sizing, a format pilot, a stratified paired-case plan, resumable downloads, a legacy-inline supplement, case-level coverage checks and a parser handoff manifest. The sample has downloaded documents on both sides of every selected funded pair. Sensitive records, files, source connection identities, cohort statistics and stakeholder correspondence remain local; none are included in this public source update.

The first paired plan selected shared FILE_DATA payloads. An independent case check then found historical award attachments whose bytes instead lived in ATTACHMENT_FILE.FILE_DATA. The supplement resolves attachment-file IDs, verifies lengths, deduplicates by the correct identity and downloads those bytes. The final check reconciles every available payload reference in the sampled narrative/institutional/award scope. A FILE_DATA-only pull must not be called complete.

New inventories carry ATTACHMENT_FILE_ID directly. Replanning from an old inventory without stable inline keys stops explicitly instead of dropping legacy files. The current run preserves the original plan plus its supplement. The canonical completed-sample artifacts are paired_documents_complete.parquet and paired_case_coverage.parquet under local storage/linkage.

## Completeness experiments

| Check | State / limit |
|---|---|
| E1 Source coverage | Inventoried variants reconciled source/joined row counts; held/excluded source families remain separately labelled. |
| E2 Orphans / inline data | Missing payloads inventoried; selected-case inline gap resolved. Full shared-store unreferenced-payload analysis remains open. |
| E3 Empty/duplicate files | Source lengths, payload identities and downloaded hashes recorded. Full historical inline-identity consolidation remains open. |
| E4 Actual format | Byte-based detection used for format and paired pilots; uncommon-format pilot proportions are not population estimates. |
| E5 Text / OCR | Native-text pilot measured. Office conversion, unsupported/encrypted files and OCR remain explicit states; OCR was not run. |
| E6 Join funnel | Live aggregate funnel passed and local context bridges built; fiscal definitions, amount allocation and other source-family contexts remain open. |
| E7 Prior staging output | Not yet reconciled; no staging writers or production entry points ran. |
| E8 Kuali UI comparison | Pending user-assisted case review. Source-ID joins do not replace this check. |
| E9 Outcomes | Distribution inspected and mapping approved. Discrepancies held for review; no predictive or causal claim. |

## Parser handoff and boundaries

The completed sample is hash-verified before generating a PDF filename manifest and immutable hash-to-research-context bridge. Office/other formats have conversion and review queues. This prepares an interface to the existing parser; it does not run procurement ingestion, local database writes, OCR, inference or external model calls. See [the sampling and extraction guide](PAIRED_SAMPLE_AND_EXTRACTION.md).

Document IDs, payload IDs and case IDs are distinct. Operational extraction preserves historical versions; the funding-analysis view must exclude award and post-decision material from predictors. Open work includes stable business attachment keys alongside ROWIDs, a source snapshot/SCN strategy, additional sources, financial version/allocation rules and UI validation. Model/service choices in planning documents need validation before deployment.
