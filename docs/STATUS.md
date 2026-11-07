# Project status and open items

_Last updated 2026-11-07. Branch: `claude/happy-clarke-plqt0c` (not yet merged to `main`)._

## Done

- **Code on GitHub:** discovery, link validation, file inventory, proposal/award
  spine, sizing, format pilot, paired-sample plan/pull/verify, legacy inline
  supplement, assembly and parser-handoff scripts, with tests (24 passing).
  Details: `docs/linkage_findings.md`, `docs/PAIRED_SAMPLE_AND_EXTRACTION.md`.
- **Outcome mapping confirmed by owner:** institutional status 1 Pending,
  2 Funded, 3 Not Funded, 5 Withdrawn, 8 Void; anything else is Unknown.
- **Paired sample downloaded on the work laptop:** funded cases with
  not-funded controls, proposal and award documents on both sides, hash
  verified. Stored in gitignored `storage/linkage/` only.

## Where the real documents live (by design)

The downloaded proposal and award files and their parquet tables are on the
work laptop only. **They must not be uploaded to GitHub.** This repo is
public and the files are internal-sensitive. If they need to be shared or
backed up, use an approved university location (e.g. university Box or
SharePoint) under SPA/AITS access controls.

## Still to do

1. **Synthetic data build**, the main next task (`docs/handoffs/codex-sample-pull.md`
   plus the private addendum kept in `storage/handoff/` on the laptop).
   Not started: no `contracts/`, `synthetic/` or `adapters/` yet. Use the
   downloaded paired sample and aggregate profiles as ground truth.
2. **Text extraction of the sample:** the `research_intake/` prototype now does
   this. It is built from the procurement intake prototype's design; see
   `docs/RESEARCH_INTAKE.md`. Tested on synthetic documents with a stub only.
   Next: run it on the laptop against `storage/linkage` with local Ollama
   models, review `review.html`, and lock the provisional field schema with SPA.
3. **Validation still open:** E7 (reconcile with existing staging tables),
   E8 (spot-check ~20 cases in the Kuali UI with the owner), full orphan and
   duplicate analysis (E2/E3), stable attachment keys instead of ROWIDs, and a
   consistent snapshot strategy.
4. **Governance:** confirm the new Kuali queries and the use of a cloud
   coding agent fit the commitments in the SPA Master Data Inventory (on-prem
   content, no new source queries).
5. **Merge:** open a PR from `claude/happy-clarke-plqt0c` to `main` once the
   above is reviewed.
