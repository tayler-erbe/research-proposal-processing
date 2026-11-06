# Proposal / award linkage implementation status

## Evidence and next step

The Claude handoff branch and existing ingestion/writer source have been reviewed. Discovery queries and proposed joins are not yet validated against a live Kuali schema. A development proposal with no award link must not automatically be labelled NOT_FUNDED: incomplete linkage, historical versions, pending/withdrawn records and migration gaps must remain distinguishable.

The initial laptop network check could not reach the configured Kuali or staging Oracle TCP ports. The external preserved configuration was used for a config-and-network-only preflight; credentials were not copied into this repository. No database authentication, source query, BLOB download, production writer or OCR run has occurred.

## Discovery changes

- Metadata and code lookups run first. Join-funnel/version aggregate scans require `--with-counts` after metadata review.
- File-bearing discovery searches all visible owner tables for BLOB columns and FILE_DATA_ID/FILE_ID, regardless of table names. Every candidate still needs an approved source adapter and join validation.
- Schema identifiers are validated. Returned metadata is bounded; hitting the limit produces an explicit failure rather than silent truncation.
- Client initialization is isolated from production helpers and accepts `--oracle-client-lib`. Per-round-trip timeout and connect timeout are configured; these are not a total runtime or resource-cost guarantee.
- Errors report driver codes rather than potentially sensitive connection text. Reports must remain under gitignored storage.
- Update year is labelled as update year, not creation year.

## Completeness experiments

| Check | State |
|---|---|
| E1 All source attachment rows accounted for | Pending schema discovery and adapter validation |
| E2 Orphans / inline BLOB coverage | Pending source-column discovery |
| E3 Empty files / duplicate identities and hashes | Pending inventory and pilot download |
| E4 Actual format vs MIME/name | Pending pilot; ZIP/OLE signatures alone do not identify Word |
| E5 Text yield / low quality / OCR | Pending extraction; OCR requires explicit approval |
| E6 Proposal → institutional proposal → award funnel | Pending verified keys and version handling |
| E7 Existing pipeline coverage | Pending staging read and owner/object-name confirmation |
| E8 Kuali UI spot checks | Pending user-assisted comparison of selected records |
| E9 Outcome plausibility | Pending lookup review and user-confirmed status mapping |

No counts or funded outcome classifications are asserted. PI names/emails and other additional personal fields have not been requested or extracted. The union must retain attachment provenance even when multiple attachment rows point at one file; binary deduplication must not erase those relationships. Proposal-award relationships belong in a many-to-many bridge to avoid inflating amounts or losing awards without development proposals.

## Run on an approved network

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-discovery.txt
.\.venv\Scripts\python.exe scripts/discover_kuali_schema.py --config "PATH_TO_EXISTING_PRIVATE_DATABASE_YAML" --preflight
.\.venv\Scripts\python.exe scripts/discover_kuali_schema.py --config "PATH_TO_EXISTING_PRIVATE_DATABASE_YAML"
```

If Oracle requires thick mode, add `--oracle-client-lib "PATH_TO_APPROVED_INSTANT_CLIENT"`. No credential values belong in command arguments. Inspect `storage/discovery/kuali_schema_report.md`; only then run a bounded `--with-counts` pass and review the observed status descriptions before defining outcomes. See the root handoff for later inventory, spine, download and extraction deliverables; those are not yet implemented.

Driver timeout behavior: [ConnectParams](https://python-oracledb.readthedocs.io/en/stable/api_manual/connect_params.html) and [connection call_timeout](https://python-oracledb.readthedocs.io/en/v3.1.1/api_manual/connection.html#Connection.call_timeout).
