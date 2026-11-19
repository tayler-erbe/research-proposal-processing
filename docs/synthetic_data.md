# Synthetic data: contract, generator, fidelity and leakage

Status (19 Nov 2026): **framework built and tested on an invented placeholder profile.**
No real data has been profiled yet. Every synthetic number produced today
reflects `synthetic/profiles/placeholder.yaml`, which is made up, not the
real Kuali population. The next step needs the work laptop (see the end of this document).

This implements steps 2, 5, 7 and 8 of [the handoff](handoffs/codex-sample-pull.md)
in a form that runs without database access. Steps 1, 3, 4 and 6 still need the laptop.

## Pieces

| Path | What it is |
|---|---|
| `contracts/canonical_model.yaml` | Canonical entities (development proposal, institutional proposal version, award version, funding link, sponsor, document), their keys and types, and the `spine` analysis view. Downstream code reads only these names. |
| `contracts/sources/kuali.yaml` | Real Kuali table and column for each canonical field, with transforms (zero-padded proposal numbers, `ACTIVE`/`ARCHIVED`, `Y`/`N`) and the confirmed status-to-outcome lookup. Names come from queries already in this repository and still need the discovery script's confirmation. |
| `configs/data_source.yaml` | The one switch: `data_source: synthetic` or `kuali` (or the `DATA_SOURCE` environment variable). |
| `adapters/mapping.py` | `load_canonical()` returns identical, typed canonical tables from either source; `to_source()` and `from_source()` convert to and from real Kuali shape; `derive()` computes campus, fiscal year (July–June) and sponsor type the same way for both; `build_select()` writes read-only SELECT text for the laptop pull. |
| `adapters/views.py` | `spine()`: one row per development proposal with outcome, award count and link-conflict flags, mirroring `scripts/build_proposal_spine.py`. |
| `synthetic/profile.py` | `profile_tables()`: the aggregate profile (shares, funded rate by sponsor type, quantiles, cardinalities, edge-case rates), used both as the generator's target and for fidelity. `disclose()`: the public-repository version (cells under 10 merged, shares rounded, amounts as 2-significant-figure quantiles). |
| `synthetic/generate.py` | Seeded generator: profile → canonical tables plus the same data in real Kuali shape (real table/column names and key formats, fresh values). Reproduces versioning, many-to-many funding links, link conflicts, missing and inline payloads, and Word files saved under `.pdf` names. |
| `synthetic/fidelity.py` | Synthetic-versus-target report: total variation distance, funded-rate gaps, quantile log ratios, rate gaps, and a ranking-agreement utility check. Thresholds in `synthetic/fidelity_thresholds.yaml`. |
| `synthetic/augment.py` | Named augmentations applied to the profile (oversample a category, raise an edge-case rate, shift the payload mix). Fidelity against the original profile flags the distortion. |
| `synthetic/leakage.py` | Before any commit of synthetic artifacts: identifier overlap, quasi-identifier near-copies and 8-token text overlap against the real local sample. Prints counts only. |
| `tests/test_synthetic.py` | Determinism, key formats, foreign keys, one current version per number, Kuali round trip, config switch, fidelity, augmentation flagging, disclosure, leakage. |

## Run it

Needs Python 3.10+ and PyYAML (`pip install -r requirements-synthetic.txt`).
pandas and pyarrow are optional: output is CSV, and parquet is read when pyarrow loads.

```
python -m synthetic.generate --seed 7 --scale 1
python -m synthetic.fidelity --data storage/synthetic/seed7_scale1
python -m unittest tests.test_synthetic
```

Output goes to `storage/synthetic/` (gitignored).

## Results on the placeholder profile (19 Nov 2026)

These numbers check only the plumbing: the generator reproducing the profile it was given.

- Scale 1 (2,000 proposals, seed 7): 26 checks. Before small groups were exempted, 25 passed and 1 failed: the requested-amount quantiles for the smallest sponsor group, which has about 200 values. At scale 5, all 26 passed (that group's gap fell from 0.29 to 0.08), so the failure was sampling noise, not bias. Groups with fewer than 300 synthetic values are now reported as not measurable instead of judged (`min_group_n`).
- An augmentation that makes one sponsor type 4 times as common fails the sponsor-type check, as intended.

## Known limits

- **The placeholder is invented.** The funded rates, amounts, version counts and file counts in it are guesses, so they say nothing about the real population.
- Sponsor type codes are synthetic labels (`FEDERAL`, `STATE`, …). The real `SPONSOR_TYPE` code-to-label lookup is still open in `contracts/sources/kuali.yaml`.
- Narrative type `535` is real. The other narrative types and the award attachment types in the generator are invented codes.
- Documents are metadata only (format, size, pages, payload location). Synthetic document text is the existing `research_intake/synthetic.py` generator. Joining it to these rows, and running the repository NLP on it (handoff step 6), is not done.
- Fidelity covers marginals, the funded-rate conditional, a few quantiles and cardinalities. It does not yet cover joint distributions beyond sponsor type, or the nearest-neighbour distance check.
- The EDW source (`contracts/sources/edw.yaml`) does not exist yet. It needs the mart names from the owner.

## Next, on the work laptop

1. Confirm the mapping: run `scripts/discover_kuali_schema.py` and correct `contracts/sources/kuali.yaml` where the names differ.
2. Pull the real-shaped tables for the sample keys into `storage/samples/<date>/kuali/`. Use `adapters.mapping.build_select()` for the column lists, restricted to the sampled keys with bind variables. Put measured document fields in `_DOCUMENT_MEASURES` from the existing inventory and pull manifest.
3. `load_canonical({'data_source': 'kuali', ...})` → `profile_tables()` → keep the full profile under `storage/`, and `disclose()` it for review. The owner reviews the disclosed profile before anything is committed under `contracts/profiles/`.
4. Generate from the disclosed profile, run `synthetic.fidelity`, and run `synthetic.leakage` against the real sample before committing any synthetic artifact.
