# 0040 — Dataset hosting: the maintainer's OneDrive stays the master; archives shared on request

**Status**: Accepted
**Type**: Ephemeral
**Date**: 2026-08-06

## Context

Where to host the canonical benchmark archives (~35 GB across four
benchmarks, in the `StructBench/{canonical,raw}` layout of ADR-0031) has
been open since ADR-0021 §5 and was flagged again in ADR-0024's
consequences. A 2026-07-09 correction already demoted it from release gate
to deferred question: benchmarks ship code + protocol + baseline numbers,
and v0.1.0 released on exactly that basis, with the archives
maintainer-held and a Zenodo direction (one record per benchmark) parked on
the roadmap.

## Decision

The current arrangement is the decision, not a stopgap (maintainer,
2026-08-06):

1. The canonical archives stay on the maintainer's institutional OneDrive,
   which remains the single master copy, in the ADR-0031 layout.
2. Access is on request: the maintainer shares OneDrive links with
   individual requesters. Public docs (the README data-availability
   paragraph, benchmark pages) state this plainly, alongside the
   ingest-your-own-data path through the LS-DYNA adapter.
3. No public hosting platform is adopted; the parked Zenodo direction is
   dropped rather than deferred.

## Alternatives considered

- **Zenodo** (the previously parked direction): DOIs and versioning, but
  per-record limits and curation overhead against ~25 GB archives, for
  demand that has not yet materialised. Revisit if a publication needs a
  citable dataset DOI.
- **HuggingFace datasets / GitHub Releases**: convenient distribution, but
  a second platform to maintain, with no strong pull today.
- **Public OneDrive share links published in the README**: frictionless,
  but the links are tied to the Curtin tenancy and rot silently;
  on-request keeps the maintainer aware of who holds the data.

## Consequences

- The v0.2 release needs no hosting work; the roadmap item closes.
- Users face one request step; the benchmark spec, cards, and baseline
  numbers remain fully public regardless.
- No dataset DOI exists. If one becomes necessary (e.g. for a paper), this
  Ephemeral ADR is updated in place with a dated note.
- The 2026-07-09 CORRECTIONS.md hosting entry is resolved by this ADR.

---

**Amendment (2026-08-28, maintainer).** The three LS-DYNA archives are now
*also* published on Hugging Face as public dataset repos —
`StructBench/wave-propagation-1d`, `StructBench/taylor-impact-2d`,
`StructBench/notch-beam-2d-impact` (CC BY 4.0; one `.h5` per case,
`cases.csv` with splits, loading/geometry parameters and a SHA-256 manifest,
the LS-DYNA input decks; tagged `v0.1.0`). This amends clauses 2–3 without
reversing clause 1: the maintainer's OneDrive stays the single master copy
(the HF repos are a distribution mirror built from it by
`tools/build_hf_bundle.py` and re-uploaded on change), on-request sharing
continues for anyone who cannot reach HF, and HF is the *live* public
channel. A Zenodo record (citable DOI) stays deferred to a dataset freeze —
the point at which a versioned snapshot is worth archiving, e.g. when a
publication needs to cite one. DeformingPlate is unaffected (not rehosted,
ADR-0042). Why now: v0.3 made StructBench a public multi-method benchmark,
and "request the data from the maintainer" was the last non-public step in
reproducing any leaderboard row.
