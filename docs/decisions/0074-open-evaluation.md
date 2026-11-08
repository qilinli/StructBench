# 0074 — Open evaluation: blind sets, computed coverage, a results registry

**Status**: Accepted (maintainer, in-session 2026-11-08)
**Type**: Durable
**Date**: 2026-11-08 (first draft 2026-11-06)
**Amends**: ADR-0019, 0025, 0026, 0043 (blind set, coverage; split names grandfathered); 0032 §4 (blind tags); 0033, 0036, 0046 (two tiers, no leaderboard); 0065 clause 1 (no rank); 0071 (scored splits; platform-generated designs); 0075 (rendering; clause 4 waits on its plan); 0040 (notes)

**Your call**:
- Two scored sets, *test* with public outputs and *blind* with outputs the platform withholds, every case read by computed coverage, not a split name. Every contract-admitted benchmark carries a blind set; shipped ones where a generator exists, DeformingPlate never. Reversing re-opens the contamination the public grids allow.
- External results enter in two tiers, self-reported by default and verified when the platform re-ran the submitter's container, without a rank. Reversing admits rows a reader cannot weigh, or restores the ranking ADR-0065 superseded.
- An admitted benchmark is published whole; the platform seeks no patent, restrictive licence or private generator; only blind outputs, pre-admission datasets and partner material are withheld. Reversing means relicensing what is granted.

## Context

The public test splits are downloaded hundreds of times a month and the grids are small, so the test cases are evident from the train ids and reproducible with a solver. Nothing tells a result trained on them from one that generalised. The split words do not transfer either: civil sweeps are factor grids, surrogates want samples, and notch-impact's held-out combinations, every level of which is in training, measure the easiest generalisation while every real design sits off the grid. Adoption comes first: a submitter's friction weighs more than the rare fraud of regenerating a withheld case, guarded, not prevented.

## Decision

1. **Blind set.** Each contract-admitted benchmark carries cases off its training points, inside the declared box and within a declared margin, with the pipeline's verification evidence. Inputs are published as the `blind` set; reference outputs are withheld. Its tag (`blind-v1`, …) sits on the card and never bumps the version; a new tag follows a declared cadence or suspected leakage, the retired set becomes public test, and scores keep their tag.
2. **Sets and coverage.** Public names: `train`, `val`, `test`, `blind`; scored cases lie off the training points. Coverage is computed per scored case from its parameters: *within* the declared box or *beyond* it, and the distance to the nearest training point in units of the training set's median nearest-neighbour spacing (on a grid, the step). An unseen categorical level or changed constant is *novelty*, hand-declared. Legacy split names stand and map for rendering; nothing is re-scored.
3. **Design.** The training design is the author's; the card declares it. Platform-generated datasets use a Sobol training set, a small factor grid as public test that also anchors the data standard's verification runs, and a random blind set.
4. **Scoring by submission.** Canonical-format predictions for the blind cases plus provenance (config, commit, seeds, checkpoint hash, write-up) and the division, *closed* (published train set only) or *open* (any data, declared). The platform computes metrics, QoIs and, once ADR-0075's plan ships it, the properties block, returns scores only, and rate-limits submissions per method per blind tag.
5. **Regeneration guard.** A prediction closer to the reference than its measured noise floor is flagged, returned, and may be asked for a container; the guard runs only where a floor is measured and never reads properties, which reference baselines satisfy by construction (ADR-0064).
6. **Two tiers.** *Self-reported* by default; *verified* when the platform executed the submitter's container on the public protocol and the blind set (the public protocol alone, marked so, where none exists). *Blessed* stays the maintainer's bar (ADR-0046).
7. **Presentation.** One results registry per benchmark, registration order, no rank: the properties block first (ADR-0075), the headline by coverage bin per set, then tier, division and provenance; everything else is a diagnostic. *Results registry* replaces *leaderboard*.
8. **Openness.** At admission the definition is published beside the archive (note on ADR-0071 clause 7); blessed checkpoints at the next code release; archives on more than one channel, freezes archived (notes on ADR-0040).
9. **Authority.** Scoring, flagging and triage are automated processes in `docs/plans/`; landing rows, releases, data tags and rotation stay on explicit instruction (ADR-0023).

## Alternatives considered

- **Execution of every submission.** Rejected: a sandbox and compute for everyone against rare, detectable fraud; kept as the verified tier.
- **Interpolation and extrapolation as split names.** Rejected: they do not transfer between grid and sampled designs; held-out combinations would pass as generalisation.
- **A ranked leaderboard.** Rejected: ADR-0065's note shows the accuracy leader inverting its rank beyond coverage.
- **Patent or non-commercial licence.** Rejected: unenforceable against research use; adoption is the value.

## Consequences

- One plan fixes the prediction format, set sizes, cadence, rate limit, coverage kernel, the contract's `withheld` splits and `grid` shorthand, the container contract and the steward processes; card and registry fields are additive, flag-first.
- `CONTRIBUTING.md`: submission contract, admission list, challenges (which may rank for their duration); a result changing task, split or protocol is another benchmark.
- Split labels in `cases.csv` and the loaders do not change; outward words: *test*, *blind*, *within*, *beyond*.
- Legacy LS-DYNA benchmarks get blind sets when solver access allows, DeformingPlate never, Abaqus datasets at admission; the card's compliance table (ADR-0065 follow-up 2, the next record drafted) shows the status.

---

**Note (2026-11-08).** The last Consequence's "the next record drafted" reads
"the plan of 2026-11-08" (`docs/plans/2026-11-08-card-fields-and-compliance-table.md`):
the maintainer ruled follow-up 2 a plan plus a card change, not an ADR. The
compliance table is built; its blind-evaluation row reads *missing* on the
three LS-DYNA benchmarks and *never* on DeformingPlate, as this record says.
