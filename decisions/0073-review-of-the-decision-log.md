# 0073 — Decisions from the maintainer's review of the decision log

**Status**: Accepted (maintainer, in writing 2026-09-29 and 2026-10-01; drafted by Claude Code)
**Type**: Durable
**Date**: 2026-10-01
**Amends**: 0010, 0012, 0026, 0035, 0043, 0046, 0055, 0064, 0065, 0066, 0067, 0068, 0071, 0072. **Accepts**: 0057, 0071, 0072.

## Context

By 2026-09-29 the log held 71 records and 86,000 words, every one drafted by
the agent. The maintainer had read some and accepted the rest as proposed,
because they were long. A triage that day read every record in full and asked
one question of each: does it hold a call that is the maintainer's to make,
which the text does not show the maintainer making? Twelve records did, and
they reduced to fifteen decisions. The maintainer answered each in writing.

This record holds the answers. It is one file rather than fifteen notes, which
is the shape the review itself called for; each amended record carries a
one-line pointer here, and the index says which records it touches.

## Decision

**Reference data.**

1. **D1 — notch-impact.** The sweep is a numerical example produced by a
   collaborator and was not validated against a physical experiment. With
   `sigy = 337` GPa in the deck, the steel impactor and supports never yield
   and act as elastic bodies. The data stand as the benchmark's reference; the
   card says both things. *Amends 0067's "left open" bullet.*
2. **D2 — Taylor's stored `global/total_energy`.** It stays as the adapter
   copies it from d3plot (kinetic plus internal, without the rigid-wall term).
   The card documents the difference; the verification row keeps reading
   `fail`. *Amends 0066's coverage note.*

**What is published, and on what terms.**

3. **D3 — `taylor_copper`.** The redistribution of the six measured outlines
   under CC BY 4.0 is accepted, on the check that the source article is CC BY
   4.0 in full, its figures are the authors' own plots, and nothing in it is
   marked as reproduced from another publication. *Accepts 0072.*
4. **D4 — the ported Transolver code.** It stays under the repository's
   Apache License 2.0, with the MIT copyright and permission notices of both
   upstream implementations (Transolver, 2024; Transolver++, 2025; THUML @
   Tsinghua University) carried in the header of `models/transolver/network.py`,
   which ships inside the installed package. No repo-root NOTICE file is
   added; `LICENSE` is untouched. *Settles 0044 clause 14.*
5. **D5 — Transolver++.** Accepted as it stands, provisional rows included.
   *Accepts 0057.*

**What the leaderboards show.**

6. **D6 — DeformingPlate RMSE.** The pooled statistic of 0043 §8 is that
   benchmark's `rollout_pos_rmse_mm` and `rollout_vm_rmse_mpa`, as its
   registry has recorded since 2026-08-21 and its card states. 0046 clause 6
   and the two `rollout_*` rows of its clause 7 are reversed for DeformingPlate;
   0055's "blessing-only" wording no longer describes it. The von Mises
   pooling is in `tools/blessing_pooled_rmse.py` (2026-10-01); re-running it on
   the six registered runs to confirm the registry's values is still to do. *Amends 0043, 0046, 0055.*
7. **D7 — `input_frames = 6`.** Confirmed, read as 0053 reads it: the shared
   seed and the start of the scored span for every model; each family's
   history window is its own. *Confirms 0035.*
8. **D8 — the notch-impact probe.** The probe stays the reported generalisation
   measure, with the card saying its two cases differ from training on several
   axes at once, so a score locates no single cause. The card's reading of the
   probe results is reduced to what was observed; the explanation in 0026's
   2026-08-15 amendment stays there as the agent's interpretation. Both draft
   amendments of 0026 are finalised. *Amends 0026.*

**What counts as a pass.**

9. **D9 — reference levels.** None is ratified. Indicators are measured and
   shown beside their published level; the reader judges. 0071 clause 6's
   third item, which proposed ratifying B-BLM-1, is corrected. *Amends 0071;
   confirms the 2026-09-21 decision in 0066.*
10. **D10 — the preflight gate.** Its semantics stand as built through part
    three (a): a passing stamp required, `--no-preflight` recorded in
    provenance, the dataset author declaring tolerances, `frame_reported` and
    `accepted_reviews`, the defaults (QoI 1 %, frame 5 %, settling margin
    25 %, increment factor 0.5, ledger closure 1e-5), and the archive stage's
    default retention of about 5 % of `.odb` files. *Confirms 0071.*
11. **D11 — a draft document as a requirement.** `input_requests_required_evidence`
    stays a requirement for LS-DYNA inputs although the conformance document
    is a draft. 0066's Abaqus rows note is finalised, and 0068 clause 8 is
    lifted for that one row on observation alone. *Amends 0066, 0068.*
12. **D12 — the yield constraint.** The structured heads keep stress at or
    below yield by construction. The reference data's own excess (up to a
    ratio of 1.00153 on Taylor) is solver and storage tolerance; any future
    property test judging a model's stress against yield must allow at least
    that. F-011 and F-002 are restated in 0064 as it used them, with no source
    in the repository. *Amends 0064.*

**Contracts and scope.**

13. **D13 — the time axis.** For data generated with this repository,
    `response/time/t` is strictly uniform and a terminal frame off the
    sampling interval is dropped at conversion. Archives converted before the
    pipeline existed are exempt. *Amends 0012.*
14. **D14 — the data standard and the flow-map code.** A benchmark admitted
    after this date must have a convergence run, a measured noise floor, a
    survivor log of discarded cases and a constitutive sensitivity study; a
    physical-test anchor and a second-solver reproduction are recorded where
    they exist and not required. The state-feedback, flow-map and
    structured-head code (0060 to 0064) is reference-baseline infrastructure
    and stays in the repository, inside a model family or as a shared module.
    *Amends 0065.*
15. **D15 — solver code in the package.** The pipeline's solver-side code may
    live under `structbench.datagen.<solver>/` so long as `import structbench`
    needs no solver. *Amends 0010; accepts 0071.*

## Alternatives considered

- **A note on each affected record and nothing else.** Rejected: fifteen
  dated notes across fourteen files is the growth the review was about, and a
  reader could not see that one review made them.
- **Leaving the accepted records as they were.** Rejected: several embedded
  calls contradicted each other or the shipped code, and a maintainer's
  approval of a record is only worth what the maintainer has read.

## Consequences

- 0057, 0071 and 0072 move to Accepted with the decisions above; 0058, 0060,
  0068, 0069 and 0070 were accepted from their summaries the same day
  (2026-10-01), with the corrections their notes record.
- Each amended record carries a dated note naming the decision above; the
  index Status column names this record.
- Still owed: the von Mises pooling script (D6); the housekeeping the triage
  listed (stale index rows and text, Durable labels on experiment records),
  which is mechanical and not decided here; `docs/VISION.md`'s rewrite
  (0065), still the maintainer's out-of-session act.
- The triage itself, with its full register, is the maintainer's private page
  of 2026-09-29 and is not in the repository.
