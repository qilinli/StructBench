# 0053 — Decouple model history from the seed / scored-span protocol

**Status**: Accepted — amends ADR-0035, supersedes the `velocity_history`
boolean of ADR-0049 (implemented on `feat/adr-0053-decouple-history`, byte-identical, pending merge)
**Type**: Durable
**Date**: 2026-08-15

## Context

`input_frames` was doing two orthogonal jobs at once:

1. the **seed / scored-span protocol** — the rollout observes `input_frames`
   ground-truth frames and scores `[input_frames, end]`, so it *must* be shared
   across model families for a common, comparable scored span; and
2. the **model's history length** — how many past frames the model actually
   consumes, an *architecture* property of each method.

ADR-0035 fused these ("the model observes exactly the frames it inputs"). That
identity holds only for GNS-style models. Concretely, at the Taylor/notch
`input_frames = 6`:

- **CGN** (← GNS, Sanchez-Gonzalez et al.) genuinely uses all 6 — its node input
  carries `input_frames − 1 = 5` finite-difference velocities. The number 6 is
  a *GNS* number.
- **MeshGraphNets** (Pfaff et al. 2021) uses history `h = 0` for the
  Hyper-Elastic Lagrangian solid (= our DeformingPlate) — node input is the node
  type only — and `h = 1` (a single velocity) for cloth, its one
  momentum-driven case. Never a multi-frame window.
- **Transolver** (Wu et al. 2024) uses `h = 0`: its structural Plasticity
  benchmark conditions on geometry + time, no velocity history.

So `input_frames = 6` was imposed as a *history length* on two families
(MGN, Transolver) that, per their own papers, do not use it — they observed 6
frames and used 1. The reference recipe encoded this by pinning the ADR-0049
`velocity_history = False` flag, which forced a binary choice (`h = 0` or the
full `h = input_frames − 1`) with no way to express the paper-faithful `h = 1`
for the momentum-driven impact benchmarks — precisely the setting ADR-0049 was
reaching for when it diagnosed the mesh-native models as "velocity-blind" on
Taylor.

## Decision

**Keep `input_frames` as the seed / scored-span protocol only, and give each
family its own history length.**

- `input_frames` (card + config) is re-documented as the **seed / scored-span**
  count. Values are unchanged (6 Taylor/notch, 2 deforming_plate); it stays
  card-pinned and shared for a common scored span. It is *no longer* the model's
  history length. **No card-schema, rollout, or scored-span change.**
- The mesh families (MGN / Transolver / GeoFLARE) replace the binary
  `velocity_history: bool` with a count **`history_frames: int`** — the number
  of trailing finite-difference velocities the model consumes from the seed
  window. `0` is Markovian (reference); `k` appends the last `k` velocities.
  Bounded `0 ≤ history_frames ≤ input_frames − 1` and rejected at config load
  otherwise (a velocity needs a preceding frame).
- The model still *receives* the `input_frames`-wide seed window but *uses* only
  the last `history_frames` velocities of it. CGN is unchanged — it is the GNS
  reference and always uses the full `input_frames − 1` window.
- The training noise scheme branches on `history_frames > 0` (was
  `velocity_history`): `0` keeps the single-frame Gaussian / MGN γ=1 target;
  `k > 0` uses the CGN random-walk + GNS adjusted-next target. The random walk
  (and its velocity features) now spans exactly the `history_frames + 1` frames
  the model uses, so intermediate `k` gets a history-matched noise scale; at
  `k = input_frames − 1` this reduces to the full window and is byte-identical
  to the pre-0053 path.

## Migration — byte-identical, no result changes, no blessed-result invalidation

| pre-0053 | post-0053 |
|---|---|
| `velocity_history = false` | `history_frames = 0` |
| `velocity_history = true` | `history_frames = input_frames − 1` |
| CGN (always full window) | unchanged (`history = input_frames − 1`, implicit) |

- All 42 in-repo mesh TOMLs were migrated by the table above; the 13
  velocity-history configs map to `history_frames = 5` (Taylor
  `input_frames = 6`), the rest to `0`.
- `read_run_record` maps legacy `config.json` records forward
  (`velocity_history` → `history_frames`, dropping the stale key) so existing
  checkpoints stay evaluable against the current dataclasses.
- Byte-identity was verified empirically: `_mesh_family_noise` at
  `history_frames = input_frames − 1` reproduces the pre-0053 velocity-history
  path bit-for-bit under a fixed seed, and at `history_frames = 0` reproduces
  the single-frame reference path; the full suite (523 passed, 6 skipped)
  covers the config round-trip, the bounds, the legacy adapter, and the new
  intermediate-`k` noise path.

## Consequences

- **Each family can declare its paper-faithful history, decoupled from the
  shared seed.** MGN/Transolver can now use `history_frames = 1` (the cloth
  convention) for the momentum-driven impact benchmarks — the principled form of
  the ADR-0049 fix — `0` for DeformingPlate and the one-shot baselines, while
  CGN keeps `5` for GNS. These new intermediate-`k` recipes are *new configs*,
  not silent changes to existing runs.
- `input_frames = 6` stays — but now honestly as the seed / scored-span, not a
  per-model history it was never used as. The Taylor/notch card's
  `protocol_rationale` (which motivates the 6 via CGN's GNS history) is now
  correctly read as a CGN-specific justification for the shared seed floor.
- **Surface changed:** `config.py` (field swap on three dataclasses, the load
  bound-check, the legacy adapter, docstrings), `cli/train.py`
  (`_mesh_family_noise` and `_transolver_pushforward` take a count and
  history-match the window; the three `build_*` derivations), the 42 mesh TOMLs,
  and tests. **Not touched:** the benchmark card schema, the rollout seed/window
  logic, the scored span, and CGN.

## Relationship to other ADRs

- **Amends ADR-0035**: `input_frames` no longer means "the model's history
  length"; it is the seed / scored-span protocol. The cross-check
  (`config.input_frames == card.input_frames`) is retained — it is a *seed*
  constraint — and remains shared for comparability.
- **Supersedes the representation** ADR-0049 introduced: the `velocity_history`
  boolean becomes the `history_frames` count. ADR-0049's substantive findings
  (noise rescale, the velocity-history *benefit*, the MGN stretch gate) stand
  unchanged; only the knob's shape and name change.
- **ADR-0051 unaffected**: `frames_per_call` (the prediction-scheme axis) is
  orthogonal to `history_frames` (the input-history axis); the one-shot
  baselines keep `history_frames = 0` + the ADR-0051 B `impact_velocity_feature`
  scalar as their faithful, velocity-blind-but-loading-aware input.
