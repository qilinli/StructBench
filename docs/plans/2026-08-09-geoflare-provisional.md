# GeoFLARE Provisional (ADR-0041 step ③) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native, provisional GeoFLARE family (`models/geoflare`) trainable and evaluable on DeformingPlate via `structbench-train` — a faithful pure-torch port of NVIDIA physicsnemo's shipped "GeoFlare" configuration (GeoTransolver with `attention_type="GALE_FA"`: GALE geometry-aware context cross-attention + FLARE low-rank self-attention), sharing the MGN/Transolver task harness.

**Architecture:** Verified identity (grounding §1): "GeoFLARE" is GeoTransolver instantiated with GALE_FA attention. The port covers the FULL shipped configuration — the ball-query multi-scale context pathway included (grounding §10: `include_local_features: true` in the shipped configs, so ball-query genuinely runs). Ragged batching uses the per-example segment convention proven in `models/transolver`. The simulator subclasses `CaseBoundSimulator`; features are byte-identical to Transolver's 18 channels for apples-to-apples comparison, plus a geometry-coordinates input consumed by the context pathway.

**Tech Stack:** Python 3.12, PyTorch only (no einops, timm, jaxtyping, transformer_engine, warp), pytest, ruff, mypy. Research basis: `scratch/2026-08-09-geoflare-grounding.md` (maintainer-local gitignored record; produced by verification workflow wf_8ed06010-0d0: 180 extracted claims, 157 adversarially confirmed, 20 refuted-with-corrections; §10 addendum verified by a follow-up raw-file agent). §-references below point there.

## Global Constraints

- **Working frame:** mm / MPa. Protocol pins (ADR-0043): `input_frames = 2`; `kinematic_types = (1, 3)`; `scripted_types = (1,)`; noise `N(0, 0.003)` NORMAL-masked, γ=1-by-construction; output `(P, dim+1)` velocity-then-aux; forward-Euler; NORMAL-masked `w_pos·‖Δv‖² + w_aux·Δaux²` loss.
- **Fidelity target = the physicsnemo code as shipped** (grounding §4, §10), not paper prose. Binding pins: parallel (not sequential) self/cross attention from the same pre-attention stream; weighted mix `w·self + (1−w)·cross` with `state_mixing = Parameter(0.0)` (σ(0)=0.5 balanced init); attention scale **1.0** at ALL THREE attention sites (both FLARE passes AND cross-attention — upstream comment: "recommended … dim_head**-0.5 but we use 1.0 because the recommended scaling is not tested yet"), implemented as MANUAL attention `softmax(q @ kᵀ · scale, dim=-1) @ v` in the house style of `models/transolver` — NOT `F.scaled_dot_product_attention`, whose `scale=` kwarg requires torch ≥ 2.1 and would silently raise the repo's declared torch ≥ 2.0 floor; `q_global = Parameter(randn(H, M, dh))` std~1.0, NO trunc_normal/orthogonal init anywhere (PyTorch defaults throughout — GeoFLARE does NOT reuse TransolverNet's `_initialize_weights` pass); context tokenizers use temperature shape **`(heads, 1, 1)`** init 0.5 **clamped [0.5, 5]** (broadcasts against the head-first `(H, N, S)` logits layout this port uses; upstream's `(1,1,H,1)` shape belongs to its `(B,N,H,S)` layout — shape follows layout, the per-head semantics is the pin) and slice-norm eps **1e-2** (physicsnemo convention — deliberately different from our thuml-faithful Transolver port's unclamped/1e-5; both are upstream-faithful to their respective references); GeometricFeatureProcessor MLP dims `[3K, 32, 16, 32]` with GELU between layers, none after the last, **tanh applied outside the MLP**; ball-query = within-radius **nearest-first top-k** (upstream torch-fallback semantics; the Warp backend is order-arbitrary and upstream disclaims neighbor order — our deterministic choice is declared), ABSOLUTE neighbor coordinates (no offsets), zero-pad `(0,0,0)` when fewer than k; no placeholder vector (that is thuml-Transolver-only); decoder = `LayerNorm(effective) → Linear(effective, out)` (single LN+Linear, no deep MLP); FFN block = `LayerNorm → Linear(h, h·4) → GELU → Linear(h·4, h)` with residual; pre-LN residual to the un-normed stream.
- **Dimension truths (grounding §10):** with defaults n_hidden=256, n_head=8, n_hidden_local=32, 2 scales: `effective_hidden = 256 + 32·2 = 320`; **block/attention `dim_head = 320//8 = 40`** but **context-tokenizer `dim_head = 256//8 = 32`** (two different values — conflating them is the likeliest porting bug); `context_dim = 3·32 = 96` (geometry tokenizer + 2 ball-query scale parts; no global part — deforming_plate has no per-run design scalars).
- **Ragged batching:** per-example segment loops for every cross-node operation (ball-query, context building, FLARE encode softmax over N, cross-attention). `q_global`/all weights shared across examples; context is PER-EXAMPLE. Killer test: batched forward ≡ per-example forwards.
- **Harness parity (declared adaptations, → ADR-0045):** functional input = Transolver's exact 18 channels `[one_hot(9), scripted_velocity(3), x_t(3), reference_coords(3)]`; geometry input = **current** positions `x_t` (noised at train / rolled-out at eval), matching every reference call site (`geometry=coords` live), standardized per example (see Design decisions); OnlineNormalizer for functional features + targets (contexts consume standardized coords, not OnlineNormalizer output); GNS noise 3e-3 (reference has none — declared); AdamW + `_lr_at_cosine` + optional clip (reference used Muon — declared, avoids the torch≥2.9 flag-first dep bump); no AMP.
- **MGN and Transolver must not change behaviour.** Allowed cross-family imports (established precedent): `OnlineNormalizer` from `models/mgn`, `build_mlp_2layer`/`_segments` from `models/transolver.network`, `collate_mesh_samples`/`mesh_static_from_trajectory` from `models/mgn`.
- **Mutation-test discipline from the start** (final-review carry-forward from the Transolver branch): every subtle fidelity property gets a hand-derived numerical pin whose sensitivity to the plausible mutation is verified (grounding §7.6 pattern).
- **Attribution:** module docstrings credit "GeoTransolver (Adams et al., arXiv:2512.20399) with FLARE attention (Puri et al., arXiv:2508.12594); crash-domain recipe arXiv:2605.27758; reference implementation github.com/NVIDIA/physicsnemo, Apache License 2.0" — licence-compatible with this repo (Apache-2.0 ↔ Apache-2.0).
- **Conventions:** ruff (88, E/F/I/UP/B), mypy strict, NumPy docstrings citing ADR-0041/ADR-0045 (0045 drafted in Task 8 — intentional forward reference), TDD, gates before every commit: `ruff format --check . && ruff check . && mypy src` + full `python -m pytest` once per task. Tests synthetic-only. Known pre-existing gate failures NOT this plan's: ruff-format `notch_beam_2d_impact/__init__.py`; mypy `datasets/normalization.py`.

## Design decisions (closures of grounding §9 gaps — Task 8 turns these into ADR-0045)

| Gap | Decision |
|---|---|
| 1 — scope | **Full shipped-GeoFlare port**, ball-query context pathway included: the shipped configs set `include_local_features: true` (grounding §10), so FLARE-only would not be "GeoFLARE". Geometry = the node coordinates themselves (co-located; every reference call site passes `geometry=coords`). No global tokenizer (`global_dim=None` — matches the crash GeoFlare config and deforming_plate has no per-run scalars). Single functional stream (reference tuple-stream machinery dropped; we always have exactly one — declared simplification). |
| 2 — context frame | **Current positions, rebuilt every forward** (train: the example's noised `x_t`; eval: the rolled-out `x_t`) — matches the reference (crash AR rebuilds from live predicted coords each step). Cost at ~1.3k nodes is negligible. |
| 3 — ball-query | Native: per example, chunked `torch.cdist`, neighbors = **nearest-first top-k within radius**, ABSOLUTE coordinates, zero-row padding when < k. Deterministic (upstream Warp order is arbitrary and disclaimed; the upstream torch fallback is exactly nearest-first — we adopt it). |
| 4 — ragged | Per-example segment loops (Transolver-proven). Highest-risk new code → the batched≡singles test plus a per-segment softmax-leak mutation pin. |
| 5 — init | PyTorch defaults + `q_global = randn` (std~1.0). NO global init pass. Own driver class (`GeoFlareNet`), not `TransolverNet`. |
| 6 — scale | `1.0` at all three SDPA sites, upstream comment quoted in the docstring. |
| 7 — sizing | Shipped-reference defaults: n_hidden=256, **n_layers=6** (base/bumper value; crash's 5 was an ~400k-node memory workaround irrelevant at ~1.3k nodes — declared), n_head=8, slice_num=128 (serves BOTH context tokenizers' S and GALE_FA's n_global_queries, as upstream wires it), mlp_ratio=4, n_hidden_local=32, dropout=0.0, weighted mixing (hardcoded — no config knob, YAGNI). Yes, M=128 global queries on ~1270-node meshes is a mild bottleneck ratio (~10%) — reference-faithful, ledgered. |
| 8 — radii frame | Reference radii `[0.05, 0.25]` live in position-NORMALIZED space (crash normalizes positions before all rollout math, grounding §5.3). Port: **standardize geometry coords per example** — `g = (x_t − mean(x_t)) / rms_std(x_t)` over the segment — and keep radii 0.05/0.25 in that standardized space; neighbor caps [8, 32] bound the effect. Declared deviation (reference used fixed train-split stats; per-example is self-contained, scale-free, and needs no stats plumbing; the effective-radii-adapt-under-deformation caveat is ledgered). The standardized coords are ALSO what the geometry tokenizer and φ_s MLPs consume (reference feeds normalized absolute coords). |
| 9 — optimizer | Harness AdamW + `_lr_at_cosine` + clip-gated-on->0. `weight_decay=1e-4` (the reference's AdamW-arm value), `max_grad_norm=0.0` (OFF — reference has no clipping; knob kept for parity with TransolverConfig). `lr_init=1e-4` (reference start_lr); cosine to `LR_SCHEDULE_FLOOR` (1e-6 vs reference end 3e-7 — declared). **No Muon** (torch≥2.9 flag-first dep; provisional latitude; deviation declared — the reference's headline numbers were Muon rows). Budget matched: batch 2, 10M steps, val_every 50k. |
| 10 — inputs | Functional 18-ch = Transolver's exactly. Geometry tensor separate (gap 2/8). Output `(P, dim+1)` velocity+stress, Euler, OnlineNormalizers, NORMAL-masked noise γ=1 — full harness parity. |
| 11 — naming | Family key `"geoflare"`, `GeoFlareConfig`, `models/geoflare/`, classes `GeoFlareNet`/`GaleFlareAttention`/`GeoFlareSimulator`. ADR-0045 records the identity chain (GALE_FA / "GeoTransolver with FLARE" / "GeoTS-FLARE" / "GeoFlare", incl. the upstream `GALE_FE` comment bug); Task 8 adds a dated note on ADR-0041 for the naming. |
| 12 — temporal | Next-step AR per ADR-0043 (protocol; not negotiable). Risk note in ADR-0045: reference validated GeoFlare only one-shot; AR was "unstable" at T=50 — our mitigations (GNS noise, quasi-static task) are declared, and no reference number anchors the comparison. |

## File Structure

- Create: `src/structbench/models/geoflare/__init__.py`, `geo_ops.py`, `context.py`, `network.py`, `simulator.py`
- Modify: `src/structbench/config.py` (GeoFlareConfig + registry), `src/structbench/cli/train.py` (imports/`__all__`, `build_geoflare_simulator`, dispatch arm, `_train_geoflare`, evaluate arm, type widenings incl. `_model_config_from_record` return and the `simulator:` concrete union — grounding §7.5 checklist items 1-6)
- Create: `configs/deforming_plate/geoflare.toml`, `configs/deforming_plate/geoflare_smoke.toml`
- Create: `decisions/0045-geoflare-provisional-adaptation.md` (draft, Proposed); Modify: `decisions/0041-*.md` (dated naming note), `decisions/README.md` (0045 row), `docs/ARCHITECTURE.md` (models/ adds geoflare/)
- Tests: `tests/models/geoflare/test_geo_ops.py`, `test_context.py`, `test_network.py`, `test_geoflare_simulator.py`; `tests/cli/test_geoflare_train_smoke.py`; additions to `tests/cli/test_train_config.py`, `tests/cli/test_train_eval.py`

---

### Task 1: `GeoFlareConfig` + registry

**Files:** Modify `src/structbench/config.py`; test additions to `tests/cli/test_train_config.py`.

**Interfaces — produces (verbatim; docstring gets a full NumPy `Attributes` section documenting every field, matching `TransolverConfig`'s convention — the abbreviated block here is a floor, not a ceiling):**

```python
@dataclass
class GeoFlareConfig:
    """Native GeoFLARE family (ADR-0041 step ③; recipe pins in ADR-0045).

    "GeoFLARE" = GeoTransolver with GALE_FA attention (GALE geometry
    cross-attention + FLARE low-rank self-attention); identity and every
    upstream-faithful pin are recorded in ADR-0045. ``weight_decay``/
    ``max_grad_norm`` are family-recipe knobs on the model config
    (``MGNConfig.noise_std`` precedent; the strict ``[train]`` schema stays
    family-uniform).
    """

    input_frames: int = 2
    dim: int = 3
    n_hidden: int = 256
    n_layers: int = 6
    n_heads: int = 8
    slice_num: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.0
    n_hidden_local: int = 32
    radius_near: float = 0.05
    radius_far: float = 0.25
    neighbors_near: int = 8
    neighbors_far: int = 32
    node_type_size: int = 9
    noise_std: float = 0.003
    normalizer_warmup_steps: int = 1000
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.0
```

`MODEL_FAMILIES` gains `"geoflare": GeoFlareConfig` (existing entries untouched). Two fixed scales as four scalar fields (strict-loader-friendly; n_scales is architecturally 2, YAGNI on generalizing).

**Steps:**
- [ ] **Step 1: Failing tests** mirroring the transolver config tests: (a) valid geoflare TOML loads, `family == "geoflare"`, full-field round-trip — diverge at least two values from defaults (e.g. `n_layers = 5`, `slice_num = 64`) so plumbed-vs-defaulted is distinguishable (Transolver-branch deferred-minor lesson); (b) unknown `[model]` key → ConfigError; (c) missing `[model]` key → ConfigError; (d) `input_frames != 2` on deforming_plate → the ADR-0035 message.
- [ ] **Step 2: Run to verify failure. Implement (dataclass + Attributes docstring + registry). Run green.**
- [ ] **Step 3: Full suite + lint gates. Commit** — `feat(config): GeoFlareConfig + geoflare family registration`.

### Task 2: Ball query + coordinate standardization (`models/geoflare/geo_ops.py`)

**Files:** Create `src/structbench/models/geoflare/__init__.py` (exports grow per task; Task 2 exports only `ball_query`, `standardize_coords`), `geo_ops.py`; test `tests/models/geoflare/test_geo_ops.py`.

**Interfaces — produces:**

```python
def standardize_coords(coords: Tensor) -> Tensor:
    """Per-example standardization for the geometry pathway (ADR-0045 gap 8).

    centered = coords - coords.mean(dim=0, keepdim=True)
    rms = centered.pow(2).mean().sqrt().clamp_min(1e-8)   # population RMS of the
    return centered / rms                                 # per-axis-centered coords
    Exact per-axis zero mean AND exact unit RMS by construction (a scalar
    isotropic scale — per-axis scaling would distort the geometry).
    """

def ball_query(coords: Tensor, radius: float, k: int) -> Tensor:
    """Nearest-first top-k neighbors within ``radius`` — (P, k, 3) ABSOLUTE
    standardized coordinates, zero-rows where fewer than k qualify.

    Semantics pinned to the reference torch fallback (grounding §10): distances
    via chunked cdist (reuse the _QUERY_CHUNK=2048 chunking idiom from
    models/mgn/mesh_ops.py — reimplemented here, per-family placement per
    ADR-0020 precedent); per query row take the k smallest distances, mask out
    entries > radius by zeroing their coordinate rows. A node IS its own
    neighbor when within radius (reference does not exclude self — distance 0
    is always nearest-first).
    """
```

Both operate on ONE example (callers loop segments).

**Steps:**
- [ ] **Step 1: Failing tests** — hand-constructed point sets: (a) exact neighbor sets and ORDER for a 5-point line with known distances (nearest-first pinned); (b) radius cutoff: a point just outside `radius` is zero-padded even when k slots remain (mutation pin: a plain top-k-ignoring-radius implementation fails); (c) fewer-than-k padding is exactly `(0,0,0)` rows; (d) ABSOLUTE coords: assert the returned rows equal the neighbors' coordinates, not offsets (mutation pin: `neighbor - query` fails); (e) self-inclusion at distance 0; (f) `standardize_coords`: exact per-axis zero mean, exact unit RMS (`g.pow(2).mean().sqrt() == 1` within float eps — holds by construction with the per-axis-centered population-RMS divisor), translation/scale invariance of the standardized result, degenerate all-identical-points case does not NaN (clamp).
- [ ] **Step 2: Run failing → implement → green. Full suite + gates. Commit** — `feat(models): geoflare geo_ops — deterministic ball query + per-example coordinate standardization`.

### Task 3: Context pathway (`models/geoflare/context.py`)

**Files:** Create `context.py`; extend `__init__.py` (add `MultiScaleContext`); test `tests/models/geoflare/test_context.py`.

**Interfaces — produces (all single-example; shapes without batch dim, flat-N convention):**

```python
class ContextTokenizer(nn.Module):
    """physicsnemo ContextProjector, single-example (grounding §10).

    __init__(dim, heads, dim_head, slice_num, dropout=0.0):
      in_project_x  = Linear(dim, heads * dim_head)
      in_project_fx = Linear(dim, heads * dim_head)
      in_project_slice = Linear(dim_head, slice_num)   # NO orthogonal init
      temperature = nn.Parameter(torch.full((heads, 1, 1), 0.5))  # (H,1,1):
      # broadcasts against the head-first (H, N, S) logits below; upstream's
      # (1,1,H,1) shape belongs to its (B,N,H,S) layout — per-head semantics
      # is what is pinned, the shape follows this port's layout.
    forward(x: (N, dim)) -> (heads, slice_num, dim_head):
      x_mid  = in_project_x(x).view(N, H, D).permute(1, 0, 2)    # (H, N, D)
      fx_mid = in_project_fx(x).view(N, H, D).permute(1, 0, 2)
      logits = in_project_slice(x_mid)                            # (H, N, S)
      w = softmax(logits / clamp(temperature, 0.5, 5.0), dim=-1)  # CLAMPED
      norm = w.sum(dim=1) + 1e-2                                  # (H, S); eps 1e-2
      token = einsum("hns,hnd->hsd", w, fx_mid) / norm.unsqueeze(-1)
      # NB the einsum is mathematically identical to upstream's matmul-with-
      # permutes aggregation (grounding §8/C8) — document as "einsum equivalent
      # to the reference matmul", never as byte-level matmul fidelity.
      return token
    """

class GeometricFeatureProcessor(nn.Module):
    """BQ -> flatten (N, 3k) -> Linear(3k, 32)+GELU+Linear(32, 16)+GELU+
    Linear(16, 32) -> tanh OUTSIDE the stack -> (N, 32). __init__(radius, k,
    n_hidden_local=32); forward(g_std: (N, 3)) calls ball_query(g_std, radius, k).
    (32/16/32 = [n_hidden_local, n_hidden_local // 2, n_hidden_local].)
    """

class MultiScaleContext(nn.Module):
    """The full context assembly for one example (grounding §10):
    __init__(n_hidden, n_heads, n_hidden_local, slice_num, radii: tuple[float, float],
             neighbors: tuple[int, int], dropout=0.0):
      dim_head_ctx = n_hidden // n_heads          # 32 at defaults — the BUILDER dim_head
      geometry_tokenizer = ContextTokenizer(3, n_heads, dim_head_ctx, slice_num)
      processors = ModuleList([GeometricFeatureProcessor(r, k, n_hidden_local)
                               for r, k in zip(radii, neighbors, strict=True)])
      scale_tokenizers = ModuleList([ContextTokenizer(n_hidden_local, n_heads,
                                     dim_head_ctx, slice_num) for _ in radii])
      context_dim (property) = 3 * dim_head_ctx    # geometry + 2 scales, no global
    forward(coords_raw: (N, 3)) -> tuple[context (H, S, 3*dim_head_ctx),
                                          local_features (N, 2*n_hidden_local)]:
      g = standardize_coords(coords_raw)
      per scale: h_s = processors[i](g); parts += [scale_tokenizers[i](h_s)]
      parts += [geometry_tokenizer(g)]     # ORDER: scale parts FIRST, geometry
      # LAST — pinned to the RAW build_context code quoted in grounding §10
      # addendum (local-extractor loop `context_parts.extend(...)` runs BEFORE
      # `context_parts.append(geometry_context)`). NB the PAPER's Eq10 (§2.6/
      # C14) lists geometry first — paper and code disagree; this port's
      # fidelity target is the code. The order is functionally absorbed by the
      # cross_k/cross_v Linear over the full context width either way; record
      # the paper-vs-code note in ADR-0045.
      context = cat(parts, dim=-1); local = cat([h_1, h_2], dim=-1)
      return context, local
    """
```

**Steps:**
- [ ] **Step 1: Failing tests**: (a) shapes at defaults: context `(8, 128, 96)`, local `(N, 64)`; (b) `ContextTokenizer` mutation pins via slice_num=1 degenerate softmax (hand-derive: token = Σfx/(N + 1e-2) — pins BOTH the eps **1e-2** (a 1e-5 mutation shifts the value measurably for small N — pick N=3 so the difference ≫ atol) AND fx-vs-x aggregation with distinct hand-set projections, mirroring the transolver mutation-test recipe); (c) temperature clamp pin — use `slice_num >= 2` with distinct per-slice logits (do NOT reuse (b)'s slice_num=1 setup, where any temperature softmaxes to 1.0 and the pin is vacuous): set `temperature.data` to 0.1 and assert the effective softmax equals the hand computation with 0.5 (clamped), not 0.1 (mutation: an unclamped port fails); (d) part ORDER pin: with hand-set weights making each tokenizer output a distinct constant, assert the concat layout is [scale1, scale2, geometry] along the last dim (the code-faithful order per grounding §10 — see the interface block's paper-vs-code note); (e) tanh boundedness of processor outputs (all in (−1,1)) plus a zero-pad propagation check (a padded neighbor row contributes zeros to the flatten, not NaN); (f) **standardization-inside-builder pin**: feed `coords` and `coords * 1000 + 5000` (large offset AND scale) and assert `MultiScaleContext.forward` returns identical context and local tensors for both (invariance holds ONLY if `standardize_coords` runs inside the builder — a raw-coords mutation silently breaks neighbor finding at mm scale while passing every shape test).
- [ ] **Step 2: Run failing → implement → green. Full suite + gates. Commit** — `feat(models): geoflare context pathway — ball-query multi-scale features + slice tokenizers`.

### Task 4: Attention + network (`models/geoflare/network.py`)

**Files:** Create `network.py`; extend `__init__.py` (add `GeoFlareNet`, `GaleFlareAttention`); test `tests/models/geoflare/test_network.py`.

**Interfaces — produces:**

```python
class GaleFlareAttention(nn.Module):
    """GALE_FA, single-example (grounding §4.1-§4.2, §10).

    __init__(dim, heads, dim_head, context_dim, slice_num, dropout=0.0):
      q_global = nn.Parameter(torch.randn(heads, slice_num, dim_head))  # std~1.0, NO re-init
      in_project_x = Linear(dim, heads * dim_head)
      self_k = Linear(dim_head, dim_head); self_v = Linear(dim_head, dim_head)
      cross_q = Linear(dim_head, dim_head)
      cross_k = Linear(context_dim, dim_head); cross_v = Linear(context_dim, dim_head)
      state_mixing = nn.Parameter(torch.tensor(0.0))
      out_linear = Linear(heads * dim_head, dim); out_dropout = Dropout(dropout)
      SCALE = 1.0  # module-level constant; all three attention sites; upstream
                   # comment ("recommended ... dim_head**-0.5 but we use 1.0
                   # because the recommended scaling is not tested yet") quoted
                   # in the docstring. MANUAL attention (house style) — torch's
                   # SDPA scale= kwarg needs torch>=2.1, above the repo floor.
    @staticmethod  # _attend(q, k, v): softmax(q @ k.transpose(-1, -2) * SCALE, dim=-1) @ v
    forward(x: (N, dim), context: (H, S_ctx, context_dim)) -> (N, dim):
      x_mid = in_project_x(x).view(N, H, D).permute(1, 0, 2)   # (H, N, D)
      k = self_k(x_mid); v = self_v(x_mid)
      z = _attend(q_global, k, v)          # FLARE encode: (H, M, D); softmax over N
      y_self = _attend(k, q_global, z)     # FLARE decode: k AS QUERIES, q_global as
                                           # keys, z as values; softmax over M
      q_c = cross_q(x_mid)
      k_c = cross_k(context); v_c = cross_v(context)           # (H, S_ctx, D)
      y_cross = _attend(q_c, k_c, v_c)
      w = torch.sigmoid(state_mixing)
      y = w * y_self + (1.0 - w) * y_cross          # weighted mix, σ(0)=0.5 init
      return out_dropout(out_linear(y.permute(1, 0, 2).reshape(N, H * D)))
    """

class GeoFlareBlock(nn.Module):
    """Pre-LN (grounding §4.4): fx = attn(ln_1(fx), ctx) + fx;
    fx = ffn(fx) + fx where ffn = Sequential(LayerNorm(hidden),
    Linear(hidden, hidden * mlp_ratio), GELU(), Linear(hidden * mlp_ratio, hidden)).
    No last-layer special head (decoder is external)."""

class GeoFlareNet(nn.Module):
    """Driver. __init__(node_in, out_size, n_hidden=256, n_layers=6, n_heads=8,
    slice_num=128, mlp_ratio=4, dropout=0.0, n_hidden_local=32,
    radii=(0.05, 0.25), neighbors=(8, 32)):
      preprocess = build_mlp_2layer(node_in, n_hidden * 2, n_hidden)  # from models/transolver
      context_builder = MultiScaleContext(...)
      effective = n_hidden + n_hidden_local * 2      # 320 at defaults
      dim_head_block = effective // n_heads          # 40 at defaults — NOT the ctx 32
      blocks = ModuleList([GeoFlareBlock(effective, n_heads, dim_head_block,
               context_builder.context_dim, slice_num, mlp_ratio, dropout) ...])
      decoder = Sequential(LayerNorm(effective), Linear(effective, out_size))
      # NO _initialize_weights pass — PyTorch defaults + randn q_global (ADR-0045).
    forward(node_feats: (ΣP, node_in), coords: (ΣP, dim),
            n_particles_per_example: Tensor | None) -> (ΣP, out_size):
      per segment (reuse _segments from models/transolver.network):
        ctx, local = context_builder(coords_seg)
        fx = cat([preprocess(feats_seg), local], -1)        # (N, 320)
        for block in blocks: fx = block(fx, ctx)            # SAME ctx every layer
        outs.append(decoder(fx))
      return cat(outs)
    """
```

Note the per-segment loop wraps the WHOLE stack here (context is per-example), unlike Transolver's per-attention-call loop — document why in the module docstring (context construction dominates; blocks at N~1.3k are cheap; simplicity beats reuse of the inner-loop pattern).

**Steps:**
- [ ] **Step 1: Failing tests**: (a) forward shape at tiny config (n_hidden=16, n_layers=2, n_heads=2, slice_num=4, n_hidden_local=4 → effective 24, dim_head_block 12, ctx dim_head 8, context_dim 24); (b) **killer / segment-leak pin**: batched forward ≡ per-example forwards (two ragged examples, eval mode, atol 1e-5) — this IS the encode-leak pin: a global (non-segmented) encode softmax over the concatenated N fails it; assert BOTH examples' outputs (no separate far-apart-coordinates construction — the encode operates on features, not coordinates, and segmenting is what's under test); (c) **decode-K-as-query pin at M=2** (M=1 is vacuous — every decode softmax over one slot is 1.0 regardless of the query tensor): H=1, **M=2**, N=2, D=1; hand-set `q_global` rows g₁≠g₂ and `self_k`/`self_v` weights so k≠v per node; hand-derive z₁, z₂ (encode) and `y[n] = Σ_m softmax(k_n·g_m)_m · z_m` (decode); assert exact values; assert sensitivity: the `v`-as-decode-queries mutation (`softmax(v_n·g_m)`) yields a hand-computed DIFFERENT y under the chosen weights (verify the difference on paper before writing the assertion); (d) **mix-direction pin**: zero BOTH `cross_v.weight` AND `cross_v.bias` (biases are nonzero by default init — zeroing only the weight leaves y_cross = bias ≠ 0) so y_cross = 0 exactly; set `state_mixing.data` so σ→0.9; assert via correct-vs-swapped COMPARISON through the same `out_linear` (compute the module output, then the hand-derived value under `w·self + (1−w)·cross` and under the swapped `w·cross + (1−w)·self` — both pass through out_linear+bias identically, so the 0.9-vs-0.1 factor on y_self is detectable); also assert `sigmoid(state_mixing) == 0.5` at init; (e) `q_global`: shape (heads, slice_num, dim_head), requires_grad, empirical std in [0.8, 1.2] (pins the no-trunc-normal decision — a 0.02-std init fails); (f) param-count formula test: derive the structural formula term-by-term in the test (preprocess + context pathway + per-block attention/FFN/LNs + decoder) at the tiny config AND at defaults with node_in=18, out=4; assert `sum(p.numel())` equals it (the reviewer will recompute independently); (g) **scale=1.0 pin, own hand case at D=4** (do NOT reuse (c)'s D=1 case — at D=1, 1.0 and dim_head**-0.5 coincide): H=1, M=1, N=2, **D=4**, hand-set weights; the encode softmax over N is scale-sensitive at D=4 (1/√4 = 0.5 shifts the logits by 2×) — hand-derive the expected z under scale 1.0 and assert; confirm on paper the dim_head**-0.5 mutation value differs.
- [ ] **Step 2: Run failing → implement → green. Full suite + gates. Commit.** Land as TWO commits for reviewability (economics guidance): first `GaleFlareAttention` + its pins (b on a bare-attention harness if convenient, c, d, e, g), then `GeoFlareBlock`/`GeoFlareNet` + integration pins (a, b at network level, f) — one task, one review, two commits: `feat(models): GaleFlareAttention — FLARE encode/decode + geometry cross-attention` then `feat(models): GeoFlareNet — segment-exact driver over the context pathway`.

### Task 5: `GeoFlareSimulator` (`models/geoflare/simulator.py`)

**Files:** Create `simulator.py`; extend `__init__.py`; test `tests/models/geoflare/test_geoflare_simulator.py`.

**Interfaces — produces** (mirror `TransolverSimulator` — read it first; deltas only):

```python
class GeoFlareSimulator(CaseBoundSimulator):
    # __init__(dim=3, n_hidden=256, n_layers=6, n_heads=8, slice_num=128,
    #          mlp_ratio=4, dropout=0.0, n_hidden_local=32,
    #          radii=(0.05, 0.25), neighbors=(8, 32), node_type_size=9,
    #          kinematic_types=(1, 3), scripted_types=(1,), device="cpu")
    # node_in = node_type_size + 3 * dim = 18 (identical to Transolver)
    # _net = GeoFlareNet(node_in, out_size=dim + 1, ...)
    # _node_normalizer = OnlineNormalizer(node_in); _target_normalizer = OnlineNormalizer(dim + 1)
    # _features(one_hot, scripted_velocity, x_t, reference_coords) — same 18-ch cat.
    # predict_positions: base tripwire → scripted vel → feats → node_normalizer →
    #   self._net(feats_norm, x_t, None)  ← ALL-POSITIONAL, always (the hook-based
    #   spy tests read args[0]/args[1] from register_forward_pre_hook, which
    #   captures positional args only) → inverse FULL (P, dim+1) BEFORE slice → Euler.
    #   NOTE: coords passed to the net are the RAW mm x_t — standardization happens
    #   inside MultiScaleContext (per example), NOT via OnlineNormalizer.
    # forward_train(x_last, next_positions, next_aux, particle_types,
    #               reference_coords, n_particles_per_example, *, accumulate):
    #   same as Transolver plus the net call becomes
    #   self._net(feats_norm, x_last, n_particles_per_example)  (all positional).
```

**Steps:**
- [ ] **Step 1: Failing tests** — mirror `test_transolver_simulator.py`'s 10 (read it), adapted: shapes on a bound case; tripwire; second-pass-without-reset raises; `_features` scripted rows == GT deltas; forward_train target semantics (γ=1); save/load round-trip pre-bind (self-contained checkpoint, no stats file); scripted⊆kinematic ValueError; train-vs-eval identical feature tensors (forward-pre-hook pattern); PLUS one geoflare-specific test: the coords tensor received by the net (spy via forward-pre-hook, positional args) is the raw `x_t` (mm frame). **Warm the node normalizer to non-identity stats first** (a few `forward_train(..., accumulate=True)` calls) — OnlineNormalizer is the identity before accumulation, so on a fresh simulator `feats[:, 12:15] == raw x_t` exactly and the feats-slice mutation would be undetectable; after warming, both mutations (feats-slice coords AND simulator-side standardized coords) are caught.
- [ ] **Step 2: Run failing → implement → green. Full suite + gates. Commit** — `feat(models): GeoFlareSimulator — stateful rollout wrapper on the shared base`.

### Task 6: Training + evaluation paths (`cli/train.py`)

**Files:** Modify `src/structbench/cli/train.py`; test additions to `tests/cli/test_train_eval.py`.

**Interfaces — produces** (grounding §7.5's six touch sites, all mechanical after the Transolver precedent — read `_train_transolver` and clone with deltas only):
- `build_geoflare_simulator(cfg, *, kinematic_types, device)` beside the others (scripted_types at class default). **It assembles the config scalars into the simulator's tuples:** `radii=(cfg.radius_near, cfg.radius_far)`, `neighbors=(cfg.neighbors_near, cfg.neighbors_far)` — this mapping lives HERE, nowhere else.
- `_train_geoflare(...)` = `_train_transolver` clone; deltas: builder, `resolved_config_dict("geoflare", ...)`, `AdamW(weight_decay=cfg.weight_decay)`, `_lr_at_cosine`, clip gated `cfg.max_grad_norm > 0` (default 0.0 = off), `forward_train(..., coords threaded via x_noisy inside the simulator — the call signature matches Transolver's exactly)`. Everything else (noise block, loss, warmup, val loop, checkpoints incl. periodic `ckpt-<step>.pt`) copied unchanged.
- `train()` dispatch: `if family == "geoflare": assert isinstance(model_cfg, GeoFlareConfig); return _train_geoflare(...)`; widen `train()`'s `model_cfg` union.
- `evaluate()`: `elif family == "geoflare":` arm (no stats file); widen the `simulator:` local to the CONCRETE union `LearnedSimulator | MeshSimulator | TransolverSimulator | GeoFlareSimulator` (NOT `CaseBoundSimulator` — `rollout()` needs `predict_positions`, which the base lacks); widen `_model_config_from_record`'s return annotation; imports + `__all__` additions (`GeoFlareConfig`, `build_geoflare_simulator` — mirrors the blessed Transolver precedent).

**Steps:**
- [ ] **Step 1: Failing tests**: (a) geoflare `config.json` record round-trips into `GeoFlareConfig`; (b) evaluate on a saved tiny geoflare checkpoint proceeds WITHOUT a stats file; (c) training-wiring test (monkeypatch AdamW kwargs — assert `weight_decay == cfg.weight_decay`; clip NOT called at the default `max_grad_norm=0.0` AND called with the configured value when a nonzero-config fixture is used; final `param_groups[0]["lr"] == _lr_at_cosine(train_cfg.training_steps - 1, train_cfg)` — the PRE-increment last step, matching the trainer's set-lr-then-increment ordering; ≥1 param changed); (d) **radii/neighbors plumbing test**: `build_geoflare_simulator` on a config with non-default `radius_near/far`/`neighbors_near/far` — assert the constructed simulator's `GeometricFeatureProcessor`s carry exactly those values (a builder that silently used class defaults passes every other test: the smoke's ~unit-scale synthetic meshes find neighbors under any radii).
- [ ] **Step 2: Run failing → implement → green. Full suite + gates. Commit** — `feat(cli): geoflare training + evaluation paths on the shared mesh harness`.

### Task 7: Configs + end-to-end smoke

**Files:** Create `configs/deforming_plate/geoflare.toml`, `geoflare_smoke.toml`; test `tests/cli/test_geoflare_train_smoke.py`; loader-test additions to `tests/cli/test_train_config.py`.

- [ ] **Step 1: Reference config** (verbatim):

```toml
# DeformingPlate — GeoFLARE PROVISIONAL reference config (ADR-0041 ③, ADR-0045).
# "GeoFLARE" = GeoTransolver with GALE_FA attention (geometry cross-attention +
# FLARE low-rank self-attention); no published rollout number exists for this
# task — the recipe is declared, not reproduced. Budget matched to the MGN and
# Transolver references (batch 2, 10M steps); lr_init = 1e-4 is the reference
# crash-example start_lr (method-native; the reference's headline rows used
# Muon, which this port deliberately does not adopt — ADR-0045).
# Not yet run.

[run]
benchmark = "deforming_plate"
seed = 1

[model]
family = "geoflare"
input_frames = 2
dim = 3
n_hidden = 256
n_layers = 6
n_heads = 8
slice_num = 128
mlp_ratio = 4
dropout = 0.0
n_hidden_local = 32
radius_near = 0.05       # standardized-coordinate units (ADR-0045 gap 8)
radius_far = 0.25
neighbors_near = 8
neighbors_far = 32
node_type_size = 9
noise_std = 0.003
normalizer_warmup_steps = 1000
weight_decay = 1e-4      # reference AdamW-arm value
max_grad_norm = 0.0      # OFF — reference example has no gradient clipping

[train]
batch_size = 2
lr_init = 1e-4
lr_decay = 0.1           # UNUSED by the geoflare family (cosine anneal,
                         # ADR-0045); present because [train] is family-uniform.
training_steps = 10_000_000
val_every = 50_000
w_pos = 1.0
w_aux = 1.0
aux_tail_weight = 0.0
train_frames = 0
```

  and `geoflare_smoke.toml` (`# NOT a baseline` header; overrides: `seed = 0`, `n_hidden = 16`, `n_layers = 2`, `n_heads = 2`, `slice_num = 4`, `n_hidden_local = 4`, `neighbors_near = 2`, `neighbors_far = 3`, `radius_near = 0.5`, `radius_far = 2.0` (relaxed so tiny synthetic meshes find neighbors in standardized space), `normalizer_warmup_steps = 5`, `training_steps = 50`, `val_every = 25`; rest as reference).
- [ ] **Step 2: End-to-end smoke test** copying `tests/cli/test_transolver_train_smoke.py`'s fixture + flow (real train + real evaluate at smoke sizes; assert `model-*.pt`, `config.json` family=="geoflare", finite in-memory metrics, `metrics-val.json` exists, no `normalization_stats.npz`, no periodic ckpt at 50 steps). Loader test: both TOMLs load via `load_run_config` (REPO_ROOT-relative pattern).
- [ ] **Step 3: Run (failures here are real integration bugs — fix in place and report). Full suite + gates. Commit** — `feat(configs): deforming_plate geoflare reference + smoke configs; end-to-end smoke test`.

### Task 8: ADR-0045 draft + docs

**Files:** Create `decisions/0045-geoflare-provisional-adaptation.md` (status **Proposed**, Type Durable, Date 2026-08-09 — maintainer finalises at merge); Modify `decisions/0041-v03-deforming-plate-multi-method.md` (dated naming note), `decisions/README.md` (0045 row, exact table format), `docs/ARCHITECTURE.md` (models/ adds `geoflare/`).

- [ ] **Step 1: Draft ADR-0045** from this plan's Design-decisions table + Global Constraints, structured Context / Decision (numbered) / Alternatives (rejected: FLARE-only context-free port; Muon + torch≥2.9 bump; dataset-stats coordinate normalization; padding/mask ragged batching; acceleration+Verlet output; tuple-stream machinery) / Consequences. Must record: the identity chain + upstream `GALE_FE` comment bug; every fidelity pin (scale=1.0 with the upstream comment, implemented as manual attention — torch-floor rationale; randn q_global no-init; clamped temperature + eps 1e-2 vs our thuml-Transolver conventions — BOTH upstream-faithful, per-family fidelity principle; parallel self/cross; 50/50 gate init; nearest-first deterministic ball query vs upstream's order-arbitrary Warp; zero-pad; absolute coords; tanh-outside MLP; two dim_heads 40/32; context part order = code-faithful scales-first-geometry-last WITH the paper-vs-code discrepancy noted (Eq10 lists geometry first; order absorbed by the cross projections either way); slice aggregation implemented as an einsum equivalent to the reference matmul (not byte-level matmul fidelity); single-stream simplification); the declared adaptations (18-ch functional parity, current-frame per-example-standardized geometry, radii in standardized units, harness AdamW/cosine/no-Muon, noise vs reference's none, matched 10M/batch-2 budget); provisional = NO numeric gate + §5 leaderboard comparison statistic; the drift-risk note (reference AR "unstable" at T=50, GeoFlare validated one-shot only — no comparability anchor); Apache-2.0↔Apache-2.0 attribution (docstring-level). Research-basis citation per the ADR-0044 corrected pattern: name the workflow id + tallies + gitignored-scratch caveat.
- [ ] **Step 2: ADR-0041 dated note** (naming: step ③'s "GeoFLARE" = GeoTransolver+GALE_FA, family key `geoflare`, ADR-0045 records the identity) + index row + ARCHITECTURE.md refresh. Every numeric in the ADR must match the code on the branch — CODE governs; discrepancies reported.
- [ ] **Step 3: Gates. Commit** — `docs: ADR-0045 draft (GeoFLARE provisional adaptation) + ADR-0041 naming note + ARCHITECTURE refresh`.

---

## Self-review notes

- Every architecture number traces to `scratch/2026-08-09-geoflare-grounding.md` §4/§6/§10; when a brief value surprises an implementer, that file governs (read-only).
- Type consistency: `GeoFlareNet(node_in, out_size, ...)` ↔ Task 5's `node_in = 18`, `out_size = dim + 1`; `forward(node_feats, coords, n_particles_per_example)` ↔ the simulator's calls; `MultiScaleContext.context_dim` ↔ `GaleFlareAttention(context_dim=...)`; four radius/neighbor scalars ↔ `radii=(radius_near, radius_far)`, `neighbors=(neighbors_near, neighbors_far)` tuple assembly in `build_geoflare_simulator`.
- The Task 4 param-count test derives its formula in-test (structural pin); the task reviewer recomputes independently (Transolver Task-3 precedent).
- Deliberately NOT in scope: `BaselineResult.provisional` + comparison table (separate plan, next); any training run (maintainer compute); Muon; 3D viz; the physicsnemo deforming_plate MGN example (noted in grounding §8/C45 — irrelevant to this port, useful someday as an MGN cross-check).
