# 0065 — StructBench is a verification-and-validation platform for learned surrogates (supersedes 0014)

**Status**: Accepted (maintainer, in-session 2026-09-15)
**Type**: Durable
**Date**: 2026-09-15

## Context

ADR-0014 (2026-05-23) placed StructBench as the *substrate layer* of a
three-layer research program — substrate, brain (foundation models), body
(agentic monitoring systems) — documented in `RESEARCH-PROGRAM.md`, and
ruled that the program informs direction but does not define StructBench's
scope. Two things have changed since.

**The program was rewritten.** `RESEARCH-PROGRAM.md` was retired in
September 2026. The maintainer's research program now lives in private
documents kept outside the repository (per the 2026-08-12 correction that
keeps research strategy out of the public tree); they guide the
maintainer's research, of which StructBench is one instrument, and much of
their content is not about StructBench. Their thesis, as far as it concerns
the platform, is that the bottleneck for learned surrogates in structural
engineering is no longer building them but establishing that they can be
trusted for an engineering decision: physical admissibility of their
outputs, trustworthy sensitivities, bounded behaviour outside training, and
reference data whose own uncertainty is quantified. That asks StructBench
to carry a verification-and-validation (V&V) role rather than an
accuracy-comparison role. The substrate/brain/body triad no longer exists
in any document.

**The code moved first.** Every ADR since late August has been about
carried state and physical consistency: complete-state channels (ADR-0059),
state feedback and its stabilisation (ADR-0060/0061), the anchored flow map
as a state-carrying simulator (ADR-0062/0063), and constitutively structured
heads that make yield admissibility and plastic-strain monotonicity hold by
construction (ADR-0064). ADR-0064 measured that the direct von Mises head
emits about a tenth physically impossible stresses, undetectably. Yet those
diagnostics reach neither the `eval/` package nor the results registries;
the public documents (VISION.md, ADR-0041's cross-method headline,
ADR-0055's relative-L2 headline) still describe a platform whose product is
an accuracy ranking. That is the drift HARNESS.md exists to prevent: the
project's identity changed in practice and the files did not follow.

Two constraints shape the decision. `VISION.md` and ADR-0014 are the only
documents that define scope, and `VISION.md` is forbidden-tier in a coding
session, so the rewrite is the maintainer's out-of-session act and this ADR
records it. And the research documents are private: they may reach the
repo only through ADRs that restate what is public-safe, never by being
read as a to-do list.

## Decision

1. **StructBench is an open platform for establishing whether learned
   surrogate models of structural response can be trusted.** `VISION.md`
   is rewritten to that identity by the maintainer (draft reviewed
   in-session 2026-09-15). Its opening paragraph becomes:

   > StructBench is an open platform for establishing whether learned
   > surrogate models of structural response can be trusted. It provides
   > benchmark problems with reference data whose uncertainty is declared,
   > evaluation protocols that test physical consistency and
   > decision-relevant behaviour alongside accuracy, and reference models
   > that show those properties are attainable — so that research groups
   > and practitioners can verify and validate data-driven simulation in a
   > consistent, reproducible way.

   Cross-method comparison is retained as a use of the platform, not as
   its purpose: researchers may still compete architectures on StructBench
   benchmarks, and the platform no longer treats that as its focus.

2. **`VISION.md` and the ADRs are the only definition of StructBench's
   scope.** The maintainer's private research documents are the origin of
   this decision, not a layer above it. They are untracked, absent from
   clones, and nothing in the repository may depend on them: anything the
   platform needs must be readable from the repository, where the private
   documents supply reasons and ADRs record decisions. They reach the repo
   only through ADRs that restate what is public-safe. Where they and the
   repository disagree, `VISION.md` and the ADRs govern StructBench, and
   the disagreement is a signal to revise one side — never a silent
   override in either direction. ADR-0014's standing reading-list item for
   a program document is retired with it; the private documents are
   consulted as context when a session's task touches scope, benchmark
   admission, or the roadmap, and only when present.

3. **The substrate-layer litmus test is kept, restated for the V&V role.**
   A proposal for new work is gated by its primary output:

   > - **Reusable V&V infrastructure** — benchmark problems; reference data
   >   with declared uncertainty (convergence, noise floor, provenance of
   >   discarded cases); the canonical solver-agnostic format and
   >   data-generation tooling; property tests and diagnostics; evaluation
   >   protocols, including the tagging of each metric with the decision
   >   it protects; and reference baselines — *including baselines built
   >   to satisfy the tested properties by construction*, whose role is to
   >   show the properties are attainable and to calibrate the tests.
   >   → **In StructBench.**
   > - **A scientific contribution** — a model whose existence is the claim
   >   of a paper, a scientific finding, a system deployed on a real asset.
   >   → **Outside StructBench**, in a separate repository or paper.
   >   StructBench may evaluate such artefacts and report their results;
   >   it does not host them.

   As in ADR-0014, *role* decides, not novelty or training cost. Under
   this test ADR-0064's structured heads are in scope as reference
   baselines for the admissibility tests, not as a research artefact.

4. **Verification is reported before accuracy is compared.** Every
   registered result will carry a properties block (constraint
   satisfaction, state sufficiency, closure, error growth with horizon, as
   applicable to the benchmark), and accuracy is read within the set of
   results that report it. The mechanism — which properties, which
   thresholds, how the registry and landing pages render them, and whether
   any property becomes a gate — is **not decided here**; it is a follow-up
   ADR amending 0032/0033/0055. This clause fixes the direction only.

5. **The layer vocabulary changes.** Where project documents need to place
   StructBench in the wider program, the three layers are *data* (is the
   simulation data itself trustworthy), *model* (is the surrogate's output
   trustworthy), and *reality* (does the simulation correspond to the
   physical world). StructBench hosts the model layer and the data-standard
   side of the data layer; the reality layer enters where a benchmark
   carries physical-test data. The substrate/brain/body terms are retired;
   their appearances in ADR-0015 and ADR-0017 are history and stand.

## Alternatives considered

- **Keep ADR-0014 and the comparison-platform identity; add property
  metrics as extra leaderboard columns.** Rejected. The public identity
  would go on saying that a better accuracy ranking is progress while the
  code, the results, and the program say otherwise. Columns without an
  identity change are exactly the slow-erosion pattern ADR-0014's own
  alternatives section warned about.

- **Start a separate V&V project and leave StructBench as the
  leaderboard.** Rejected. The V&V tests need the same reference data,
  canonical format, protocols, and baselines; two repositories would
  duplicate the substrate and split a small community. The platform is
  the right home precisely because the tests are infrastructure.

- **Fold the private research documents into `VISION.md`.** Rejected,
  for ADR-0014's reason and one more: it erases the program/platform
  separation the litmus test depends on, and it would move private
  research strategy into the public tree, which the 2026-08-12 correction
  forbids.

- **Amend ADR-0014 in place with a dated note.** Rejected. This is a
  reversal of its framing — the program's demands now shape what the
  platform measures, and the layer model it rested on is gone. Durable
  ADRs are superseded on genuine reversals (decisions/README.md).

## Consequences

- **`VISION.md` is rewritten by the maintainer outside a coding session**
  from the reviewed draft (`scratch/2026-09-15-vision-draft.md`, local).
  The ADR was accepted in-session on 2026-09-15 ahead of that rewrite, so
  ADR-0014 is superseded from this date and `VISION.md` carries its
  earlier wording until the maintainer applies the draft; `CLAUDE.md`'s
  snapshot says so until then.

- **Reading list and manual.** ADR-0009 (ephemeral) takes a dated note
  removing `RESEARCH-PROGRAM.md` and `research/FINDINGS.md` from the
  always-read items 3–4 (both retired) and adding the maintainer's private
  research documents to the *conditional* list — read when the task
  touches scope, benchmark admission, or the roadmap; skipped without
  error when absent. `CLAUDE.md`'s reading list and *Project snapshot*
  paragraph are updated to match, and the `WORKFLOW.md` publication note
  records that the retired files and the private research documents live
  outside the repository.

- **Follow-up ADRs, each flag-first and not decided here:**
  1. *Protocol* — the properties block, decision tags on metrics, and the
     rendering of both in registries and landing pages; amends
     0032/0033/0055.
  2. *Benchmark card* — structural scale, load type, autonomy, parameter
     space, ground-truth regime with its uncertainty, and a data-standards
     compliance table per benchmark; amends 0027.
  3. *Task definition* — the benchmark task becomes trajectory prediction
     from initial state and parameters with the prediction scheme declared,
     and a restart task family that only state-carrying simulators can
     serve is added; amends 0019/0025/0026/0043. Agreed in principle
     2026-09-15.
  4. *Roadmap* — v0.4 is reframed around 1–3, a gradient benchmark, and a
     metal component-tier benchmark; the crash benchmark becomes an
     assembly-tier question. README Roadmap section.

- **Existing benchmarks and registered results stand.** They are not
  withdrawn or re-scored by this ADR. Under the data standards the V&V
  role asks for, none of the four shipped benchmarks would be admitted today
  (only Taylor has a convergence run and a measured noise floor; none has a
  survivor log, a physical-test anchor, a cross-solver reproduction, or a
  constitutive sensitivity study). Follow-up 2 makes that visible as a
  compliance table rather than hiding it; existing benchmarks are
  grandfathered and new entries meet the standard.

- **Unchanged.** HARNESS.md, PRINCIPLES.md, the case schema, ADR-0004's
  solver-agnostic commitment, ADR-0023's git authority, and the
  prediction-scheme axis (ADR-0050–0054). The platform still ships no
  solver, no general ML framework, and no model whose existence is a
  paper's claim.

## Note on follow-up 1: what the properties block is for (2026-09-23, maintainer + agent)

Clause 4 commits to a properties block and fixes the direction only. This
note records the question it answers, so that follow-up 1 is drafted against
a stated purpose rather than assembled from whatever metrics exist. It
decides nothing: no property, threshold, rendering or gate is settled here.

**The platform should be able to answer whether a surrogate is trustworthy
enough to enter a structural-engineering workflow — design optimisation, a
digital twin.** That question does not reduce to a ranking, because it is
three independent questions:

1. **Accuracy** — how close the prediction is to the reference. What the
   leaderboard reports today (ADR-0055).
2. **Physics-consistency** — whether the prediction violates laws it cannot
   legitimately violate, *however close it is*. Independent of 1 by
   demonstration: ADR-0064 measured the accuracy leader emitting 10.3 %
   physically impossible stresses, with D2 at 6.4 % and D3 at 42.8 % before
   enforcement. Accurate and inadmissible at once.
3. **Fitness for a downstream use** — which neither of the first two
   settles, because the failure modes differ by use. An optimiser walks
   off-distribution by construction and needs sensitivities to be right; a
   digital twin runs horizons far beyond anything scored and never has
   ground truth to check against.

That a single ranking misleads for use 3 is already visible in this
repository. On notch-impact, `rollout_rel_l2_disp` on the interpolation
split orders Transolver-TC 0.0352 < Transolver-AR 0.0641 < MGN 0.2245 <
CGN 0.2827; on the four-axis OOD probe the order **inverts** — CGN 0.5905 <
MGN 0.7672 < TC 1.047 < AR 1.185 — and a relative L2 above one means the
error exceeds the signal. The model that wins in-distribution by roughly
eight times is the one to trust least where an optimiser would take it.

Two structural observations, recorded because they worsen with time rather
than with scope:

- **The same property is already implemented twice.** Yield admissibility
  is a kernel over plain arrays in `verification/kernels.py`, serving
  reference-data verification, and is enforced again inside
  `models/transolver/simulator.py`, serving one model family. One
  definition, two implementations, no shared contract. Each further family
  adds another copy.
- **Nothing in the repository can say "do not trust this particular
  prediction."** Every metric in `eval/metrics.py` and every benchmark QoI
  requires the ground truth, and there is no calibration or uncertainty
  machinery anywhere in `src/`. The physics-consistency properties are the
  only checks computable on a prediction alone, which is what would make
  them runnable in a deployment at all.

The open research question this frames, and which the platform is unusually
well placed to answer with four benchmarks, held-out and OOD splits, several
families and two prediction schemes: **do the properties computable without
ground truth predict the error that needs it?** If they do, there is a
runtime trust signal and the trustworthiness question is answerable. If they
do not, physics-consistency is hygiene worth enforcing but is not evidence
of trustworthiness — and that is a finding worth reporting either way.

A note is the right weight for this: it is direction, not decision, and
follow-up 1 remains flag-first and undrafted.
