# CLAUDE.md

*The operational manual for Claude Code working on this project. Read at the start of every session.*

---

## Purpose

This file is the entry point for Claude Code sessions on StructBench. Most rules and conventions live in other documents; this file points to them and covers the pieces that don't have a home elsewhere.

HARNESS.md carries the principles (*why* rules exist); this file carries the mechanisms (*what* to do). When editing either, content drifting across that boundary should be routed back to its proper home. For what the project is, see VISION.md.

---

## Project snapshot

StructBench is an open platform for establishing whether learned surrogate models of structural response can be trusted — benchmark problems with reference data of declared uncertainty, property tests and evaluation protocols, and reference models that show the tested properties are attainable (ADR-0065, Accepted 2026-09-15, superseding ADR-0014's substrate framing; `docs/VISION.md` keeps its earlier accuracy-comparison wording until the maintainer applies the reviewed rewrite out-of-session). Run by Qilin Li (human) and Claude Code (agent) together under the philosophy in HARNESS.md.

Current stage: **v0.3.0 shipped** — tagged and released on GitHub 2026-08-27 (v0.2.0 2026-08-06, v0.1.0 2026-07-09). The public repo carries four benchmarks — Taylor 2D (ADR-0019), wave-1D (ADR-0025), notch-impact (ADR-0026, 250 µs scored horizon ADR-0039; notch-bend descoped ADR-0056, excluded and removed from the tree 2026-09-23), and `DeformingPlate` (ADR-0041/0042/0043 — the first 3D benchmark, on public MeshGraphNets data) — and four native model families (CGN, MGN, Transolver with an off-by-default Transolver++ variant per ADR-0057 Proposed, GeoFLARE) under one pipeline and one protocol, with cross-method leaderboards on Taylor, notch-impact, and DeformingPlate (wave-1D's registry is CGN-only). Blessed baselines: CGN on Taylor (s1), wave-1D (x1-s1), notch-impact (h250c-s1); MGN on DeformingPlate (noise-fixed, reproduces the published band); Transolver/GeoFLARE ship provisional (ADR-0046), and the DP registry is a family × scheme matrix. Relative L2 is the headline metric (ADR-0055); prediction schemes are an explicit axis (ADR-0051/0053/0054 — time-conditioning is the operators' native scheme). The substrate is built end to end: canonical case schema + HDF5 I/O in `core/`, the general LS-DYNA adapter (ADR-0016) plus the `tfrecord` ingestion adapter (ADR-0042), benchmark cards with generated landing pages (ADR-0027/0036), grouped run configs and per-benchmark results registries (ADR-0032/0033/0046), and the config-driven pipeline (`structbench-train`). The three LS-DYNA canonical archives are public on Hugging Face (`StructBench/{wave-propagation-1d,taylor-impact-2d,notch-beam-2d-impact}`, dataset tag `v0.1.0`, built by `tools/build_hf_bundle.py`; ADR-0040 amended 2026-08-28 — the maintainer's OneDrive stays the master and on-request sharing continues; DeformingPlate is download-and-convert, not rehosted, ADR-0042); blessed-checkpoint archives live in the gitignored `models/` (private, paths recorded in the cards). After v0.3: RC beam stays deferred, a crash benchmark is a v0.4 candidate gated on public data, and the segmented beam stays parked (ADR-0015). See the Roadmap section of README.md for sequencing. **Scope redefinition in progress (2026-09-15)**: ADR-0065 (Accepted) moves the platform's identity to verification and validation of learned surrogates — verification reported before accuracy is compared, cross-method comparison kept as a use rather than the purpose. Its follow-ups (properties block in the registries, card fields and compliance table, task redefinition with a restart family, v0.4 roadmap reframe) are listed there and not yet drafted; existing benchmarks and registered results stand.

**Reference-data verification (ADR-0066, Accepted 2026-09-21) — stages 1 and 2 built and Taylor's record published (2026-09-21).** The `verification/` module (layered `core ← datasets ← verification ← {eval, benchmarks} ← cli`) holds deterministic checks that judge whether a reference *simulation run* is trustworthy. The ADR states a run-evidence requirement (E1–E10) that repository data generation is built to and contributions are measured against; separates `measure` (threshold-free) from `judge` (never touches data); has five verdicts (`pass` / `fail` / `review` / `not_applicable` / `not_assessable` with a typed reason naming the missing evidence); treats solution-verification limits (energy balance, hourglass, added mass) as indicators with sourced reference levels scoped by run traits, where an exceedance is `review` and the call is a person's; and checks units only against things outside the unit label (declared anchors, plausibility ranges, dimensionless groups). The legacy LS-DYNA archives are a test bed for the instrument, never a design input (CORRECTIONS 2026-09-21). It discharges none of ADR-0065's four follow-ups. Design and the 156-claim source dossier are `docs/plans/2026-09-21-reference-data-verification-*`. **Built:** the evidence records in `core/evidence.py`; the fail-closed LS-DYNA readers in `core/io/lsdyna_run.py` (`read_input_facts` for the keyword input, `read_run_evidence` for message-file and global-statistics *text*); the quantity catalogue as data (63 rows, 43 implemented; every row also declares `bears_on` — which artefact a violation condemns: the solver input, the stored response, the run record, or the benchmark's own declaration — a definition, not a measurement, so it stays out of the JSON and nothing was re-measured for it; ADR-0066 catalogue note 2026-09-22); criteria and `judge`; a byte-stable JSON record with a reader-facing Markdown report regenerated from it (`verification/markdown.py`; plain-language titles and `bears_on` live on the catalogue rows), linked from the benchmark's landing page. **The instrument met its second benchmark 2026-09-23**: all 110 notch-impact cases measured, judged and published, now uniform at 16 pass / 1 finding / 8 measured-but-unjudged / 23 not checked / 15 not applicable per case — and the widening's real yield was three defects Taylor could not expose: two in the message-file reader (a warning's own prose counted as a solver error; a `Revision:` line read only in its `SVN Version:` form) and one in the units measures (`not_applicable`, a claim that the input states no strength or modulus, was returned for an input whose material cards the reader could not parse — now `not_assessable / unparsable`). All three fixed, with Taylor's re-measured record content-identical. **The report was reworked for readability 2026-09-22** (presentation only; no verdict, bound, criterion or measurement changed, and the Taylor counts are unchanged): the summary and each finding name the artefact the finding lands in, grouped by artefact; a per-quantity `Spread across cases` (lowest / median / highest / worst case, the median a measured value and never interpolated) replaces the per-case table, which did not scale to notch's 110 or DeformingPlate's 1200 cases; the `What could not be checked` section is gone because every row already carried its reason; ratios within one percent of unity carry their deviation and sub-thousandth percentages read `< 0.001 %`, while bounds are formatted exactly and never approximated. Taylor's report went 225 to 206 lines, 4643 to 3673 words; and `python -m structbench.cli.datacheck measure|judge`. Run evidence reaches the CLI only as the whitelisted JSON record that per-dataset glue writes (`data_generation/lsdyna/2D-Copper-Bar-Taylor-Impact/collect_run_evidence.py`), never as a run directory. On the Taylor test bed the instrument reproduces the facts recorded before it existed and reports two archive-level failures identical across all 33 cases (a non-monotone hardening knot, and the rigid wall's shell, an element block no input part owns; the terminal off-interval frame is reported as a fact, not judged); the energy rows are measured but get no verdict, because no sourced level covers a particle formulation of unknown conservation class. **Coverage widened 2026-09-23 (ADR-0066 coverage note, DRAFT)**: `stored_globals_match_ledger` is built and found that Taylor's stored `global/total_energy` is not the solver's total — it equals kinetic+internal to 2e-6 while the printed total also carries the rigid-wall term, so the two disagree by 0.6-1.2 % and the row reads `fail`; the adapter copies d3plot's channel verbatim, so it is a mismatch between two solver outputs and what to do about it (schema, every archive) is the maintainer's. `input_requests_required_evidence` is new and reads `STANDARD_INPUT_BLOCK.md` back off the deck, gated on the model's features, closing a silent-acceptance path (an uncomputed `*CONTROL_ENERGY` term never reaches `glstat`, and the ledger identity is built over the terms present, so it was accepted as complete); both sweeps fail it — Taylor omits `*DATABASE_RWFORC` despite its rigid wall, notch computes every energy term and never requests `glstat`. `read_input_facts` gained `energy_terms_computed`, `databases_requested` and `damping_defined`. Units anchors now live on `BenchmarkCard.units_anchors`; Taylor declares one (copper 8.9e3 kg/m3, 2 digits) and notch one (concrete 2.4e3 kg/m3), both passing — one anchor pins mass against length but not time, and a second needs a value the maintainer stands behind. Taylor now: 20 pass / 4 findings / 17 unjudged / 7 not checked / 15 n/a. **No sourced reference level is ratified** (maintainer decision 2026-09-21): indicators are reported as measurements with the published level shown for context, and verdicts come from definitional requirements and instrument tolerances only. The published records are `docs/datachecks/taylor_impact_2d.{json,md}` and, since 2026-09-23, `docs/datachecks/notch_beam_2d_impact.{json,md}` (ADR-0066 widening note); a data-free test keeps each `.md` equal to what its `.json` and the criteria generate. Working output goes to the gitignored `runs/datachecks/`. **Open:** the kinetic-energy closure is measured on sample instants found in the files and has no tolerance yet; the other three E9 rows (internal-energy closure, sampling clock, stored globals) are not built; the instrument has met Taylor and notch-impact, and a third benchmark is not scheduled (the maintainer declined wave-1D for now, 2026-09-21, and its runs kept no `glstat` either); **ADR-0067 (Accepted and built 2026-09-23) settles the two material classes** — `concrete_damage` and `elastic_plastic_kinematic`, four of five fields measured rather than recalled (the K&C slot is the scaled damage measure on exactly [0, 2] across 22 cases; monotone with zero decreases in ~66M samples; no `*EOS_*` card; `yield_law` `not_assessable` because von Mises is not a function of the stored scalar — fully damaged C50 concrete reaches 413 MPa where confinement reaches 422 — and because `a0 = -0.05` has the solver generate the surface coefficients, so they are absent from the input). The build moved eight notch rows off `unsupported`: `state_variable_min` and `state_variable_decrease_max` to **pass** — this ADR's monotonicity claim holding on all 110 cases, not just the 22 it was measured from — `eos_closure` to `not_applicable`, and the five yield rows to `not_assessable / not_available_from_solver`. That last needed a gate fix the build exposed: a class saying `yield_law="not_assessable"` answered the trait gate the same as a material with no yield surface at all, so the rows read `not_applicable` — implying yield was irrelevant to a concrete impact benchmark rather than unevaluable (ADR-0067 build note). **A claim audit followed (ADR-0066 claim-audit note 2026-09-23)**: twelve sites where the module reports a quantity absent or inapplicable, eleven true, one not — `input_dimensionless_groups_plausible` read `unsupported`, blaming the instrument for a ratio the *input* does not state, and `unsupported` is in `PLATFORM_REASONS` so the row also exempted the data from its own gap. Fixed. Four such misreports in one week: **`not_applicable`, and the reason on an absence, are claims about the data and must be checked like any other.** Notch now: 18 pass / 1 finding / 8 unjudged / 19 not checked / 17 n/a. Of the 19 still unchecked, 13 are `source_missing` (no ledger, no time-step history, no load resultants) and beyond any class. The *card layouts* landed 2026-09-23 without an ADR (a layout says what numbers a card holds; only a class says what they mean) and were worth five rows: notch's stored density now matches its input to 2e-16 and its declared density anchor passes, so `kg-mm-ms` is confirmed from outside the label. Verification keys on the deck parsed fresh at measure time, so adding a class needs no re-conversion; the instrument tolerances are provisional; ratifying any reference level, and one unpublished conformance run with the standard LS-DYNA input block (`data_generation/lsdyna/STANDARD_INPUT_BLOCK.md`, a draft whose open points that run settles) with the standard input block remain the maintainer's to decide. Real-data acceptance tests are env-gated (`STRUCTBENCH_DATA_ROOT`, `STRUCTBENCH_TAYLOR_RUN_DIR`) and skip when unset.

**Data generation is a StructBench capability (ADR-0071, Proposed; part one built and merged 2026-09-27).** Abaqus/Explicit is the second solver (ADR-0068, Proposed) and its pipeline, first built as shared scripts under `data_generation/abaqus/` (ADR-0069, Proposed: four stages, the `abaqus-npz/1` intermediate that the solver's own Python writes, scipy as the `datagen` extra), now lives in the package as `structbench.datagen` with the per-solver subpackage `datagen/abaqus` (`deck` writers; `odb_export.py`, Python-3.10 package data handed to `abaqus python`). One entry point, `structbench-datagen`, carries the stages `new check preflight generate run follow export convert verify converge archive` (`verify` runs the ADR-0066 instrument; it was `validate` until ADR-0072, 2026-09-27, gave that word to comparison with experiment). **A dataset is a definition the pipeline consumes, never code in the repository**: `dataset.toml` (tables `[dataset] [declaration] [fixed] [variables] [regions] [splits] [limits] [levels] [pilot] [qoi] [retention]`; splits marked `probe = true` are pilots and convergence sets, the rest run at `[levels].production`, which `plan_cases` fills in) and `problem.py` (hooks `input_deck(params, variant)`, `mesh`, `qoi`, optional `feasible`). `new` scaffolds one from the public example (`datagen/examples/abaqus_conformance`, the single-rod conformance case), and `check` validates it without a solver: the tables, everything `generate` would refuse, a byte-stable deck (rebuilt in a fresh interpreter under another hash seed, so a deck that walks a set or dict is caught), nesting mesh levels, QoI names against the declaration — every fault a sentence naming the field, never a traceback. Provenance is `abaqus-provenance/3`: the deck's sha256 plus LF-normalised hashes of both definition files, StructBench's version always and its commit only from a checkout, the dataset repository's commit and dirtiness, and since part two (b) the `preflight` record (the stamp's hash and date, `{"skipped": true}` under `--no-preflight`, null for a probe split) and a preflight case's `probe` role. Writer arguments (`explicit_step(scale_factor=)`, `contact_damping`) replaced the string edits datasets made to their decks. Docs: `docs/DATA_GENERATION.md` (a stub until part two), `docs/datagen/abaqus-conformance.md` (the Abaqus standard input block, moved from `data_generation/abaqus/`), `docs/datagen/abaqus-lessons.md` (the first Abaqus dataset's lessons as platform guidance with a pre-production checklist, study numbers removed; added 2026-09-27), the layering in ARCHITECTURE.md (`datagen` depends on `core`, `verification` and `validation` only, held by an import-boundary test since part two moved `declared_from_toml`, `measure_cases` and `input_facts_for` into `verification/dataset.py`). The maintainer's private datasets stay out of the repository and are passed by path (`--dataset <dir>`); the first one was migrated to the contract with every deck that ran rebuilding byte for byte against its provenance, and no dataset name, split, case id or constant may appear in public files — commit *trees*, not just the working tree, are scanned before a merge. The Abaqus work also gave verification a definitional requirement, `plastic_dissipation_excess_max` (plastic dissipation may not exceed internal energy, tolerance 1e-5; ADR-0066 energy-account note), which exposed kinematic contact creating energy at the full stable increment — a defect three earlier sweeps had passed silently — and the run adapter `core/io/abaqus_run.py` (end-of-step landing row dropped). **Part two (a) built 2026-09-27:** `structbench-datagen converge` — runs paired across run roots by their parameters (the production run stands for its level, duplicates named), each QoI extrapolated from three levels in constant ratio, every stored field of each coarser level measured against the finest, a byte-stable `convergence-record/1` with Markdown — on the engine now in `verification/convergence.py` (Richardson with statuses, nested-node matching, restriction with axisymmetric or planar weights per `[levels].symmetry`, field errors on cases); the headline metric `relative_l2_pooled` moved to `verification/kernels.py` with `eval.metrics` re-exporting it; the layering exception is gone (`verification/dataset.py` holds `declared_from_toml`, `measure_cases`, `input_facts_for`; `datagen` imports `core`, `verification`, `validation` only, held by a boundary test); `problem.py` may import its siblings. **Part two (b) built 2026-09-28 (plan `docs/plans/2026-09-27-datagen-platform-plan-2b.md`):** `structbench-datagen preflight` — the pilot split's points become a case set (every pilot at every `[levels].pilot` level but the finest, the `fine_cases` there too; the pilots at production with the solver increment scaled by `[pilot].increment_factors`; one pilot exported at `frame_factor` times the stored frame interval; one conformance run with `*ENERGY OUTPUT, VARIABLE=ALL`, widened by the writer itself), run in `<work-root>/<name>/preflight/` as a sweep of its own with ids suffixed `-L<level>`, `-T<value>`, `-F`, `-E`, and ten steps turn the records of `run`, `export`, `convert`, `verify` and `converge` into ADR-0066 verdicts: deck regression (byte-stable in a fresh interpreter, and every probe key changes the deck), feasibility, conformance (the ledger identity closes to 1e-5 and no term outside it is non-zero), space (each QoI at production within `[qoi].tolerance` of its extrapolation; `review` without an observed order), increment, frame (the midpoint-interpolation error of the finer export, on node fields and globals, against `frame_tolerance`), duration (settling of the QoIs and the separation frame of `contact_force_global` against `settling_margin`), energy, budget (median bytes and wall per production-level pilot × production cases against the free space plus `min_free_gb`) and verification (no fail outside `accepted_gaps`). The stamp `preflight-stamp/1` binds the verdicts to both definition hashes and to the sibling modules `problem.py` imported; `generate` refuses a split that is not a probe without a passing stamp for the current definition (`--no-preflight` records the omission); the production-level pilots and the conformance run go first and a step that does not pass there launches nothing more; re-running resumes and a preflight folder holding runs of another definition (either file or a sibling, decks changed or not) is refused, never cleared. The review's fix pass (2026-09-28) also made every probe key a `[fixed]` constant that no split may sample, pin or point (the increment probe is `not_assessable` without it), the frame step judge the clock `1 / frame_factor` names, non-finite values never pass, a `qoi()` that raises on a prefix a finding rather than a crash (`check` exercises a one-frame case), and acceleration reported but not judged by the frame step (a ruling the maintainer may reverse). With it: the temporal measures in `verification/temporal.py` (settling, separation, rise time, midpoint-interpolation and common-instant errors); `run --min-free-gb` (default the stamp's) halting cleanly with exit 3 and printing the stamp's estimate; `follow` exporting and converting finished cases beside a run; `converge --anchor` and `--cases`; `export.py` split out of `cli.py`; and the whole chain tested against a fake solver script (`tests/datagen/fake_solver.py`). No real Abaqus run of the stage has happened yet; that is the maintainer's to schedule. **Owed by part three:** the Abaqus `input_requests_required_evidence` and hourglass rows, ratifying the energy-gain indicator, the card generator, the full guide; the whole-branch review's eleven minors from part one were deferred and are recorded here and nowhere else: `check` nests each level against the first only; the QoI-name check is order-sensitive; `[pilot].fine_cases` and `accepted_gaps` are not checked against planned case ids or catalogue rows; `new .` derives an empty name; `[declaration]` is validated only by `validate`, not by `check`; `provenance.json` is written CRLF on Windows; a stale module path in `core/io/abaqus.py`; a relative link and an old collect command in the conformance document, and `sampling.py`'s docstring still says `sweep.toml`; one paragraph of LS-DYNA context lost from `data_generation/README.md`; a missing scipy raises an ImportError instead of an install hint, and exit codes 2 and 3 are used inconsistently across stages; and the private dataset's `problem.py` still carries literal keyword text for probe splits. Material classes for Abaqus: ADR-0070 (`elastic_plastic_isotropic`, Proposed).

**Validation against experiments (ADR-0072, Proposed; built 2026-09-27).** `structbench.validation` sits beside `verification` and depends on `core` only: reference-experiment sets as package data with their provenance (sources with DOIs checked against Crossref, whether each was consulted directly, the licence the curves travel under, the extraction tool and its checks, caveats); measures that apply the *same* function to a measured outline and to a canonical case's deformed boundary (`measures/taylor.py`: L_f, R_f, W_f at the source's fractions, the outline chained from the mesh's boundary edges, a settled-tail band on the length); `compare` (pairs of test and canonical case per variant, deviations per test, min/median/max per variant, `aborted`/`missing`/`unmeasurable` named and never skipped); a byte-stable `validation-record/1` JSON with a Markdown rendering; and one command, `structbench-validate --reference NAME --pairs pairs.toml --setup setup.toml --out DIR` (`--list` names the sets). **It reports and never judges** — no acceptance level is ratified, for ADR-0066's reason. The first set, `taylor_copper`, is the six copper Taylor tests compiled by Zelepugin, Cherepanov & Pakhnutova (2023, CC BY 4.0; tests from Wilkins & Guinan 1973, Gust 1982 and Zelepugin et al. 2022 as that paper reports them), outlines read exactly from the paper's vector figures by `tools/validation/digitize_zelepugin2023.py` (PDF path as argument; the PDF is not in the repository; the source's own simulated curves are not redistributed); a test holds the stored measures to the functions on the stored outline. The datagen stage `validate` became `verify` in the same ADR, so *verify* is the instrument and *validate* is comparison with experiment everywhere. A dataset's record stays with the dataset until admission (the first Abaqus dataset's is private, regenerated with the public command from re-run probe cases). **Open:** a `BenchmarkCard` field and landing-page link with the first public record; an LS-DYNA Taylor record needs runs at the experiments' conditions (the maintainer's call); a figure in the reference companion; a `/2` format for time-resolved measurements.

---

## Session workflow

### Starting a session

Read these files, in order, before any work begins:

1. `CLAUDE.md` (this file).
2. `docs/VISION.md`.
3. `docs/HARNESS.md`.
4. `docs/PRINCIPLES.md`.
5. `docs/CORRECTIONS.md` — all entries marked `active`.
6. `decisions/README.md` — the ADR index.
7. `docs/WORKFLOW.md` — session venues and multi-machine git workflow; identify your venue before making any change.

Then, conditionally based on the session's task:

- `docs/ARCHITECTURE.md` — if the task touches the package structure, module interfaces, or the case schema.
- Specific ADRs from the index — whichever are relevant to the task.
- The Roadmap section of `README.md` — if the session is about planning or scoping.
- The maintainer's private research documents — kept **outside the repository**, never tracked; consulted only when the task touches scope, benchmark admission, or the roadmap, and only when present (the maintainer points the session at them). *Context-only: they guide the maintainer's research, of which StructBench is one instrument, and do not define StructBench scope — that is `docs/VISION.md` and the ADRs alone (ADR-0065).*

If the session's task is not clear from the opening message, ask the human what the session is for before beginning work.

Target: the full start-of-session reading should take under 10 minutes of agent time.

### During a session

- **Default to asking when ambiguous.** Silent resolution of ambiguity is how invariants erode.
- **Draft ADRs immediately when decisions are made**, not at end of session.
- **Flag scope expansion.** If the task has grown beyond what was originally requested, say so.
- **Break complex work into checkpoints.** Pause for confirmation at natural boundaries.
- **When corrected, ask whether to log to `CORRECTIONS.md`.**

### Ending a session

- Commit changes to a feature branch; `main` moves only on the human's explicit in-session instruction (ADR-0023).
- Unfinished work persists as `WIP:`-prefixed commits on a feature branch, or as dated notes in `scratch/`.
- After session end, a reader of the repo files (without chat history) should be able to reconstruct what was decided and what was done.
- No formal session summary required; commit messages serve as the record.

---

## Authority tiers

Four tiers govern what Claude Code can do. When in doubt, default to the more restrictive tier.

### Unilateral — do without asking

- Writing, refactoring, or deleting code within existing modules, if the public API doesn't change and no dependencies are added.
- Writing or updating tests.
- Running tests, linters, formatters, the CLI.
- Creating or modifying docstrings and code comments.
- Fixing obvious bugs with local fixes.
- Installing already-approved dependencies in a local environment.
- Reading any file in the repo.
- Writing scratch notes in `scratch/` (gitignored).

### Flag-first — propose and wait for confirmation

- Adding, removing, or upgrading any dependency.
- Modifying the public API of any module.
- Modifying the case schema in any way.
- Creating new top-level modules or new files at repo root.
- Drafting or modifying an ADR (the human finalises).
- Architectural changes affecting how modules interact.
- Deletions exceeding ~50 lines of non-trivial code.
- Running anything with real compute cost — propose with estimated cost and runtime.
- Changes to git state affecting history.

### On explicit instruction — execute when the human directs it in-session

*(Added by ADR-0023, amending ADR-0006.)* These never happen as part of unprompted work — `main` moves only by the human's word — but when the human explicitly instructs them in the session ("merge it", "push"), Claude Code executes them directly instead of handing commands back.

- Merging a feature branch into `main`.
- Pushing to the remote.
- Committing directly to `main`.
- Hugging Face data-release actions on the `StructBench/*` dataset repos — create a repo, upload, flip public, create a *data* tag (ADR-0023, amended 2026-08-28). The token stays the human's: they log in, Claude only invokes the CLI.

### Forbidden — refuse even if asked in-session

These require deliberate human action outside a normal coding session.

- Publishing *code* releases — GitHub releases, code version tags, uploading to PyPI or Zenodo. (Dataset releases on Hugging Face are on-instruction, above.)
- Modifying `LICENSE`, `HARNESS.md`, or `VISION.md` during a coding session.
- Rewriting git history on shared branches.
- Accepting or merging third-party pull requests.
- Changing repository settings.
- Handling secrets, credentials, API keys, or SSH keys.
- Asserting facts about external sources without verification.

---

## Corrections handling

Small corrections that don't warrant an ADR are logged in `CORRECTIONS.md`. Format and workflow specified in that file's header. In-session behaviour:

- When the human corrects something that could plausibly recur, ask: *"should I log this to `CORRECTIONS.md`?"*
- On confirmation, add the entry before continuing.
- Active entries are read at session start and inform behaviour throughout the session.

---

## Where other rules live

- **Coding conventions** (Python version, style, testing, documentation, logging, git): `docs/PRINCIPLES.md`.
- **Repository structure and package layout**: `docs/ARCHITECTURE.md`.
- **Case schema**: `docs/ARCHITECTURE.md`.
- **Dependency policy and approved list**: `docs/PRINCIPLES.md`, with individual additions recorded as ADRs.
- **ADR format and process**: `decisions/README.md`.
- **Session venues and multi-machine git workflow**: `docs/WORKFLOW.md`.
- **Long-term trajectory**: the Roadmap section of `README.md`.

If a rule seems missing from all of these, flag it rather than guess. It may belong in one of the existing documents, or it may indicate a gap the harness doesn't yet cover.

---

## Common situations

**The human asks me to do something that crosses into flag-first territory.** Propose with reasoning, wait for confirmation.

**The human asks me to do something forbidden.** Refuse, explain why, suggest the correct out-of-session path.

**I want to do something the rules don't cover.** Flag the ambiguity. The resolution is either a quick human answer or a new `CORRECTIONS.md` entry.

**I realise mid-task that the approach is wrong.** Stop, say so, propose an alternative. No silent pivots.

**The rules here conflict with `VISION.md` or `HARNESS.md`.** The philosophical documents take precedence. Flag the conflict so this file can be revised.

**The rules here feel wrong for the current task.** Flag it. Do not override silently. If the rule is genuinely a bad fit, the path is to revise this file.
