# Before production: lessons from the first Abaqus dataset

*Platform guidance under ADR-0069 and ADR-0071, distilled on 2026-09-27 from
the private record of the first Abaqus/Explicit dataset — an axisymmetric rod
on a rigid wall, several hundred short runs, produced four times before the
setup was right. The study's own numbers stay with the study; what travels
here is what happened in kind, the rule it left, and what the pipeline now
does about it. The solver facts themselves are in
[`abaqus-conformance.md`](abaqus-conformance.md); the stages and the dataset
contract in [`../DATA_GENERATION.md`](../DATA_GENERATION.md).*

The pipeline never failed. The *setup* did, four times: a severity limit
chosen after production, an energy term the deck did not request, a mesh
chosen before its convergence was measured, a contact enforcement that created
energy without a solver warning, a fix checked on one axis and not the other,
and a disk that filled. ADR-0071's preflight gate exists because of this list.
This page keeps the reasoning, so the checks are understood rather than obeyed.

---

## 1. What the solver actually writes

**Treat every keyword and every output as a hypothesis until one small run
shows it, and name the file each fact came from.** The first conformance run
contradicted the plan on nine points: the wording of the completion and abort
banners; that Explicit writes its diagnostics to the `.sta`, not the `.msg`;
that the `.sta` increment table is the stable-increment history; that
`TIME MARKS=YES` gives the frames on the grid *plus a duplicate end-of-step
frame*; that stored fields are float32 even with `double=both`; that CAX4R has
one integration point; that a rigid body's reference node carries no field
output; that 2D coordinates come with three columns; and that `odbAccess`
returns `None` for a missing label. *Now:* the conformance document records
each fact with its source, and the adapter (`core/io/abaqus.py`) is built to
them. `check` cannot see solver output, which is why the preflight's first step
is one run.

**A "same data" claim is checked on every stored series, across all runs.**
"The duplicate end frame is identical" had been checked on displacement only;
at conversion the acceleration differed in every run, and a handful of runs
differed by float32 last-bit amounts in other series. *Now:* the adapter drops
the duplicate frame only after checking every series but the acceleration to a
stated tolerance, and says so in the case.

**Request `*ENERGY OUTPUT, VARIABLE=ALL` once, establish the identity, then
request exactly the terms it needs.** The ten energy terms first requested did
not add up to the total; one diagnostic run with everything requested found the
missing one (the contact penalty work, non-zero even with kinematic contact
against an analytical wall). With it the identity closes to float32. *Now:* the
example deck requests it, and ADR-0071 extends
`input_requests_required_evidence` to Abaqus decks so an unrequested term is a
finding rather than a silent gap.

**Licence text sits in run files.** The runner's log and the `.dat` header
name the licensee, host, site and expiry. *Now:* `archive` never copies the
runner's log, redacts the `.dat` and `.msg` headers and any licence line, and
the evidence readers keep only numbers and version tokens. Scan any new file
type for licence lines before archiving or sharing it.

## 2. Physics and feasibility

**Declare feasibility in the model; do not shrink the box by hand.** Pure
Lagrangian elements at the severe corner of the box inverted near the impact
face and the runs aborted. Probing the boundary and declaring a limit in
`feasible()` keeps every split filling from its own Sobol draw, removes only
the infeasible sliver, and states the limit in the card. *Now:* the `[limits]`
table and the `feasible` hook of the dataset contract.

**Test a fitted boundary on the points it is meant to protect.** The first
fitted limit, linear in a shape ratio, was wrong: of the production points it
called at risk most aborted, including two it had called safe. A constant in a
dimensionless severity group fitted the probes, and the feasible points nearest
it were run before production. *Now:* the preflight's feasibility step runs the
pilots at the production level and prints the limit with the pilots' values.

**The limit belongs to one mesh.** A point that passed at the production mesh
aborted at half the element size. Choose convergence cases away from the
limit, or report the ones that abort.

**Look at an animation before production.** "No mushroom" turned out to be a
frictionless wall's flared lip, and a colleague's expectation came from a
different model; the answer was to keep the wall frictionless and write the
reason down. *Now:* `viz.fringe` draws filled elements for exactly this look.

**Contact type and the energy-gate definition are one decision, made before
production.** Kinematic contact removes the impact-face row's kinetic energy
at impact (an amount that scales with the element size); penalty contact
removed that loss but its wall force did not converge with the mesh; kinematic
contact with the stable increment halved was clean on both counts. The choice
was settled only by running the same cases at two meshes and two increments.
*Now:* the preflight's energy step runs at more than one stable-increment scale
(ADR-0071, note of 2026-09-27).

**Mesh size is elements across the radius, not millimetres**, so the element
count stays comparable across the box. **Use quantities of interest that
converge**: a corner peak of plastic strain does not; the same strain averaged
over a fixed region at the face does.

**The hardening table must reach the largest plastic strain the box produces.**
Near-limit runs went well past the first guess; the table was extended and the
near-limit points re-run.

**Source material constants early, from papers you hold**, and name a synthetic
family for what it is rather than after the metal whose elastic constants it
borrows.

## 3. Conversion and verification

**Stored names must not carry mesh-dependent numbers.** A wall reaction first
named by its reference node's label changed name with the mesh numbering, so
no single declared field list could fit the sweep. It became
`reaction_force_2_reference_node`.

**Take vocabulary from existing cards, not from a plan's prose.** The plan
said one discretisation word; the cards say `FEM`, and field names carry their
prefixes (`node/…`, `solid/…`, `global/…`).

**When a check is added, re-measure the published records and look at what
changed.** A new sampling-clock check would have failed every published
LS-DYNA case, for a cause another row already reports without judging; it now
counts on-grid frames only. Only the new rows may change.

**`total_energy` means different things to different solvers.** Abaqus's
ETOTAL subtracts external work; the canonical `total_energy` for Abaqus is
ETOTAL plus ALLWK, so that it is the energy the model holds. The adapter states
the convention.

**Some evidence a solver never writes.** Abaqus prints no solver revision or
parallel layout, so `solver_identity_complete` fails on every Abaqus case; it
is an accepted gap (`[pilot].accepted_gaps`) until the row is re-scoped for
Abaqus in ADR-0071's part three. Say so in the card rather than hide the row.

**Keep the fresh review at the end of every plan.** The independent review of
the conversion and verification code found what per-task tests had missed:
nodal stress that could pass as integration-point stress, two materials in one
part merged, one bad case stopping the whole conversion, a report blaming a
missing field for a missing history, ODB retention that depended on which
split was named, licence lines copied into the archive, and a verification
that silently skipped runs with no canonical file.

## 4. Working method

**Settle the filters and the output requests before production; case numbers
depend on them.** A case number is its position among the points that pass
the split's filters, so tightening a limit renumbered most of the sweep, not
just the cases it removed; adding one output term changed every deck. If they
must change, move the old sweep aside, re-run it whole, and use the unchanged
cases as a byte-for-byte regression. *Now:* `check` demands a byte-stable deck,
`generate` refuses to overwrite a case that ran, and the preflight stamp
re-arms whenever the definition changes.

**Verify on real output.** Several "expected" values written into plans from
memory were wrong (the end frame, the energy identity, the vocabulary). Run a
probe first, then write the rule.

**Record every deviation with a ruling** — what, why, cost if wrong — and flag
for the maintainer any decision they had already confirmed.

**Decide the mesh on the fields the surrogate learns, before production.** The
first mesh was chosen up front and checked on the quantities of interest only,
and production ran twice on it. A convergence set on nested meshes (so fields
compare node for node) showed the quantities converged but the stress field did
not: after the rod leaves the wall, stress is a low-energy elastic vibration
that every mesh places slightly differently, and a relative L2 counts that
heavily. The finest level was ruled out on cost and on the impossibility of
measuring its own error. Measure runtime at each level too: severe cases slow
down far more than the nominal factor. *Now:* `[levels]`, the convergence
engine of ADR-0071's part two, and the preflight's resolution step (space,
time, duration).

**The energy account can fail silently. Check it; do not only read it.** A
contact enforcement created energy in a few runs of one production version,
with no warning or error in any solver file; the instrument measured the
residual, but energy balance is an indicator with no ratified level, so it
judged nothing and the sweep read clean. Where a definitional inequality
exists, make it a requirement. *Now:* `plastic_dissipation_excess_max`
(plastic dissipation may not exceed internal energy) is a requirement of the
ADR-0066 instrument and fails every such run. Before production, list which
energy checks are requirements and which are only measured; a measured-only
check is one nobody runs.

**A fix must be checked on the axis of the problem it replaces.** The first
remedy for that defect passed a sample on energy, quantities of interest and
the new check — and doubled the stress discretisation uncertainty, because its
contact stiffness followed the element size. The double check the maintainer
asked for is what caught it. After changing any solver setting, re-run the
h / h/2 pair on a few cases before the sweep.

**Budget the disk per version.** Superseded production versions kept "for
now" filled the drive during a re-run; the runner's per-case records survived
and the affected cases re-ran, but only after the maintainer chose what to
delete. Before a re-run, compare the free space with the size of the version
being replaced; prune superseded ODBs first (the npz and the h5 keep the
data) and keep only their small tables. *Now:* `[pilot].min_free_gb`, the
runner's free-space check of part two, `archive --prune-odb`.

**Windows and shell quirks.** Heredocs in Git Bash mangle backslashes, so
patch scripts are files; `Path.write_text` writes CRLF, which once changed the
hash of a definition file (the loader now hashes with LF endings); long jobs
run in the background.

**Private details and licence text.** Scan the *commit trees* of the range to
be pushed — not the working tree — for the dataset name, case prefix, split
names, sweep values and local paths; tests use invented values; the archive
redacts what the solver wrote about its licence.

## 5. Pre-production checklist

1. **Conformance first**: one run with `*ENERGY OUTPUT, VARIABLE=ALL`; the
   duplicate end frame checked on every series; every new output's position
   confirmed in the ODB.
2. **Energy output**: request every term the identity needs, establish the
   identity, decide which energy checks are requirements.
3. **Feasibility**: probe the boundary, fit it, test the fit on the at-risk
   points, declare it in `feasible()`; remember it belongs to one mesh.
4. **Look**: an animation of a typical and an extreme case.
5. **Names**: stable stored names; the `[declaration]` written at the start.
6. **Runtime**: measured on real cases at each level, early.
7. **Contact and the energy gate**: settled together, at two increments.
8. **Material constants**: sourced and cited before production.
9. **Retention**: `[retention]` set before any pruning.
10. **Resolution**: the convergence set on nested meshes, the frame interval
    and the horizon settled from the pilots (ADR-0071, note of 2026-09-27) —
    `structbench-datagen preflight` runs all three and writes the stamp
    `generate` needs; `[qoi].tolerance`, `[pilot].settling_margin` and
    `contact_force_global` are the dataset's part of it.
11. **After any solver-setting change**: the h / h/2 pair again.
12. **Disk**: budgeted per version.
13. **Before every push or share**: the private-detail and licence scans.
