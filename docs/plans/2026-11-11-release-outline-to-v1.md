# Release outline to v1.0

*A long-term reference (maintainer, 2026-11-11), not a schedule. The README
Roadmap is the planning home: when a release is picked up, its criteria below
become that release's checklist there. This file holds the whole arc so that a
later release is not designed from scratch, and it changes by dated notes.*

## Rules for the arc

1. A release is what a reader can cite and rely on. It is closed by the
   criteria below, not by a feature count.
2. No new task family enters before v0.6. The platform's capabilities are
   ahead of its evidence; v0.5 and v0.6 close that gap.
3. Platform work only (ADR-0065). What a release enables outside the
   repository is not recorded here.
4. v1.0 is defined by criteria that depend on other people (a second group, a
   peer-reviewed description). It ships when they hold, however many 0.x
   releases that takes.

## The releases

| Release | Theme | Done when |
|---|---|---|
| **v0.5** | The citable platform: nothing new, what v0.4 claims is made true and stable | `docs/VISION.md` carries the ADR-0065 identity, and `CITATION.cff` and the BibTeX entry say the same · every registered row reports its properties or states why not · the repeatability tolerance of an evaluation is declared · the gradient family has rows from a second model family · per-case coverage figures are on the landing pages · a registered row is reproduced from its public checkpoint by the documented command alone · the release is archived with a DOI, the first |
| **v0.6** | The first benchmark that meets the data standard | the first contract dataset is admitted with a full compliance table (ADR-0073 D14), a blind set and a restart declaration · the first result from outside this repository passes the intake (ADR-0074) |
| **v0.7** | The width of the truth | one case reproduced in a second solver · the uncertainty budget of a reference dataset published as a record · the component-tier benchmark only if its case is made (Roadmap, *Later*) |
| **v0.8** | Used by someone else | a result or a dataset from another group passes the intake · one blind rotation is complete |
| **v0.9** | The freeze | the case schema, card, registry, dataset contract and commands are declared stable · shipped benchmarks superseded by a standard-meeting one are closed to new rows |
| **v1.0** | The standard | two benchmarks meet the data standard at two structural scales · blind evaluation is operating · verified results from two groups are registered · the platform is described in a peer-reviewed publication |

## Why this order

- **v0.5 before anything new.** v0.4 shipped the instrument; three rows
  report no properties, graph-network rollouts differ by about 1.5 % between
  two evaluations of one checkpoint, the gradient family has one model, and
  the vision and citation texts still describe the accuracy-comparison
  identity. A citable release should not carry those.
- **v0.6 is the first proof.** No shipped benchmark meets the data standard
  (the landing pages' compliance tables say so). The platform's claim that
  the standard is attainable rests on the first one that does.
- **v0.7 answers "how wide is the reference".** A second-solver reproduction
  and a published uncertainty budget are the two numbers that bound what any
  model error means.
- **v0.8 and v1.0 depend on others,** which is why they are criteria and not
  work items.
