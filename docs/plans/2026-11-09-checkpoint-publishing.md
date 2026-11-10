# Plan: publishing the blessed checkpoints (ADR-0074 clause 8)

*Written 2026-11-09 before the work on `feat/checkpoint-publishing`. ADR-0074
clause 8 decides that blessed checkpoints are published at the next code
release; ADR-0037 set the move up ("upload the existing bundle and rewrite
the pointer"); ADR-0040 chose Hugging Face as the public channel for the
archives. This plan fixes the mechanics the records delegate and builds the
tool that stages a release's checkpoints. The upload itself needs the
`models/` mirror, which lives on DUG and the maintainer's OneDrive, so the
tool is exercised here on synthetic bundles and runs for real at release
time. Platform implementation only.*

## Choices (the plan's, not the records')

1. **Channel.** One Hugging Face *model* repository per benchmark,
   `StructBench/<slug>` with the dataset repos' slugs (`taylor-impact-2d`,
   `wave-propagation-1d`, `notch-beam-2d-impact`) plus `deforming-plate`:
   the DeformingPlate *data* is not rehosted (ADR-0042, no stated licence)
   but the checkpoint StructBench trained on it is StructBench's own
   artifact under the repository licence. A model repo and a dataset repo
   may share a name.
2. **What is uploaded.** The minimum that runs and checks the model
   (maintainer, 2026-11-10, replacing this plan's first choice of the whole
   bundle): the registered checkpoint, `config.json`,
   `normalization_stats.npz` when the run has one, the `metrics-<split>.json`
   of the row's scored splits and a `SHA256SUMS` of those files, under the
   bundle name `<family>-<run_commit>/` at the repo root, plus a generated
   model-card `README.md`. Logs, `provenance/`, rollouts and the
   unregistered checkpoints stay in the private bundle (they are working
   records and carry local paths); the JSON files are rewritten without
   `data_root` and with `checkpoint_path` reduced to the file name, and any
   other absolute path stops the staging. The private bundle of ADR-0037 is
   unchanged and remains the evidence unit.
3. **Which rows.** The blessed rows of every registry (ADR-0074 clause 8
   says *blessed*); provisional rows only on `--include-provisional`, which
   the release session decides. A row whose pointer is already a URL is
   reported as published and skipped.
4. **Tag and pointer.** The model repo is tagged with the code release
   (`v0.4.0`), and the registry pointer becomes
   `https://huggingface.co/StructBench/<slug>/resolve/<tag>/<bundle>/<file>`,
   the immutable form (ADR-0037 clause 4: a public URL after publication).
   The landing pages then drop the private-archive marker by themselves.
5. **Verification before upload.** A row is staged only when its bundle
   exists under the mirror, `SHA256SUMS` lists the checkpoint, the file's
   digest equals both the listed one and the registry's
   `checkpoint_sha256`; a registry row without a digest is a problem, not
   a pass.
   A row whose private bundle is not assembled is staged from its run
   directory (`--run-dir <bundle>=<dir>`): no `SHA256SUMS` is asked of it,
   and the registry digest alone identifies the checkpoint.
6. **Who runs it.** ADR-0023's 2026-08-28 amendment puts Hugging Face
   data-release actions on the `StructBench/*` *dataset* repos in the
   on-instruction tier. Model repos are not named there. Until the
   maintainer extends that note, the tool writes the `hf` commands to a
   script and the maintainer runs them; the token stays the human's.

## Build

`tools/publish_checkpoints.py` (a standalone script like
`build_hf_bundle.py`, ADR-0010): `parse_pointer`, `read_sha256sums`,
`verify_bundle`, `public_url`, `render_model_readme`, `plan_release`,
`stage`; `main` with `--benchmark`, `--models-root`, `--out`, `--tag`,
`--include-provisional`, `--check`. `stage` writes `<out>/<benchmark>/
README.md`, `publish.sh` (create, upload per bundle, upload the README,
flip public, tag), the published folder of each row (`stage_bundle`) and
`pointers.json` (label → old pointer, new URL,
digest), and exits non-zero when any row fails verification. Tests on a
synthetic mirror in `tests/tools/test_publish_checkpoints.py`. The README
Roadmap's *Later* item "checkpoint-publishing workflow" moves into the
v0.4 release-mechanics bullet; the DUG runbook's blessing checklist gains
the publication step.

## Not in this plan

Running the upload; rewriting the registry sources (the release session
edits the four `results` blocks from `pointers.json` and regenerates the
pages); the on-instruction note on ADR-0023.
