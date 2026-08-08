# `deforming_plate` — download-and-convert

Converts the MeshGraphNets `deforming_plate` dataset (Pfaff et al. 2021,
COMSOL hyperelastic-plate simulations) from its native tfrecord format to
canonical StructBench HDF5 (ADR-0042). Not part of the importable
`structbench` package (ADR-0010) — a standalone script that imports the
installed package.

## No-rehost policy

The dataset carries **no redistribution licence**. Only the DeepMind
`meshgraphnets` *code* is Apache-2.0; the hosted data itself states no terms.
StructBench therefore ships **download-and-convert, not rehost**
(ADR-0042 §2a):

- **StructBench does not host or redistribute the raw tfrecords.** You
  download them yourself, from source, into a local scratch directory.
- `convert.py` reads that local copy and writes canonical `.h5` files
  locally. Only the converted output — never the raw tfrecords — is meant to
  end up in the maintainer's `canonical/` archive.

## 1. Set up a throwaway TensorFlow environment

`structbench` itself never depends on TensorFlow — it is imported lazily
inside `read_deforming_plate` only (ADR-0042 §2a), so `import structbench`
and this script's `--help` both work without it. Reading tfrecords does need
it, so create a disposable env with both `tensorflow` and `structbench`
installed, e.g. (mirrors the project's existing `uv`/`.venv` pattern used for
`lasso-python`):

```bash
uv venv .venv-tf
uv pip install --python .venv-tf tensorflow
uv pip install --python .venv-tf -e .
```

Discard `.venv-tf` when you're done; nothing about it is part of the
project's normal dev environment.

## 2. Download the dataset from source

Either the DeepMind `meshgraphnets` repo's helper script:

```bash
bash download_dataset.sh deforming_plate <scratch>/deforming_plate
```

(`download_dataset.sh` lives in the `meshgraphnets` subdirectory of
`google-deepmind/deepmind-research` on GitHub) — or fetch the four files
directly from the source GCS bucket:

```
https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json
https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/train.tfrecord
https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/valid.tfrecord
https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord
```

Either way, `<scratch>/deforming_plate/` should end up holding `meta.json`
plus `train.tfrecord` (1000 trajectories), `valid.tfrecord` (100), and
`test.tfrecord` (100); 400 quasi-static load-step frames each, ~1271 nodes
average (verified facts, ADR-0042).

## 3. Convert

```bash
.venv-tf/Scripts/python convert.py \
    --data-root <scratch>/deforming_plate \
    --out <scratch>/canonical/deforming_plate \
    --split valid --limit 2
```

Smoke a couple of cases first (`--split valid --limit 2`) before running the
full 1000/100/100 conversion (drop `--split`/`--limit` for that). Flags:

- `--data-root` — dir holding `meta.json` + `{train,valid,test}.tfrecord`.
- `--out` — output directory for canonical `.h5`.
- `--split {train,valid,test}` — convert one source split only (default: all
  three).
- `--limit N` — convert only the first `N` trajectories per split.
- `--overwrite` — reconvert cases whose output `.h5` already exists.

Output filenames are `<canon_split>_<i>.h5`, e.g. `val_0000.h5`,
`val_0001.h5`, ... — MeshGraphNets' own `valid` split is renamed to `val` to
match StructBench's platform-wide train/val/test convention (see
`convert.py`'s `SPLITS` mapping); `train`/`test` pass through unchanged.

## Units are provisional

`convert.py`'s `SOURCE_UNITS` constant is a **placeholder**
(`"kg-m-s"`) — `world_pos`/`stress` units are undocumented in `meta.json` and
the paper. A later, human-run task measures the true convention against real
downloaded data and patches the constant before any converted archive is
trusted or blessed (ADR-0042 §2b). Do not treat output from the current
`SOURCE_UNITS` value as verified SI until that measurement has happened.

## See also

- ADR-0042 (`decisions/0042-schema-020-per-node-fields-nodal-fe-ingestion.md`)
  — the schema/ingestion decision this converter implements.
- `structbench.core.io.meshgraphnets` — the pure `build_deforming_plate_case`
  assembly and the lazy-TF `read_deforming_plate` reader this script drives.
