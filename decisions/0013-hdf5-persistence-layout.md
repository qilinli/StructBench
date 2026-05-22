# 0013 — HDF5 persistence layout for the case schema

**Status**: Accepted
**Type**: Durable
**Date**: 2026-05-22

## Context

ADR-0011 fixed the case vocabulary; ADR-0012 fixed the field-level structure and the validity tiers. Both explicitly deferred the concrete persistence layer — "the on-disk HDF5 layout (groups vs flat datasets, exact path spellings, dtype choices, attribute conventions)" — to a separate ADR. This is that ADR (Stage 3).

The platform is already committed to HDF5 as the canonical container (ADR-0004, ARCHITECTURE.md). What remained open was *how* the agreed field set is laid out inside an HDF5 file: the library used to read and write it, the group/dataset/attribute mapping, dtypes, compression, string handling, and how heterogeneous solver-native data is stored.

The layout was designed against the v0.1 anchor cases: the 2D SPH Taylor impact in `Taylor.k` (≈4800 particles, `MAT_ELASTIC_PLASTIC_HYDRO` + `EOS_GRUNEISEN`, an initial-velocity card, a planar rigidwall, one 4800-member node set) and the RC beam impact problem (ADR-0003). These exercise degenerate single-node SPH connectivity, a material carrying embedded hardening curves plus a linked equation of state, and large node sets — the awkward cases a layout must handle cleanly.

## Decision

### Library

**h5py** is the canonical reader/writer. It is the standard NumPy-native binding to HDF5 (groups, datasets, attributes), and its object model maps one-to-one onto ADR-0012's structure. **numpy** is a direct dependency alongside it (the schema's arrays are NumPy arrays). Both are recorded as approved runtime dependencies in PRINCIPLES.md upon acceptance of this ADR. PyTables (unnecessary table machinery) and zarr (not HDF5; would reopen ADR-0004/0011/0012) were rejected.

### File granularity

One case = one HDF5 file, consistent with ADR-0011. Packing many cases into one container is a `datasets/`-module concern and is out of scope here.

### Group / dataset / attribute layout

All paths are lowercase `snake_case`, matching ADR-0012. Heterogeneous or possibly-large content is stored as **datasets**; small scalars are stored as **attributes**.

**Metadata** — group `/metadata`:
- Scalar metadata as attributes on `/metadata`: `case_id`, `schema_version`, `units_convention` (`"SI"`), `dimension` (`2`|`3`), and optional `source_units`, `asset_id`, `dataset_id`.
- `/metadata/provenance` — subgroup; `solver_name`, `solver_version`, `generation_date` as attributes. Required when solver-generated.
- `/metadata/source_deck` — variable-length UTF-8 string **dataset** (gzip-compressed), optional. A dataset, not an attribute, because decks are large (`Taylor.k` is ~560 KB; HDF5 attributes cap near 64 KB).

**Geometry and topology**:
- `/nodes/coords` — float64, `(n_nodes, dim)`.
- `/nodes/node_id` — int64, `(n_nodes,)`.
- `/elements/<type>/connectivity` — int64, `(n_elem, n_nodes_per_elem)`, 0-indexed into `/nodes`. SPH is the degenerate `(n_elem, 1)` case.
- `/elements/<type>/element_id`, `/elements/<type>/part_id` — int64, `(n_elem,)`.
- `/parts` — group of parallel datasets: `part_id`, `section_id`, `material_id` (int64), optional `title` (vlen UTF-8).
- `/sections/<section_id>` — per-section group; `source_params` JSON string dataset (+ a `canonical_type` attribute where a clean mapping exists). Detailed section field shape is a follow-on (per ADR-0012).

**Materials** — group `/materials`, parallel datasets indexed alike:
- `material_id` — int64.
- `canonical_model` — vlen UTF-8; empty string `""` denotes null (no clean canonical mapping). HDF5 has no native null; the empty-string sentinel is the documented convention.
- `source_model` — vlen UTF-8, always populated.
- `source_params` — vlen UTF-8 JSON string, one per material, always populated. Solver sub-models linked to the material (EOS, hourglass) are nested inside this JSON by the adapter — e.g. the Taylor copper material carries its 16-point `eps`/`es` curves and the Grüneisen EOS within one `source_params` record.

**Constraints, loading, initial conditions**:
- `/boundary_conditions`, `/loading`, `/initial_conditions` — named groups, each holding entries as vlen UTF-8 JSON records at the layout level. The detailed per-entry field shape is the follow-on ADR-0012 named; this ADR fixes only that they are JSON-record groups for now. The Taylor `*RIGIDWALL_PLANAR` and `*INITIAL_VELOCITY` cards land in `/loading` and `/initial_conditions` respectively.
- `/time_curves/<curve_id>` — float64 dataset `(n_points, 2)` for `(t, value)`. Separate datasets accommodate per-curve length.
- `/sets/node/<set_id>` and `/sets/element/<set_id>` — int64 1D datasets of member ids; per-set datasets accommodate variable membership. The Taylor 4800-member node set is `/sets/node/1`.

**Response** — group `/response`:
- `/response/time/t` — float64, `(n_frames,)`. Single global axis (ADR-0012).
- `/response/node/{displacement,velocity,acceleration}` — float32, `(n_frames, n_nodes, dim)`.
- `/response/element/<type>/{stress,strain,damage,…}` — float32; tensor fields Voigt-symmetric `(n_frames, n_elem, n_components)`, scalar fields `(n_frames, n_elem)`.
- `/response/global/<name>` — float32, `(n_frames,)`, one dataset per scalar (`kinetic_energy`, `internal_energy`, `contact_force`, reactions, …).
- `/response/sensor` — reserved; internal shape deferred to SHM scope.

**Sensors**: `/sensors` — reserved; internal shape deferred to SHM scope.

### Dtypes

- **Geometry and the time axis** (`/nodes/coords`, `/time_curves`, `/response/time/t`) — **float64**. Small and precision-sensitive.
- **Bulk response fields** (`/response/node/*`, `/response/element/*`, `/response/global/*`) — **float32**. These dominate file size and float32 is the standard ML training precision.
- **Identity and connectivity** (`*_id`, `connectivity`) — **int64**.
- **Strings** — HDF5 variable-length **UTF-8** (`h5py.string_dtype(encoding="utf-8")`).

### Compression and chunking

Large datasets (all of `/response/*`, `/metadata/source_deck`) are **chunked and gzip-compressed (level 4)**. Response arrays are chunked along the frame axis — chunk shape `(c, n_nodes, dim)` with `c` chosen so a chunk is on the order of ~1 MB — so transition-based training can read frame ranges without loading the whole array. gzip is chosen over lzf for portability (readable by any HDF5 tool) and over no-compression for size. Small datasets are left uncompressed.

### Heterogeneous solver-native data

`source_params` (per material/section) and `source_deck` are stored as **JSON strings** (the deck as a verbatim text blob). This handles arbitrary, ragged, per-model solver dictionaries with trivial roundtrip. The tradeoff — these fields are opaque to a pure-HDF5 browser — is accepted; a JSON schema for `source_params` is a possible future follow-on.

### Schema version

This ADR pins the initial `schema_version` to **`"0.1.0"`**. Additive field changes are minor-version bumps; structural or breaking changes are major bumps requiring a superseding ADR (consistent with ADR-0012).

## Alternatives considered

- **PyTables / zarr instead of h5py.** Rejected. PyTables adds query/index machinery a single-case file does not need; zarr is not HDF5 and would reopen ADR-0004/0011/0012.
- **float64 everywhere.** Rejected. The response arrays dominate size; float32 halves them at precision that ML training does not miss. Geometry and the time axis stay float64.
- **Compound (table) dtypes for `parts`/`materials`.** Rejected in favour of parallel per-column datasets. Compound dtypes mix awkwardly with variable-length strings, complicate partial reads, and gain nothing here.
- **Native nested HDF5 groups for `source_params`.** Rejected. Solver-native parameter sets are heterogeneous and ragged across models; mapping arbitrary dicts to groups is brittle. JSON strings roundtrip cleanly.
- **lzf or no compression.** Rejected. lzf is faster but weaker and h5py-specific; no compression bloats response-heavy files. gzip is the portable, archival-friendly middle.
- **Many cases per file / sharded layout.** Rejected here. ADR-0011 fixes one case = one file; multi-case packing belongs to `datasets/`.
- **Storing `source_deck` as an attribute.** Rejected. Decks exceed HDF5's attribute size ceiling; they are datasets.
- **A dedicated null marker dataset for `canonical_model`.** Rejected in favour of the empty-string sentinel, documented — simpler and adequate.

## Consequences

- **`core/io` is now implementable** against a concrete spec: readers, writers, and the schema validator (group presence per ADR-0012 tiers; dtype and shape per this ADR) can be built.
- **h5py and numpy become the first approved runtime dependencies**, recorded in PRINCIPLES.md's approved-runtime table referencing this ADR.
- **The adapter contract is extended**: write geometry/time as float64 and response as float32, JSON-encode `source_params` and the source deck, gzip-chunk large arrays, resolve solver sub-model linkages (EOS, hourglass) into the owning material's `source_params`.
- **Streaming reads are supported**: frame-axis chunking lets datasets/loaders pull transitions without materialising whole response arrays.
- **JSON opacity is a known tradeoff**: `source_params`/`source_deck` are not browsable as native HDF5 structure; a `source_params` JSON schema may follow.
- **Follow-ons remain**, each likely its own ADR or field-shape note: detailed `boundary_conditions`/`loading`/`sections` record shapes, the canonical-material enum contents (ADR-0012), sensor representation when SHM lands, and any future `source_params` JSON schema.
