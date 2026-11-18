# abaqus_conformance — the example dataset definition

One copper-like rod striking a frictionless rigid wall, the conformance case of
`docs/datagen/abaqus-conformance.md`. `structbench-datagen new` copies this
directory as the template a new dataset starts from.

The definition's directory must be a git repository with a commit: every case's
provenance records the commit it was generated from.

    structbench-datagen check     <this dir>
    structbench-datagen preflight --dataset <this dir> --work-root <runs>
    structbench-datagen generate  --dataset <this dir> --work-root <runs>
    structbench-datagen run       --sweep <runs>/abaqus_conformance
    structbench-datagen export    --sweep <runs>/abaqus_conformance
    structbench-datagen convert   --sweep <runs>/abaqus_conformance
    structbench-datagen verify    --sweep <runs>/abaqus_conformance --dataset <this dir>
    structbench-datagen converge  --sweep <runs>/abaqus_conformance --dataset <this dir>
    structbench-datagen archive   --sweep <runs>/abaqus_conformance --dataset <this dir> --data-root <tree>

`generate` opens a split that is not a probe only against a passing preflight
stamp for the current definition. The stages are described in
`docs/DATA_GENERATION.md`.
