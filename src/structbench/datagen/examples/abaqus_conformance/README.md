# abaqus_conformance — the example dataset definition

One copper-like rod striking a frictionless rigid wall, the conformance case of
`docs/datagen/abaqus-conformance.md`. `structbench-datagen new` copies this
directory as the template a new dataset starts from.

    structbench-datagen check    <this dir>
    structbench-datagen generate --dataset <this dir> --work-root <runs>
    structbench-datagen run      --sweep <runs>/abaqus_conformance
    structbench-datagen export   --sweep <runs>/abaqus_conformance
    structbench-datagen convert  --sweep <runs>/abaqus_conformance
    structbench-datagen validate --sweep <runs>/abaqus_conformance --dataset <this dir>
    structbench-datagen archive  --sweep <runs>/abaqus_conformance --dataset <this dir> --data-root <tree>

`preflight`, `converge` and `card` arrive with parts two and three of ADR-0071.
