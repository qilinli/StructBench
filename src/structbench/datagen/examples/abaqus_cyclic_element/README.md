# abaqus_cyclic_element — the second example dataset definition

A block of a combined-hardening material (one backstress, an isotropic part)
taken through one cycle of axial strain; cases may start from a non-zero
equivalent plastic strain and backstress. It is the test bed of what
`docs/datagen/abaqus-conformance.md`, "Combined hardening", establishes: the
material cards, the initial state, the backstress in the export and in the
canonical file. At level 1 a case is one element and runs in seconds.

    structbench-datagen check    <this dir>
    structbench-datagen generate --dataset <this dir> --work-root <runs> --split pilot
    structbench-datagen run      --sweep <runs>/abaqus_cyclic_element
    structbench-datagen export   --sweep <runs>/abaqus_cyclic_element
    structbench-datagen convert  --sweep <runs>/abaqus_cyclic_element
    structbench-datagen verify   --sweep <runs>/abaqus_cyclic_element --dataset <this dir>

A new dataset starts from the first example, `abaqus_conformance`, whose
`dataset.toml` explains every key.
