# abaqus_drop_mass — the third example dataset definition

A short cylinder upset between a fixed rigid base and a dropped rigid plate.
The plate is an analytical rigid surface; its reference node carries the drop
mass as a point mass and the initial velocity, and is in the field output, so
the force on the plate is the mass times its stored acceleration. The mesh
carries a small imperfection: its quads are not rectangles, and its levels
nest by index. It is the test bed of `docs/datagen/abaqus-conformance.md`,
"A rigid body with a point mass". At level 1 a case runs in half a minute.

    structbench-datagen check     <this dir>
    structbench-datagen preflight --dataset <this dir> --work-root <runs>

Reading the mass from a converted case:

    from structbench.core.io.abaqus_run import abaqus_point_masses
    masses = abaqus_point_masses(case.metadata.source_deck,
                                 source_units=case.metadata.source_units)
    # {node id: kilograms}; the node is in case.nodes.node_id

A new dataset starts from the first example, `abaqus_conformance`, whose
`dataset.toml` explains every key.
