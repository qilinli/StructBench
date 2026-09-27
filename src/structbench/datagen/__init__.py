"""Data generation: from a dataset definition to verified canonical cases.

Solver-agnostic stages live here (sampling, generate, run, convert, validate,
archive, the definition contract and its scaffold); solver-specific writers and
exporters live in a subpackage per solver (``abaqus``). Nothing here is
imported by the rest of the package: ``datagen`` sits beside ``eval`` and
``benchmarks`` in the layering (ADR-0071).
"""
