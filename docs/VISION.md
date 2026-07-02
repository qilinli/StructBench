# VISION.md

*What StructBench is, who it serves, and where it is going.*

---

## What StructBench is

StructBench is an open platform for data-driven structural engineering. It provides standardised benchmark problems, reference models, and — over time — reusable deployment tools that let research groups and practitioners apply machine learning methods to structural analysis and health monitoring in a consistent, reproducible way.

At its current stage, the platform's focus is on benchmarks and reference implementations for surrogate modelling of structures under dynamic and extreme loading. Its scope will expand to include multi-modal structural health monitoring and end-to-end deployment workflows as the platform matures.

## Why it exists

Machine learning research in structural engineering is currently fragmented. Individual groups publish results on custom datasets with non-standardised evaluation protocols, making it difficult to compare methods, reproduce claimed results, or assess whether the field is making real progress. Reference implementations are rarely released or maintained. The path from a research prototype to practical deployment on a real asset is rebuilt from scratch for each project.

StructBench addresses this by establishing a shared substrate: benchmark problems with reproducible data generation, reference models with released checkpoints, a solver-agnostic data format that allows contributions across different simulation backends, and evaluation protocols that let results be compared meaningfully across methods.

## Who it serves

Three audiences. **Researchers in machine learning for structural engineering**, who need standardised benchmarks to evaluate new methods and meaningful baselines to compare against. **Structural engineers and research groups**, who want to apply existing ML methods to their own assets without building every component from scratch. **Industry partners**, at a later stage, as the deployment tools mature — though the platform remains a research-first artifact, with commercial applications treated as a secondary consideration.

## What StructBench is not

Not a new finite element solver — it uses existing solvers, commercial or open, as data generation backends and focuses on the ML layer above them. Not a general-purpose machine learning framework — it builds on existing libraries rather than re-implementing them. Not a single-solver or single-structure ecosystem — the data format is solver-agnostic and scope expansion is part of the plan. Not a replacement for domain expertise — the platform supports engineers and researchers, but does not encode the judgment their work requires.

## Long-term trajectory

StructBench is intended as a durable contribution: a platform that accumulates benchmarks, models, and deployment tools over years, and that other groups can build on, cite, and contribute to. Its ambition is to be useful — to lower the cost of doing rigorous ML research in structural engineering, and to shorten the path from research method to practical application. Whether it becomes widely adopted depends on community uptake, which in turn depends on the platform remaining open, honest about its limitations, and focused on real engineering problems rather than methodological novelty for its own sake.
