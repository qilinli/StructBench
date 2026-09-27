"""Validation against experiments (ADR-0072).

Reference-experiment sets with their provenance, measures that apply the same
function to a measured outline and to a canonical case, a comparison that
reports deviations and never judges them, and a byte-stable record with a
Markdown rendering, behind ``structbench-validate``. Depends on ``core`` only.
Verification -- whether a reference run is internally sound -- is the
``verification`` package (ADR-0066); this one asks whether a setup reproduces
what was measured.
"""
