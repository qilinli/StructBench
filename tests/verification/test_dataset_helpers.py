"""The dataset-level helpers live in verification; the CLI and datagen import them."""


def test_the_helpers_import_from_verification_and_the_cli_re_exports_them():
    from structbench.cli import datacheck
    from structbench.verification import dataset

    assert datacheck.declared_from_toml is dataset.declared_from_toml
    assert datacheck.measure_cases is dataset.measure_cases
    assert datacheck.input_facts_for is dataset.input_facts_for


def test_the_card_vocabulary_is_one_object_in_core():
    from typing import get_args

    from structbench.benchmarks.card import Discretisation
    from structbench.core.evidence import CardDiscretisation

    assert Discretisation is CardDiscretisation
    assert set(get_args(CardDiscretisation)) == {"SPH", "FEM", "coupled"}
