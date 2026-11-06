"""The shared by-path importer for dataset and benchmark definitions."""

import sys

import pytest

from structbench.core.definition_module import (
    DefinitionModuleError,
    definition_sha256s,
    file_sha256,
    load_definition_module,
)


def _write(directory, name, text):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text, encoding="utf-8")


def test_loads_module_and_records_siblings(tmp_path):
    _write(tmp_path, "helper.py", "VALUE = 7\n")
    _write(tmp_path, "thing.py", "import helper\nANSWER = helper.VALUE * 6\n")
    before = set(sys.modules)
    module = load_definition_module(tmp_path, "thing.py", qualifier="test_thing")
    assert module.ANSWER == 42
    assert module.__siblings__ == ("helper.py",)
    # The module stays registered under its qualified name (dataclasses
    # resolve annotations through sys.modules); the sibling does not.
    assert set(sys.modules) - before == {module.__name__}
    assert "helper" not in sys.modules
    assert str(tmp_path) not in sys.path


def test_no_bytecode_is_written(tmp_path):
    _write(tmp_path, "thing.py", "X = 1\n")
    load_definition_module(tmp_path, "thing.py", qualifier="test_thing")
    assert not (tmp_path / "__pycache__").exists()


def test_missing_file_is_a_definition_error(tmp_path):
    with pytest.raises(DefinitionModuleError, match="thing.py: not found"):
        load_definition_module(tmp_path, "thing.py", qualifier="t")


def test_import_error_is_a_definition_error(tmp_path):
    _write(tmp_path, "thing.py", "raise RuntimeError('boom')\n")
    with pytest.raises(DefinitionModuleError, match="RuntimeError: boom"):
        load_definition_module(tmp_path, "thing.py", qualifier="t")
    assert not any(n.startswith("t_") for n in sys.modules)


def test_sha256s_cover_file_and_siblings_lf_normalised(tmp_path):
    _write(tmp_path, "helper.py", "VALUE = 7\n")
    (tmp_path / "thing.py").write_bytes(b"import helper\r\nX = helper.VALUE\r\n")
    module = load_definition_module(tmp_path, "thing.py", qualifier="t")
    hashes = definition_sha256s(tmp_path, "thing.py", module)
    assert list(hashes) == ["thing.py", "helper.py"]
    lf = tmp_path / "lf.py"
    lf.write_bytes(b"import helper\nX = helper.VALUE\n")
    assert hashes["thing.py"] == file_sha256(lf)
