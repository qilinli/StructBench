"""structbench-validate: --list, and a record written from pairs and setup."""

from structbench.validation import cli

STATUS_ONLY_PAIRS = b"""\
reference = "taylor_copper"

[variants]
a = "A"

[[pair]]
variant = "a"
test = "3"
status = "aborted"
reason = "none run"
"""


def test_list_names_the_shipped_sets(capsys):
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "taylor_copper" in out and "162" in out


def test_writes_json_and_markdown(tmp_path, capsys):
    (tmp_path / "pairs.toml").write_bytes(STATUS_ONLY_PAIRS)
    (tmp_path / "setup.toml").write_bytes(
        b'[setup]\nsolver = "none"\n\ncaveats = ["synthetic"]\n'
    )
    rc = cli.main(
        [
            "--reference",
            "taylor_copper",
            "--pairs",
            str(tmp_path / "pairs.toml"),
            "--setup",
            str(tmp_path / "setup.toml"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "taylor_copper.json").is_file()
    md = (tmp_path / "out" / "taylor_copper.md").read_text(encoding="utf-8")
    assert "10.3390/ma16155452" in md and "synthetic" in md
    assert "0 of 1" in capsys.readouterr().out


def test_a_bad_pairs_file_exits_two_with_one_line(tmp_path, capsys):
    (tmp_path / "pairs.toml").write_bytes(
        STATUS_ONLY_PAIRS.replace(b'test = "3"', b'test = "42"')
    )
    (tmp_path / "setup.toml").write_bytes(b"[setup]\n")
    rc = cli.main(
        [
            "--reference",
            "taylor_copper",
            "--pairs",
            str(tmp_path / "pairs.toml"),
            "--setup",
            str(tmp_path / "setup.toml"),
            "--out",
            str(tmp_path / "o"),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2 and "42" in err and err.count("\n") == 1


def test_a_pairs_file_for_another_set_is_refused(tmp_path, capsys):
    (tmp_path / "pairs.toml").write_bytes(
        STATUS_ONLY_PAIRS.replace(b"taylor_copper", b"other")
    )
    (tmp_path / "setup.toml").write_bytes(b"[setup]\n")
    rc = cli.main(
        [
            "--reference",
            "taylor_copper",
            "--pairs",
            str(tmp_path / "pairs.toml"),
            "--setup",
            str(tmp_path / "setup.toml"),
            "--out",
            str(tmp_path / "o"),
        ]
    )
    assert rc == 2 and "other" in capsys.readouterr().err
