import csv

import pytest

from scripts.compare_void_size_functions import main


def base_args(output_csv) -> list[str]:
    return [
        "tests/fixtures/pinocchio_pair_a.out",
        "tests/fixtures/pinocchio_pair_b.out",
        "tests/fixtures/vide_voidDesc_all_small.out",
        "tests/fixtures/vide_voidDesc_all_small.out",
        "--box-size",
        "10.0",
        "--rho-bar",
        "1.0",
        "--linking-length",
        "0.3",
        "--min-cluster-members",
        "2",
        "--bins",
        "2",
        "--output-csv",
        str(output_csv),
    ]


def test_compare_void_size_functions_script_writes_finder_and_vide_csv(tmp_path, capsys) -> None:
    output_csv = tmp_path / "vsf.csv"

    exit_code = main(base_args(output_csv))

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Target A:" in captured.out
    assert "Target B:" in captured.out
    assert f"Wrote {output_csv}" in captured.out

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert set(rows[0]) == {
        "label",
        "source",
        "target",
        "bin_min_mpc_h",
        "bin_max_mpc_h",
        "bin_center_mpc_h",
        "count",
        "density_dndlnr_per_mpc_h3",
    }
    assert {(row["source"], row["target"]) for row in rows} == {
        ("finder", "A"),
        ("finder", "B"),
        ("vide", "A"),
        ("vide", "B"),
    }
    counts = {}
    for row in rows:
        key = (row["source"], row["target"])
        counts[key] = counts.get(key, 0) + int(row["count"])
    assert counts[("finder", "A")] == 1
    assert counts[("finder", "B")] == 1
    assert counts[("vide", "A")] == 2
    assert counts[("vide", "B")] == 2


def test_compare_void_size_functions_script_accepts_linking_factor(tmp_path, capsys) -> None:
    output_csv = tmp_path / "vsf_factor.csv"
    args = base_args(output_csv)
    length_index = args.index("--linking-length")
    del args[length_index : length_index + 2]
    args.extend(["--linking-factor", "0.05"])

    exit_code = main(args)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Linking: factor=0.05" in captured.out
    assert output_csv.exists()


def test_compare_void_size_functions_script_rejects_mixed_linking_modes(tmp_path) -> None:
    output_csv = tmp_path / "vsf.csv"
    args = base_args(output_csv)
    args.extend(["--linking-factor", "0.05"])

    with pytest.raises(SystemExit):
        main(args)
