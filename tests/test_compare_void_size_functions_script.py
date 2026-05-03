import csv
import math

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


def write_cosmology(path) -> None:
    lines = [
        "# Cosmological quantities used in PINOCCHIO (h=1.000000)",
        "#15: smoothing scale R (Mpc)",
        "#16: Gaussian-filtered mass variance sigma_G^2(R)",
        "#18: d Log sigma_G^2 / d Log R",
    ]
    for radius, sigma2 in (
        (0.001, 16.0),
        (0.01, 8.0),
        (0.1, 4.0),
        (0.5, 2.0),
        (1.0, 1.0),
        (3.0, 0.4),
        (10.0, 0.1),
        (100.0, 0.01),
    ):
        columns = [1.0] * 20
        columns[14] = radius
        columns[15] = sigma2
        columns[17] = -1.0
        lines.append(" ".join(f"{value:.8e}" for value in columns))
    path.write_text("\n".join(lines), encoding="utf-8")


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


def test_compare_void_size_functions_script_writes_theory_rows(tmp_path, capsys) -> None:
    output_csv = tmp_path / "vsf_theory.csv"
    cosmology = tmp_path / "pinocchio.cosmology.out"
    write_cosmology(cosmology)
    args = base_args(output_csv)
    args.extend(["--theory", "vdn-svdw", "--cosmology-file", str(cosmology)])

    exit_code = main(args)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Theory: vdn-svdw" in captured.out

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 12
    assert {(row["source"], row["target"]) for row in rows if row["source"] == "vdn-svdw"} == {
        ("vdn-svdw", "A"),
        ("vdn-svdw", "B"),
    }
    for row in rows:
        if row["source"] == "vdn-svdw":
            assert row["count"] == ""
            density = float(row["density_dndlnr_per_mpc_h3"])
            assert math.isnan(density) or density >= 0.0


def test_compare_void_size_functions_script_accepts_fixed_linear_bins_and_summary(
    tmp_path,
) -> None:
    output_csv = tmp_path / "vsf_linear.csv"
    summary_csv = tmp_path / "summary.csv"
    args = base_args(output_csv)
    args.extend(
        [
            "--binning",
            "linear",
            "--bin-min",
            "0.5",
            "--bin-max",
            "2.5",
            "--summary-csv",
            str(summary_csv),
        ]
    )

    exit_code = main(args)

    assert exit_code == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {float(row["bin_min_mpc_h"]) for row in rows} == {0.5, 1.5}
    assert {float(row["bin_max_mpc_h"]) for row in rows} == {1.5, 2.5}

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert {(row["source"], row["target"]) for row in summary_rows} == {
        ("finder", "A"),
        ("finder", "B"),
        ("vide", "A"),
        ("vide", "B"),
    }
    vide_a = next(row for row in summary_rows if row["source"] == "vide" and row["target"] == "A")
    assert vide_a["count"] == "2"
    assert float(vide_a["median_mpc_h"]) == 1.5


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
