import csv
import math

import numpy as np

from scripts.debug_theory_vsf import main, read_vsf_csv, theory_implied_count


def write_vsf_csv(path) -> None:
    rows = [
        {
            "label": "fixture",
            "source": "finder",
            "target": "A",
            "bin_min_mpc_h": 1.0,
            "bin_max_mpc_h": 2.0,
            "bin_center_mpc_h": math.sqrt(2.0),
            "count": 3,
            "density_dndlnr_per_mpc_h3": 2.0e-4,
        },
        {
            "label": "fixture",
            "source": "vide",
            "target": "A",
            "bin_min_mpc_h": 1.0,
            "bin_max_mpc_h": 2.0,
            "bin_center_mpc_h": math.sqrt(2.0),
            "count": 2,
            "density_dndlnr_per_mpc_h3": 1.0e-4,
        },
        {
            "label": "fixture",
            "source": "vdn-svdw",
            "target": "A",
            "bin_min_mpc_h": 1.0,
            "bin_max_mpc_h": 2.0,
            "bin_center_mpc_h": math.sqrt(2.0),
            "count": "",
            "density_dndlnr_per_mpc_h3": 1.0e-1,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_cosmology(path) -> None:
    lines = [
        "# Cosmological quantities used in PINOCCHIO (h=1.000000)",
        "#15: smoothing scale R (Mpc)",
        "#16: Gaussian-filtered mass variance sigma_G^2(R)",
        "#18: d Log sigma_G^2 / d Log R",
    ]
    for radius, sigma2 in ((0.1, 4.0), (0.5, 2.0), (1.0, 1.0), (3.0, 0.4)):
        columns = [1.0] * 20
        columns[14] = radius
        columns[15] = sigma2
        columns[17] = -1.0
        lines.append(" ".join(f"{value:.8e}" for value in columns))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_read_vsf_csv_parses_blank_theory_counts(tmp_path) -> None:
    path = tmp_path / "vsf.csv"
    write_vsf_csv(path)

    rows = read_vsf_csv(path)

    assert len(rows) == 3
    assert rows[2].source == "vdn-svdw"
    assert rows[2].count is None


def test_theory_implied_count_matches_density_volume_dlnr() -> None:
    implied = theory_implied_count(
        density_dndlnr_per_mpc_h3=0.1,
        bin_min_mpc_h=1.0,
        bin_max_mpc_h=2.0,
        box_size_mpc_h=10.0,
    )

    np.testing.assert_allclose(implied, 0.1 * 10.0**3 * math.log(2.0))


def test_debug_theory_vsf_script_reports_ratios_and_decomposition(tmp_path, capsys) -> None:
    vsf = tmp_path / "vsf.csv"
    cosmology = tmp_path / "pinocchio.cosmology.out"
    write_vsf_csv(vsf)
    write_cosmology(cosmology)

    exit_code = main(
        [
            str(vsf),
            "--box-size",
            "10.0",
            "--cosmology-file",
            str(cosmology),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "theory/vide" in captured.out
    assert "theory/finder" in captured.out
    assert "1.00000000e+03" in captured.out
    assert "decomposition,A" in captured.out
