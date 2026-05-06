import csv

import numpy as np

from scripts.match_n256_void_centers import main, periodic_pairwise_distances


def test_periodic_pairwise_distances_uses_minimum_image() -> None:
    distances = periodic_pairwise_distances(
        np.array([[9.8, 0.0, 0.0]]),
        np.array([[0.1, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        box_size_mpc_h=10.0,
    )

    np.testing.assert_allclose(distances, [[0.3, 4.8]])


def test_match_void_centers_script_writes_match_and_summary_csv(tmp_path) -> None:
    output_csv = tmp_path / "matches.csv"
    summary_csv = tmp_path / "summary.csv"

    exit_code = main(
        [
            "--catalog-a",
            "tests/fixtures/pinocchio_pair_a.out",
            "--catalog-b",
            "tests/fixtures/pinocchio_pair_b.out",
            "--vide-desc-a",
            "tests/fixtures/vide_voidDesc_all_small.out",
            "--vide-desc-b",
            "tests/fixtures/vide_voidDesc_all_small.out",
            "--vide-centers-a",
            "tests/fixtures/vide_centers_all_small.out",
            "--vide-centers-b",
            "tests/fixtures/vide_centers_all_small.out",
            "--vide-macrocenters-a",
            "tests/fixtures/vide_macrocenters_all_small.out",
            "--vide-macrocenters-b",
            "tests/fixtures/vide_macrocenters_all_small.out",
            "--box-size",
            "10.0",
            "--rho-bar",
            "1.0",
            "--linking-length",
            "0.3",
            "--radius-a0",
            "1.0",
            "--radius-alpha",
            "1.0",
            "--adjacency-factor",
            "1.0",
            "--merge-score-mode",
            "weighted",
            "--merge-threshold",
            "0.2",
            "--bridge-radius-factor",
            "0.5",
            "--bridge-weight",
            "0.3",
            "--compatibility-weight",
            "0.1",
            "--target",
            "A",
            "--output-csv",
            str(output_csv),
            "--summary-csv",
            str(summary_csv),
        ]
    )

    assert exit_code == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["target"] == "A"
    assert rows[0]["vide_variant"] == "all"
    assert float(rows[0]["center_distance_mpc_h"]) > 0.0
    assert float(rows[0]["distance_over_min_reff"]) > 0.0
    assert summary_rows[0]["target"] == "A"
    assert summary_rows[0]["vide_variant"] == "all"
    assert summary_rows[0]["finder_count"] == "1"
    assert summary_rows[0]["matched_finder_count"] == "1"
