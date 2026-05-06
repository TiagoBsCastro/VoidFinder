import csv
from pathlib import Path

import numpy as np

from scripts.plot_n256_void_slice import (
    resolve_target_vide_paths,
    periodic_axis_distance,
    slice_mask,
    main,
)


def test_periodic_axis_distance_uses_minimum_image() -> None:
    distances = periodic_axis_distance(
        np.array([1.0, 9.0, 5.0]),
        center_mpc_h=0.0,
        box_size_mpc_h=10.0,
    )

    np.testing.assert_allclose(distances, [1.0, 1.0, 5.0])


def test_slice_mask_can_include_sphere_intersections() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    radii = np.array([1.0, 3.0])

    centers_only, _ = slice_mask(
        positions_mpc_h=positions,
        radii_mpc_h=radii,
        axis="z",
        center_mpc_h=0.0,
        thickness_mpc_h=2.0,
        box_size_mpc_h=10.0,
        include_intersections=False,
    )
    intersections, _ = slice_mask(
        positions_mpc_h=positions,
        radii_mpc_h=radii,
        axis="z",
        center_mpc_h=0.0,
        thickness_mpc_h=2.0,
        box_size_mpc_h=10.0,
        include_intersections=True,
    )

    np.testing.assert_array_equal(centers_only, [True, False])
    np.testing.assert_array_equal(intersections, [True, True])


def test_resolve_target_vide_paths_switches_variant_prefixes() -> None:
    desc, centers, macrocenters = resolve_target_vide_paths(
        desc_path=Path("sample/voidDesc_all_demo.out"),
        centers_path=Path("sample/untrimmed_centers_all_demo.out"),
        macrocenters_path=Path("sample/macrocenters_all_demo.out"),
        variant="untrimmed_dencut",
    )

    assert desc == Path("sample/untrimmed_dencut_voidDesc_all_demo.out")
    assert centers == Path("sample/untrimmed_dencut_centers_all_demo.out")
    assert macrocenters == Path("sample/untrimmed_dencut_macrocenters_all_demo.out")


def test_plot_void_slice_script_writes_csv_and_png(tmp_path) -> None:
    output_png = tmp_path / "slice.png"
    output_csv = tmp_path / "slice.csv"

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
            "--vide-center-kind",
            "macrocenter",
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
            "--slice-axis",
            "z",
            "--slice-center",
            "3.0",
            "--slice-thickness",
            "10.0",
            "--show-nearest-matches",
            "--max-match-lines",
            "2",
            "--label-count",
            "1",
            "--output",
            str(output_png),
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    assert output_png.exists()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["method"] for row in rows} == {"finder", "vide"}
    assert {row["target"] for row in rows} == {"A"}
    assert any(row["file_void_id"] == "0" for row in rows)
