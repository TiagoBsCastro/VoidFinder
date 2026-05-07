import csv
from pathlib import Path

import numpy as np

from scripts.debug_n256_vide_region import (
    SpatialVoidCatalog,
    halo_region_rows,
    main,
    variant_path,
    void_region_rows,
)


def test_variant_path_adds_vide_catalog_prefix() -> None:
    path = Path("sample/voidDesc_all_demo.out")

    assert variant_path(path, "all") == path
    assert variant_path(path, "untrimmed") == Path("sample/untrimmed_voidDesc_all_demo.out")
    assert variant_path(
        Path("sample/untrimmed_voidDesc_all_demo.out"),
        "trimmed_nodencut",
    ) == Path("sample/trimmed_nodencut_voidDesc_all_demo.out")


def test_void_region_rows_reports_3d_and_projected_coverage() -> None:
    catalog = SpatialVoidCatalog(
        method="vide",
        target="A",
        catalog_variant="default",
        position_mode="final",
        center_kind="center",
        positions_mpc_h=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        radii_mpc_h=np.array([2.0, 1.0]),
        void_ids=np.array([10, 11]),
        file_void_ids=np.array([20, 21]),
    )

    rows = void_region_rows(
        catalog=catalog,
        point_mpc_h=np.array([1.0, 0.0, 0.0]),
        slice_axis="z",
        slice_center_mpc_h=0.0,
        slice_thickness_mpc_h=2.0,
        box_size_mpc_h=10.0,
        top_n=1,
    )

    assert rows[0]["contains_point_3d_count"] == 1
    assert rows[0]["covers_point_projected_count"] == 1
    assert rows[0]["file_void_id"] == 20
    assert rows[0]["margin_3d_mpc_h"] == -1.0


def test_halo_region_rows_counts_periodic_projected_disks() -> None:
    rows = halo_region_rows(
        target="A",
        position_mode="final",
        positions_mpc_h=np.array(
            [[9.8, 0.0, 0.0], [0.4, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 0.0, 3.0]]
        ),
        point_mpc_h=np.array([0.0, 0.0, 0.0]),
        slice_axis="z",
        slice_center_mpc_h=0.0,
        slice_thickness_mpc_h=2.0,
        box_size_mpc_h=10.0,
        radii_mpc_h=[0.5],
    )

    assert rows[0]["halo_slab_count"] == 3
    assert rows[0]["position_mode"] == "final"
    assert rows[0]["halo_count"] == 2


def test_debug_region_script_writes_csv_with_small_fixtures(tmp_path) -> None:
    output = tmp_path / "debug.csv"

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
            "--target",
            "A",
            "--slice-axis",
            "z",
            "--slice-center",
            "0.0",
            "--slice-thickness",
            "10.0",
            "--halo-radius",
            "1.0",
            "--vide-variant",
            "all",
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
            "--skip-finder",
            "--output-csv",
            str(output),
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert {row["row_type"] for row in rows} == {"halo_count", "void_margin"}
    assert {row["position_mode"] for row in rows} == {"final"}
