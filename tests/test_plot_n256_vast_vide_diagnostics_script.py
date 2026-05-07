from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.plot_n256_vast_vide_diagnostics import (
    estimate_periodic_union_volume,
    read_vast_catalog,
    select_slice_rows,
)


def test_estimate_periodic_union_volume_handles_boundary_crossing_sphere() -> None:
    volume = estimate_periodic_union_volume(
        np.asarray([[9.8, 5.0, 5.0]], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        box_size_mpc_h=10.0,
        voxel_size_mpc_h=0.25,
    )

    np.testing.assert_allclose(volume, 4.0 * np.pi / 3.0, rtol=0.08)


def test_read_vast_catalog_estimates_reff_from_holes(tmp_path: Path) -> None:
    run_dir = tmp_path / "n256"
    run_dir.mkdir(parents=True)
    with (run_dir / "vast_voids_maximal.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "void_id",
                "x_mpc_h",
                "y_mpc_h",
                "z_mpc_h",
                "maximal_radius_mpc_h",
                "n_holes",
                "source_row",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "void_id": 7,
                "x_mpc_h": 1.0,
                "y_mpc_h": 2.0,
                "z_mpc_h": 3.0,
                "maximal_radius_mpc_h": 1.0,
                "n_holes": 1,
                "source_row": 0,
            }
        )
    with (run_dir / "vast_voids_holes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["hole_id", "void_id", "x_mpc_h", "y_mpc_h", "z_mpc_h", "radius_mpc_h"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "hole_id": 0,
                "void_id": 7,
                "x_mpc_h": 1.0,
                "y_mpc_h": 2.0,
                "z_mpc_h": 3.0,
                "radius_mpc_h": 1.0,
            }
        )

    catalog = read_vast_catalog(
        tmp_path,
        target="A",
        box_size_mpc_h=10.0,
        voxel_size_mpc_h=0.25,
        max_voxels_per_void=100_000,
    )

    assert catalog.void_ids.tolist() == [7]
    np.testing.assert_allclose(catalog.maximal_radii_mpc_h, [1.0])
    np.testing.assert_allclose(catalog.reff_radii_mpc_h, [1.0], rtol=0.08)


def test_select_slice_rows_uses_sphere_intersections() -> None:
    args = type(
        "Args",
        (),
        {
            "slice_axis": "z",
            "slice_center": 5.0,
            "slice_thickness": 2.0,
            "box_size": 10.0,
        },
    )()
    rows = select_slice_rows(
        method="vast",
        target="A",
        positions_mpc_h=np.asarray([[1.0, 1.0, 5.0], [2.0, 2.0, 7.5]], dtype=np.float64),
        radii_mpc_h=np.asarray([1.0, 1.0], dtype=np.float64),
        void_ids=np.asarray([1, 2], dtype=np.int64),
        args=args,
    )

    assert rows.void_ids.tolist() == [1]
    np.testing.assert_allclose(rows.display_radii_mpc_h, [1.0])
