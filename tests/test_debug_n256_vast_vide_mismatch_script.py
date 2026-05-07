from __future__ import annotations

import numpy as np

from scripts.debug_n256_vast_vide_mismatch import (
    match_summary_rows,
    nearest_match_rows,
    periodic_nearest,
    radius_summary_row,
    vsf_rows,
)


def test_periodic_nearest_matches_across_box_boundary() -> None:
    distances, indices = periodic_nearest(
        np.asarray([[9.8, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[0.1, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64),
        box_size_mpc_h=10.0,
    )

    np.testing.assert_allclose(distances, [0.3])
    assert indices.tolist() == [0]


def test_nearest_match_rows_reports_radius_overlap_metrics() -> None:
    rows = nearest_match_rows(
        target="A",
        variant="all",
        direction="vast_to_vide",
        query_source="vast",
        reference_source="vide",
        radius_mode="reff",
        query_positions_mpc_h=np.asarray([[9.8, 0.0, 0.0]], dtype=np.float64),
        query_radii_mpc_h=np.asarray([1.0], dtype=np.float64),
        query_ids=np.asarray([10], dtype=np.int64),
        reference_positions_mpc_h=np.asarray([[0.1, 0.0, 0.0]], dtype=np.float64),
        reference_radii_mpc_h=np.asarray([2.0], dtype=np.float64),
        reference_ids=np.asarray([20], dtype=np.int64),
        box_size_mpc_h=10.0,
    )

    assert rows[0]["query_id"] == 10
    assert rows[0]["reference_id"] == 20
    assert rows[0]["center_inside_reference_radius"] == 1
    assert float(rows[0]["sphere_intersection_margin_mpc_h"]) > 0.0

    summary = match_summary_rows(rows)
    assert summary[0]["count"] == 1
    assert summary[0]["center_inside_reference_fraction"] == 1.0


def test_radius_and_vsf_rows_use_shared_bins() -> None:
    radii = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    summary = radius_summary_row(
        target="A",
        variant="all",
        source="vast",
        radius_mode="maximal",
        radii_mpc_h=radii,
    )
    rows = vsf_rows(
        target="A",
        variant="all",
        source="vast",
        radius_mode="maximal",
        radii_mpc_h=radii,
        box_size_mpc_h=10.0,
        edges_mpc_h=np.asarray([1.0, 3.0, 5.0], dtype=np.float64),
    )

    assert summary["count"] == 3
    assert rows[0]["count"] == 2
    assert rows[0]["count_ge_bin_min"] == 3
    assert rows[1]["count"] == 1
