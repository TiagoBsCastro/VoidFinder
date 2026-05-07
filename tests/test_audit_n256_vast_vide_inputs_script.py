from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.audit_n256_vast_vide_inputs import (
    TracerSnapshot,
    catalog_summary_row,
    compare_snapshots,
    duplicate_position_count,
    nearest_neighbor_distances,
)


def _snapshot(name: str, ids, positions, *, box_size: float = 10.0) -> TracerSnapshot:
    ids_array = np.asarray(ids, dtype=np.int64)
    return TracerSnapshot(
        source_name=name,
        ids=ids_array,
        positions_mpc_h=np.asarray(positions, dtype=np.float64),
        velocities_km_s=np.zeros((len(ids_array), 3), dtype=np.float64),
        masses_msun_h=np.ones(len(ids_array), dtype=np.float64),
        path=Path(f"{name}.csv"),
        box_size_mpc_h=box_size,
    )


def test_compare_snapshots_reports_exact_matching_tracers() -> None:
    left = _snapshot("pinocchio", [3, 1], [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0]])
    right = _snapshot("vide", [1, 3], [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])

    row = compare_snapshots(target="A", left=left, right=right)

    assert row["common_id_count"] == 2
    assert row["position_match_count_1e-6"] == 2
    assert row["position_delta_max_mpc_h"] == 0.0


def test_compare_snapshots_detects_shifted_or_scaled_positions() -> None:
    left = _snapshot("pinocchio", [1, 2], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    right = _snapshot("vide", [1, 2], [[2.0, 1.0, 1.0], [4.0, 2.0, 2.0]])

    row = compare_snapshots(target="A", left=left, right=right)

    assert row["common_id_count"] == 2
    assert row["position_match_count_1e-6"] == 0
    assert float(row["position_delta_max_mpc_h"]) > 1.0


def test_catalog_summary_reports_duplicates_and_periodic_neighbor_scales() -> None:
    snapshot = _snapshot(
        "pinocchio",
        [1, 2, 3, 4],
        [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
    )

    row = catalog_summary_row("A", snapshot)
    nn1 = nearest_neighbor_distances(snapshot.positions_mpc_h, box_size_mpc_h=10.0, neighbor_index=1)

    assert row["duplicate_position_count"] == 1
    assert duplicate_position_count(snapshot.positions_mpc_h) == 1
    assert np.min(nn1) == 0.0
