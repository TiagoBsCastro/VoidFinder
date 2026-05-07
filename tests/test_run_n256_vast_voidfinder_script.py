from __future__ import annotations

import numpy as np

from scripts.run_n256_vast_voidfinder import (
    parse_args,
    load_final_halo_positions,
    normalize_vast_tables,
    requested_min_maximal_radii,
    run_label,
)


def test_load_final_halo_positions_uses_pinocchio_final_columns() -> None:
    ids, positions, velocities, masses = load_final_halo_positions(
        "tests/fixtures/pinocchio_pair_a.out",
        box_size_mpc_h=10.0,
    )

    assert ids.tolist() == [101, 102]
    np.testing.assert_allclose(positions, [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0]])
    np.testing.assert_allclose(velocities, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(masses, [1.0e12, 1.0e12])


def test_normalize_vast_tables_writes_stable_maximal_and_hole_rows() -> None:
    maximal_table = {
        "x": [1.0, 5.0],
        "y": [2.0, 6.0],
        "z": [3.0, 7.0],
        "radius": [4.0, 8.0],
        "void": [10, 20],
    }
    holes_table = {
        "x": [1.0, 1.5, 5.0],
        "y": [2.0, 2.5, 6.0],
        "z": [3.0, 3.5, 7.0],
        "radius": [4.0, 2.0, 8.0],
        "void": [10, 10, 20],
    }

    maximal_rows, hole_rows = normalize_vast_tables(maximal_table, holes_table)

    assert maximal_rows[0]["void_id"] == 10
    assert maximal_rows[0]["n_holes"] == 2
    assert maximal_rows[1]["void_id"] == 20
    assert maximal_rows[1]["maximal_radius_mpc_h"] == 8.0
    assert hole_rows[1]["hole_id"] == 1
    assert hole_rows[1]["void_id"] == 10


def test_vast_runner_labels_radius_and_wall_field_variants() -> None:
    assert run_label("n256", wall_field_separation=False, min_maximal_radius=10.0) == "n256"
    assert run_label("n256", wall_field_separation=False, min_maximal_radius=15.0) == "n256_rmin15"
    assert run_label("n256", wall_field_separation=True, min_maximal_radius=10.0) == "n256_wall"
    assert run_label("n256", wall_field_separation=True, min_maximal_radius=12.5) == "n256_wall_rmin12p5"

    args = parse_args(["--min-maximal-radius", "15", "--min-maximal-radius", "20"])
    assert requested_min_maximal_radii(args) == (15.0, 20.0)
