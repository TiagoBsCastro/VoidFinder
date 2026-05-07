import numpy as np

from pinocchio_voids.catalog import HaloCatalog
from pinocchio_voids.io import read_paired_pinocchio_halo_catalogs


FIXTURE_A = "tests/fixtures/pinocchio_pair_a.out"
FIXTURE_B = "tests/fixtures/pinocchio_pair_b.out"


def test_read_paired_pinocchio_halo_catalogs_returns_canonical_catalogs() -> None:
    paired = read_paired_pinocchio_halo_catalogs(
        FIXTURE_A,
        FIXTURE_B,
        box_size_mpc_h=10.0,
    )

    assert paired.box_size_mpc_h == 10.0
    assert paired.position_mode == "final"
    assert paired.path_a.name == "pinocchio_pair_a.out"
    assert paired.path_b.name == "pinocchio_pair_b.out"
    assert isinstance(paired.catalog_a, HaloCatalog)
    assert isinstance(paired.catalog_b, HaloCatalog)
    assert paired.catalog_a.ids.tolist() == [101, 102]
    assert paired.catalog_b.ids.tolist() == [201, 202]
    np.testing.assert_allclose(paired.catalog_a.positions_mpc_h[:, 0], [9.9, 0.1])
    np.testing.assert_allclose(paired.catalog_b.masses_msun_h, [3.0e12, 3.0e12])


def test_read_paired_pinocchio_halo_catalogs_can_use_initial_positions() -> None:
    paired = read_paired_pinocchio_halo_catalogs(
        FIXTURE_A,
        FIXTURE_B,
        box_size_mpc_h=10.0,
        position_mode="initial",
    )

    assert paired.position_mode == "initial"
    assert paired.catalog_a.position_mode == "initial"
    assert paired.catalog_b.position_mode == "initial"
    np.testing.assert_allclose(paired.catalog_a.positions_mpc_h[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(paired.catalog_b.positions_mpc_h[:, 0], [0.0, 0.0])
