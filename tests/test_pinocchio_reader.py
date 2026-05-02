import numpy as np
import pytest

from pinocchio_voids.io import PinocchioCatalogError, read_pinocchio_halo_catalog


FIXTURE = "tests/fixtures/pinocchio_catalog_small.out"


def test_read_pinocchio_halo_catalog() -> None:
    catalog = read_pinocchio_halo_catalog(FIXTURE)

    assert len(catalog) == 3
    assert catalog.group_ids.tolist() == [101, 102, 103]
    assert catalog.final_positions_mpc_h.shape == (3, 3)
    assert catalog.velocities_km_s[1].tolist() == [-10.0, -20.0, -30.0]
    assert catalog.n_particles.tolist() == [20, 50, 100]
    np.testing.assert_allclose(catalog.masses_msun_h, [1.0e13, 2.5e13, 5.0e13])


def test_read_pinocchio_halo_catalog_wraps_final_positions() -> None:
    catalog = read_pinocchio_halo_catalog(
        FIXTURE,
        box_size_mpc_h=256.0,
        wrap_positions=True,
    )

    np.testing.assert_allclose(catalog.final_positions_mpc_h[-1], [4.0, 255.0, 1.0])


def test_read_pinocchio_halo_catalog_rejects_wrong_column_count(tmp_path) -> None:
    invalid_catalog = tmp_path / "invalid.out"
    invalid_catalog.write_text("1 2 3\n", encoding="utf-8")

    with pytest.raises(PinocchioCatalogError, match="12 columns"):
        read_pinocchio_halo_catalog(invalid_catalog)
