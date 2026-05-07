import numpy as np
import pytest

from pinocchio_voids.catalog import HaloCatalog
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


def test_pinocchio_catalog_converts_to_canonical_halo_catalog() -> None:
    pinocchio_catalog = read_pinocchio_halo_catalog(FIXTURE)

    halo_catalog = pinocchio_catalog.to_halo_catalog(box_size_mpc_h=256.0)

    assert isinstance(halo_catalog, HaloCatalog)
    assert halo_catalog.box_size_mpc_h == 256.0
    assert halo_catalog.position_mode == "final"
    assert halo_catalog.ids.tolist() == [101, 102, 103]
    np.testing.assert_allclose(halo_catalog.positions_mpc_h[-1], [4.0, 255.0, 1.0])
    np.testing.assert_allclose(halo_catalog.masses_msun_h, [1.0e13, 2.5e13, 5.0e13])


def test_pinocchio_catalog_can_use_initial_positions() -> None:
    pinocchio_catalog = read_pinocchio_halo_catalog(FIXTURE)

    halo_catalog = pinocchio_catalog.to_halo_catalog(
        box_size_mpc_h=256.0,
        position_mode="initial",
    )

    assert halo_catalog.position_mode == "initial"
    np.testing.assert_allclose(halo_catalog.positions_mpc_h[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(halo_catalog.positions_mpc_h[-1], [13.0, 14.0, 15.0])


def test_pinocchio_initial_position_conversion_wraps_selected_columns(tmp_path) -> None:
    catalog_path = tmp_path / "initial_wrap.out"
    catalog_path.write_text(
        "1 1.0e12 260 -1 257 4 5 6 0 0 0 10\n",
        encoding="utf-8",
    )
    pinocchio_catalog = read_pinocchio_halo_catalog(catalog_path)

    halo_catalog = pinocchio_catalog.to_halo_catalog(
        box_size_mpc_h=256.0,
        position_mode="initial",
    )

    np.testing.assert_allclose(halo_catalog.positions_mpc_h[0], [4.0, 255.0, 1.0])


def test_pinocchio_catalog_rejects_unknown_position_mode() -> None:
    pinocchio_catalog = read_pinocchio_halo_catalog(FIXTURE)

    with pytest.raises(PinocchioCatalogError, match="position_mode"):
        pinocchio_catalog.to_halo_catalog(
            box_size_mpc_h=256.0,
            position_mode="lagrangian",  # type: ignore[arg-type]
        )


def test_pinocchio_conversion_requires_box_size() -> None:
    pinocchio_catalog = read_pinocchio_halo_catalog(FIXTURE)

    with pytest.raises(PinocchioCatalogError, match="box_size_mpc_h"):
        pinocchio_catalog.to_halo_catalog()
