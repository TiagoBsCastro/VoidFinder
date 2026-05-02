import numpy as np
import pytest

from pinocchio_voids.catalog import (
    CatalogValidationError,
    HaloCatalog,
    TracerCatalog,
)


def test_halo_catalog_validates_and_freezes_arrays() -> None:
    catalog = HaloCatalog(
        ids=[1, 2],
        masses_msun_h=[1.0e13, 2.0e13],
        positions_mpc_h=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        velocities_km_s=[[10.0, 20.0, 30.0], [-10.0, -20.0, -30.0]],
        particle_counts=[20, 40],
        box_size_mpc_h=256.0,
    )

    assert len(catalog) == 2
    assert catalog.position_unit == "Mpc/h"
    assert catalog.mass_unit == "Msun/h"
    assert catalog.velocity_unit == "km/s"
    assert not catalog.positions_mpc_h.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        catalog.positions_mpc_h[0, 0] = 99.0


def test_halo_catalog_rejects_inconsistent_shapes() -> None:
    with pytest.raises(CatalogValidationError, match="positions_mpc_h"):
        HaloCatalog(
            ids=[1, 2],
            masses_msun_h=[1.0e13, 2.0e13],
            positions_mpc_h=[[1.0, 2.0, 3.0]],
            velocities_km_s=[[10.0, 20.0, 30.0], [-10.0, -20.0, -30.0]],
            particle_counts=[20, 40],
            box_size_mpc_h=256.0,
        )


def test_halo_catalog_converts_to_tracer_catalog() -> None:
    halos = HaloCatalog(
        ids=[1, 2],
        masses_msun_h=[1.0e13, 2.0e13],
        positions_mpc_h=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        velocities_km_s=[[10.0, 20.0, 30.0], [-10.0, -20.0, -30.0]],
        particle_counts=[20, 40],
        box_size_mpc_h=256.0,
    )

    tracers = halos.to_tracer_catalog()

    assert isinstance(tracers, TracerCatalog)
    assert tracers.weight_unit == "Msun/h"
    np.testing.assert_allclose(tracers.weights, halos.masses_msun_h)
    np.testing.assert_allclose(tracers.positions_mpc_h, halos.positions_mpc_h)


def test_tracer_catalog_defaults_to_unit_weights() -> None:
    tracers = TracerCatalog(
        ids=[1, 2],
        positions_mpc_h=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        box_size_mpc_h=128.0,
    )

    np.testing.assert_allclose(tracers.weights, [1.0, 1.0])
    assert tracers.velocities_km_s is None
