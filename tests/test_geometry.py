import numpy as np
import pytest

from pinocchio_voids.geometry import (
    GeometryError,
    PeriodicGeometryError,
    minimum_image_displacement,
    periodic_center_of_mass,
    periodic_distance,
    sphere_volume_from_radius,
    spherical_equivalent_radius_from_volume,
)


def test_minimum_image_displacement_crosses_box_boundary() -> None:
    displacement = minimum_image_displacement([9.5, 1.0, 1.0], [0.5, 1.0, 1.0], 10.0)

    np.testing.assert_allclose(displacement, [1.0, 0.0, 0.0])
    assert periodic_distance([9.5, 1.0, 1.0], [0.5, 1.0, 1.0], 10.0) == pytest.approx(1.0)


def test_periodic_center_of_mass_handles_boundary_crossing_pair() -> None:
    center = periodic_center_of_mass(
        [[9.9, 2.0, 3.0], [0.1, 2.0, 3.0]],
        box_size_mpc_h=10.0,
        weights=[1.0, 1.0],
    )

    np.testing.assert_allclose(center, [0.0, 2.0, 3.0], atol=1.0e-12)


def test_periodic_center_of_mass_rejects_ambiguous_axis() -> None:
    with pytest.raises(PeriodicGeometryError, match="ambiguous"):
        periodic_center_of_mass(
            [[2.5, 0.0, 0.0], [7.5, 0.0, 0.0]],
            box_size_mpc_h=10.0,
        )


def test_spherical_equivalent_radius_matches_sphere_volume() -> None:
    radii = np.asarray([1.0, 2.0, 3.0])

    volumes = sphere_volume_from_radius(radii)
    recovered = spherical_equivalent_radius_from_volume(volumes)

    np.testing.assert_allclose(
        volumes,
        (4.0 * np.pi / 3.0) * radii**3,
    )
    np.testing.assert_allclose(recovered, radii)
    assert spherical_equivalent_radius_from_volume((4.0 * np.pi / 3.0) * 4.0**3) == pytest.approx(4.0)


def test_spherical_equivalent_radius_rejects_non_positive_volume() -> None:
    with pytest.raises(GeometryError, match="positive finite"):
        spherical_equivalent_radius_from_volume([1.0, 0.0])
