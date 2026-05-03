import numpy as np
import pytest

from pinocchio_voids.theory import (
    TheoryError,
    compute_vdn_svdw_factors,
    compute_vdn_svdw_size_function,
    read_pinocchio_cosmology,
    svdw_first_crossing_fraction,
)


def write_cosmology(path) -> None:
    rows = [
        (0.5, 4.0, -1.0),
        (1.0, 2.0, -1.0),
        (2.0, 1.0, -1.0),
        (4.0, 0.5, -1.0),
        (8.0, 0.25, -1.0),
    ]
    lines = [
        "# Cosmological quantities used in PINOCCHIO (h=1.000000)",
        "#15: smoothing scale R (Mpc)",
        "#16: Gaussian-filtered mass variance sigma_G^2(R)",
        "#18: d Log sigma_G^2 / d Log R",
    ]
    for radius, sigma2, derivative in rows:
        columns = [1.0] * 20
        columns[14] = radius
        columns[15] = sigma2
        columns[17] = derivative
        lines.append(" ".join(f"{value:.8e}" for value in columns))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_read_pinocchio_cosmology_parses_scale_dependent_columns(tmp_path) -> None:
    path = tmp_path / "pinocchio.cosmology.out"
    write_cosmology(path)

    cosmology = read_pinocchio_cosmology(path)

    assert cosmology.h == 1.0
    np.testing.assert_allclose(cosmology.smoothing_radii_mpc, [0.5, 1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(cosmology.sigma2, [4.0, 2.0, 1.0, 0.5, 0.25])
    np.testing.assert_allclose(cosmology.dlog_sigma2_dlog_r, [-1.0] * 5)


def test_vdn_svdw_size_function_is_finite_on_supported_bins(tmp_path) -> None:
    path = tmp_path / "pinocchio.cosmology.out"
    write_cosmology(path)
    cosmology = read_pinocchio_cosmology(path)

    theory = compute_vdn_svdw_size_function([1.0, 2.0, 4.0], cosmology)

    assert theory.model == "vdn-svdw"
    assert np.all(np.isfinite(theory.density_dndlnr_per_mpc_h3))
    assert np.all(theory.density_dndlnr_per_mpc_h3 > 0.0)


def test_vdn_svdw_size_function_masks_bins_outside_cosmology_range(tmp_path) -> None:
    path = tmp_path / "pinocchio.cosmology.out"
    write_cosmology(path)
    cosmology = read_pinocchio_cosmology(path)

    theory = compute_vdn_svdw_size_function([0.01, 1.0, 4.0], cosmology)

    assert np.isnan(theory.density_dndlnr_per_mpc_h3[0])
    assert np.isfinite(theory.density_dndlnr_per_mpc_h3[1])


def test_vdn_svdw_factors_expose_intermediate_terms(tmp_path) -> None:
    path = tmp_path / "pinocchio.cosmology.out"
    write_cosmology(path)
    cosmology = read_pinocchio_cosmology(path)

    factors = compute_vdn_svdw_factors([2.0], cosmology)

    np.testing.assert_allclose(factors.eulerian_radii_mpc_h, [2.0])
    assert factors.valid.tolist() == [True]
    assert factors.lagrangian_radii_mpc_h[0] < factors.eulerian_radii_mpc_h[0]
    assert np.isfinite(factors.sigma[0])
    assert np.isfinite(factors.first_crossing_fraction[0])
    assert np.isfinite(factors.density_dndlnr_per_mpc_h3[0])


def test_svdw_first_crossing_rejects_invalid_barriers() -> None:
    with pytest.raises(TheoryError, match="delta_v_linear"):
        svdw_first_crossing_fraction([1.0], delta_v_linear=2.7)
