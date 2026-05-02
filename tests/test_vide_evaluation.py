import numpy as np
import pytest

from pinocchio_voids.evaluation import (
    EvaluationError,
    compare_void_size_functions,
    compute_void_size_function,
)
from pinocchio_voids.io import VideCatalogError, read_vide_void_desc


FIXTURE = "tests/fixtures/vide_voidDesc_all_small.out"


def test_read_vide_void_desc_computes_effective_radii() -> None:
    catalog = read_vide_void_desc(FIXTURE)

    assert len(catalog) == 2
    assert catalog.void_ids.tolist() == [0, 1]
    assert catalog.summary == "4 particles, 2 voids."
    np.testing.assert_allclose(catalog.effective_radii_mpc_h, [1.0, 2.0])


def test_read_vide_void_desc_rejects_missing_void_volume(tmp_path) -> None:
    invalid = tmp_path / "voidDesc_invalid.out"
    invalid.write_text("summary\nVoid# Radius\n0 1.0\n", encoding="utf-8")

    with pytest.raises(VideCatalogError, match="VoidVol"):
        read_vide_void_desc(invalid)


def test_compute_void_size_function_counts_per_log_radius_volume() -> None:
    size_function = compute_void_size_function(
        [1.0, 2.0],
        box_size_mpc_h=10.0,
        bins=[0.5, 1.5, 2.5],
    )

    np.testing.assert_allclose(size_function.bin_centers_mpc_h, [np.sqrt(0.75), np.sqrt(3.75)])
    np.testing.assert_array_equal(size_function.counts, [1, 1])
    expected_density = np.array([1.0, 1.0]) / (
        10.0**3 * np.diff(np.log([0.5, 1.5, 2.5]))
    )
    np.testing.assert_allclose(size_function.density_dndlnr_per_mpc_h3, expected_density)


def test_compare_void_size_functions_uses_shared_bins() -> None:
    comparison = compare_void_size_functions(
        [1.0, 2.0],
        [1.0, 1.2, 2.4],
        box_size_mpc_h=10.0,
        bins=[0.5, 1.5, 2.5],
    )

    np.testing.assert_array_equal(comparison.predicted.counts, [1, 1])
    np.testing.assert_array_equal(comparison.reference.counts, [2, 1])
    np.testing.assert_allclose(
        comparison.predicted.bin_edges_mpc_h,
        comparison.reference.bin_edges_mpc_h,
    )
    assert comparison.count_l1_difference == 1


def test_compute_void_size_function_rejects_non_positive_radii() -> None:
    with pytest.raises(EvaluationError, match="positive finite radii"):
        compute_void_size_function([1.0, 0.0], box_size_mpc_h=10.0)
