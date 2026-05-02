import numpy as np

from pinocchio_voids.calibration import (
    mean_halo_spacing_mpc_h,
    score_direction_against_vide,
    sweep_geometry_parameters,
)
from pinocchio_voids.catalog import HaloCatalog
from pinocchio_voids.io import read_vide_void_desc
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    FinalVoidCatalog,
    PairedVoidFinderConfig,
    ProtovoidCatalog,
    SourceClusterCatalog,
    run_paired_halo_void_finder,
)


def make_halo_catalog(
    positions: list[list[float]],
    masses: list[float],
    *,
    box_size_mpc_h: float = 10.0,
) -> HaloCatalog:
    halo_count = len(positions)
    return HaloCatalog(
        ids=np.arange(1, halo_count + 1),
        masses_msun_h=masses,
        positions_mpc_h=positions,
        velocities_km_s=np.zeros((halo_count, 3)),
        particle_counts=np.ones(halo_count, dtype=np.int64),
        box_size_mpc_h=box_size_mpc_h,
    )


def test_score_direction_against_vide_uses_predicted_and_reference_counts() -> None:
    catalog_a = make_halo_catalog([[9.9, 0.0, 0.0], [0.1, 0.0, 0.0]], [1.0, 1.0])
    catalog_b = make_halo_catalog([[4.9, 0.0, 0.0], [5.1, 0.0, 0.0]], [3.0, 3.0])
    config = PairedVoidFinderConfig(
        linking_length_mpc_h=0.3,
        min_cluster_members=2,
        reference_rho_bar_msun_h_mpc3=1.0,
    )
    result = run_paired_halo_void_finder(catalog_a, catalog_b, config=config)
    reference = read_vide_void_desc("tests/fixtures/vide_voidDesc_all_small.out")

    score = score_direction_against_vide(
        result.voids_a,
        reference,
        box_size_mpc_h=10.0,
        bins=2,
    )

    assert score.target_label == "A"
    assert score.predicted_void_count == 1
    assert score.reference_void_count == 2
    assert score.count_l1_difference == 1


def test_sweep_geometry_parameters_sorts_by_total_l1() -> None:
    catalog_a = make_halo_catalog(
        [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]],
        [1.0, 1.0, 2.0],
    )
    catalog_b = make_halo_catalog(
        [[4.9, 0.0, 0.0], [5.1, 0.0, 0.0], [8.0, 0.0, 0.0]],
        [3.0, 3.0, 2.0],
    )
    reference = read_vide_void_desc("tests/fixtures/vide_voidDesc_all_small.out")

    results = sweep_geometry_parameters(
        catalog_a,
        catalog_b,
        reference_a=reference,
        reference_b=reference,
        reference_rho_bar_msun_h_mpc3=1.0,
        linking_lengths_mpc_h=[0.3, 4.0],
        radius_a0_values=[1.0],
        radius_alpha_values=[1.0],
        adjacency_factors=[1.0],
        min_cluster_members=1,
        bins=2,
    )

    assert len(results) == 2
    assert results[0].total_count_l1_difference <= results[1].total_count_l1_difference
    assert {result.config.linking_length_mpc_h for result in results} == {0.3, 4.0}


def test_mean_halo_spacing_uses_catalog_volume_per_halo() -> None:
    catalog = make_halo_catalog(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        [1.0, 1.0],
        box_size_mpc_h=8.0,
    )

    np.testing.assert_allclose(mean_halo_spacing_mpc_h(catalog), np.cbrt(8.0**3 / 2.0))


def test_sweep_geometry_parameters_accepts_mean_spacing_linking_factor() -> None:
    catalog_a = make_halo_catalog(
        [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]],
        [1.0, 1.0, 2.0],
    )
    catalog_b = make_halo_catalog(
        [[4.9, 0.0, 0.0], [5.1, 0.0, 0.0], [8.0, 0.0, 0.0], [8.2, 0.0, 0.0]],
        [3.0, 3.0, 2.0, 2.0],
    )
    reference = read_vide_void_desc("tests/fixtures/vide_voidDesc_all_small.out")

    results = sweep_geometry_parameters(
        catalog_a,
        catalog_b,
        reference_a=reference,
        reference_b=reference,
        reference_rho_bar_msun_h_mpc3=1.0,
        linking_lengths_mpc_h=[],
        linking_length_mean_spacing_factors=[0.05],
        radius_a0_values=[1.0],
        radius_alpha_values=[1.0],
        adjacency_factors=[1.0],
        min_cluster_members=1,
        bins=2,
    )

    assert len(results) == 1
    result = results[0]
    assert result.linking_mode == "mean_spacing"
    assert result.linking_value == 0.05
    np.testing.assert_allclose(
        result.source_a_linking_length_mpc_h,
        0.05 * mean_halo_spacing_mpc_h(catalog_a),
    )
    np.testing.assert_allclose(
        result.source_b_linking_length_mpc_h,
        0.05 * mean_halo_spacing_mpc_h(catalog_b),
    )
    assert result.config.linking_length_mpc_h == result.source_a_linking_length_mpc_h
    assert result.config.source_b_linking_length_mpc_h == result.source_b_linking_length_mpc_h


def test_score_direction_handles_zero_predicted_voids() -> None:
    reference = read_vide_void_desc("tests/fixtures/vide_voidDesc_all_small.out")
    empty_result = DirectionalVoidFinderResult(
        source_label="A",
        target_label="B",
        source_clusters=SourceClusterCatalog((), box_size_mpc_h=10.0),
        protovoids=ProtovoidCatalog((), box_size_mpc_h=10.0),
        adjacency_edges=(),
        voids=FinalVoidCatalog((), box_size_mpc_h=10.0, target_label="B"),
    )

    score = score_direction_against_vide(
        empty_result,
        reference,
        box_size_mpc_h=10.0,
        bins=2,
    )

    assert score.predicted_void_count == 0
    assert score.reference_void_count == 2
    assert score.predicted_reference_fraction == 0.0
    assert score.count_l1_difference == 2
    assert score.guarded_count_l1_difference == 4
    assert score.is_degenerate_underprediction
