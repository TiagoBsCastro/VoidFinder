import numpy as np
import pytest

from pinocchio_voids.catalog import HaloCatalog
from pinocchio_voids.voidfinder import (
    PairedVoidFinderConfig,
    Protovoid,
    ProtovoidCatalog,
    SourceCluster,
    SourceClusterCatalog,
    bridge_density_score,
    build_protovoid_adjacency,
    compatibility_score,
    find_source_clusters,
    merge_protovoids,
    protovoid_radius_from_mass,
    run_paired_halo_void_finder,
    source_clusters_to_protovoids,
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


def test_periodic_fof_clusters_across_box_boundary() -> None:
    catalog = make_halo_catalog(
        [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 5.0]],
        [1.0, 1.0, 10.0],
    )

    clusters = find_source_clusters(
        catalog,
        linking_length_mpc_h=0.3,
        min_cluster_members=2,
    )

    assert len(clusters) == 1
    cluster = clusters.clusters[0]
    assert cluster.member_indices.tolist() == [0, 1]
    assert cluster.richness == 2
    assert cluster.total_mass_msun_h == pytest.approx(2.0)
    np.testing.assert_allclose(cluster.center_mpc_h, [0.0, 0.0, 0.0], atol=1.0e-12)
    assert cluster.effective_radius_mpc_h == pytest.approx(0.1)
    assert cluster.shape_tensor_mpc_h2.shape == (3, 3)
    assert cluster.axis_ratio >= 1.0
    assert cluster.max_member_distance_mpc_h == pytest.approx(0.1)
    assert cluster.rms_radius_over_linking_length == pytest.approx(0.1 / 0.3)


def test_source_cluster_quality_cuts_filter_extended_clusters() -> None:
    catalog = make_halo_catalog(
        [[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.4, 0.0, 0.0]],
        [1.0, 1.0, 1.0],
    )

    clusters = find_source_clusters(
        catalog,
        linking_length_mpc_h=0.3,
        min_cluster_members=2,
        max_cluster_axis_ratio=2.0,
    )

    assert len(clusters) == 0


def test_source_clusters_map_to_spherical_protovoids() -> None:
    rho_bar = 2.0
    mass = (4.0 / 3.0) * np.pi * rho_bar * 2.0**3
    clusters = SourceClusterCatalog(
        (
            SourceCluster(
                id=0,
                member_indices=[0, 1],
                total_mass_msun_h=mass,
                center_mpc_h=[1.0, 2.0, 3.0],
                richness=2,
                effective_radius_mpc_h=0.5,
            ),
        ),
        box_size_mpc_h=10.0,
        source_label="A",
    )

    protovoids = source_clusters_to_protovoids(
        clusters,
        radius_a0=1.5,
        radius_alpha=1.0,
        reference_rho_bar_msun_h_mpc3=rho_bar,
        target_label="B",
    )

    assert len(protovoids) == 1
    assert protovoids.target_label == "B"
    assert protovoid_radius_from_mass(
        mass,
        radius_a0=1.5,
        radius_alpha=1.0,
        reference_rho_bar_msun_h_mpc3=rho_bar,
    ) == pytest.approx(3.0)
    assert protovoids.protovoids[0].radius_mpc_h == pytest.approx(3.0)
    np.testing.assert_allclose(protovoids.protovoids[0].center_mpc_h, [1.0, 2.0, 3.0])


def test_graph_adjacency_and_connected_component_merging() -> None:
    protovoids = ProtovoidCatalog(
        (
            Protovoid(0, 10, [9.8, 0.0, 0.0], 0.3, 1.0),
            Protovoid(1, 11, [0.2, 0.0, 0.0], 0.3, 2.0),
            Protovoid(2, 12, [5.0, 0.0, 0.0], 0.3, 4.0),
        ),
        box_size_mpc_h=10.0,
        target_label="A",
    )

    edges = build_protovoid_adjacency(protovoids, adjacency_factor=1.0)
    final_voids = merge_protovoids(protovoids, edges)

    assert len(edges) == 1
    assert (edges[0].protovoid_i, edges[0].protovoid_j) == (0, 1)
    assert edges[0].distance_mpc_h == pytest.approx(0.4)
    assert edges[0].geometric_score == pytest.approx(1.0 - 0.4 / 0.6)
    assert len(final_voids) == 2
    merged = final_voids.voids[0]
    assert merged.member_protovoid_ids.tolist() == [0, 1]
    assert merged.source_cluster_ids.tolist() == [10, 11]
    assert merged.total_source_mass_msun_h == pytest.approx(3.0)
    assert merged.effective_radius_mpc_h == pytest.approx((2.0 * 0.3**3) ** (1.0 / 3.0))
    np.testing.assert_allclose(merged.center_mpc_h, [0.0, 0.0, 0.0], atol=1.0e-12)
    assert len(final_voids.voids[0].member_protovoid_ids) == 2


def test_weighted_merge_threshold_blocks_weak_adjacency() -> None:
    protovoids = ProtovoidCatalog(
        (
            Protovoid(0, 10, [0.0, 0.0, 0.0], 1.0, 1.0),
            Protovoid(1, 11, [1.8, 0.0, 0.0], 1.0, 1.0),
        ),
        box_size_mpc_h=10.0,
        target_label="A",
    )

    edges = build_protovoid_adjacency(
        protovoids,
        adjacency_factor=1.0,
        merge_score_mode="weighted",
        geom_weight=1.0,
        merge_threshold=0.5,
    )
    final_voids = merge_protovoids(protovoids, edges)

    assert len(edges) == 1
    assert edges[0].geometric_score == pytest.approx(0.1)
    assert edges[0].merge_score == pytest.approx(0.1)
    assert edges[0].passes_merge_threshold is False
    assert len(final_voids) == 2


def test_bridge_density_score_can_promote_weighted_merge() -> None:
    source_catalog = make_halo_catalog(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [1.0, 1.0, 100.0],
    )
    source_clusters = SourceClusterCatalog(
        (
            SourceCluster(0, [0], 1.0, [0.0, 0.0, 0.0], 1, 0.0),
            SourceCluster(1, [1], 1.0, [4.0, 0.0, 0.0], 1, 0.0),
        ),
        box_size_mpc_h=10.0,
        source_label="A",
    )
    protovoids = ProtovoidCatalog(
        (
            Protovoid(0, 0, [0.0, 0.0, 0.0], 2.1, 1.0),
            Protovoid(1, 1, [4.0, 0.0, 0.0], 2.1, 1.0),
        ),
        box_size_mpc_h=10.0,
        target_label="B",
    )

    bridge = bridge_density_score(
        source_catalog,
        source_clusters.clusters[0],
        source_clusters.clusters[1],
        bridge_radius_factor=1.0,
        bridge_min_radius_mpc_h=0.2,
        bridge_density_mode="mass",
    )
    edges = build_protovoid_adjacency(
        protovoids,
        adjacency_factor=1.0,
        source_catalog=source_catalog,
        source_clusters=source_clusters,
        merge_score_mode="weighted",
        geom_weight=0.0,
        bridge_weight=1.0,
        merge_threshold=0.5,
        bridge_radius_factor=1.0,
        bridge_min_radius_mpc_h=0.2,
    )
    final_voids = merge_protovoids(protovoids, edges, source_clusters=source_clusters)

    assert bridge > 0.5
    assert edges[0].bridge_score == pytest.approx(bridge)
    assert edges[0].passes_merge_threshold is True
    assert len(final_voids) == 1
    assert final_voids.voids[0].mean_merge_score == pytest.approx(edges[0].merge_score)
    assert final_voids.voids[0].total_source_richness == 2


def test_compatibility_score_uses_radius_and_richness_similarity() -> None:
    score = compatibility_score(
        Protovoid(0, 0, [0.0, 0.0, 0.0], 2.0, 1.0),
        Protovoid(1, 1, [1.0, 0.0, 0.0], 4.0, 1.0),
        SourceCluster(0, [0, 1], 1.0, [0.0, 0.0, 0.0], 2, 0.0),
        SourceCluster(1, [2, 3, 4, 5], 1.0, [1.0, 0.0, 0.0], 4, 0.0),
    )

    assert score == pytest.approx(0.7 * 0.5 + 0.3 * 0.5)


def test_paired_pipeline_runs_symmetrically_on_tiny_catalogs() -> None:
    catalog_a = make_halo_catalog(
        [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0]],
        [1.0, 1.0],
    )
    catalog_b = make_halo_catalog(
        [[4.9, 0.0, 0.0], [5.1, 0.0, 0.0]],
        [3.0, 3.0],
    )
    config = PairedVoidFinderConfig(
        linking_length_mpc_h=0.3,
        min_cluster_members=2,
        reference_rho_bar_msun_h_mpc3=1.0,
        radius_a0=1.0,
        radius_alpha=1.0,
        adjacency_factor=1.0,
    )

    result = run_paired_halo_void_finder(
        catalog_a,
        catalog_b,
        config=config,
        label_a="A",
        label_b="B",
    )

    assert result.voids_b.source_label == "A"
    assert result.voids_b.target_label == "B"
    assert result.voids_a.source_label == "B"
    assert result.voids_a.target_label == "A"
    assert len(result.voids_b.source_clusters) == 1
    assert len(result.voids_a.source_clusters) == 1
    assert len(result.voids_b.voids) == 1
    assert len(result.voids_a.voids) == 1
    assert result.voids_b.voids.voids[0].total_source_mass_msun_h == pytest.approx(2.0)
    assert result.voids_a.voids.voids[0].total_source_mass_msun_h == pytest.approx(6.0)


def test_paired_pipeline_can_use_source_specific_linking_lengths() -> None:
    catalog_a = make_halo_catalog(
        [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0]],
        [1.0, 1.0],
    )
    catalog_b = make_halo_catalog(
        [[4.6, 0.0, 0.0], [5.4, 0.0, 0.0]],
        [3.0, 3.0],
    )
    config = PairedVoidFinderConfig(
        linking_length_mpc_h=0.3,
        source_b_linking_length_mpc_h=1.0,
        min_cluster_members=2,
        reference_rho_bar_msun_h_mpc3=1.0,
    )

    result = run_paired_halo_void_finder(catalog_a, catalog_b, config=config)

    assert len(result.voids_b.source_clusters) == 1
    assert len(result.voids_a.source_clusters) == 1
    assert result.voids_b.voids.voids[0].total_source_mass_msun_h == pytest.approx(2.0)
    assert result.voids_a.voids.voids[0].total_source_mass_msun_h == pytest.approx(6.0)
