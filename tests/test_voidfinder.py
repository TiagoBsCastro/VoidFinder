import numpy as np
import pytest

from pinocchio_voids.catalog import HaloCatalog
from pinocchio_voids.voidfinder import (
    PairedVoidFinderConfig,
    Protovoid,
    ProtovoidCatalog,
    SourceCluster,
    SourceClusterCatalog,
    build_protovoid_adjacency,
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
