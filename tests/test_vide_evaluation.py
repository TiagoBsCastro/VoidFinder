import numpy as np
import pytest

from pinocchio_voids.evaluation import (
    EvaluationError,
    compare_void_size_functions,
    compute_void_size_function,
)
from pinocchio_voids.io import (
    VideCatalogError,
    normalize_vide_catalog_variant,
    read_vide_input_tracers,
    read_vide_particle_volumes,
    read_vide_void_centers,
    read_vide_void_desc,
    read_vide_void_macrocenters,
    read_vide_void_zones,
    read_vide_zone_particles,
    resolve_vide_catalog_variant_path,
    strip_vide_catalog_variant_prefix,
    vide_catalog_variant_output_suffix,
    vide_particle_ids_for_file_void_id,
)


FIXTURE = "tests/fixtures/vide_voidDesc_all_small.out"


def test_read_vide_void_desc_computes_effective_radii() -> None:
    catalog = read_vide_void_desc(FIXTURE)

    assert len(catalog) == 2
    assert catalog.void_ids.tolist() == [0, 1]
    assert catalog.summary == "4 particles, 2 voids."
    np.testing.assert_allclose(catalog.effective_radii_mpc_h, [1.0, 2.0])


def test_read_vide_void_centers_reads_positions_and_file_ids() -> None:
    catalog = read_vide_void_centers("tests/fixtures/vide_centers_all_small.out")

    assert len(catalog) == 2
    np.testing.assert_allclose(catalog.positions_mpc_h, [[1.0, 2.0, 3.0], [8.0, 7.0, 6.0]])
    np.testing.assert_array_equal(catalog.file_void_ids, [0, 1])
    np.testing.assert_allclose(catalog.radii_mpc_h, [1.0, 2.0])


def test_read_vide_void_macrocenters_reads_positions_and_file_ids() -> None:
    catalog = read_vide_void_macrocenters("tests/fixtures/vide_macrocenters_all_small.out")

    assert len(catalog) == 2
    np.testing.assert_allclose(
        catalog.positions_mpc_h,
        [[1.5, 2.5, 3.5], [8.5, 7.5, 6.5]],
    )
    np.testing.assert_array_equal(catalog.file_void_ids, [0, 1])


def test_vide_catalog_variant_path_resolution_is_deterministic() -> None:
    path = "sample/untrimmed_voidDesc_all_demo.out"

    assert normalize_vide_catalog_variant("default") == "all"
    assert strip_vide_catalog_variant_prefix("untrimmed_dencut_voidDesc_all_demo.out") == (
        "voidDesc_all_demo.out"
    )
    assert resolve_vide_catalog_variant_path(path, "all").as_posix() == (
        "sample/voidDesc_all_demo.out"
    )
    assert resolve_vide_catalog_variant_path(path, "trimmed_nodencut").as_posix() == (
        "sample/trimmed_nodencut_voidDesc_all_demo.out"
    )
    assert vide_catalog_variant_output_suffix("all") == ""
    assert vide_catalog_variant_output_suffix("untrimmed") == "_untrimmed"


def test_read_vide_binary_membership_and_input_tracers(tmp_path) -> None:
    volume_path = tmp_path / "vol_sample.dat"
    volume_path.write_bytes(
        np.asarray([4], dtype=np.int32).tobytes()
        + np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
    )
    void_zone_path = tmp_path / "voidZone_sample.dat"
    void_zone_path.write_bytes(
        np.asarray([2, 2, 0, 1, 1, 2], dtype=np.int32).tobytes()
    )
    zone_particle_path = tmp_path / "voidPart_sample.dat"
    zone_particle_path.write_bytes(
        np.asarray([4, 3, 2, 0, 2, 1, 1, 1, 3], dtype=np.int32).tobytes()
    )
    tracer_path = tmp_path / "sample_z0.0.dat"
    tracer_path.write_text(
        "\n".join(
            [
                "10.0",
                "0.3",
                "0.7",
                "0.0",
                "4",
                "10 1 2 3 0 0 0 100",
                "11 4 5 6 0 0 0 200",
                "12 7 8 9 0 0 0 300",
                "13 2 3 4 0 0 0 400",
            ]
        ),
        encoding="utf-8",
    )

    volumes = read_vide_particle_volumes(volume_path)
    void_zones = read_vide_void_zones(void_zone_path)
    zone_particles = read_vide_zone_particles(zone_particle_path)
    tracers = read_vide_input_tracers(tracer_path)

    np.testing.assert_allclose(volumes.volumes, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(void_zones.zones_for_file_void_id(0), [0, 1])
    np.testing.assert_array_equal(zone_particles.particles_for_zone_id(0), [0, 2])
    np.testing.assert_array_equal(
        vide_particle_ids_for_file_void_id(
            file_void_id=0,
            void_zones=void_zones,
            zone_particles=zone_particles,
        ),
        [0, 1, 2],
    )
    np.testing.assert_allclose(tracers.positions_mpc_h[[0, 1, 2]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_read_vide_void_desc_converts_normalized_volumes_from_sample_info(tmp_path) -> None:
    catalog_path = tmp_path / "voidDesc_all_sample.out"
    catalog_path.write_text(
        "\n".join(
            [
                "4 particles, 1 void.",
                "Void# FileVoid# CoreParticle CoreDens ZoneVol Zone#Part Void#Zones VoidVol Void#Part VoidDensContrast VoidProb",
                "0 0 10 0.3 4.1887902047863905 1 1 4.1887902047863905 1 1.1 0.2",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "sample_info.txt").write_text(
        "Estimated mean tracer separation (Mpc/h): 2.0\n",
        encoding="utf-8",
    )

    catalog = read_vide_void_desc(catalog_path)

    assert catalog.volume_scale_mpc_h3 == 8.0
    np.testing.assert_allclose(catalog.void_volumes_mpc_h3, [4.1887902047863905 * 8.0])
    np.testing.assert_allclose(catalog.effective_radii_mpc_h, [2.0])


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


def test_compare_void_size_functions_allows_empty_predicted_side() -> None:
    comparison = compare_void_size_functions(
        [],
        [1.0, 2.0],
        box_size_mpc_h=10.0,
        bins=2,
    )

    np.testing.assert_array_equal(comparison.predicted.counts, [0, 0])
    np.testing.assert_array_equal(comparison.reference.counts, [1, 1])
    assert comparison.count_l1_difference == 2


def test_compute_void_size_function_rejects_non_positive_radii() -> None:
    with pytest.raises(EvaluationError, match="positive finite radii"):
        compute_void_size_function([1.0, 0.0], box_size_mpc_h=10.0)
