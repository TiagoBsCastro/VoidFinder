import numpy as np

from scripts.plot_n256_halo_void_slice import (
    display_radii_for_slice,
    halo_slice_mask,
    main,
    parse_args,
    read_calibration_best_fit,
    resolve_finder_parameters,
    vide_member_positions_for_rows,
    void_slice_mask,
)
from scripts.plot_n256_void_slice import VoidSliceRows


def _small_cli(output_dir) -> list[str]:
    return [
        "--catalog-a",
        "tests/fixtures/pinocchio_pair_a.out",
        "--catalog-b",
        "tests/fixtures/pinocchio_pair_b.out",
        "--vide-desc-a",
        "tests/fixtures/vide_voidDesc_all_small.out",
        "--vide-desc-b",
        "tests/fixtures/vide_voidDesc_all_small.out",
        "--vide-centers-a",
        "tests/fixtures/vide_centers_all_small.out",
        "--vide-centers-b",
        "tests/fixtures/vide_centers_all_small.out",
        "--vide-macrocenters-a",
        "tests/fixtures/vide_macrocenters_all_small.out",
        "--vide-macrocenters-b",
        "tests/fixtures/vide_macrocenters_all_small.out",
        "--box-size",
        "10.0",
        "--rho-bar",
        "1.0",
        "--linking-length",
        "0.3",
        "--radius-a0",
        "1.0",
        "--radius-alpha",
        "1.0",
        "--adjacency-factor",
        "1.0",
        "--merge-score-mode",
        "weighted",
        "--merge-threshold",
        "0.2",
        "--bridge-radius-factor",
        "0.5",
        "--bridge-weight",
        "0.3",
        "--compatibility-weight",
        "0.1",
        "--slice-axis",
        "z",
        "--slice-center",
        "0.0",
        "--slice-thickness",
        "10.0",
        "--output-dir",
        str(output_dir),
    ]


def test_halo_slice_mask_uses_periodic_minimum_image() -> None:
    mask, distances = halo_slice_mask(
        positions_mpc_h=np.array([[9.9, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        axis="x",
        center_mpc_h=0.0,
        thickness_mpc_h=1.0,
        box_size_mpc_h=10.0,
    )

    np.testing.assert_array_equal(mask, [True, True, False])
    np.testing.assert_allclose(distances, [0.1, 0.2, 5.0])


def test_void_slice_mask_can_select_sphere_intersections() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5], [0.0, 0.0, 4.0]])
    radii = np.array([1.0, 2.0, 1.0])

    centers_only, _ = void_slice_mask(
        positions_mpc_h=positions,
        radii_mpc_h=radii,
        axis="z",
        center_mpc_h=0.0,
        thickness_mpc_h=2.0,
        box_size_mpc_h=10.0,
        selection="centers",
    )
    intersections, _ = void_slice_mask(
        positions_mpc_h=positions,
        radii_mpc_h=radii,
        axis="z",
        center_mpc_h=0.0,
        thickness_mpc_h=2.0,
        box_size_mpc_h=10.0,
        selection="intersections",
    )

    np.testing.assert_array_equal(centers_only, [True, False, False])
    np.testing.assert_array_equal(intersections, [True, True, False])


def test_display_radii_can_use_slab_cross_sections() -> None:
    display_radii = display_radii_for_slice(
        radii_mpc_h=np.array([2.0, 2.0, 2.0]),
        distance_to_slice_mpc_h=np.array([0.5, 1.5, 3.0]),
        thickness_mpc_h=2.0,
        radius_mode="cross-section",
    )

    np.testing.assert_allclose(display_radii, [2.0, np.sqrt(3.75), 0.0])


def test_calibration_summary_values_and_cli_overrides(tmp_path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "\n".join(
            [
                "parameter,best_fit",
                "linking_factor,0.11",
                "radius_a0,4.5",
                "radius_alpha,0.95",
                "adjacency_factor,0.33",
            ]
        ),
        encoding="utf-8",
    )

    assert read_calibration_best_fit(summary)["radius_a0"] == 4.5
    args = parse_args(["--calibration-summary", str(summary), "--radius-a0", "5.0"])
    parameters = resolve_finder_parameters(args)

    assert parameters["linking_factor"] == 0.11
    assert parameters["radius_a0"] == 5.0
    assert parameters["radius_alpha"] == 0.95
    assert parameters["adjacency_factor"] == 0.33


def test_halo_void_slice_script_writes_default_two_pngs(tmp_path) -> None:
    exit_code = main(_small_cli(tmp_path))

    assert exit_code == 0
    assert (tmp_path / "n256_halo_slice_finder.png").exists()
    assert (tmp_path / "n256_halo_slice_vide.png").exists()
    assert (tmp_path / "n256_halo_slice_finder.csv").exists()
    assert (tmp_path / "n256_halo_slice_vide.csv").exists()


def test_halo_void_slice_script_writes_four_pngs_for_both_targets(tmp_path) -> None:
    exit_code = main(_small_cli(tmp_path) + ["--target", "both"])

    assert exit_code == 0
    for target in ("a", "b"):
        assert (tmp_path / f"n256_halo_slice_target_{target}_finder.png").exists()
        assert (tmp_path / f"n256_halo_slice_target_{target}_vide.png").exists()
        assert (tmp_path / f"n256_halo_slice_target_{target}_finder.csv").exists()
        assert (tmp_path / f"n256_halo_slice_target_{target}_vide.csv").exists()


def test_halo_void_slice_script_center_selection_writes_diagnostic_csv(tmp_path) -> None:
    exit_code = main(_small_cli(tmp_path) + ["--void-selection", "centers"])

    assert exit_code == 0
    csv_path = tmp_path / "n256_halo_slice_vide.csv"
    text = csv_path.read_text(encoding="utf-8")
    assert "display_radius_mpc_h" in text
    assert "distance_to_slice_mpc_h" in text


def test_vide_member_positions_for_rows_selects_slab_members(tmp_path) -> None:
    input_path = tmp_path / "sample_z0.0.dat"
    input_path.write_text(
        "\n".join(
            [
                "10.0",
                "0.3",
                "0.7",
                "0.0",
                "3",
                "10 1 2 0 0 0 0 100",
                "11 4 5 4 0 0 0 100",
                "12 8 9 0 0 0 0 100",
            ]
        ),
        encoding="utf-8",
    )
    void_zones = tmp_path / "voidZone_sample.dat"
    void_zones.write_bytes(np.asarray([1, 2, 0, 1], dtype=np.int32).tobytes())
    zone_particles = tmp_path / "voidPart_sample.dat"
    zone_particles.write_bytes(
        np.asarray([3, 2, 1, 0, 2, 1, 2], dtype=np.int32).tobytes()
    )
    rows = VoidSliceRows(
        method="vide",
        target="A",
        positions_mpc_h=np.array([[0.0, 0.0, 0.0]]),
        radii_mpc_h=np.array([1.0]),
        void_ids=np.array([0]),
        file_void_ids=np.array([0]),
        distance_to_slice_mpc_h=np.array([0.0]),
        center_kind="center",
    )
    args = parse_args(
        [
            "--box-size",
            "10.0",
            "--slice-axis",
            "z",
            "--slice-center",
            "0.0",
            "--slice-thickness",
            "2.0",
        ]
    )

    positions = vide_member_positions_for_rows(
        rows=rows,
        input_path=input_path,
        void_zones_path=void_zones,
        zone_particles_path=zone_particles,
        args=args,
    )

    np.testing.assert_allclose(positions, [[1.0, 2.0, 0.0], [8.0, 9.0, 0.0]])
