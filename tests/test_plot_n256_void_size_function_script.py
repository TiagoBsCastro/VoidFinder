import argparse
from pathlib import Path

from scripts.plot_n256_void_size_function import N256_RUN, _run_args


def make_args(*, include_theory: bool) -> argparse.Namespace:
    return argparse.Namespace(
        box_size=256.0,
        rho_bar=8.63025e10,
        bins=12,
        binning="log",
        bin_min=None,
        bin_max=None,
        paper_bins=False,
        include_theory=include_theory,
        vide_variant="all",
        output_dir=Path("runs/void-statistics"),
        linking_factor=N256_RUN["linking_factor"],
        radius_a0=N256_RUN["radius_a0"],
        radius_alpha=N256_RUN["radius_alpha"],
        adjacency_factor=N256_RUN["adjacency_factor"],
        merge_score_mode="geometry_only",
        merge_threshold="0.0",
        geom_weight="1.0",
        bridge_weight="0.0",
        compatibility_weight="0.0",
        bridge_radius_factor="0.5",
        bridge_min_radius="0.0",
        bridge_delta_scale="1.0",
        bridge_density_mode="mass",
    )


def test_plot_driver_does_not_include_theory_by_default() -> None:
    command = _run_args(make_args(include_theory=False))

    assert "--theory" not in command
    assert "--cosmology-file" not in command
    assert "runs/void-statistics/n256_finder_vide_vsf.csv" in command


def test_plot_driver_can_include_theory_overlay() -> None:
    command = _run_args(make_args(include_theory=True))

    assert command[command.index("--theory") + 1] == "vdn-svdw"
    assert "--cosmology-file" in command
    assert "runs/void-statistics/n256_finder_vide_theory_vsf.csv" in command


def test_plot_driver_can_switch_vide_variant() -> None:
    args = make_args(include_theory=False)
    args.vide_variant = "trimmed_nodencut"

    command = _run_args(args)

    assert any(
        entry.endswith("trimmed_nodencut_voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out")
        for entry in command
    )
    assert "runs/void-statistics/n256_finder_vide_vsf_trimmed_nodencut.csv" in command
