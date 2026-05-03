import argparse
from pathlib import Path

from scripts.plot_n128_n256_void_size_functions import _run_args


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
        output_dir=Path("runs/void-statistics"),
        n128_linking_factor="0.15",
        n256_linking_factor="0.13",
        n128_radius_a0="4.5",
        n128_radius_alpha="1.05",
        n128_adjacency_factor="0.40",
        n256_radius_a0="6.5",
        n256_radius_alpha="1.0",
        n256_adjacency_factor="0.30",
    )


def test_plot_driver_does_not_include_theory_by_default() -> None:
    command = _run_args("n256", make_args(include_theory=False))

    assert "--theory" not in command
    assert "--cosmology-file" not in command
    assert "runs/void-statistics/n256_finder_vide_vsf.csv" in command


def test_plot_driver_can_include_theory_overlay() -> None:
    command = _run_args("n256", make_args(include_theory=True))

    assert command[command.index("--theory") + 1] == "vdn-svdw"
    assert "--cosmology-file" in command
    assert "runs/void-statistics/n256_finder_vide_theory_vsf.csv" in command
