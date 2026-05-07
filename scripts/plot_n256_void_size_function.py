#!/usr/bin/env python
"""Generate n256 finder-vs-VIDE void size-function plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from pinocchio_voids.io import (
    PINOCCHIO_POSITION_MODES,
    VIDE_CATALOG_VARIANTS,
    pinocchio_position_mode_output_suffix,
    resolve_vide_catalog_variant_path,
    vide_catalog_variant_output_suffix,
)

try:
    from scripts.compare_void_size_functions import main as compare_main
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from compare_void_size_functions import main as compare_main


N256_RUN = {
    "catalog_a": "runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out",
    "catalog_b": "runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out",
    "vide_a": "runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out",
    "vide_b": "runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out",
    "cosmology": "runs/pinocchio-lowres/n256/pinocchio.lowres_n256.cosmology.out",
    "linking_factor": "0.14605092780899798",
    "radius_a0": "6.14700029037185",
    "radius_alpha": "0.9313222316465706",
    "adjacency_factor": "0.5240470713979322",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot n256 finder-vs-VIDE void size functions."
    )
    parser.add_argument("--box-size", type=float, default=256.0, help="Box size in Mpc/h.")
    parser.add_argument(
        "--position-mode",
        choices=PINOCCHIO_POSITION_MODES,
        default="final",
        help="PINOCCHIO coordinate columns used by the finder.",
    )
    parser.add_argument(
        "--rho-bar",
        type=float,
        default=8.63025e10,
        help="Mean matter density used by the protovoid radius mapping.",
    )
    parser.add_argument("--bins", type=int, default=12, help="Number of radius bins.")
    parser.add_argument(
        "--binning",
        choices=("log", "linear"),
        default="log",
        help="Radius bin spacing for generated comparison plots.",
    )
    parser.add_argument("--bin-min", type=float, help="Optional fixed lower radius edge in Mpc/h.")
    parser.add_argument("--bin-max", type=float, help="Optional fixed upper radius edge in Mpc/h.")
    parser.add_argument(
        "--paper-bins",
        action="store_true",
        help="Use Lepinzan et al.-style 17 linear bins from 10 to 80 Mpc/h.",
    )
    parser.add_argument(
        "--include-theory",
        action="store_true",
        help="Overlay the Vdn/SVdW theoretical void size function.",
    )
    parser.add_argument(
        "--vide-variant",
        choices=VIDE_CATALOG_VARIANTS,
        default="all",
        help="VIDE catalog variant used for the reference VSF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/void-statistics"),
        help="Directory for generated CSV and PNG outputs.",
    )
    parser.add_argument(
        "--linking-factor",
        default=N256_RUN["linking_factor"],
        help="Source halo mean-spacing linking factor.",
    )
    parser.add_argument(
        "--radius-a0",
        default=N256_RUN["radius_a0"],
        help="Protovoid radius normalization.",
    )
    parser.add_argument(
        "--radius-alpha",
        default=N256_RUN["radius_alpha"],
        help="Protovoid radius slope.",
    )
    parser.add_argument(
        "--adjacency-factor",
        default=N256_RUN["adjacency_factor"],
        help="Adjacency threshold multiplier.",
    )
    parser.add_argument(
        "--merge-score-mode",
        choices=("geometry_only", "weighted"),
        default="geometry_only",
        help="Use all adjacency edges or threshold weighted merge scores.",
    )
    parser.add_argument("--merge-threshold", default="0.0", help="Weighted merge threshold.")
    parser.add_argument("--geom-weight", default="1.0", help="Geometric score weight.")
    parser.add_argument("--bridge-weight", default="0.0", help="Bridge-density score weight.")
    parser.add_argument(
        "--compatibility-weight",
        default="0.0",
        help="Compatibility score weight.",
    )
    parser.add_argument(
        "--bridge-radius-factor",
        default="0.5",
        help="Bridge capsule radius factor.",
    )
    parser.add_argument(
        "--bridge-min-radius",
        default="0.0",
        help="Minimum bridge capsule radius in Mpc/h.",
    )
    parser.add_argument(
        "--bridge-delta-scale",
        default="1.0",
        help="Overdensity scale for bridge scoring.",
    )
    parser.add_argument(
        "--bridge-density-mode",
        choices=("number", "mass", "both"),
        default="mass",
        help="Halo density field used for bridge scoring.",
    )
    return parser.parse_args()


def _run_args(args: argparse.Namespace) -> list[str]:
    binning = "linear" if args.paper_bins else args.binning
    bins = 17 if args.paper_bins else args.bins
    bin_min = 10.0 if args.paper_bins else args.bin_min
    bin_max = 80.0 if args.paper_bins else args.bin_max
    if args.paper_bins:
        suffix = "paper_bins_theory_vsf" if args.include_theory else "paper_bins_vsf"
    else:
        suffix = "finder_vide_theory_vsf" if args.include_theory else "finder_vide_vsf"
    output_stem = args.output_dir / (
        f"n256_{suffix}"
        f"{vide_catalog_variant_output_suffix(args.vide_variant)}"
        f"{pinocchio_position_mode_output_suffix(args.position_mode)}"
    )
    command = [
        N256_RUN["catalog_a"],
        N256_RUN["catalog_b"],
        str(resolve_vide_catalog_variant_path(N256_RUN["vide_a"], args.vide_variant)),
        str(resolve_vide_catalog_variant_path(N256_RUN["vide_b"], args.vide_variant)),
        "--box-size",
        str(args.box_size),
        "--position-mode",
        args.position_mode,
        "--rho-bar",
        str(args.rho_bar),
        "--linking-factor",
        str(args.linking_factor),
        "--radius-a0",
        str(args.radius_a0),
        "--radius-alpha",
        str(args.radius_alpha),
        "--adjacency-factor",
        str(args.adjacency_factor),
        "--merge-score-mode",
        args.merge_score_mode,
        "--merge-threshold",
        str(args.merge_threshold),
        "--geom-weight",
        str(args.geom_weight),
        "--bridge-weight",
        str(args.bridge_weight),
        "--compatibility-weight",
        str(args.compatibility_weight),
        "--bridge-radius-factor",
        str(args.bridge_radius_factor),
        "--bridge-min-radius",
        str(args.bridge_min_radius),
        "--bridge-delta-scale",
        str(args.bridge_delta_scale),
        "--bridge-density-mode",
        args.bridge_density_mode,
        "--bins",
        str(bins),
        "--binning",
        binning,
        "--label",
        (
            f"n256 scored merge ({args.position_mode})"
            if args.merge_score_mode == "weighted"
            else f"n256 geometry-only ({args.position_mode})"
        ),
        "--output-csv",
        str(output_stem.with_suffix(".csv")),
        "--output-plot",
        str(output_stem.with_suffix(".png")),
        "--summary-csv",
        str(output_stem.with_name(output_stem.name + "_summary").with_suffix(".csv")),
    ]
    if args.include_theory:
        command.extend(["--theory", "vdn-svdw", "--cosmology-file", N256_RUN["cosmology"]])
    if bin_min is not None or bin_max is not None:
        command.extend(["--bin-min", str(bin_min), "--bin-max", str(bin_max)])
    return command


def main() -> int:
    args = parse_args()
    print("Generating n256 void size-function comparison")
    compare_main(_run_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
