#!/usr/bin/env python
"""Generate n128/n256 finder-vs-VIDE void size-function plots."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.compare_void_size_functions import main as compare_main
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from compare_void_size_functions import main as compare_main


RUNS = {
    "n128": {
        "catalog_a": "runs/pinocchio-lowres/n128/pinocchio.0.0000.lowres_n128.catalog.out",
        "catalog_b": "runs/pinocchio-lowres/n128_paired/pinocchio.0.0000.lowres_n128_paired.catalog.out",
        "vide_a": "runs/vide-lowres/n128_pair/n128/outputs/pinocchio_n128_ss1.0/sample_pinocchio_n128_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n128_ss1.0_z0.00_d00.out",
        "vide_b": "runs/vide-lowres/n128_pair/n128_paired/outputs/pinocchio_n128_paired_ss1.0/sample_pinocchio_n128_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n128_paired_ss1.0_z0.00_d00.out",
        "cosmology": "runs/pinocchio-lowres/n128/pinocchio.lowres_n128.cosmology.out",
        "linking_factor": "0.15",
        "radius_a0": "5.0",
        "radius_alpha": "1.0",
        "adjacency_factor": "0.50",
    },
    "n256": {
        "catalog_a": "runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out",
        "catalog_b": "runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out",
        "vide_a": "runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out",
        "vide_b": "runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out",
        "cosmology": "runs/pinocchio-lowres/n256/pinocchio.lowres_n256.cosmology.out",
        "linking_factor": "0.12",
        "radius_a0": "6.0",
        "radius_alpha": "1.0",
        "adjacency_factor": "0.35",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot n128/n256 finder-vs-VIDE void size functions."
    )
    parser.add_argument("--box-size", type=float, default=256.0, help="Box size in Mpc/h.")
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
        "--output-dir",
        type=Path,
        default=Path("runs/void-statistics"),
        help="Directory for generated CSV and PNG outputs.",
    )
    parser.add_argument(
        "--only",
        choices=tuple(RUNS),
        action="append",
        help="Restrict generation to one resolution. May be passed more than once.",
    )
    parser.add_argument(
        "--n128-linking-factor",
        default=RUNS["n128"]["linking_factor"],
        help="Mean-spacing linking factor for n128.",
    )
    parser.add_argument(
        "--n256-linking-factor",
        default=RUNS["n256"]["linking_factor"],
        help="Mean-spacing linking factor for n256.",
    )
    for name in RUNS:
        parser.add_argument(
            f"--{name}-radius-a0",
            default=RUNS[name]["radius_a0"],
            help=f"Protovoid radius normalization for {name}.",
        )
        parser.add_argument(
            f"--{name}-radius-alpha",
            default=RUNS[name]["radius_alpha"],
            help=f"Protovoid radius slope for {name}.",
        )
        parser.add_argument(
            f"--{name}-adjacency-factor",
            default=RUNS[name]["adjacency_factor"],
            help=f"Adjacency threshold multiplier for {name}.",
        )
    return parser.parse_args()


def _run_args(name: str, args: argparse.Namespace) -> list[str]:
    run = RUNS[name]
    linking_factor = getattr(args, f"{name}_linking_factor")
    radius_a0 = getattr(args, f"{name}_radius_a0")
    radius_alpha = getattr(args, f"{name}_radius_alpha")
    adjacency_factor = getattr(args, f"{name}_adjacency_factor")
    binning = "linear" if args.paper_bins else args.binning
    bins = 17 if args.paper_bins else args.bins
    bin_min = 10.0 if args.paper_bins else args.bin_min
    bin_max = 80.0 if args.paper_bins else args.bin_max
    if args.paper_bins:
        suffix = "paper_bins_theory_vsf" if args.include_theory else "paper_bins_vsf"
    else:
        suffix = "finder_vide_theory_vsf" if args.include_theory else "finder_vide_vsf"
    output_stem = args.output_dir / f"{name}_{suffix}"
    command = [
        run["catalog_a"],
        run["catalog_b"],
        run["vide_a"],
        run["vide_b"],
        "--box-size",
        str(args.box_size),
        "--rho-bar",
        str(args.rho_bar),
        "--linking-factor",
        str(linking_factor),
        "--radius-a0",
        str(radius_a0),
        "--radius-alpha",
        str(radius_alpha),
        "--adjacency-factor",
        str(adjacency_factor),
        "--bins",
        str(bins),
        "--binning",
        binning,
        "--label",
        f"{name} geometry-only",
        "--output-csv",
        str(output_stem.with_suffix(".csv")),
        "--output-plot",
        str(output_stem.with_suffix(".png")),
        "--summary-csv",
        str(output_stem.with_name(output_stem.name + "_summary").with_suffix(".csv")),
    ]
    if args.include_theory:
        command.extend(["--theory", "vdn-svdw", "--cosmology-file", run["cosmology"]])
    if bin_min is not None or bin_max is not None:
        command.extend(["--bin-min", str(bin_min), "--bin-max", str(bin_max)])
    return command


def main() -> int:
    args = parse_args()
    selected = args.only or list(RUNS)
    for name in selected:
        print(f"Generating {name} void size-function comparison")
        compare_main(_run_args(name, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
