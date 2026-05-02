#!/usr/bin/env python
"""Compare void size functions from the paired finder and VIDE catalogs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pinocchio_voids.calibration import mean_halo_spacing_mpc_h, score_direction_against_vide
from pinocchio_voids.evaluation import VoidSizeFunctionComparison
from pinocchio_voids.io import read_paired_pinocchio_halo_catalogs, read_vide_void_desc
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    PairedVoidFinderConfig,
    run_paired_halo_void_finder,
)


@dataclass(frozen=True)
class DirectionComparison:
    """Void size-function comparison for one paired target direction."""

    target: str
    finder_result: DirectionalVoidFinderResult
    comparison: VoidSizeFunctionComparison
    predicted_void_count: int
    reference_void_count: int
    raw_count_l1: int
    guarded_count_l1: int
    density_l1: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare finder and VIDE void size functions for a paired run."
    )
    parser.add_argument("catalog_a", type=Path, help="PINOCCHIO halo catalog for realization A.")
    parser.add_argument("catalog_b", type=Path, help="PINOCCHIO halo catalog for realization B.")
    parser.add_argument("vide_a", type=Path, help="VIDE voidDesc reference for target A.")
    parser.add_argument("vide_b", type=Path, help="VIDE voidDesc reference for target B.")
    parser.add_argument("--box-size", type=float, required=True, help="Periodic box size in Mpc/h.")
    parser.add_argument(
        "--rho-bar",
        type=float,
        required=True,
        help="Mean matter density used by the protovoid radius mapping.",
    )
    linking = parser.add_mutually_exclusive_group()
    linking.add_argument(
        "--linking-length",
        type=float,
        help="Fixed source-cluster linking length in Mpc/h. Defaults to 8.0.",
    )
    linking.add_argument(
        "--linking-factor",
        type=float,
        help="Source halo mean-spacing factor used to resolve A/B linking lengths.",
    )
    parser.add_argument(
        "--min-cluster-members",
        type=int,
        default=2,
        help="Minimum halo count for source clusters.",
    )
    parser.add_argument(
        "--min-cluster-mass",
        type=float,
        default=0.0,
        help="Minimum source-cluster mass in Msun/h.",
    )
    parser.add_argument("--radius-a0", type=float, default=1.0, help="Protovoid radius normalization.")
    parser.add_argument("--radius-alpha", type=float, default=1.0, help="Protovoid radius slope.")
    parser.add_argument(
        "--adjacency-factor",
        type=float,
        default=1.0,
        help="Adjacency threshold multiplier for protovoid merging.",
    )
    parser.add_argument(
        "--min-predicted-fraction",
        type=float,
        default=0.25,
        help="Guard threshold for degenerate underprediction summaries.",
    )
    parser.add_argument("--bins", type=int, default=12, help="Number of shared log-radius bins.")
    parser.add_argument(
        "--label",
        default="geometry-only",
        help="Run label written to CSV rows and plot legends.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/void-statistics/void_size_functions.csv"),
        help="Output CSV table path.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Optional output plot path. Requires matplotlib.",
    )
    return parser.parse_args(argv)


def _resolve_linking_lengths(args: argparse.Namespace, paired) -> tuple[float, float, str, float]:
    if args.linking_factor is not None:
        factor = float(args.linking_factor)
        return (
            factor * mean_halo_spacing_mpc_h(paired.catalog_a),
            factor * mean_halo_spacing_mpc_h(paired.catalog_b),
            "factor",
            factor,
        )

    linking_length = 8.0 if args.linking_length is None else float(args.linking_length)
    return linking_length, linking_length, "fixed", linking_length


def _compare_direction(
    *,
    target: str,
    finder_result: DirectionalVoidFinderResult,
    vide_path: Path,
    box_size_mpc_h: float,
    bins: int,
    min_predicted_fraction: float,
) -> DirectionComparison:
    reference = read_vide_void_desc(vide_path)
    score = score_direction_against_vide(
        finder_result,
        reference,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
        min_predicted_fraction=min_predicted_fraction,
    )
    return DirectionComparison(
        target=target,
        finder_result=finder_result,
        comparison=score.size_function,
        predicted_void_count=score.predicted_void_count,
        reference_void_count=score.reference_void_count,
        raw_count_l1=score.count_l1_difference,
        guarded_count_l1=score.guarded_count_l1_difference,
        density_l1=score.density_l1_difference,
    )


def build_comparisons(args: argparse.Namespace) -> tuple[DirectionComparison, DirectionComparison, str, float, float, float]:
    paired = read_paired_pinocchio_halo_catalogs(
        args.catalog_a,
        args.catalog_b,
        box_size_mpc_h=args.box_size,
    )
    link_a, link_b, linking_mode, linking_value = _resolve_linking_lengths(args, paired)
    config = PairedVoidFinderConfig(
        linking_length_mpc_h=link_a,
        source_b_linking_length_mpc_h=link_b,
        min_cluster_members=args.min_cluster_members,
        min_cluster_mass_msun_h=args.min_cluster_mass,
        reference_rho_bar_msun_h_mpc3=args.rho_bar,
        radius_a0=args.radius_a0,
        radius_alpha=args.radius_alpha,
        adjacency_factor=args.adjacency_factor,
    )
    result = run_paired_halo_void_finder(paired.catalog_a, paired.catalog_b, config=config)
    comparison_a = _compare_direction(
        target="A",
        finder_result=result.voids_a,
        vide_path=args.vide_a,
        box_size_mpc_h=args.box_size,
        bins=args.bins,
        min_predicted_fraction=args.min_predicted_fraction,
    )
    comparison_b = _compare_direction(
        target="B",
        finder_result=result.voids_b,
        vide_path=args.vide_b,
        box_size_mpc_h=args.box_size,
        bins=args.bins,
        min_predicted_fraction=args.min_predicted_fraction,
    )
    return comparison_a, comparison_b, linking_mode, linking_value, link_a, link_b


def _rows_for_size_function(
    *,
    label: str,
    source: str,
    target: str,
    size_function,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for r_min, r_max, r_mid, count, density in zip(
        size_function.bin_edges_mpc_h[:-1],
        size_function.bin_edges_mpc_h[1:],
        size_function.bin_centers_mpc_h,
        size_function.counts,
        size_function.density_dndlnr_per_mpc_h3,
    ):
        rows.append(
            {
                "label": label,
                "source": source,
                "target": target,
                "bin_min_mpc_h": r_min,
                "bin_max_mpc_h": r_max,
                "bin_center_mpc_h": r_mid,
                "count": int(count),
                "density_dndlnr_per_mpc_h3": density,
            }
        )
    return rows


def write_csv(
    path: Path,
    *,
    label: str,
    comparisons: Sequence[DirectionComparison],
) -> None:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        rows.extend(
            _rows_for_size_function(
                label=label,
                source="finder",
                target=comparison.target,
                size_function=comparison.comparison.predicted,
            )
        )
        rows.extend(
            _rows_for_size_function(
                label=label,
                source="vide",
                target=comparison.target,
                size_function=comparison.comparison.reference,
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    path: Path,
    *,
    label: str,
    comparisons: Sequence[DirectionComparison],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --output-plot") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(comparisons),
        figsize=(6.0 * len(comparisons), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, comparison in zip(axes[0], comparisons):
        for source, size_function in (
            ("finder", comparison.comparison.predicted),
            ("VIDE", comparison.comparison.reference),
        ):
            axis.step(
                size_function.bin_centers_mpc_h,
                size_function.density_dndlnr_per_mpc_h3,
                where="mid",
                marker="o",
                label=f"{label} {source}",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$R_\mathrm{eff}$ [$h^{-1}\,\mathrm{Mpc}$]")
        axis.set_ylabel(r"$dN / d\ln R / V$ [$(h^{-1}\,\mathrm{Mpc})^{-3}$]")
        axis.set_title(f"Target {comparison.target}")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(path, dpi=180)


def _print_summary(
    *,
    comparisons: Sequence[DirectionComparison],
    linking_mode: str,
    linking_value: float,
    link_a: float,
    link_b: float,
) -> None:
    print(
        f"Linking: {linking_mode}={linking_value:g} "
        f"(source A/B lengths {link_a:g}/{link_b:g} Mpc/h)"
    )
    for comparison in comparisons:
        print(
            f"Target {comparison.target}: "
            f"finder={comparison.predicted_void_count} "
            f"VIDE={comparison.reference_void_count} "
            f"raw_l1={comparison.raw_count_l1} "
            f"guarded_l1={comparison.guarded_count_l1} "
            f"density_l1={comparison.density_l1:.6e}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    comparison_a, comparison_b, linking_mode, linking_value, link_a, link_b = build_comparisons(args)
    comparisons = (comparison_a, comparison_b)

    write_csv(args.output_csv, label=args.label, comparisons=comparisons)
    if args.output_plot is not None:
        write_plot(args.output_plot, label=args.label, comparisons=comparisons)

    _print_summary(
        comparisons=comparisons,
        linking_mode=linking_mode,
        linking_value=linking_value,
        link_a=link_a,
        link_b=link_b,
    )
    print(f"Wrote {args.output_csv}")
    if args.output_plot is not None:
        print(f"Wrote {args.output_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
