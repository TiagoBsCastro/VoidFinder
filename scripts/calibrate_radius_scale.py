#!/usr/bin/env python
"""Calibrate finder radius scale against fixed VIDE paper-bin VSFs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from pinocchio_voids.calibration import (
    PairedRadiusCalibrationResult,
    sweep_radius_scale_parameters,
)
from pinocchio_voids.io import read_paired_pinocchio_halo_catalogs, read_vide_void_desc


DEFAULT_LINKING_FACTORS = (0.06, 0.08, 0.10, 0.12, 0.15)
DEFAULT_RADIUS_A0 = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
DEFAULT_RADIUS_ALPHA = (0.9, 1.0, 1.1)
DEFAULT_ADJACENCY = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
PAPER_MASS_CUT_MSUN_H = 1.0e13


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank radius-scale parameter rows against fixed VIDE paper-bin VSFs."
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
    parser.add_argument(
        "--linking-factor",
        type=float,
        action="append",
        help="Source halo mean-spacing factor to test. Defaults to the n128/n256 grid.",
    )
    parser.add_argument(
        "--linking-length",
        type=float,
        action="append",
        default=[],
        help="Fixed source-cluster linking length in Mpc/h. Repeat to test multiple values.",
    )
    parser.add_argument(
        "--min-cluster-members",
        type=int,
        action="append",
        help="Minimum halo count for source clusters. Defaults to 1 and 2.",
    )
    parser.add_argument(
        "--min-cluster-mass",
        type=float,
        action="append",
        help="Minimum source-cluster mass in Msun/h. Defaults to 0 plus optional paper cut.",
    )
    parser.add_argument(
        "--no-paper-mass-cut",
        action="store_true",
        help="Do not add the 1e13 Msun/h paper-motivated mass cut to the default grid.",
    )
    parser.add_argument(
        "--radius-a0",
        type=float,
        action="append",
        help="Protovoid radius normalization to test. Defaults to 2 through 7.",
    )
    parser.add_argument(
        "--radius-alpha",
        type=float,
        action="append",
        help="Protovoid radius slope to test. Defaults to 0.9, 1.0, and 1.1.",
    )
    parser.add_argument(
        "--adjacency-factor",
        type=float,
        action="append",
        help="Adjacency threshold multiplier to test. Defaults to 0.05 through 1.0.",
    )
    parser.add_argument("--bins", type=int, default=17, help="Number of fixed linear radius bins.")
    parser.add_argument("--bin-min", type=float, default=10.0, help="Lower radius edge in Mpc/h.")
    parser.add_argument("--bin-max", type=float, default=80.0, help="Upper radius edge in Mpc/h.")
    parser.add_argument(
        "--min-predicted-fraction",
        type=float,
        default=0.10,
        help="Degeneracy threshold for in-bin finder/VIDE count fraction.",
    )
    parser.add_argument(
        "--max-source-clusters",
        type=int,
        default=5000,
        help="Skip cached rows with more source clusters than this. Use 0 to disable.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of ranked rows to print.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/void-statistics/radius_scale_calibration.csv"),
        help="Output ranked calibration CSV.",
    )
    return parser.parse_args(argv)


def _values_or_default(values: Sequence[float] | None, default: Sequence[float]) -> tuple[float, ...]:
    return tuple(default if values is None else values)


def _member_values(values: Sequence[int] | None) -> tuple[int, ...]:
    return (1, 2) if values is None else tuple(values)


def _linking_factors(args: argparse.Namespace) -> tuple[float, ...]:
    if args.linking_factor is not None:
        return tuple(args.linking_factor)
    if args.linking_length:
        return ()
    return DEFAULT_LINKING_FACTORS


def _mass_values(args: argparse.Namespace, paired) -> tuple[float, ...]:
    if args.min_cluster_mass is not None:
        return tuple(args.min_cluster_mass)
    values = [0.0]
    if not args.no_paper_mass_cut:
        max_a = float(np.max(paired.catalog_a.masses_msun_h)) if len(paired.catalog_a) else 0.0
        max_b = float(np.max(paired.catalog_b.masses_msun_h)) if len(paired.catalog_b) else 0.0
        if max_a >= PAPER_MASS_CUT_MSUN_H and max_b >= PAPER_MASS_CUT_MSUN_H:
            values.append(PAPER_MASS_CUT_MSUN_H)
    return tuple(values)


def _fixed_linear_edges(*, bins: int, lower: float, upper: float) -> np.ndarray:
    if bins < 1:
        raise SystemExit("--bins must be at least 1")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise SystemExit("--bin-min/--bin-max must be positive finite increasing edges")
    return np.linspace(float(lower), float(upper), bins + 1)


def _float(value: float) -> float | str:
    return "" if not np.isfinite(value) else float(value)


def _row(result: PairedRadiusCalibrationResult, rank: int) -> dict[str, object]:
    config = result.config
    score_a = result.score_a
    score_b = result.score_b
    return {
        "rank": rank,
        "is_degenerate": result.is_degenerate,
        "linking_mode": result.linking_mode,
        "linking_value": result.linking_value,
        "source_a_linking_length_mpc_h": result.source_a_linking_length_mpc_h,
        "source_b_linking_length_mpc_h": result.source_b_linking_length_mpc_h,
        "min_cluster_members": config.min_cluster_members,
        "min_cluster_mass_msun_h": config.min_cluster_mass_msun_h,
        "radius_a0": config.radius_a0,
        "radius_alpha": config.radius_alpha,
        "adjacency_factor": config.adjacency_factor,
        "source_a_clusters": result.source_a_cluster_count,
        "source_b_clusters": result.source_b_cluster_count,
        "target_a_protovoids": result.protovoid_a_count,
        "target_b_protovoids": result.protovoid_b_count,
        "target_a_edges": result.edge_a_count,
        "target_b_edges": result.edge_b_count,
        "total_density_l1": result.total_density_l1_difference,
        "total_count_l1": result.total_count_l1_difference,
        "total_median_radius_abs_error_mpc_h": _float(
            result.total_median_radius_abs_error_mpc_h
        ),
        "total_in_bin_count_abs_error": result.total_in_bin_count_abs_error,
        "target_a_finder_total": score_a.finder_summary.count,
        "target_a_finder_in_bins": score_a.size_score.predicted_void_count,
        "target_a_vide_total": score_a.reference_summary.count,
        "target_a_vide_in_bins": score_a.size_score.reference_void_count,
        "target_a_finder_median_mpc_h": _float(score_a.finder_summary.median_mpc_h),
        "target_a_vide_median_mpc_h": _float(score_a.reference_summary.median_mpc_h),
        "target_a_median_abs_error_mpc_h": _float(score_a.median_radius_abs_error_mpc_h),
        "target_a_density_l1": score_a.size_score.density_l1_difference,
        "target_a_count_l1": score_a.size_score.count_l1_difference,
        "target_a_zero_in_bin": score_a.is_zero_in_bin,
        "target_b_finder_total": score_b.finder_summary.count,
        "target_b_finder_in_bins": score_b.size_score.predicted_void_count,
        "target_b_vide_total": score_b.reference_summary.count,
        "target_b_vide_in_bins": score_b.size_score.reference_void_count,
        "target_b_finder_median_mpc_h": _float(score_b.finder_summary.median_mpc_h),
        "target_b_vide_median_mpc_h": _float(score_b.reference_summary.median_mpc_h),
        "target_b_median_abs_error_mpc_h": _float(score_b.median_radius_abs_error_mpc_h),
        "target_b_density_l1": score_b.size_score.density_l1_difference,
        "target_b_count_l1": score_b.size_score.count_l1_difference,
        "target_b_zero_in_bin": score_b.is_zero_in_bin,
    }


def write_csv(path: Path, results: Sequence[PairedRadiusCalibrationResult]) -> None:
    rows = [_row(result, rank) for rank, result in enumerate(results, start=1)]
    if not rows:
        raise SystemExit("No calibration rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(results: Sequence[PairedRadiusCalibrationResult], *, top: int) -> None:
    print(
        "rank,degenerate,linking,min_members,min_mass,a0,alpha,adj,"
        "A_in/VIDE,A_med/VIDE,B_in/VIDE,B_med/VIDE,density_l1"
    )
    for rank, result in enumerate(results[:top], start=1):
        config = result.config
        print(
            f"{rank},{result.is_degenerate},"
            f"{result.linking_mode}:{result.linking_value:g},"
            f"{config.min_cluster_members},{config.min_cluster_mass_msun_h:g},"
            f"{config.radius_a0:g},{config.radius_alpha:g},{config.adjacency_factor:g},"
            f"{result.score_a.size_score.predicted_void_count}/"
            f"{result.score_a.size_score.reference_void_count},"
            f"{result.score_a.finder_summary.median_mpc_h:.3g}/"
            f"{result.score_a.reference_summary.median_mpc_h:.3g},"
            f"{result.score_b.size_score.predicted_void_count}/"
            f"{result.score_b.size_score.reference_void_count},"
            f"{result.score_b.finder_summary.median_mpc_h:.3g}/"
            f"{result.score_b.reference_summary.median_mpc_h:.3g},"
            f"{result.total_density_l1_difference:.8e}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paired = read_paired_pinocchio_halo_catalogs(
        args.catalog_a,
        args.catalog_b,
        box_size_mpc_h=args.box_size,
    )
    reference_a = read_vide_void_desc(args.vide_a)
    reference_b = read_vide_void_desc(args.vide_b)
    bin_edges = _fixed_linear_edges(
        bins=args.bins,
        lower=args.bin_min,
        upper=args.bin_max,
    )
    max_source_clusters = None if args.max_source_clusters == 0 else args.max_source_clusters
    results = sweep_radius_scale_parameters(
        paired.catalog_a,
        paired.catalog_b,
        reference_a=reference_a,
        reference_b=reference_b,
        reference_rho_bar_msun_h_mpc3=args.rho_bar,
        linking_lengths_mpc_h=tuple(args.linking_length),
        linking_length_mean_spacing_factors=_linking_factors(args),
        min_cluster_members_values=_member_values(args.min_cluster_members),
        min_cluster_mass_values_msun_h=_mass_values(args, paired),
        radius_a0_values=_values_or_default(args.radius_a0, DEFAULT_RADIUS_A0),
        radius_alpha_values=_values_or_default(args.radius_alpha, DEFAULT_RADIUS_ALPHA),
        adjacency_factors=_values_or_default(args.adjacency_factor, DEFAULT_ADJACENCY),
        bins=bin_edges,
        radius_min_mpc_h=args.bin_min,
        radius_max_mpc_h=args.bin_max,
        min_predicted_fraction=args.min_predicted_fraction,
        max_source_clusters=max_source_clusters,
    )
    write_csv(args.output_csv, results)
    _print_rows(results, top=args.top)
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
