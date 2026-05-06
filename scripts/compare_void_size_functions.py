#!/usr/bin/env python
"""Compare void size functions from the paired finder and VIDE catalogs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from pinocchio_voids.calibration import mean_halo_spacing_mpc_h, score_direction_against_vide
from pinocchio_voids.evaluation import VoidSizeFunctionComparison
from pinocchio_voids.io import read_paired_pinocchio_halo_catalogs, read_vide_void_desc
from pinocchio_voids.theory import (
    PinocchioCosmologyTable,
    TheoreticalVoidSizeFunction,
    compute_vdn_svdw_size_function,
    read_pinocchio_cosmology,
)
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
    finder_radii_mpc_h: tuple[float, ...]
    reference_radii_mpc_h: tuple[float, ...]
    theory: TheoreticalVoidSizeFunction | None = None


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
        "--merge-score-mode",
        choices=("geometry_only", "weighted"),
        default="geometry_only",
        help="Use all adjacency edges or threshold weighted merge scores.",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.0,
        help="Minimum weighted merge score required to merge an adjacency edge.",
    )
    parser.add_argument(
        "--geom-weight",
        type=float,
        default=1.0,
        help="Weight applied to the geometric merge score.",
    )
    parser.add_argument(
        "--bridge-weight",
        type=float,
        default=0.0,
        help="Weight applied to the source-catalog bridge-density score.",
    )
    parser.add_argument(
        "--compatibility-weight",
        type=float,
        default=0.0,
        help="Weight applied to the source-cluster compatibility score.",
    )
    parser.add_argument(
        "--bridge-radius-factor",
        type=float,
        default=0.5,
        help="Bridge capsule radius factor for weighted merging.",
    )
    parser.add_argument(
        "--bridge-min-radius",
        type=float,
        default=0.0,
        help="Minimum bridge capsule radius in Mpc/h.",
    )
    parser.add_argument(
        "--bridge-delta-scale",
        type=float,
        default=1.0,
        help="Overdensity scale used to map bridge density to a 0..1 score.",
    )
    parser.add_argument(
        "--bridge-density-mode",
        choices=("number", "mass", "both"),
        default="mass",
        help="Halo density field used for bridge scoring.",
    )
    parser.add_argument(
        "--min-predicted-fraction",
        type=float,
        default=0.25,
        help="Guard threshold for degenerate underprediction summaries.",
    )
    parser.add_argument("--bins", type=int, default=12, help="Number of radius bins.")
    parser.add_argument(
        "--binning",
        choices=("log", "linear"),
        default="log",
        help="Radius bin spacing. Without --bin-min/--bin-max, log uses shared data bounds.",
    )
    parser.add_argument(
        "--bin-min",
        type=float,
        help="Optional fixed lower radius bin edge in Mpc/h.",
    )
    parser.add_argument(
        "--bin-max",
        type=float,
        help="Optional fixed upper radius bin edge in Mpc/h.",
    )
    parser.add_argument(
        "--theory",
        choices=("vdn-svdw",),
        help="Optional theoretical void size-function model to overlay.",
    )
    parser.add_argument(
        "--cosmology-file",
        type=Path,
        help="PINOCCHIO cosmology table used by --theory.",
    )
    parser.add_argument(
        "--delta-v-linear",
        type=float,
        default=-2.7,
        help="Linear void barrier used by Vdn/SVdW theory.",
    )
    parser.add_argument(
        "--delta-c-linear",
        type=float,
        default=1.686,
        help="Linear collapse barrier used by Vdn/SVdW theory.",
    )
    parser.add_argument(
        "--delta-v-nonlinear",
        type=float,
        default=-0.8,
        help="Non-linear enclosed void density contrast used by Vdn/SVdW theory.",
    )
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
        "--summary-csv",
        type=Path,
        help="Optional radius-summary CSV path for finder and VIDE radii.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Optional output plot path. Requires matplotlib.",
    )
    return parser.parse_args(argv)


def resolve_bins(args: argparse.Namespace) -> int | np.ndarray:
    """Return either shared automatic bin count or explicit radius edges."""

    if args.bins < 1:
        raise SystemExit("--bins must be at least 1")
    has_min = args.bin_min is not None
    has_max = args.bin_max is not None
    if has_min != has_max:
        raise SystemExit("--bin-min and --bin-max must be provided together")
    if not has_min:
        return args.bins

    lower = float(args.bin_min)
    upper = float(args.bin_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise SystemExit("--bin-min/--bin-max must be positive finite increasing edges")
    if args.binning == "linear":
        return np.linspace(lower, upper, args.bins + 1)
    return np.geomspace(lower, upper, args.bins + 1)


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


def _load_theory_cosmology(args: argparse.Namespace) -> PinocchioCosmologyTable | None:
    if args.theory is None:
        return None
    if args.cosmology_file is None:
        raise SystemExit("--cosmology-file is required with --theory")
    return read_pinocchio_cosmology(args.cosmology_file)


def _compute_theory(
    *,
    args: argparse.Namespace,
    cosmology: PinocchioCosmologyTable | None,
    bin_edges_mpc_h,
) -> TheoreticalVoidSizeFunction | None:
    if args.theory is None:
        return None
    if args.theory == "vdn-svdw" and cosmology is not None:
        return compute_vdn_svdw_size_function(
            bin_edges_mpc_h,
            cosmology,
            delta_v_linear=args.delta_v_linear,
            delta_c_linear=args.delta_c_linear,
            delta_v_nonlinear=args.delta_v_nonlinear,
        )
    raise SystemExit(f"Unsupported theory model: {args.theory}")


def _compare_direction(
    *,
    args: argparse.Namespace,
    target: str,
    finder_result: DirectionalVoidFinderResult,
    vide_path: Path,
    box_size_mpc_h: float,
    bins: int | ArrayLike,
    min_predicted_fraction: float,
    theory_cosmology: PinocchioCosmologyTable | None,
) -> DirectionComparison:
    reference = read_vide_void_desc(vide_path)
    finder_radii = tuple(void.effective_radius_mpc_h for void in finder_result.voids)
    reference_radii = tuple(float(radius) for radius in reference.effective_radii_mpc_h)
    score = score_direction_against_vide(
        finder_result,
        reference,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
        min_predicted_fraction=min_predicted_fraction,
    )
    theory = _compute_theory(
        args=args,
        cosmology=theory_cosmology,
        bin_edges_mpc_h=score.size_function.reference.bin_edges_mpc_h,
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
        finder_radii_mpc_h=finder_radii,
        reference_radii_mpc_h=reference_radii,
        theory=theory,
    )


def build_comparisons(args: argparse.Namespace) -> tuple[DirectionComparison, DirectionComparison, str, float, float, float]:
    paired = read_paired_pinocchio_halo_catalogs(
        args.catalog_a,
        args.catalog_b,
        box_size_mpc_h=args.box_size,
    )
    theory_cosmology = _load_theory_cosmology(args)
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
        merge_score_mode=args.merge_score_mode,
        merge_threshold=args.merge_threshold,
        geom_weight=args.geom_weight,
        bridge_weight=args.bridge_weight,
        compatibility_weight=args.compatibility_weight,
        bridge_radius_factor=args.bridge_radius_factor,
        bridge_min_radius_mpc_h=args.bridge_min_radius,
        bridge_delta_scale=args.bridge_delta_scale,
        bridge_density_mode=args.bridge_density_mode,
    )
    result = run_paired_halo_void_finder(paired.catalog_a, paired.catalog_b, config=config)
    bins = resolve_bins(args)
    comparison_a = _compare_direction(
        args=args,
        target="A",
        finder_result=result.voids_a,
        vide_path=args.vide_a,
        box_size_mpc_h=args.box_size,
        bins=bins,
        min_predicted_fraction=args.min_predicted_fraction,
        theory_cosmology=theory_cosmology,
    )
    comparison_b = _compare_direction(
        args=args,
        target="B",
        finder_result=result.voids_b,
        vide_path=args.vide_b,
        box_size_mpc_h=args.box_size,
        bins=bins,
        min_predicted_fraction=args.min_predicted_fraction,
        theory_cosmology=theory_cosmology,
    )
    return comparison_a, comparison_b, linking_mode, linking_value, link_a, link_b


def _radius_summary_row(
    *,
    label: str,
    source: str,
    target: str,
    radii_mpc_h: Sequence[float],
) -> dict[str, object]:
    radii = np.asarray(radii_mpc_h, dtype=np.float64)
    row: dict[str, object] = {
        "label": label,
        "source": source,
        "target": target,
        "count": int(radii.size),
        "min_mpc_h": "",
        "p10_mpc_h": "",
        "median_mpc_h": "",
        "p90_mpc_h": "",
        "max_mpc_h": "",
        "count_10_80_mpc_h": int(np.count_nonzero((radii >= 10.0) & (radii <= 80.0))),
    }
    if radii.size:
        row.update(
            {
                "min_mpc_h": float(np.min(radii)),
                "p10_mpc_h": float(np.percentile(radii, 10.0)),
                "median_mpc_h": float(np.median(radii)),
                "p90_mpc_h": float(np.percentile(radii, 90.0)),
                "max_mpc_h": float(np.max(radii)),
            }
        )
    return row


def write_radius_summary_csv(
    path: Path,
    *,
    label: str,
    comparisons: Sequence[DirectionComparison],
) -> None:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        rows.append(
            _radius_summary_row(
                label=label,
                source="finder",
                target=comparison.target,
                radii_mpc_h=comparison.finder_radii_mpc_h,
            )
        )
        rows.append(
            _radius_summary_row(
                label=label,
                source="vide",
                target=comparison.target,
                radii_mpc_h=comparison.reference_radii_mpc_h,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _rows_for_theory(
    *,
    label: str,
    target: str,
    theory: TheoreticalVoidSizeFunction,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for r_min, r_max, r_mid, density in zip(
        theory.bin_edges_mpc_h[:-1],
        theory.bin_edges_mpc_h[1:],
        theory.bin_centers_mpc_h,
        theory.density_dndlnr_per_mpc_h3,
    ):
        rows.append(
            {
                "label": label,
                "source": theory.model,
                "target": target,
                "bin_min_mpc_h": r_min,
                "bin_max_mpc_h": r_max,
                "bin_center_mpc_h": r_mid,
                "count": "",
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
        if comparison.theory is not None:
            rows.extend(
                _rows_for_theory(
                    label=label,
                    target=comparison.target,
                    theory=comparison.theory,
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
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
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
        if comparison.theory is not None:
            density = comparison.theory.density_dndlnr_per_mpc_h3
            valid = density > 0.0
            axis.plot(
                comparison.theory.bin_centers_mpc_h[valid],
                density[valid],
                linestyle="--",
                color="black",
                label="Vdn/SVdW theory",
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
    theory: str | None,
) -> None:
    print(
        f"Linking: {linking_mode}={linking_value:g} "
        f"(source A/B lengths {link_a:g}/{link_b:g} Mpc/h)"
    )
    if theory is not None:
        print(f"Theory: {theory}")
    for comparison in comparisons:
        print(
            f"Target {comparison.target}: "
            f"finder_in_bins={comparison.predicted_void_count}/{len(comparison.finder_radii_mpc_h)} "
            f"VIDE_in_bins={comparison.reference_void_count}/{len(comparison.reference_radii_mpc_h)} "
            f"raw_l1={comparison.raw_count_l1} "
            f"guarded_l1={comparison.guarded_count_l1} "
            f"density_l1={comparison.density_l1:.6e}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    comparison_a, comparison_b, linking_mode, linking_value, link_a, link_b = build_comparisons(args)
    comparisons = (comparison_a, comparison_b)

    write_csv(args.output_csv, label=args.label, comparisons=comparisons)
    if args.summary_csv is not None:
        write_radius_summary_csv(args.summary_csv, label=args.label, comparisons=comparisons)
    if args.output_plot is not None:
        write_plot(args.output_plot, label=args.label, comparisons=comparisons)

    _print_summary(
        comparisons=comparisons,
        linking_mode=linking_mode,
        linking_value=linking_value,
        link_a=link_a,
        link_b=link_b,
        theory=args.theory,
    )
    print(f"Wrote {args.output_csv}")
    if args.summary_csv is not None:
        print(f"Wrote {args.summary_csv}")
    if args.output_plot is not None:
        print(f"Wrote {args.output_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
