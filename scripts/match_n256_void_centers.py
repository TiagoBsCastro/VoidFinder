#!/usr/bin/env python
"""Nearest-neighbor center diagnostics for n256 finder and VIDE voids."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pinocchio_voids.calibration import mean_halo_spacing_mpc_h
from pinocchio_voids.io import (
    PINOCCHIO_POSITION_MODES,
    VIDE_CATALOG_VARIANTS,
    pinocchio_position_mode_output_suffix,
    read_paired_pinocchio_halo_catalogs,
    vide_catalog_variant_output_suffix,
)
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    PairedVoidFinderConfig,
    run_paired_halo_void_finder,
)

try:
    from scripts.plot_n256_void_slice import (
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        N256_RUN,
        add_full_algorithm_arguments,
        full_algorithm_kwargs_from_args,
        load_vide_spatial_catalog,
        periodic_pairwise_distances,
        resolve_target_vide_paths,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from plot_n256_void_slice import (
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        N256_RUN,
        add_full_algorithm_arguments,
        full_algorithm_kwargs_from_args,
        load_vide_spatial_catalog,
        periodic_pairwise_distances,
        resolve_target_vide_paths,
    )


MATCH_COLUMNS = [
    "target",
    "position_mode",
    "vide_center_kind",
    "vide_variant",
    "finder_void_id",
    "finder_x_mpc_h",
    "finder_y_mpc_h",
    "finder_z_mpc_h",
    "finder_reff_mpc_h",
    "finder_member_protovoid_count",
    "finder_total_source_mass_msun_h",
    "vide_void_id",
    "vide_file_void_id",
    "vide_x_mpc_h",
    "vide_y_mpc_h",
    "vide_z_mpc_h",
    "vide_reff_mpc_h",
    "center_distance_mpc_h",
    "distance_over_finder_reff",
    "distance_over_vide_reff",
    "distance_over_min_reff",
    "radius_ratio_finder_over_vide",
]


SUMMARY_COLUMNS = [
    "target",
    "position_mode",
    "vide_center_kind",
    "vide_variant",
    "finder_count",
    "vide_count",
    "matched_finder_count",
    "fraction_distance_lt_finder_reff",
    "fraction_distance_lt_vide_reff",
    "fraction_distance_lt_min_reff",
    "median_center_distance_mpc_h",
    "median_distance_over_min_reff",
    "p90_distance_over_min_reff",
]


@dataclass(frozen=True)
class FinderSpatialCatalog:
    """Finder void positions, radii, and metadata for center matching."""

    positions_mpc_h: NDArray[np.float64]
    radii_mpc_h: NDArray[np.float64]
    void_ids: NDArray[np.int64]
    member_counts: NDArray[np.int64]
    total_source_masses_msun_h: NDArray[np.float64]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match n256 finder void centers to nearest VIDE void centers."
    )
    parser.add_argument("--catalog-a", type=Path, default=Path(N256_RUN["catalog_a"]))
    parser.add_argument("--catalog-b", type=Path, default=Path(N256_RUN["catalog_b"]))
    parser.add_argument("--vide-desc-a", type=Path, default=Path(N256_RUN["vide_a"]))
    parser.add_argument("--vide-desc-b", type=Path, default=Path(N256_RUN["vide_b"]))
    parser.add_argument("--vide-centers-a", type=Path, default=DEFAULT_VIDE_CENTERS_A)
    parser.add_argument("--vide-centers-b", type=Path, default=DEFAULT_VIDE_CENTERS_B)
    parser.add_argument("--vide-macrocenters-a", type=Path, default=DEFAULT_VIDE_MACROCENTERS_A)
    parser.add_argument("--vide-macrocenters-b", type=Path, default=DEFAULT_VIDE_MACROCENTERS_B)
    parser.add_argument(
        "--vide-center-kind",
        choices=("center", "macrocenter"),
        default="center",
        help="VIDE position convention used for nearest-neighbor matching.",
    )
    parser.add_argument(
        "--vide-variant",
        choices=VIDE_CATALOG_VARIANTS,
        default="all",
        help="VIDE catalog variant used for voidDesc, centers, and macrocenters.",
    )
    parser.add_argument("--box-size", type=float, default=256.0)
    parser.add_argument(
        "--position-mode",
        choices=PINOCCHIO_POSITION_MODES,
        default="final",
        help="PINOCCHIO coordinate columns used by the finder.",
    )
    parser.add_argument("--rho-bar", type=float, default=8.63025e10)
    linking = parser.add_mutually_exclusive_group()
    linking.add_argument("--linking-length", type=float)
    linking.add_argument("--linking-factor", type=float, default=float(N256_RUN["linking_factor"]))
    parser.add_argument("--min-cluster-members", type=int, default=2)
    parser.add_argument("--min-cluster-mass", type=float, default=0.0)
    parser.add_argument("--radius-a0", type=float, default=float(N256_RUN["radius_a0"]))
    parser.add_argument("--radius-alpha", type=float, default=float(N256_RUN["radius_alpha"]))
    parser.add_argument(
        "--adjacency-factor",
        type=float,
        default=float(N256_RUN["adjacency_factor"]),
    )
    add_full_algorithm_arguments(parser)
    parser.add_argument("--target", choices=("A", "B", "both"), default="both")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output match CSV. Defaults depend on --vide-center-kind.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output summary CSV. Defaults depend on --vide-center-kind.",
    )
    return parser.parse_args(argv)


def _default_output_paths(
    vide_center_kind: str,
    vide_variant: str,
    position_mode: str,
) -> tuple[Path, Path]:
    suffix = (
        f"{vide_catalog_variant_output_suffix(vide_variant)}"
        f"{pinocchio_position_mode_output_suffix(position_mode)}"
    )
    if vide_center_kind == "macrocenter":
        return (
            Path(f"runs/void-statistics/n256_void_macrocenter_matches{suffix}.csv"),
            Path(f"runs/void-statistics/n256_void_macrocenter_match_summary{suffix}.csv"),
        )
    return (
        Path(f"runs/void-statistics/n256_void_center_matches{suffix}.csv"),
        Path(f"runs/void-statistics/n256_void_center_match_summary{suffix}.csv"),
    )


def _resolve_linking_lengths(args: argparse.Namespace, paired) -> tuple[float, float]:
    if args.linking_length is not None:
        return float(args.linking_length), float(args.linking_length)
    factor = float(args.linking_factor)
    return (
        factor * mean_halo_spacing_mpc_h(paired.catalog_a),
        factor * mean_halo_spacing_mpc_h(paired.catalog_b),
    )


def finder_spatial_catalog(result: DirectionalVoidFinderResult) -> FinderSpatialCatalog:
    """Convert a directional finder result to center-matching arrays."""

    return FinderSpatialCatalog(
        positions_mpc_h=np.asarray(
            [void.center_mpc_h for void in result.voids],
            dtype=np.float64,
        ).reshape((-1, 3)),
        radii_mpc_h=np.asarray(
            [void.effective_radius_mpc_h for void in result.voids],
            dtype=np.float64,
        ),
        void_ids=np.asarray([void.id for void in result.voids], dtype=np.int64),
        member_counts=np.asarray(
            [len(void.member_protovoid_ids) for void in result.voids],
            dtype=np.int64,
        ),
        total_source_masses_msun_h=np.asarray(
            [void.total_source_mass_msun_h for void in result.voids],
            dtype=np.float64,
        ),
    )


def build_match_rows(
    *,
    target: str,
    position_mode: str,
    finder: FinderSpatialCatalog,
    vide,
    box_size_mpc_h: float,
    vide_variant: str = "all",
) -> list[dict[str, float | int | str]]:
    """Build finder-to-nearest-VIDE object rows."""

    if len(finder.radii_mpc_h) == 0 or len(vide.radii_mpc_h) == 0:
        return []

    distances = periodic_pairwise_distances(
        finder.positions_mpc_h,
        vide.positions_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(distances.shape[0]), nearest_indices]

    rows: list[dict[str, float | int | str]] = []
    for finder_index, vide_index, distance in zip(
        range(len(finder.radii_mpc_h)),
        nearest_indices,
        nearest_distances,
        strict=True,
    ):
        finder_radius = float(finder.radii_mpc_h[finder_index])
        vide_radius = float(vide.radii_mpc_h[vide_index])
        min_radius = min(finder_radius, vide_radius)
        rows.append(
            {
                "target": target,
                "position_mode": position_mode,
                "vide_center_kind": vide.center_kind,
                "vide_variant": vide_variant,
                "finder_void_id": int(finder.void_ids[finder_index]),
                "finder_x_mpc_h": float(finder.positions_mpc_h[finder_index, 0]),
                "finder_y_mpc_h": float(finder.positions_mpc_h[finder_index, 1]),
                "finder_z_mpc_h": float(finder.positions_mpc_h[finder_index, 2]),
                "finder_reff_mpc_h": finder_radius,
                "finder_member_protovoid_count": int(finder.member_counts[finder_index]),
                "finder_total_source_mass_msun_h": float(
                    finder.total_source_masses_msun_h[finder_index]
                ),
                "vide_void_id": int(vide.void_ids[vide_index]),
                "vide_file_void_id": int(vide.file_void_ids[vide_index]),
                "vide_x_mpc_h": float(vide.positions_mpc_h[vide_index, 0]),
                "vide_y_mpc_h": float(vide.positions_mpc_h[vide_index, 1]),
                "vide_z_mpc_h": float(vide.positions_mpc_h[vide_index, 2]),
                "vide_reff_mpc_h": vide_radius,
                "center_distance_mpc_h": float(distance),
                "distance_over_finder_reff": float(distance / finder_radius),
                "distance_over_vide_reff": float(distance / vide_radius),
                "distance_over_min_reff": float(distance / min_radius),
                "radius_ratio_finder_over_vide": float(finder_radius / vide_radius),
            }
        )
    return rows


def _summary_row(
    *,
    target: str,
    position_mode: str,
    vide_center_kind: str,
    vide_variant: str = "all",
    finder_count: int,
    vide_count: int,
    rows: Sequence[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    if not rows:
        return {
            "target": target,
            "position_mode": position_mode,
            "vide_center_kind": vide_center_kind,
            "vide_variant": vide_variant,
            "finder_count": finder_count,
            "vide_count": vide_count,
            "matched_finder_count": 0,
            "fraction_distance_lt_finder_reff": "",
            "fraction_distance_lt_vide_reff": "",
            "fraction_distance_lt_min_reff": "",
            "median_center_distance_mpc_h": "",
            "median_distance_over_min_reff": "",
            "p90_distance_over_min_reff": "",
        }

    distance = np.asarray([float(row["center_distance_mpc_h"]) for row in rows])
    over_finder = np.asarray([float(row["distance_over_finder_reff"]) for row in rows])
    over_vide = np.asarray([float(row["distance_over_vide_reff"]) for row in rows])
    over_min = np.asarray([float(row["distance_over_min_reff"]) for row in rows])
    return {
        "target": target,
        "position_mode": position_mode,
        "vide_center_kind": vide_center_kind,
        "vide_variant": vide_variant,
        "finder_count": finder_count,
        "vide_count": vide_count,
        "matched_finder_count": len(rows),
        "fraction_distance_lt_finder_reff": float(np.mean(over_finder < 1.0)),
        "fraction_distance_lt_vide_reff": float(np.mean(over_vide < 1.0)),
        "fraction_distance_lt_min_reff": float(np.mean(over_min < 1.0)),
        "median_center_distance_mpc_h": float(np.median(distance)),
        "median_distance_over_min_reff": float(np.median(over_min)),
        "p90_distance_over_min_reff": float(np.percentile(over_min, 90.0)),
    }


def _write_csv(path: Path, *, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    default_output_csv, default_summary_csv = _default_output_paths(
        args.vide_center_kind,
        args.vide_variant,
        args.position_mode,
    )
    output_csv = args.output_csv if args.output_csv is not None else default_output_csv
    summary_csv = args.summary_csv if args.summary_csv is not None else default_summary_csv
    paired = read_paired_pinocchio_halo_catalogs(
        args.catalog_a,
        args.catalog_b,
        box_size_mpc_h=args.box_size,
        position_mode=args.position_mode,
    )
    linking_a, linking_b = _resolve_linking_lengths(args, paired)
    config = PairedVoidFinderConfig(
        linking_length_mpc_h=linking_a,
        source_b_linking_length_mpc_h=linking_b,
        min_cluster_members=args.min_cluster_members,
        min_cluster_mass_msun_h=args.min_cluster_mass,
        reference_rho_bar_msun_h_mpc3=args.rho_bar,
        radius_a0=args.radius_a0,
        radius_alpha=args.radius_alpha,
        adjacency_factor=args.adjacency_factor,
        **full_algorithm_kwargs_from_args(args),
    )
    result = run_paired_halo_void_finder(paired.catalog_a, paired.catalog_b, config=config)
    requested_targets = ("A", "B") if args.target == "both" else (args.target,)

    match_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for target in requested_targets:
        finder_result = result.voids_a if target == "A" else result.voids_b
        vide_desc = args.vide_desc_a if target == "A" else args.vide_desc_b
        vide_centers = args.vide_centers_a if target == "A" else args.vide_centers_b
        vide_macrocenters = (
            args.vide_macrocenters_a if target == "A" else args.vide_macrocenters_b
        )
        vide_desc, vide_centers, vide_macrocenters = resolve_target_vide_paths(
            desc_path=vide_desc,
            centers_path=vide_centers,
            macrocenters_path=vide_macrocenters,
            variant=args.vide_variant,
        )
        finder = finder_spatial_catalog(finder_result)
        vide = load_vide_spatial_catalog(
            desc_path=vide_desc,
            centers_path=vide_centers,
            macrocenters_path=vide_macrocenters,
            center_kind=args.vide_center_kind,
        )
        target_rows = build_match_rows(
            target=target,
            position_mode=args.position_mode,
            finder=finder,
            vide=vide,
            box_size_mpc_h=args.box_size,
            vide_variant=args.vide_variant,
        )
        target_summary = _summary_row(
            target=target,
            position_mode=args.position_mode,
            vide_center_kind=args.vide_center_kind,
            vide_variant=args.vide_variant,
            finder_count=len(finder.radii_mpc_h),
            vide_count=len(vide.radii_mpc_h),
            rows=target_rows,
        )
        match_rows.extend(target_rows)
        summary_rows.append(target_summary)
        median = target_summary["median_distance_over_min_reff"]
        fraction = target_summary["fraction_distance_lt_min_reff"]
        print(
            f"Target {target}: finder={len(finder.radii_mpc_h)} VIDE={len(vide.radii_mpc_h)} "
            f"median d/min(Reff)={median} fraction d<min(Reff)={fraction}"
        )

    _write_csv(output_csv, fieldnames=MATCH_COLUMNS, rows=match_rows)
    _write_csv(summary_csv, fieldnames=SUMMARY_COLUMNS, rows=summary_rows)
    print(f"Wrote {output_csv}")
    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
