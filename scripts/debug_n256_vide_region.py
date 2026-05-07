#!/usr/bin/env python
"""Diagnose finder and VIDE coverage around a point in the n256 box."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pinocchio_voids.calibration import mean_halo_spacing_mpc_h
from pinocchio_voids.geometry import periodic_distance
from pinocchio_voids.io import (
    PINOCCHIO_POSITION_MODES,
    VIDE_CATALOG_VARIANTS,
    pinocchio_position_mode_output_suffix,
    read_paired_pinocchio_halo_catalogs,
    resolve_vide_catalog_variant_path,
)
from pinocchio_voids.voidfinder import PairedVoidFinderConfig, run_paired_halo_void_finder

try:
    from scripts.plot_n256_halo_void_slice import (
        DEFAULT_CALIBRATION_SUMMARY,
        read_calibration_best_fit,
    )
    from scripts.plot_n256_void_slice import (
        AXIS_INDEX,
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        N256_RUN,
        PLANE_AXES,
        VideSpatialCatalog,
        add_full_algorithm_arguments,
        full_algorithm_kwargs_from_args,
        load_vide_spatial_catalog,
        periodic_axis_distance,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from plot_n256_halo_void_slice import DEFAULT_CALIBRATION_SUMMARY, read_calibration_best_fit
    from plot_n256_void_slice import (
        AXIS_INDEX,
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        N256_RUN,
        PLANE_AXES,
        VideSpatialCatalog,
        add_full_algorithm_arguments,
        full_algorithm_kwargs_from_args,
        load_vide_spatial_catalog,
        periodic_axis_distance,
    )

@dataclass(frozen=True)
class SpatialVoidCatalog:
    """Simple spatial void catalog used by the region diagnostics."""

    method: str
    target: str
    catalog_variant: str
    position_mode: str
    center_kind: str
    positions_mpc_h: NDArray[np.float64]
    radii_mpc_h: NDArray[np.float64]
    void_ids: NDArray[np.int64]
    file_void_ids: NDArray[np.int64]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether finder and VIDE voids cover a point in the n256 box."
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
        help="VIDE position convention to audit.",
    )
    parser.add_argument(
        "--vide-variant",
        action="append",
        choices=VIDE_CATALOG_VARIANTS,
        default=None,
        help="VIDE catalog variant to audit. Repeat to select several variants.",
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=DEFAULT_CALIBRATION_SUMMARY,
        help="Joint MCMC summary CSV used for default finder parameters when present.",
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
    linking.add_argument("--linking-factor", type=float)
    parser.add_argument("--min-cluster-members", type=int, default=2)
    parser.add_argument("--min-cluster-mass", type=float, default=0.0)
    parser.add_argument("--radius-a0", type=float)
    parser.add_argument("--radius-alpha", type=float)
    parser.add_argument("--adjacency-factor", type=float)
    add_full_algorithm_arguments(parser, summary_defaults=True)
    parser.add_argument("--target", choices=("A", "B", "both"), default="A")
    parser.add_argument("--x", type=float, default=100.0)
    parser.add_argument("--y", type=float, default=100.0)
    parser.add_argument("--z", type=float, default=128.0)
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-center", type=float, default=128.0)
    parser.add_argument("--slice-thickness", type=float, default=20.0)
    parser.add_argument(
        "--halo-radius",
        type=float,
        action="append",
        default=None,
        help="Projected disk radius for halo-density checks. Repeat for several radii.",
    )
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument(
        "--skip-finder",
        action="store_true",
        help="Only audit halos and VIDE catalogs, without running the finder.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/void-statistics/n256_vide_region_debug.csv"),
    )
    return parser.parse_args(argv)


def variant_path(path: Path, variant: str) -> Path:
    """Return the matching VIDE path for a catalog variant."""

    return resolve_vide_catalog_variant_path(path, variant)


def _variant_label(variant: str) -> str:
    return "all" if variant == "default" else variant


def point_from_args(args: argparse.Namespace) -> NDArray[np.float64]:
    return np.asarray([args.x, args.y, args.z], dtype=np.float64)


def projected_distances_to_point(
    positions_mpc_h: NDArray[np.float64],
    point_mpc_h: NDArray[np.float64],
    *,
    slice_axis: str,
    box_size_mpc_h: float,
) -> NDArray[np.float64]:
    """Return periodic 2D distances in the plotted plane."""

    plane_x, plane_y = PLANE_AXES[slice_axis]
    dx = (
        positions_mpc_h[:, plane_x]
        - point_mpc_h[plane_x]
        + 0.5 * box_size_mpc_h
    ) % box_size_mpc_h - 0.5 * box_size_mpc_h
    dy = (
        positions_mpc_h[:, plane_y]
        - point_mpc_h[plane_y]
        + 0.5 * box_size_mpc_h
    ) % box_size_mpc_h - 0.5 * box_size_mpc_h
    return np.hypot(dx, dy)


def projected_sphere_radii(
    *,
    positions_mpc_h: NDArray[np.float64],
    radii_mpc_h: NDArray[np.float64],
    slice_axis: str,
    slice_center_mpc_h: float,
    slice_thickness_mpc_h: float,
    box_size_mpc_h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return display radii used by the halo-slice sphere overlay."""

    axis_distances = periodic_axis_distance(
        positions_mpc_h[:, AXIS_INDEX[slice_axis]],
        center_mpc_h=slice_center_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    outside_slab = np.maximum(0.0, axis_distances - 0.5 * slice_thickness_mpc_h)
    display_radii = np.sqrt(np.maximum(0.0, radii_mpc_h**2 - outside_slab**2))
    return display_radii, axis_distances


def halo_region_rows(
    *,
    target: str,
    position_mode: str,
    positions_mpc_h: NDArray[np.float64],
    point_mpc_h: NDArray[np.float64],
    slice_axis: str,
    slice_center_mpc_h: float,
    slice_thickness_mpc_h: float,
    box_size_mpc_h: float,
    radii_mpc_h: Sequence[float],
) -> list[dict[str, float | int | str]]:
    """Count halos in projected disks around the diagnostic point."""

    axis_distances = periodic_axis_distance(
        positions_mpc_h[:, AXIS_INDEX[slice_axis]],
        center_mpc_h=slice_center_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    slab = axis_distances <= 0.5 * slice_thickness_mpc_h
    slab_positions = positions_mpc_h[slab]
    projected = projected_distances_to_point(
        slab_positions,
        point_mpc_h,
        slice_axis=slice_axis,
        box_size_mpc_h=box_size_mpc_h,
    )
    rows: list[dict[str, float | int | str]] = []
    for radius in radii_mpc_h:
        area = np.pi * float(radius) ** 2
        expected = len(slab_positions) * area / box_size_mpc_h**2
        count = int(np.count_nonzero(projected <= float(radius)))
        rows.append(
            {
                "row_type": "halo_count",
                "target": target,
                "method": "halos",
                "catalog_variant": "",
                "position_mode": position_mode,
                "center_kind": "",
                "metric": f"projected_disk_{float(radius):g}",
                "rank": "",
                "void_id": "",
                "file_void_id": "",
                "x_mpc_h": "",
                "y_mpc_h": "",
                "z_mpc_h": "",
                "reff_mpc_h": "",
                "distance_3d_mpc_h": "",
                "margin_3d_mpc_h": "",
                "distance_projected_mpc_h": "",
                "display_radius_mpc_h": "",
                "margin_projected_mpc_h": "",
                "axis_distance_mpc_h": "",
                "contains_point_3d_count": "",
                "covers_point_projected_count": "",
                "halo_slab_count": int(len(slab_positions)),
                "halo_count": count,
                "halo_uniform_expected": float(expected),
                "halo_count_over_expected": float(count / expected) if expected > 0.0 else "",
            }
        )
    return rows


def void_region_rows(
    *,
    catalog: SpatialVoidCatalog,
    point_mpc_h: NDArray[np.float64],
    slice_axis: str,
    slice_center_mpc_h: float,
    slice_thickness_mpc_h: float,
    box_size_mpc_h: float,
    top_n: int,
) -> list[dict[str, float | int | str]]:
    """Build nearest-margin rows for one void catalog."""

    if len(catalog.radii_mpc_h) == 0:
        return []
    d3 = np.asarray(
        periodic_distance(catalog.positions_mpc_h, point_mpc_h, box_size_mpc_h),
        dtype=np.float64,
    )
    margin_3d = d3 - catalog.radii_mpc_h
    display_radii, axis_distances = projected_sphere_radii(
        positions_mpc_h=catalog.positions_mpc_h,
        radii_mpc_h=catalog.radii_mpc_h,
        slice_axis=slice_axis,
        slice_center_mpc_h=slice_center_mpc_h,
        slice_thickness_mpc_h=slice_thickness_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    projected = projected_distances_to_point(
        catalog.positions_mpc_h,
        point_mpc_h,
        slice_axis=slice_axis,
        box_size_mpc_h=box_size_mpc_h,
    )
    margin_projected = projected - display_radii
    selected: list[tuple[str, int]] = []
    for metric, margins in (("nearest_3d_margin", margin_3d), ("nearest_projected_margin", margin_projected)):
        limit = min(max(int(top_n), 0), len(margins))
        selected.extend((metric, int(index)) for index in np.argsort(margins)[:limit])

    contains_count = int(np.count_nonzero(margin_3d <= 0.0))
    projected_count = int(np.count_nonzero(margin_projected <= 0.0))
    rows: list[dict[str, float | int | str]] = []
    for metric, index in selected:
        rows.append(
            {
                "row_type": "void_margin",
                "target": catalog.target,
                "method": catalog.method,
                "catalog_variant": catalog.catalog_variant,
                "position_mode": catalog.position_mode,
                "center_kind": catalog.center_kind,
                "metric": metric,
                "rank": int(sum(1 for m, _ in selected[: selected.index((metric, index))] if m == metric) + 1),
                "void_id": int(catalog.void_ids[index]),
                "file_void_id": "" if int(catalog.file_void_ids[index]) < 0 else int(catalog.file_void_ids[index]),
                "x_mpc_h": float(catalog.positions_mpc_h[index, 0]),
                "y_mpc_h": float(catalog.positions_mpc_h[index, 1]),
                "z_mpc_h": float(catalog.positions_mpc_h[index, 2]),
                "reff_mpc_h": float(catalog.radii_mpc_h[index]),
                "distance_3d_mpc_h": float(d3[index]),
                "margin_3d_mpc_h": float(margin_3d[index]),
                "distance_projected_mpc_h": float(projected[index]),
                "display_radius_mpc_h": float(display_radii[index]),
                "margin_projected_mpc_h": float(margin_projected[index]),
                "axis_distance_mpc_h": float(axis_distances[index]),
                "contains_point_3d_count": contains_count,
                "covers_point_projected_count": projected_count,
                "halo_slab_count": "",
                "halo_count": "",
                "halo_uniform_expected": "",
                "halo_count_over_expected": "",
            }
        )
    return rows


def _spatial_from_finder_result(
    result,
    *,
    target: str,
    position_mode: str,
) -> SpatialVoidCatalog:
    positions = np.asarray([void.center_mpc_h for void in result.voids], dtype=np.float64).reshape((-1, 3))
    radii = np.asarray([void.effective_radius_mpc_h for void in result.voids], dtype=np.float64)
    void_ids = np.asarray([void.id for void in result.voids], dtype=np.int64)
    return SpatialVoidCatalog(
        method="finder",
        target=target,
        catalog_variant="",
        position_mode=position_mode,
        center_kind="center",
        positions_mpc_h=positions,
        radii_mpc_h=radii,
        void_ids=void_ids,
        file_void_ids=np.full(void_ids.shape, -1, dtype=np.int64),
    )


def _spatial_from_vide(
    spatial: VideSpatialCatalog,
    *,
    target: str,
    catalog_variant: str,
) -> SpatialVoidCatalog:
    return SpatialVoidCatalog(
        method="vide",
        target=target,
        catalog_variant=catalog_variant,
        position_mode="final",
        center_kind=spatial.center_kind,
        positions_mpc_h=spatial.positions_mpc_h,
        radii_mpc_h=spatial.radii_mpc_h,
        void_ids=spatial.void_ids,
        file_void_ids=spatial.file_void_ids,
    )


def _target_vide_paths(args: argparse.Namespace, target: str) -> tuple[Path, Path, Path]:
    if target == "A":
        return args.vide_desc_a, args.vide_centers_a, args.vide_macrocenters_a
    return args.vide_desc_b, args.vide_centers_b, args.vide_macrocenters_b


def _resolve_finder_parameters(args: argparse.Namespace) -> dict[str, float]:
    summary = read_calibration_best_fit(args.calibration_summary)
    defaults = {
        "linking_factor": float(N256_RUN["linking_factor"]),
        "radius_a0": float(N256_RUN["radius_a0"]),
        "radius_alpha": float(N256_RUN["radius_alpha"]),
        "adjacency_factor": float(N256_RUN["adjacency_factor"]),
    }
    values = defaults | {name: summary[name] for name in defaults if name in summary}
    for name, value in (
        ("linking_factor", args.linking_factor),
        ("radius_a0", args.radius_a0),
        ("radius_alpha", args.radius_alpha),
        ("adjacency_factor", args.adjacency_factor),
    ):
        if value is not None:
            values[name] = float(value)
    return values


def _run_finder_spatial_catalogs(args: argparse.Namespace, paired) -> dict[str, SpatialVoidCatalog]:
    parameters = _resolve_finder_parameters(args)
    summary = read_calibration_best_fit(args.calibration_summary)
    if args.linking_length is not None:
        linking_a = float(args.linking_length)
        linking_b = float(args.linking_length)
    else:
        linking_a = parameters["linking_factor"] * mean_halo_spacing_mpc_h(paired.catalog_a)
        linking_b = parameters["linking_factor"] * mean_halo_spacing_mpc_h(paired.catalog_b)
    config = PairedVoidFinderConfig(
        linking_length_mpc_h=linking_a,
        source_b_linking_length_mpc_h=linking_b,
        min_cluster_members=args.min_cluster_members,
        min_cluster_mass_msun_h=args.min_cluster_mass,
        reference_rho_bar_msun_h_mpc3=args.rho_bar,
        radius_a0=parameters["radius_a0"],
        radius_alpha=parameters["radius_alpha"],
        adjacency_factor=parameters["adjacency_factor"],
        **full_algorithm_kwargs_from_args(args, summary=summary),
    )
    result = run_paired_halo_void_finder(paired.catalog_a, paired.catalog_b, config=config)
    return {
        "A": _spatial_from_finder_result(
            result.voids_a,
            target="A",
            position_mode=args.position_mode,
        ),
        "B": _spatial_from_finder_result(
            result.voids_b,
            target="B",
            position_mode=args.position_mode,
        ),
    }


def _write_rows(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: Sequence[dict[str, float | int | str]]) -> None:
    halo_rows = [row for row in rows if row["row_type"] == "halo_count"]
    for row in halo_rows:
        print(
            f"Target {row['target']} halos {row['metric']}: "
            f"count={row['halo_count']} expected={float(row['halo_uniform_expected']):.2f} "
            f"ratio={row['halo_count_over_expected']}"
        )
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row["row_type"] != "void_margin":
            continue
        key = (
            str(row["target"]),
            str(row["method"]),
            str(row["catalog_variant"]),
            str(row["center_kind"]),
        )
        if key in seen:
            continue
        seen.add(key)
        best_3d = next(
            item
            for item in rows
            if item["row_type"] == "void_margin"
            and item["target"] == row["target"]
            and item["method"] == row["method"]
            and item["catalog_variant"] == row["catalog_variant"]
            and item["center_kind"] == row["center_kind"]
            and item["metric"] == "nearest_3d_margin"
        )
        best_projected = next(
            item
            for item in rows
            if item["row_type"] == "void_margin"
            and item["target"] == row["target"]
            and item["method"] == row["method"]
            and item["catalog_variant"] == row["catalog_variant"]
            and item["center_kind"] == row["center_kind"]
            and item["metric"] == "nearest_projected_margin"
        )
        variant = f" {row['catalog_variant']}" if row["catalog_variant"] else ""
        print(
            f"Target {row['target']} {row['method']}{variant} ({row['center_kind']}): "
            f"contains_3d={row['contains_point_3d_count']} "
            f"covers_projected={row['covers_point_projected_count']} "
            f"best_3d_margin={float(best_3d['margin_3d_mpc_h']):.2f} "
            f"best_projected_margin={float(best_projected['margin_projected_mpc_h']):.2f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    point = point_from_args(args)
    halo_radii = args.halo_radius if args.halo_radius is not None else [10.0, 20.0, 30.0, 40.0]
    variants = args.vide_variant if args.vide_variant is not None else list(VIDE_CATALOG_VARIANTS)
    if (
        args.output_csv == Path("runs/void-statistics/n256_vide_region_debug.csv")
        and args.position_mode != "final"
    ):
        args.output_csv = args.output_csv.with_name(
            f"{args.output_csv.stem}{pinocchio_position_mode_output_suffix(args.position_mode)}"
            f"{args.output_csv.suffix}"
        )
    paired = read_paired_pinocchio_halo_catalogs(
        args.catalog_a,
        args.catalog_b,
        box_size_mpc_h=args.box_size,
        position_mode=args.position_mode,
    )
    requested_targets = ("A", "B") if args.target == "both" else (args.target,)
    finder_catalogs = {} if args.skip_finder else _run_finder_spatial_catalogs(args, paired)

    rows: list[dict[str, float | int | str]] = []
    for target in requested_targets:
        target_catalog = paired.catalog_a if target == "A" else paired.catalog_b
        rows.extend(
            halo_region_rows(
                target=target,
                position_mode=args.position_mode,
                positions_mpc_h=target_catalog.positions_mpc_h,
                point_mpc_h=point,
                slice_axis=args.slice_axis,
                slice_center_mpc_h=args.slice_center,
                slice_thickness_mpc_h=args.slice_thickness,
                box_size_mpc_h=args.box_size,
                radii_mpc_h=halo_radii,
            )
        )
        if target in finder_catalogs:
            rows.extend(
                void_region_rows(
                    catalog=finder_catalogs[target],
                    point_mpc_h=point,
                    slice_axis=args.slice_axis,
                    slice_center_mpc_h=args.slice_center,
                    slice_thickness_mpc_h=args.slice_thickness,
                    box_size_mpc_h=args.box_size,
                    top_n=args.top_n,
                )
            )
        desc_path, centers_path, macrocenters_path = _target_vide_paths(args, target)
        for variant in variants:
            variant_desc = variant_path(desc_path, variant)
            variant_centers = variant_path(centers_path, variant)
            variant_macrocenters = variant_path(macrocenters_path, variant)
            required_center_path = variant_macrocenters if args.vide_center_kind == "macrocenter" else variant_centers
            if not variant_desc.exists() or not required_center_path.exists():
                print(f"Skipping missing VIDE {target} {variant}: {variant_desc}")
                continue
            spatial = load_vide_spatial_catalog(
                desc_path=variant_desc,
                centers_path=variant_centers,
                macrocenters_path=variant_macrocenters,
                center_kind=args.vide_center_kind,
            )
            rows.extend(
                void_region_rows(
                    catalog=_spatial_from_vide(
                        spatial,
                        target=target,
                        catalog_variant=_variant_label(variant),
                    ),
                    point_mpc_h=point,
                    slice_axis=args.slice_axis,
                    slice_center_mpc_h=args.slice_center,
                    slice_thickness_mpc_h=args.slice_thickness,
                    box_size_mpc_h=args.box_size,
                    top_n=args.top_n,
                )
            )

    _write_rows(args.output_csv, rows)
    _print_summary(rows)
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
