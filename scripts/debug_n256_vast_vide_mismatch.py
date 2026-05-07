#!/usr/bin/env python
"""Quantify why n256 VAST and VIDE void catalogs disagree."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.exists():
        sys.path.insert(0, str(import_path))

from pinocchio_voids.io import VIDE_CATALOG_VARIANTS

try:
    from scripts.plot_n256_vast_vide_diagnostics import (
        VastCatalog,
        read_vast_catalog,
        read_vide_spatial,
        requested_targets,
        vast_radii,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from plot_n256_vast_vide_diagnostics import (
        VastCatalog,
        read_vast_catalog,
        read_vide_spatial,
        requested_targets,
        vast_radii,
    )


DEFAULT_OUTPUT_PREFIX = Path("runs/void-statistics/n256_vast_vide_mismatch")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write n256 VAST-vs-VIDE radius, VSF, and center-match diagnostics."
    )
    parser.add_argument("--vast-root", type=Path, default=Path("runs/vast-voidfinder"))
    parser.add_argument("--vast-run-suffix", default="")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--vide-variant",
        action="append",
        choices=VIDE_CATALOG_VARIANTS,
        help="VIDE variant to compare. Repeat to select multiple; default is all variants.",
    )
    parser.add_argument("--target", choices=("A", "B", "both"), default="both")
    parser.add_argument("--box-size", type=float, default=256.0)
    parser.add_argument("--bins", type=int, default=17)
    parser.add_argument("--bin-min", type=float, default=10.0)
    parser.add_argument("--bin-max", type=float, default=80.0)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--max-voxels-per-void", type=int, default=3_000_000)
    return parser.parse_args(argv)


def requested_variants(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.vide_variant) if args.vide_variant else VIDE_CATALOG_VARIANTS


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def radius_summary_row(
    *,
    target: str,
    variant: str,
    source: str,
    radius_mode: str,
    radii_mpc_h: NDArray[np.float64],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "target": target,
        "vide_variant": variant,
        "source": source,
        "radius_mode": radius_mode,
        "count": int(len(radii_mpc_h)),
        "min_mpc_h": "",
        "p10_mpc_h": "",
        "median_mpc_h": "",
        "p90_mpc_h": "",
        "max_mpc_h": "",
    }
    if len(radii_mpc_h):
        row.update(
            {
                "min_mpc_h": float(np.min(radii_mpc_h)),
                "p10_mpc_h": float(np.percentile(radii_mpc_h, 10.0)),
                "median_mpc_h": float(np.median(radii_mpc_h)),
                "p90_mpc_h": float(np.percentile(radii_mpc_h, 90.0)),
                "max_mpc_h": float(np.max(radii_mpc_h)),
            }
        )
    return row


def vsf_rows(
    *,
    target: str,
    variant: str,
    source: str,
    radius_mode: str,
    radii_mpc_h: NDArray[np.float64],
    box_size_mpc_h: float,
    edges_mpc_h: NDArray[np.float64],
) -> list[dict[str, float | int | str]]:
    counts, _ = np.histogram(radii_mpc_h, bins=edges_mpc_h)
    centers = np.sqrt(edges_mpc_h[:-1] * edges_mpc_h[1:])
    dlnr = np.diff(np.log(edges_mpc_h))
    density = counts / (box_size_mpc_h**3 * dlnr)
    rows: list[dict[str, float | int | str]] = []
    cumulative = np.asarray([np.count_nonzero(radii_mpc_h >= edge) for edge in edges_mpc_h[:-1]])
    for r_min, r_max, r_mid, count, count_ge, dn_dlnr_dv in zip(
        edges_mpc_h[:-1],
        edges_mpc_h[1:],
        centers,
        counts,
        cumulative,
        density,
        strict=True,
    ):
        rows.append(
            {
                "target": target,
                "vide_variant": variant,
                "source": source,
                "radius_mode": radius_mode,
                "bin_min_mpc_h": float(r_min),
                "bin_max_mpc_h": float(r_max),
                "bin_center_mpc_h": float(r_mid),
                "count": int(count),
                "count_ge_bin_min": int(count_ge),
                "density_dndlnr_per_mpc_h3": float(dn_dlnr_dv),
            }
        )
    return rows


def periodic_nearest(
    query_positions_mpc_h: NDArray[np.float64],
    reference_positions_mpc_h: NDArray[np.float64],
    *,
    box_size_mpc_h: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    query = np.asarray(query_positions_mpc_h, dtype=np.float64) % box_size_mpc_h
    reference = np.asarray(reference_positions_mpc_h, dtype=np.float64) % box_size_mpc_h
    if len(query) == 0 or len(reference) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    tree = cKDTree(reference, boxsize=box_size_mpc_h)
    distances, indices = tree.query(query, k=1)
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def nearest_match_rows(
    *,
    target: str,
    variant: str,
    direction: str,
    query_source: str,
    reference_source: str,
    radius_mode: str,
    query_positions_mpc_h: NDArray[np.float64],
    query_radii_mpc_h: NDArray[np.float64],
    query_ids: NDArray[np.int64],
    reference_positions_mpc_h: NDArray[np.float64],
    reference_radii_mpc_h: NDArray[np.float64],
    reference_ids: NDArray[np.int64],
    box_size_mpc_h: float,
) -> list[dict[str, float | int | str]]:
    distances, indices = periodic_nearest(
        query_positions_mpc_h,
        reference_positions_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    rows: list[dict[str, float | int | str]] = []
    for query_index, distance in enumerate(distances):
        reference_index = int(indices[query_index])
        query_radius = float(query_radii_mpc_h[query_index])
        reference_radius = float(reference_radii_mpc_h[reference_index])
        radius_sum = query_radius + reference_radius
        min_radius = min(query_radius, reference_radius)
        rows.append(
            {
                "target": target,
                "vide_variant": variant,
                "direction": direction,
                "query_source": query_source,
                "reference_source": reference_source,
                "radius_mode": radius_mode,
                "query_id": int(query_ids[query_index]),
                "reference_id": int(reference_ids[reference_index]),
                "distance_mpc_h": float(distance),
                "query_radius_mpc_h": query_radius,
                "reference_radius_mpc_h": reference_radius,
                "distance_over_query_radius": float(distance / query_radius),
                "distance_over_reference_radius": float(distance / reference_radius),
                "distance_over_min_radius": float(distance / min_radius),
                "center_inside_query_radius": int(distance <= query_radius),
                "center_inside_reference_radius": int(distance <= reference_radius),
                "sphere_intersection_margin_mpc_h": float(radius_sum - distance),
                "overlap_proxy": float(max(0.0, radius_sum - distance) / radius_sum),
            }
        )
    return rows


def match_summary_rows(
    rows: Sequence[Mapping[str, float | int | str]]
) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, float | int | str]]] = {}
    for row in rows:
        key = (
            str(row["target"]),
            str(row["vide_variant"]),
            str(row["direction"]),
            str(row["radius_mode"]),
        )
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, float | int | str]] = []
    for (target, variant, direction, radius_mode), group_rows in groups.items():
        distances = np.asarray([float(row["distance_mpc_h"]) for row in group_rows])
        ratios = np.asarray([float(row["distance_over_min_radius"]) for row in group_rows])
        overlaps = np.asarray([float(row["overlap_proxy"]) for row in group_rows])
        summaries.append(
            {
                "target": target,
                "vide_variant": variant,
                "direction": direction,
                "radius_mode": radius_mode,
                "count": int(len(group_rows)),
                "median_distance_mpc_h": float(np.median(distances)),
                "p90_distance_mpc_h": float(np.percentile(distances, 90.0)),
                "median_distance_over_min_radius": float(np.median(ratios)),
                "p90_distance_over_min_radius": float(np.percentile(ratios, 90.0)),
                "center_inside_reference_fraction": float(
                    np.mean([int(row["center_inside_reference_radius"]) for row in group_rows])
                ),
                "positive_sphere_intersection_fraction": float(np.mean(overlaps > 0.0)),
                "median_overlap_proxy": float(np.median(overlaps)),
            }
        )
    return summaries


def write_rows(path: Path, rows: Sequence[Mapping[str, float | int | str]]) -> None:
    if not rows:
        raise SystemExit(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_vide_variant(target: str, variant: str):
    try:
        return read_vide_spatial(target, variant=variant)
    except (OSError, SystemExit) as exc:
        print(f"Skipping missing or invalid VIDE variant {variant} target {target}: {exc}")
        return None


def add_variant_rows(
    *,
    target: str,
    variant: str,
    vast: VastCatalog,
    vide,
    edges_mpc_h: NDArray[np.float64],
    args: argparse.Namespace,
    radius_rows: list[dict[str, float | int | str]],
    vsf_output_rows: list[dict[str, float | int | str]],
    match_rows: list[dict[str, float | int | str]],
) -> None:
    radius_modes = ("maximal", "reff")
    for radius_mode in radius_modes:
        vast_values = vast_radii(vast, radius_mode)
        radius_rows.append(
            radius_summary_row(
                target=target,
                variant=variant,
                source="vast",
                radius_mode=radius_mode,
                radii_mpc_h=vast_values,
            )
        )
        radius_rows.append(
            radius_summary_row(
                target=target,
                variant=variant,
                source="vide",
                radius_mode=radius_mode,
                radii_mpc_h=vide.radii_mpc_h,
            )
        )
        vsf_output_rows.extend(
            vsf_rows(
                target=target,
                variant=variant,
                source="vast",
                radius_mode=radius_mode,
                radii_mpc_h=vast_values,
                box_size_mpc_h=float(args.box_size),
                edges_mpc_h=edges_mpc_h,
            )
        )
        vsf_output_rows.extend(
            vsf_rows(
                target=target,
                variant=variant,
                source="vide",
                radius_mode=radius_mode,
                radii_mpc_h=vide.radii_mpc_h,
                box_size_mpc_h=float(args.box_size),
                edges_mpc_h=edges_mpc_h,
            )
        )
        match_rows.extend(
            nearest_match_rows(
                target=target,
                variant=variant,
                direction="vast_to_vide",
                query_source="vast",
                reference_source="vide",
                radius_mode=radius_mode,
                query_positions_mpc_h=vast.positions_mpc_h,
                query_radii_mpc_h=vast_values,
                query_ids=vast.void_ids,
                reference_positions_mpc_h=vide.positions_mpc_h,
                reference_radii_mpc_h=vide.radii_mpc_h,
                reference_ids=vide.void_ids,
                box_size_mpc_h=float(args.box_size),
            )
        )
        match_rows.extend(
            nearest_match_rows(
                target=target,
                variant=variant,
                direction="vide_to_vast",
                query_source="vide",
                reference_source="vast",
                radius_mode=radius_mode,
                query_positions_mpc_h=vide.positions_mpc_h,
                query_radii_mpc_h=vide.radii_mpc_h,
                query_ids=vide.void_ids,
                reference_positions_mpc_h=vast.positions_mpc_h,
                reference_radii_mpc_h=vast_values,
                reference_ids=vast.void_ids,
                box_size_mpc_h=float(args.box_size),
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    targets = requested_targets(args)
    variants = requested_variants(args)
    edges = np.linspace(float(args.bin_min), float(args.bin_max), int(args.bins) + 1)
    vast_catalogs = {
        target: read_vast_catalog(
            args.vast_root,
            target=target,
            run_suffix=str(args.vast_run_suffix),
            box_size_mpc_h=float(args.box_size),
            voxel_size_mpc_h=float(args.voxel_size),
            max_voxels_per_void=int(args.max_voxels_per_void),
        )
        for target in targets
    }

    radius_rows: list[dict[str, float | int | str]] = []
    vsf_output_rows: list[dict[str, float | int | str]] = []
    match_rows: list[dict[str, float | int | str]] = []
    for target in targets:
        vast = vast_catalogs[target]
        for variant in variants:
            vide = load_vide_variant(target, variant)
            if vide is None:
                continue
            add_variant_rows(
                target=target,
                variant=variant,
                vast=vast,
                vide=vide,
                edges_mpc_h=edges,
                args=args,
                radius_rows=radius_rows,
                vsf_output_rows=vsf_output_rows,
                match_rows=match_rows,
            )

    write_rows(output_path(args.output_prefix, "_radius_summary.csv"), radius_rows)
    write_rows(output_path(args.output_prefix, "_vsf.csv"), vsf_output_rows)
    write_rows(output_path(args.output_prefix, "_nearest_matches.csv"), match_rows)
    write_rows(output_path(args.output_prefix, "_nearest_match_summary.csv"), match_summary_rows(match_rows))
    print(f"Wrote {output_path(args.output_prefix, '_radius_summary.csv')}")
    print(f"Wrote {output_path(args.output_prefix, '_nearest_match_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
