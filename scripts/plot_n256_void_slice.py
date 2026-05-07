#!/usr/bin/env python
"""Plot a 2D n256 box slice with finder and VIDE void circles."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from pinocchio_voids.calibration import mean_halo_spacing_mpc_h
from pinocchio_voids.geometry import minimum_image_displacement, periodic_distance
from pinocchio_voids.io import (
    PINOCCHIO_POSITION_MODES,
    VIDE_CATALOG_VARIANTS,
    pinocchio_position_mode_output_suffix,
    read_paired_pinocchio_halo_catalogs,
    read_vide_void_centers,
    read_vide_void_desc,
    read_vide_void_macrocenters,
    resolve_vide_catalog_variant_path,
    vide_catalog_variant_output_suffix,
)
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    PairedVoidFinderConfig,
    run_paired_halo_void_finder,
)

try:
    from scripts.plot_n256_void_size_function import N256_RUN
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from plot_n256_void_size_function import N256_RUN


DEFAULT_VIDE_CENTERS_A = Path(
    "runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/"
    "sample_pinocchio_n256_ss1.0_z0.00_d00/"
    "centers_all_pinocchio_n256_ss1.0_z0.00_d00.out"
)
DEFAULT_VIDE_CENTERS_B = Path(
    "runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/"
    "sample_pinocchio_n256_paired_ss1.0_z0.00_d00/"
    "centers_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out"
)
DEFAULT_VIDE_MACROCENTERS_A = Path(
    "runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/"
    "sample_pinocchio_n256_ss1.0_z0.00_d00/"
    "macrocenters_all_pinocchio_n256_ss1.0_z0.00_d00.out"
)
DEFAULT_VIDE_MACROCENTERS_B = Path(
    "runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/"
    "sample_pinocchio_n256_paired_ss1.0_z0.00_d00/"
    "macrocenters_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out"
)
DEFAULT_VIDE_INPUT_A = Path("runs/vide-lowres/n256/examples/pinocchio_n256_z0.0.dat")
DEFAULT_VIDE_INPUT_B = Path(
    "runs/vide-lowres/n256_paired/examples/pinocchio_n256_paired_z0.0.dat"
)
DEFAULT_VIDE_VOID_ZONES_A = Path(
    "runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/"
    "sample_pinocchio_n256_ss1.0_z0.00_d00/"
    "voidZone_pinocchio_n256_ss1.0_z0.00_d00.dat"
)
DEFAULT_VIDE_VOID_ZONES_B = Path(
    "runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/"
    "sample_pinocchio_n256_paired_ss1.0_z0.00_d00/"
    "voidZone_pinocchio_n256_paired_ss1.0_z0.00_d00.dat"
)
DEFAULT_VIDE_ZONE_PARTICLES_A = Path(
    "runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/"
    "sample_pinocchio_n256_ss1.0_z0.00_d00/"
    "voidPart_pinocchio_n256_ss1.0_z0.00_d00.dat"
)
DEFAULT_VIDE_ZONE_PARTICLES_B = Path(
    "runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/"
    "sample_pinocchio_n256_paired_ss1.0_z0.00_d00/"
    "voidPart_pinocchio_n256_paired_ss1.0_z0.00_d00.dat"
)
DEFAULT_OUTPUT = Path("runs/void-statistics/n256_void_slice_comparison.png")
DEFAULT_OUTPUT_CSV = Path("runs/void-statistics/n256_void_slice_comparison.csv")
FULL_ALGORITHM_DEFAULTS = {
    "merge_score_mode": "geometry_only",
    "merge_threshold": 0.0,
    "geom_weight": 1.0,
    "bridge_weight": 0.0,
    "compatibility_weight": 0.0,
    "bridge_radius_factor": 0.5,
    "bridge_min_radius_mpc_h": 0.0,
    "bridge_delta_scale": 1.0,
    "bridge_density_mode": "mass",
}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
PLANE_AXES = {
    "x": (1, 2),
    "y": (0, 2),
    "z": (0, 1),
}
AXIS_LABELS = ("x", "y", "z")


def add_full_algorithm_arguments(
    parser: argparse.ArgumentParser,
    *,
    summary_defaults: bool = False,
) -> None:
    """Add shared full finder algorithm parameters to an n256 diagnostic CLI."""

    def _default(name: str):
        return None if summary_defaults else FULL_ALGORITHM_DEFAULTS[name]

    parser.add_argument(
        "--merge-score-mode",
        choices=("geometry_only", "weighted"),
        default=_default("merge_score_mode"),
        help="Use all adjacency edges or threshold weighted merge scores.",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=_default("merge_threshold"),
        help="Minimum weighted merge score required to merge an adjacency edge.",
    )
    parser.add_argument(
        "--geom-weight",
        type=float,
        default=_default("geom_weight"),
        help="Weight applied to the geometric merge score.",
    )
    parser.add_argument(
        "--bridge-weight",
        type=float,
        default=_default("bridge_weight"),
        help="Weight applied to the source-catalog bridge-density score.",
    )
    parser.add_argument(
        "--compatibility-weight",
        type=float,
        default=_default("compatibility_weight"),
        help="Weight applied to the source-cluster compatibility score.",
    )
    parser.add_argument(
        "--bridge-radius-factor",
        type=float,
        default=_default("bridge_radius_factor"),
        help="Bridge capsule radius factor for weighted merging.",
    )
    parser.add_argument(
        "--bridge-min-radius",
        dest="bridge_min_radius_mpc_h",
        type=float,
        default=_default("bridge_min_radius_mpc_h"),
        help="Minimum bridge capsule radius in Mpc/h.",
    )
    parser.add_argument(
        "--bridge-delta-scale",
        type=float,
        default=_default("bridge_delta_scale"),
        help="Overdensity scale used to map bridge density to a 0..1 score.",
    )
    parser.add_argument(
        "--bridge-density-mode",
        choices=("number", "mass", "both"),
        default=_default("bridge_density_mode"),
        help="Halo density field used for bridge scoring.",
    )


def full_algorithm_kwargs_from_args(
    args: argparse.Namespace,
    *,
    summary: dict[str, float | str] | None = None,
) -> dict[str, float | str]:
    """Resolve shared full finder algorithm keyword arguments."""

    resolved = dict(FULL_ALGORITHM_DEFAULTS)
    if summary is not None:
        for key in resolved:
            if key in summary:
                resolved[key] = summary[key]
    for key in resolved:
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = value
    return resolved


@dataclass(frozen=True)
class VoidSliceRows:
    """Void circles selected for a 2D slice."""

    method: str
    target: str
    position_mode: str
    positions_mpc_h: NDArray[np.float64]
    radii_mpc_h: NDArray[np.float64]
    void_ids: NDArray[np.int64]
    file_void_ids: NDArray[np.int64]
    distance_to_slice_mpc_h: NDArray[np.float64]
    center_kind: str = ""


@dataclass(frozen=True)
class VideSpatialCatalog:
    """VIDE centers joined to corrected ``voidDesc`` effective radii."""

    positions_mpc_h: NDArray[np.float64]
    radii_mpc_h: NDArray[np.float64]
    void_ids: NDArray[np.int64]
    file_void_ids: NDArray[np.int64]
    center_kind: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot finder and VIDE void Reff circles in a 2D n256 box slice."
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
        help="VIDE position convention to plot and match.",
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
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-center", type=float, default=128.0)
    parser.add_argument("--slice-thickness", type=float, default=20.0)
    parser.add_argument(
        "--include-intersections",
        action="store_true",
        help="Include voids whose sphere intersects the slice, not only centers in the slice.",
    )
    parser.add_argument(
        "--show-nearest-matches",
        action="store_true",
        help="Draw finder-to-nearest-VIDE center links for selected slice objects.",
    )
    parser.add_argument(
        "--max-match-lines",
        type=int,
        default=20,
        help="Maximum nearest-match links to draw per target.",
    )
    parser.add_argument(
        "--label-count",
        type=int,
        default=0,
        help="Label the largest N finder and VIDE circles per target.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    return parser.parse_args(argv)


def periodic_pairwise_distances(
    positions_a_mpc_h: NDArray[np.float64],
    positions_b_mpc_h: NDArray[np.float64],
    *,
    box_size_mpc_h: float,
) -> NDArray[np.float64]:
    """Return all minimum-image distances between two position arrays."""

    positions_a = np.asarray(positions_a_mpc_h, dtype=np.float64).reshape((-1, 3))
    positions_b = np.asarray(positions_b_mpc_h, dtype=np.float64).reshape((-1, 3))
    if positions_a.size == 0 or positions_b.size == 0:
        return np.empty((positions_a.shape[0], positions_b.shape[0]), dtype=np.float64)
    return np.asarray(
        periodic_distance(positions_a[:, np.newaxis, :], positions_b[np.newaxis, :, :], box_size_mpc_h),
        dtype=np.float64,
    )


def periodic_axis_distance(
    values_mpc_h: NDArray[np.float64],
    *,
    center_mpc_h: float,
    box_size_mpc_h: float,
) -> NDArray[np.float64]:
    """Return minimum-image scalar distance from a slice center."""

    delta = (values_mpc_h - center_mpc_h + 0.5 * box_size_mpc_h) % box_size_mpc_h
    return np.abs(delta - 0.5 * box_size_mpc_h)


def slice_mask(
    *,
    positions_mpc_h: NDArray[np.float64],
    radii_mpc_h: NDArray[np.float64],
    axis: str,
    center_mpc_h: float,
    thickness_mpc_h: float,
    box_size_mpc_h: float,
    include_intersections: bool,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Select voids for a periodic slab."""

    if thickness_mpc_h <= 0.0 or not np.isfinite(thickness_mpc_h):
        raise ValueError("slice thickness must be positive and finite")
    axis_index = AXIS_INDEX[axis]
    distances = periodic_axis_distance(
        positions_mpc_h[:, axis_index],
        center_mpc_h=center_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    limit = 0.5 * thickness_mpc_h
    if include_intersections:
        limit = limit + radii_mpc_h
    return distances <= limit, distances


def _resolve_linking_lengths(args: argparse.Namespace, paired) -> tuple[float, float]:
    if args.linking_length is not None:
        return float(args.linking_length), float(args.linking_length)
    factor = float(args.linking_factor)
    return (
        factor * mean_halo_spacing_mpc_h(paired.catalog_a),
        factor * mean_halo_spacing_mpc_h(paired.catalog_b),
    )


def _finder_rows(
    result: DirectionalVoidFinderResult,
    *,
    target: str,
    args: argparse.Namespace,
) -> VoidSliceRows:
    positions = np.asarray(
        [void.center_mpc_h for void in result.voids],
        dtype=np.float64,
    ).reshape((-1, 3))
    radii = np.asarray([void.effective_radius_mpc_h for void in result.voids], dtype=np.float64)
    void_ids = np.asarray([void.id for void in result.voids], dtype=np.int64)
    file_ids = np.full(void_ids.shape, -1, dtype=np.int64)
    mask, distances = slice_mask(
        positions_mpc_h=positions,
        radii_mpc_h=radii,
        axis=args.slice_axis,
        center_mpc_h=args.slice_center,
        thickness_mpc_h=args.slice_thickness,
        box_size_mpc_h=args.box_size,
        include_intersections=args.include_intersections,
    )
    return VoidSliceRows(
        method="finder",
        target=target,
        position_mode=args.position_mode,
        positions_mpc_h=positions[mask],
        radii_mpc_h=radii[mask],
        void_ids=void_ids[mask],
        file_void_ids=file_ids[mask],
        distance_to_slice_mpc_h=distances[mask],
    )


def load_vide_spatial_catalog(
    *,
    desc_path: Path,
    centers_path: Path,
    macrocenters_path: Path,
    center_kind: str,
) -> VideSpatialCatalog:
    desc = read_vide_void_desc(desc_path)
    if center_kind == "center":
        centers = read_vide_void_centers(centers_path)
    elif center_kind == "macrocenter":
        centers = read_vide_void_macrocenters(macrocenters_path)
    else:
        raise ValueError(f"Unknown VIDE center kind: {center_kind}")

    file_desc_ids = desc.column("FileVoid#").astype(np.int64)
    void_ids = desc.void_ids.astype(np.int64)
    radius_by_file_id = dict(zip(file_desc_ids, desc.effective_radii_mpc_h, strict=True))
    void_id_by_file_id = dict(zip(file_desc_ids, void_ids, strict=True))

    positions: list[NDArray[np.float64]] = []
    radii: list[float] = []
    matched_void_ids: list[int] = []
    matched_file_ids: list[int] = []
    for center_position, file_id in zip(
        centers.positions_mpc_h,
        centers.file_void_ids,
        strict=True,
    ):
        if int(file_id) not in radius_by_file_id:
            continue
        positions.append(center_position)
        radii.append(float(radius_by_file_id[int(file_id)]))
        matched_void_ids.append(int(void_id_by_file_id[int(file_id)]))
        matched_file_ids.append(int(file_id))
    if not positions:
        raise SystemExit(f"No VIDE {center_kind}s matched voidDesc FileVoid# values: {desc_path}")

    return VideSpatialCatalog(
        positions_mpc_h=np.asarray(positions, dtype=np.float64),
        radii_mpc_h=np.asarray(radii, dtype=np.float64),
        void_ids=np.asarray(matched_void_ids, dtype=np.int64),
        file_void_ids=np.asarray(matched_file_ids, dtype=np.int64),
        center_kind=center_kind,
    )


def _vide_rows(
    *,
    desc_path: Path,
    centers_path: Path,
    macrocenters_path: Path,
    target: str,
    args: argparse.Namespace,
) -> VoidSliceRows:
    spatial = load_vide_spatial_catalog(
        desc_path=desc_path,
        centers_path=centers_path,
        macrocenters_path=macrocenters_path,
        center_kind=args.vide_center_kind,
    )
    mask, distances = slice_mask(
        positions_mpc_h=spatial.positions_mpc_h,
        radii_mpc_h=spatial.radii_mpc_h,
        axis=args.slice_axis,
        center_mpc_h=args.slice_center,
        thickness_mpc_h=args.slice_thickness,
        box_size_mpc_h=args.box_size,
        include_intersections=args.include_intersections,
    )
    return VoidSliceRows(
        method="vide",
        target=target,
        position_mode=args.position_mode,
        positions_mpc_h=spatial.positions_mpc_h[mask],
        radii_mpc_h=spatial.radii_mpc_h[mask],
        void_ids=spatial.void_ids[mask],
        file_void_ids=spatial.file_void_ids[mask],
        distance_to_slice_mpc_h=distances[mask],
        center_kind=spatial.center_kind,
    )


def resolve_target_vide_paths(
    *,
    desc_path: Path,
    centers_path: Path,
    macrocenters_path: Path,
    variant: str,
) -> tuple[Path, Path, Path]:
    """Resolve VIDE catalog paths for a selected catalog variant."""

    return (
        resolve_vide_catalog_variant_path(desc_path, variant),
        resolve_vide_catalog_variant_path(centers_path, variant),
        resolve_vide_catalog_variant_path(macrocenters_path, variant),
    )


def _project(positions_mpc_h: NDArray[np.float64], axis: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    plane_x, plane_y = PLANE_AXES[axis]
    return positions_mpc_h[:, plane_x], positions_mpc_h[:, plane_y]


def _periodic_circle_centers(
    *,
    x: float,
    y: float,
    radius: float,
    box_size_mpc_h: float,
) -> Iterable[tuple[float, float]]:
    x_offsets = [0.0]
    y_offsets = [0.0]
    if x - radius < 0.0:
        x_offsets.append(box_size_mpc_h)
    if x + radius > box_size_mpc_h:
        x_offsets.append(-box_size_mpc_h)
    if y - radius < 0.0:
        y_offsets.append(box_size_mpc_h)
    if y + radius > box_size_mpc_h:
        y_offsets.append(-box_size_mpc_h)
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            yield x + x_offset, y + y_offset


def write_slice_csv(path: Path, rows: Sequence[VoidSliceRows]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "target",
                "position_mode",
                "void_id",
                "file_void_id",
                "x_mpc_h",
                "y_mpc_h",
                "z_mpc_h",
                "reff_mpc_h",
                "distance_to_slice_mpc_h",
                "center_kind",
            ],
        )
        writer.writeheader()
        for group in rows:
            for position, radius, void_id, file_void_id, distance in zip(
                group.positions_mpc_h,
                group.radii_mpc_h,
                group.void_ids,
                group.file_void_ids,
                group.distance_to_slice_mpc_h,
                strict=True,
            ):
                writer.writerow(
                    {
                        "method": group.method,
                        "target": group.target,
                        "position_mode": group.position_mode,
                        "void_id": int(void_id),
                        "file_void_id": "" if int(file_void_id) < 0 else int(file_void_id),
                        "x_mpc_h": float(position[0]),
                        "y_mpc_h": float(position[1]),
                        "z_mpc_h": float(position[2]),
                        "reff_mpc_h": float(radius),
                        "distance_to_slice_mpc_h": float(distance),
                        "center_kind": group.center_kind,
                    }
                )


def _draw_nearest_match_lines(
    axis,
    *,
    finder_rows: VoidSliceRows,
    vide_rows: VoidSliceRows,
    slice_axis: str,
    box_size_mpc_h: float,
    max_match_lines: int,
) -> None:
    if max_match_lines <= 0:
        return
    if len(finder_rows.positions_mpc_h) == 0 or len(vide_rows.positions_mpc_h) == 0:
        return
    distances = periodic_pairwise_distances(
        finder_rows.positions_mpc_h,
        vide_rows.positions_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(distances.shape[0]), nearest_indices]
    selected_finder = np.argsort(nearest_distances)[:max_match_lines]
    plane_x, plane_y = PLANE_AXES[slice_axis]
    starts = finder_rows.positions_mpc_h[selected_finder]
    ends = vide_rows.positions_mpc_h[nearest_indices[selected_finder]]
    displacements = minimum_image_displacement(starts, ends, box_size_mpc_h)
    start_x = starts[:, plane_x]
    start_y = starts[:, plane_y]
    end_x = start_x + displacements[:, plane_x]
    end_y = start_y + displacements[:, plane_y]
    for x0, y0, x1, y1 in zip(start_x, start_y, end_x, end_y, strict=True):
        axis.plot(
            [float(x0), float(x1)],
            [float(y0), float(y1)],
            color="0.25",
            linewidth=0.75,
            alpha=0.48,
            zorder=1,
        )


def _label_largest_voids(
    axis,
    *,
    group: VoidSliceRows,
    slice_axis: str,
    label_count: int,
) -> None:
    if label_count <= 0 or len(group.radii_mpc_h) == 0:
        return
    xs, ys = _project(group.positions_mpc_h, slice_axis)
    selected = np.argsort(group.radii_mpc_h)[::-1][:label_count]
    prefix = "F" if group.method == "finder" else "V"
    for index in selected:
        raw_id = group.void_ids[index] if group.method == "finder" else group.file_void_ids[index]
        axis.annotate(
            f"{prefix}{int(raw_id)}",
            (float(xs[index]), float(ys[index])),
            xytext=(3.0, 3.0),
            textcoords="offset points",
            fontsize=6.5,
            color="0.18",
            zorder=5,
        )


def write_slice_plot(
    path: Path,
    *,
    rows_by_target: dict[str, tuple[VoidSliceRows, VoidSliceRows]],
    args: argparse.Namespace,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch

    targets = list(rows_by_target)
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(7.0 * len(targets), 6.5),
        squeeze=False,
        constrained_layout=True,
    )
    plane_x, plane_y = PLANE_AXES[args.slice_axis]
    colors = {"finder": "tab:blue", "vide": "tab:orange"}
    linestyles = {"finder": "-", "vide": "--"}
    for axis, target in zip(axes[0], targets, strict=True):
        finder_rows, vide_rows = rows_by_target[target]
        if args.show_nearest_matches:
            _draw_nearest_match_lines(
                axis,
                finder_rows=finder_rows,
                vide_rows=vide_rows,
                slice_axis=args.slice_axis,
                box_size_mpc_h=args.box_size,
                max_match_lines=args.max_match_lines,
            )
        for group in rows_by_target[target]:
            xs, ys = _project(group.positions_mpc_h, args.slice_axis)
            for x_value, y_value, radius in zip(xs, ys, group.radii_mpc_h, strict=True):
                for circle_x, circle_y in _periodic_circle_centers(
                    x=float(x_value),
                    y=float(y_value),
                    radius=float(radius),
                    box_size_mpc_h=args.box_size,
                ):
                    axis.add_patch(
                        Circle(
                            (circle_x, circle_y),
                            float(radius),
                            fill=False,
                            linewidth=1.1,
                            edgecolor=colors[group.method],
                            linestyle=linestyles[group.method],
                            alpha=0.72,
                        )
                    )
            axis.scatter(
                xs,
                ys,
                s=10,
                color=colors[group.method],
                alpha=0.8,
                zorder=4,
            )
            _label_largest_voids(
                axis,
                group=group,
                slice_axis=args.slice_axis,
                label_count=args.label_count,
            )
        axis.set_xlim(0.0, args.box_size)
        axis.set_ylim(0.0, args.box_size)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"{AXIS_LABELS[plane_x]} [Mpc/h]")
        axis.set_ylabel(f"{AXIS_LABELS[plane_y]} [Mpc/h]")
        finder_count = len(rows_by_target[target][0].radii_mpc_h)
        vide_count = len(rows_by_target[target][1].radii_mpc_h)
        axis.set_title(
            f"Target {target}: {args.slice_axis}-slice {args.slice_center:g} +/- {0.5 * args.slice_thickness:g} Mpc/h\n"
            f"finder {finder_count} ({args.position_mode}), VIDE {vide_count} ({args.vide_center_kind})"
        )
        legend_handles = [
            Patch(facecolor="none", edgecolor=colors["finder"], label="finder"),
            Patch(facecolor="none", edgecolor=colors["vide"], linestyle="--", label="VIDE"),
        ]
        if args.show_nearest_matches:
            legend_handles.append(Line2D([0], [0], color="0.25", linewidth=0.75, label="nearest"))
        axis.legend(handles=legend_handles, loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suffix = (
        f"{vide_catalog_variant_output_suffix(args.vide_variant)}"
        f"{pinocchio_position_mode_output_suffix(args.position_mode)}"
    )
    if suffix:
        if args.output == DEFAULT_OUTPUT:
            args.output = args.output.with_name(f"{args.output.stem}{suffix}{args.output.suffix}")
        if args.output_csv == DEFAULT_OUTPUT_CSV:
            args.output_csv = args.output_csv.with_name(
                f"{args.output_csv.stem}{suffix}{args.output_csv.suffix}"
            )
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

    rows_by_target: dict[str, tuple[VoidSliceRows, VoidSliceRows]] = {}
    all_rows: list[VoidSliceRows] = []
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
        finder_rows = _finder_rows(finder_result, target=target, args=args)
        vide_rows = _vide_rows(
            desc_path=vide_desc,
            centers_path=vide_centers,
            macrocenters_path=vide_macrocenters,
            target=target,
            args=args,
        )
        rows_by_target[target] = (finder_rows, vide_rows)
        all_rows.extend([finder_rows, vide_rows])
        print(
            f"Target {target}: finder={len(finder_rows.radii_mpc_h)} "
            f"VIDE={len(vide_rows.radii_mpc_h)} circles in slice"
        )

    write_slice_csv(args.output_csv, all_rows)
    write_slice_plot(args.output, rows_by_target=rows_by_target, args=args)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
