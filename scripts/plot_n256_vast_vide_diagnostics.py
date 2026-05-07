#!/usr/bin/env python
"""Compare VAST.VoidFinder n256 outputs against VIDE VSF and slab diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.exists():
        sys.path.insert(0, str(import_path))

from pinocchio_voids.geometry import (
    minimum_image_displacement,
    spherical_equivalent_radius_from_volume,
)
from pinocchio_voids.io import VIDE_CATALOG_VARIANTS

try:
    from scripts.plot_n256_void_slice import (
        AXIS_INDEX,
        AXIS_LABELS,
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        N256_RUN,
        PLANE_AXES,
        _periodic_circle_centers,
        load_vide_spatial_catalog,
        resolve_target_vide_paths,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from plot_n256_void_slice import (
        AXIS_INDEX,
        AXIS_LABELS,
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        N256_RUN,
        PLANE_AXES,
        _periodic_circle_centers,
        load_vide_spatial_catalog,
        resolve_target_vide_paths,
    )


VAST_RUNS = {
    "A": {"name": "n256"},
    "B": {"name": "n256_paired"},
}
DEFAULT_OUTPUT_DIR = Path("runs/void-statistics")
DEFAULT_VAST_ROOT = Path("runs/vast-voidfinder")


@dataclass(frozen=True)
class VastCatalog:
    """Normalized VAST void catalog for one target."""

    target: str
    positions_mpc_h: NDArray[np.float64]
    maximal_radii_mpc_h: NDArray[np.float64]
    reff_radii_mpc_h: NDArray[np.float64]
    void_ids: NDArray[np.int64]
    holes_by_void_id: Mapping[int, tuple[NDArray[np.float64], NDArray[np.float64]]]


@dataclass(frozen=True)
class SliceRows:
    """Selected void rows for a projected slab."""

    method: str
    target: str
    positions_mpc_h: NDArray[np.float64]
    radii_mpc_h: NDArray[np.float64]
    display_radii_mpc_h: NDArray[np.float64]
    void_ids: NDArray[np.int64]
    distance_to_slice_mpc_h: NDArray[np.float64]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot n256 VAST.VoidFinder vs VIDE void size functions and slabs."
    )
    parser.add_argument("--vast-root", type=Path, default=DEFAULT_VAST_ROOT)
    parser.add_argument(
        "--vast-run-suffix",
        default="",
        help="Optional suffix appended to VAST run directories, e.g. _wall or _rmin15.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--vide-variant",
        choices=VIDE_CATALOG_VARIANTS,
        default="all",
        help="VIDE variant used for the reference catalog.",
    )
    parser.add_argument(
        "--radius-mode",
        choices=("maximal", "reff", "both"),
        default="both",
        help="VAST radius definition to compare against VIDE.",
    )
    parser.add_argument("--target", choices=("A", "B", "both"), default="both")
    parser.add_argument("--box-size", type=float, default=256.0)
    parser.add_argument("--bins", type=int, default=17)
    parser.add_argument("--bin-min", type=float, default=10.0)
    parser.add_argument("--bin-max", type=float, default=80.0)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Voxel size in Mpc/h for deterministic VAST hole-union Reff estimates.",
    )
    parser.add_argument(
        "--max-voxels-per-void",
        type=int,
        default=3_000_000,
        help="Switch to deterministic Monte Carlo if one void bounding box exceeds this many voxels.",
    )
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-center", type=float, default=128.0)
    parser.add_argument("--slice-thickness", type=float, default=20.0)
    return parser.parse_args(argv)


def requested_targets(args: argparse.Namespace) -> tuple[str, ...]:
    return ("A", "B") if args.target == "both" else (args.target,)


def requested_radius_modes(args: argparse.Namespace) -> tuple[str, ...]:
    return ("maximal", "reff") if args.radius_mode == "both" else (args.radius_mode,)


def vast_run_name(target: str, suffix: str = "") -> str:
    return f"{VAST_RUNS[target]['name']}{suffix}"


def output_run_token(suffix: str) -> str:
    cleaned = str(suffix).strip("_")
    return f"{cleaned}_" if cleaned else ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"Cannot read required VAST CSV: {path}") from exc


def _as_float(row: Mapping[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Invalid or missing {name!r} in VAST CSV row: {row}") from exc


def _as_int(row: Mapping[str, str], name: str) -> int:
    try:
        return int(float(row[name]))
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Invalid or missing {name!r} in VAST CSV row: {row}") from exc


def estimate_periodic_union_volume(
    centers_mpc_h: NDArray[np.float64],
    radii_mpc_h: NDArray[np.float64],
    *,
    box_size_mpc_h: float,
    voxel_size_mpc_h: float,
    max_voxels: int = 3_000_000,
) -> float:
    """Estimate the union volume of periodic spheres with a deterministic lattice."""

    centers = np.asarray(centers_mpc_h, dtype=np.float64).reshape((-1, 3))
    radii = np.asarray(radii_mpc_h, dtype=np.float64).reshape((-1,))
    if len(centers) != len(radii):
        raise ValueError("centers and radii must have matching lengths")
    if len(radii) == 0:
        return 0.0
    if voxel_size_mpc_h <= 0.0 or not np.isfinite(voxel_size_mpc_h):
        raise ValueError("voxel_size_mpc_h must be positive and finite")
    if box_size_mpc_h <= 0.0 or not np.isfinite(box_size_mpc_h):
        raise ValueError("box_size_mpc_h must be positive and finite")
    if np.any(radii <= 0.0) or not np.all(np.isfinite(radii)):
        raise ValueError("radii_mpc_h must be positive and finite")

    reference = centers[0]
    unwrapped = reference + minimum_image_displacement(reference, centers, box_size_mpc_h)
    lower = np.min(unwrapped - radii[:, np.newaxis], axis=0)
    upper = np.max(unwrapped + radii[:, np.newaxis], axis=0)
    starts = np.floor(lower / voxel_size_mpc_h) * voxel_size_mpc_h + 0.5 * voxel_size_mpc_h
    stops = np.ceil(upper / voxel_size_mpc_h) * voxel_size_mpc_h
    axes = [np.arange(starts[i], stops[i], voxel_size_mpc_h, dtype=np.float64) for i in range(3)]
    shape = tuple(len(axis) for axis in axes)
    voxel_count = int(np.prod(shape, dtype=np.int64))
    if voxel_count == 0:
        return 0.0

    if voxel_count > max_voxels:
        rng = np.random.default_rng(12345)
        samples = rng.uniform(lower, upper, size=(max_voxels, 3))
        inside = _points_inside_any_sphere(samples, unwrapped, radii)
        return float(np.prod(upper - lower) * np.mean(inside))

    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    inside = _points_inside_any_sphere(points, unwrapped, radii)
    return float(np.count_nonzero(inside) * voxel_size_mpc_h**3)


def _points_inside_any_sphere(
    points_mpc_h: NDArray[np.float64],
    centers_mpc_h: NDArray[np.float64],
    radii_mpc_h: NDArray[np.float64],
) -> NDArray[np.bool_]:
    inside = np.zeros(points_mpc_h.shape[0], dtype=np.bool_)
    for center, radius in zip(centers_mpc_h, radii_mpc_h, strict=True):
        offsets = points_mpc_h - center
        inside |= np.einsum("ij,ij->i", offsets, offsets) <= radius**2
    return inside


def _holes_by_void(
    rows: Sequence[Mapping[str, str]]
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    grouped_positions: dict[int, list[list[float]]] = {}
    grouped_radii: dict[int, list[float]] = {}
    for row in rows:
        void_id = _as_int(row, "void_id")
        if void_id < 0:
            continue
        grouped_positions.setdefault(void_id, []).append(
            [_as_float(row, "x_mpc_h"), _as_float(row, "y_mpc_h"), _as_float(row, "z_mpc_h")]
        )
        grouped_radii.setdefault(void_id, []).append(_as_float(row, "radius_mpc_h"))
    return {
        void_id: (
            np.asarray(grouped_positions[void_id], dtype=np.float64),
            np.asarray(grouped_radii[void_id], dtype=np.float64),
        )
        for void_id in grouped_positions
    }


def read_vast_catalog(
    root: Path,
    *,
    target: str,
    run_suffix: str = "",
    box_size_mpc_h: float,
    voxel_size_mpc_h: float,
    max_voxels_per_void: int,
    compute_reff: bool = True,
) -> VastCatalog:
    """Read normalized VAST maximal and holes CSV files."""

    run_dir = root / vast_run_name(target, suffix=run_suffix)
    maximal_rows = _read_csv(run_dir / "vast_voids_maximal.csv")
    hole_rows = _read_csv(run_dir / "vast_voids_holes.csv")
    holes_by_void = _holes_by_void(hole_rows)

    void_ids = np.asarray([_as_int(row, "void_id") for row in maximal_rows], dtype=np.int64)
    positions = np.asarray(
        [
            [_as_float(row, "x_mpc_h"), _as_float(row, "y_mpc_h"), _as_float(row, "z_mpc_h")]
            for row in maximal_rows
        ],
        dtype=np.float64,
    )
    maximal_radii = np.asarray(
        [_as_float(row, "maximal_radius_mpc_h") for row in maximal_rows],
        dtype=np.float64,
    )
    if compute_reff:
        reff_radii = np.empty_like(maximal_radii)
        for index, (void_id, maximal_radius) in enumerate(zip(void_ids, maximal_radii, strict=True)):
            holes = holes_by_void.get(int(void_id))
            if holes is None:
                volume = 4.0 * np.pi * maximal_radius**3 / 3.0
            else:
                hole_positions, hole_radii = holes
                volume = estimate_periodic_union_volume(
                    hole_positions,
                    hole_radii,
                    box_size_mpc_h=box_size_mpc_h,
                    voxel_size_mpc_h=voxel_size_mpc_h,
                    max_voxels=max_voxels_per_void,
                )
            reff_radii[index] = float(spherical_equivalent_radius_from_volume(volume))
    else:
        reff_radii = np.full_like(maximal_radii, np.nan)

    return VastCatalog(
        target=target,
        positions_mpc_h=positions,
        maximal_radii_mpc_h=maximal_radii,
        reff_radii_mpc_h=reff_radii,
        void_ids=void_ids,
        holes_by_void_id=holes_by_void,
    )


def vast_radii(catalog: VastCatalog, radius_mode: str) -> NDArray[np.float64]:
    if radius_mode == "maximal":
        return catalog.maximal_radii_mpc_h
    if radius_mode == "reff":
        return catalog.reff_radii_mpc_h
    raise ValueError(f"Unknown radius mode: {radius_mode}")


def _target_vide_paths(target: str) -> tuple[Path, Path, Path]:
    if target == "A":
        return Path(N256_RUN["vide_a"]), DEFAULT_VIDE_CENTERS_A, DEFAULT_VIDE_MACROCENTERS_A
    return Path(N256_RUN["vide_b"]), DEFAULT_VIDE_CENTERS_B, DEFAULT_VIDE_MACROCENTERS_B


def read_vide_spatial(target: str, *, variant: str):
    desc, centers, macrocenters = _target_vide_paths(target)
    resolved_desc, resolved_centers, resolved_macrocenters = resolve_target_vide_paths(
        desc_path=desc,
        centers_path=centers,
        macrocenters_path=macrocenters,
        variant=variant,
    )
    return load_vide_spatial_catalog(
        desc_path=resolved_desc,
        centers_path=resolved_centers,
        macrocenters_path=resolved_macrocenters,
        center_kind="center",
    )


def radius_summary(label: str, source: str, target: str, radii: NDArray[np.float64]) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "label": label,
        "source": source,
        "target": target,
        "count": int(len(radii)),
        "min_mpc_h": "",
        "p10_mpc_h": "",
        "median_mpc_h": "",
        "p90_mpc_h": "",
        "max_mpc_h": "",
        "count_10_80_mpc_h": int(np.count_nonzero((radii >= 10.0) & (radii <= 80.0))),
    }
    if len(radii):
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


def write_vsf_products(
    *,
    catalogs: Mapping[str, VastCatalog],
    radius_mode: str,
    targets: Sequence[str],
    args: argparse.Namespace,
) -> None:
    edges = np.linspace(float(args.bin_min), float(args.bin_max), int(args.bins) + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dlnr = np.diff(np.log(edges))
    volume = float(args.box_size) ** 3
    label = f"vast-{radius_mode}"
    output_stem = (
        args.output_dir
        / f"n256_vast_{output_run_token(args.vast_run_suffix)}vide_{args.vide_variant}_vsf_{radius_mode}"
    )
    rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for target in targets:
        vast = catalogs[target]
        vast_values = vast_radii(vast, radius_mode)
        vide = read_vide_spatial(target, variant=args.vide_variant)
        for source, radii in (("vast", vast_values), ("vide", vide.radii_mpc_h)):
            counts, _ = np.histogram(radii, bins=edges)
            density = counts / (volume * dlnr)
            for r_min, r_max, r_mid, count, dn_dlnr_dv in zip(
                edges[:-1], edges[1:], centers, counts, density, strict=True
            ):
                rows.append(
                    {
                        "label": label,
                        "source": source,
                        "target": target,
                        "bin_min_mpc_h": float(r_min),
                        "bin_max_mpc_h": float(r_max),
                        "bin_center_mpc_h": float(r_mid),
                        "count": int(count),
                        "density_dndlnr_per_mpc_h3": float(dn_dlnr_dv),
                    }
                )
            summary_rows.append(radius_summary(label, source, target, radii))

    write_dicts(output_stem.with_suffix(".csv"), rows)
    write_dicts(output_stem.with_name(output_stem.name + "_summary").with_suffix(".csv"), summary_rows)
    write_vsf_plot(output_stem.with_suffix(".png"), rows, targets=targets, radius_mode=radius_mode)
    print(f"Wrote {output_stem.with_suffix('.csv')}")
    print(f"Wrote {output_stem.with_suffix('.png')}")


def write_dicts(path: Path, rows: Sequence[Mapping[str, float | int | str]]) -> None:
    if not rows:
        raise SystemExit(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_vsf_plot(
    path: Path,
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    targets: Sequence[str],
    radius_mode: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(6.2 * len(targets), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"vast": "tab:green", "vide": "tab:orange"}
    for axis, target in zip(axes[0], targets, strict=True):
        for source in ("vast", "vide"):
            selected = [row for row in rows if row["target"] == target and row["source"] == source]
            x = [float(row["bin_center_mpc_h"]) for row in selected]
            y = [float(row["density_dndlnr_per_mpc_h3"]) for row in selected]
            counts = [int(row["count"]) for row in selected]
            axis.step(
                x,
                y,
                where="mid",
                marker="o",
                color=colors[source],
                label=f"{source} ({sum(counts)})",
            )
        axis.set_yscale("log")
        axis.set_xlabel(r"$R$ [$h^{-1}\,\mathrm{Mpc}$]")
        axis.set_ylabel(r"$dN/d\ln R/V$ [$(h^{-1}\,\mathrm{Mpc})^{-3}$]")
        axis.set_title(f"Target {target}: VAST {radius_mode} vs VIDE")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def periodic_axis_distance(
    values_mpc_h: NDArray[np.float64],
    *,
    center_mpc_h: float,
    box_size_mpc_h: float,
) -> NDArray[np.float64]:
    delta = (values_mpc_h - center_mpc_h + 0.5 * box_size_mpc_h) % box_size_mpc_h
    return np.abs(delta - 0.5 * box_size_mpc_h)


def select_slice_rows(
    *,
    method: str,
    target: str,
    positions_mpc_h: NDArray[np.float64],
    radii_mpc_h: NDArray[np.float64],
    void_ids: NDArray[np.int64],
    args: argparse.Namespace,
) -> SliceRows:
    distances = periodic_axis_distance(
        positions_mpc_h[:, AXIS_INDEX[args.slice_axis]],
        center_mpc_h=args.slice_center,
        box_size_mpc_h=args.box_size,
    )
    half_thickness = 0.5 * args.slice_thickness
    mask = distances <= half_thickness + radii_mpc_h
    selected_distances = distances[mask]
    outside = np.maximum(0.0, selected_distances - half_thickness)
    display_radii = np.sqrt(np.maximum(0.0, radii_mpc_h[mask] ** 2 - outside**2))
    return SliceRows(
        method=method,
        target=target,
        positions_mpc_h=positions_mpc_h[mask],
        radii_mpc_h=radii_mpc_h[mask],
        display_radii_mpc_h=display_radii,
        void_ids=void_ids[mask],
        distance_to_slice_mpc_h=selected_distances,
    )


def write_slice_products(
    *,
    catalogs: Mapping[str, VastCatalog],
    radius_mode: str,
    targets: Sequence[str],
    args: argparse.Namespace,
) -> None:
    rows_by_target: dict[str, tuple[SliceRows, SliceRows]] = {}
    csv_rows: list[dict[str, float | int | str]] = []
    for target in targets:
        vast = catalogs[target]
        vide = read_vide_spatial(target, variant=args.vide_variant)
        vast_rows = select_slice_rows(
            method="vast",
            target=target,
            positions_mpc_h=vast.positions_mpc_h,
            radii_mpc_h=vast_radii(vast, radius_mode),
            void_ids=vast.void_ids,
            args=args,
        )
        vide_rows = select_slice_rows(
            method="vide",
            target=target,
            positions_mpc_h=vide.positions_mpc_h,
            radii_mpc_h=vide.radii_mpc_h,
            void_ids=vide.void_ids,
            args=args,
        )
        rows_by_target[target] = (vast_rows, vide_rows)
        csv_rows.extend(slice_csv_rows(vast_rows))
        csv_rows.extend(slice_csv_rows(vide_rows))
        print(
            f"Target {target} slab {radius_mode}: "
            f"VAST={len(vast_rows.radii_mpc_h)} VIDE={len(vide_rows.radii_mpc_h)}"
        )

    output_stem = (
        args.output_dir
        / f"n256_vast_{output_run_token(args.vast_run_suffix)}vide_{args.vide_variant}_void_slice_{radius_mode}"
    )
    write_dicts(output_stem.with_suffix(".csv"), csv_rows)
    write_slice_plot(output_stem.with_suffix(".png"), rows_by_target, args=args, radius_mode=radius_mode)
    print(f"Wrote {output_stem.with_suffix('.csv')}")
    print(f"Wrote {output_stem.with_suffix('.png')}")


def slice_csv_rows(rows: SliceRows) -> list[dict[str, float | int | str]]:
    output = []
    for position, radius, display_radius, void_id, distance in zip(
        rows.positions_mpc_h,
        rows.radii_mpc_h,
        rows.display_radii_mpc_h,
        rows.void_ids,
        rows.distance_to_slice_mpc_h,
        strict=True,
    ):
        output.append(
            {
                "method": rows.method,
                "target": rows.target,
                "void_id": int(void_id),
                "x_mpc_h": float(position[0]),
                "y_mpc_h": float(position[1]),
                "z_mpc_h": float(position[2]),
                "radius_mpc_h": float(radius),
                "display_radius_mpc_h": float(display_radius),
                "distance_to_slice_mpc_h": float(distance),
            }
        )
    return output


def _project(positions_mpc_h: NDArray[np.float64], axis: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    plane_x, plane_y = PLANE_AXES[axis]
    return positions_mpc_h[:, plane_x], positions_mpc_h[:, plane_y]


def write_slice_plot(
    path: Path,
    rows_by_target: Mapping[str, tuple[SliceRows, SliceRows]],
    *,
    args: argparse.Namespace,
    radius_mode: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch

    targets = list(rows_by_target)
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(7.0 * len(targets), 6.5),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"vast": "tab:green", "vide": "tab:orange"}
    linestyles = {"vast": "-", "vide": "--"}
    plane_x, plane_y = PLANE_AXES[args.slice_axis]
    for axis, target in zip(axes[0], targets, strict=True):
        for group in rows_by_target[target]:
            xs, ys = _project(group.positions_mpc_h, args.slice_axis)
            for x_value, y_value, radius in zip(xs, ys, group.display_radii_mpc_h, strict=True):
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
                            alpha=0.75,
                        )
                    )
            axis.scatter(xs, ys, s=12, color=colors[group.method], alpha=0.85, zorder=3)
        vast_count = len(rows_by_target[target][0].radii_mpc_h)
        vide_count = len(rows_by_target[target][1].radii_mpc_h)
        axis.set_xlim(0.0, args.box_size)
        axis.set_ylim(0.0, args.box_size)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"{AXIS_LABELS[plane_x]} [Mpc/h]")
        axis.set_ylabel(f"{AXIS_LABELS[plane_y]} [Mpc/h]")
        axis.set_title(
            f"Target {target}: VAST {radius_mode} vs VIDE {args.vide_variant}\n"
            f"{args.slice_axis}={args.slice_center:g} +/- {0.5 * args.slice_thickness:g}, "
            f"VAST {vast_count}, VIDE {vide_count}"
        )
        axis.legend(
            handles=[
                Patch(facecolor="none", edgecolor=colors["vast"], label="VAST"),
                Patch(facecolor="none", edgecolor=colors["vide"], linestyle="--", label="VIDE"),
            ],
            loc="upper right",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    targets = requested_targets(args)
    radius_modes = requested_radius_modes(args)
    catalogs = {
        target: read_vast_catalog(
            args.vast_root,
            target=target,
            run_suffix=str(args.vast_run_suffix),
            box_size_mpc_h=float(args.box_size),
            voxel_size_mpc_h=float(args.voxel_size),
            max_voxels_per_void=int(args.max_voxels_per_void),
            compute_reff="reff" in radius_modes,
        )
        for target in targets
    }
    for radius_mode in radius_modes:
        write_vsf_products(catalogs=catalogs, radius_mode=radius_mode, targets=targets, args=args)
        write_slice_products(catalogs=catalogs, radius_mode=radius_mode, targets=targets, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
