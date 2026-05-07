#!/usr/bin/env python
"""Plot n256 halo slice scatter with finder or VIDE void circles."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
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
    read_vide_input_tracers,
    read_vide_void_zones,
    read_vide_zone_particles,
    vide_catalog_variant_output_suffix,
    vide_particle_ids_for_file_void_id,
)
from pinocchio_voids.voidfinder import PairedVoidFinderConfig, run_paired_halo_void_finder

try:
    from scripts.plot_n256_void_slice import (
        AXIS_INDEX,
        AXIS_LABELS,
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_INPUT_A,
        DEFAULT_VIDE_INPUT_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        DEFAULT_VIDE_VOID_ZONES_A,
        DEFAULT_VIDE_VOID_ZONES_B,
        DEFAULT_VIDE_ZONE_PARTICLES_A,
        DEFAULT_VIDE_ZONE_PARTICLES_B,
        N256_RUN,
        PLANE_AXES,
        VoidSliceRows,
        _periodic_circle_centers,
        _project,
        add_full_algorithm_arguments,
        full_algorithm_kwargs_from_args,
        load_vide_spatial_catalog,
        periodic_axis_distance,
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
        DEFAULT_VIDE_INPUT_A,
        DEFAULT_VIDE_INPUT_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        DEFAULT_VIDE_VOID_ZONES_A,
        DEFAULT_VIDE_VOID_ZONES_B,
        DEFAULT_VIDE_ZONE_PARTICLES_A,
        DEFAULT_VIDE_ZONE_PARTICLES_B,
        N256_RUN,
        PLANE_AXES,
        VoidSliceRows,
        _periodic_circle_centers,
        _project,
        add_full_algorithm_arguments,
        full_algorithm_kwargs_from_args,
        load_vide_spatial_catalog,
        periodic_axis_distance,
        resolve_target_vide_paths,
    )


DEFAULT_CALIBRATION_SUMMARY = Path("runs/void-statistics/n256_joint_mcmc_summary.csv")
DEFAULT_OUTPUT_PREFIX = "n256_halo_slice"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target halos in a 2D n256 slice with finder and VIDE void circles."
    )
    parser.add_argument("--catalog-a", type=Path, default=Path(N256_RUN["catalog_a"]))
    parser.add_argument("--catalog-b", type=Path, default=Path(N256_RUN["catalog_b"]))
    parser.add_argument("--vide-desc-a", type=Path, default=Path(N256_RUN["vide_a"]))
    parser.add_argument("--vide-desc-b", type=Path, default=Path(N256_RUN["vide_b"]))
    parser.add_argument("--vide-centers-a", type=Path, default=DEFAULT_VIDE_CENTERS_A)
    parser.add_argument("--vide-centers-b", type=Path, default=DEFAULT_VIDE_CENTERS_B)
    parser.add_argument("--vide-macrocenters-a", type=Path, default=DEFAULT_VIDE_MACROCENTERS_A)
    parser.add_argument("--vide-macrocenters-b", type=Path, default=DEFAULT_VIDE_MACROCENTERS_B)
    parser.add_argument("--vide-input-a", type=Path, default=DEFAULT_VIDE_INPUT_A)
    parser.add_argument("--vide-input-b", type=Path, default=DEFAULT_VIDE_INPUT_B)
    parser.add_argument("--vide-void-zones-a", type=Path, default=DEFAULT_VIDE_VOID_ZONES_A)
    parser.add_argument("--vide-void-zones-b", type=Path, default=DEFAULT_VIDE_VOID_ZONES_B)
    parser.add_argument("--vide-zone-particles-a", type=Path, default=DEFAULT_VIDE_ZONE_PARTICLES_A)
    parser.add_argument("--vide-zone-particles-b", type=Path, default=DEFAULT_VIDE_ZONE_PARTICLES_B)
    parser.add_argument(
        "--vide-center-kind",
        choices=("center", "macrocenter"),
        default="center",
        help="VIDE position convention to plot.",
    )
    parser.add_argument(
        "--vide-variant",
        choices=VIDE_CATALOG_VARIANTS,
        default="all",
        help="VIDE catalog variant used for voidDesc, centers, and macrocenters.",
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
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-center", type=float, default=128.0)
    parser.add_argument("--slice-thickness", type=float, default=20.0)
    parser.add_argument(
        "--void-selection",
        choices=("intersections", "centers"),
        default="intersections",
        help="Select voids by slab intersection or by center-in-slab membership.",
    )
    parser.add_argument(
        "--circle-radius-mode",
        choices=("cross-section", "reff"),
        default="cross-section",
        help="Draw slice cross-section radii or full Reff circles.",
    )
    parser.add_argument(
        "--vide-overlay",
        choices=("spheres", "members", "both"),
        default="spheres",
        help="Draw VIDE as Reff sphere cross-sections, member tracers, or both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/void-statistics"),
        help="Directory for generated halo-slice PNGs.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Filename prefix for generated halo-slice PNGs.",
    )
    return parser.parse_args(argv)


def read_calibration_best_fit(path: Path) -> dict[str, float]:
    """Read parameter best-fit values from a joint MCMC summary CSV."""

    if not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"Cannot read calibration summary: {path}") from exc
    values: dict[str, float] = {}
    for row in rows:
        parameter = row.get("parameter", "")
        best_fit = row.get("best_fit", "")
        if not parameter or not best_fit:
            continue
        try:
            values[parameter] = float(best_fit)
        except ValueError as exc:
            raise SystemExit(f"Invalid best_fit value in {path}: {best_fit}") from exc
    return values


def resolve_finder_parameters(args: argparse.Namespace) -> dict[str, float]:
    """Resolve finder parameters from CLI overrides, summary CSV, then run defaults."""

    summary = read_calibration_best_fit(args.calibration_summary)
    defaults = {
        "linking_factor": float(N256_RUN["linking_factor"]),
        "radius_a0": float(N256_RUN["radius_a0"]),
        "radius_alpha": float(N256_RUN["radius_alpha"]),
        "adjacency_factor": float(N256_RUN["adjacency_factor"]),
    }
    values = defaults | {
        name: summary[name]
        for name in defaults
        if name in summary
    }
    if args.linking_factor is not None:
        values["linking_factor"] = float(args.linking_factor)
    if args.radius_a0 is not None:
        values["radius_a0"] = float(args.radius_a0)
    if args.radius_alpha is not None:
        values["radius_alpha"] = float(args.radius_alpha)
    if args.adjacency_factor is not None:
        values["adjacency_factor"] = float(args.adjacency_factor)
    return values


def halo_slice_mask(
    *,
    positions_mpc_h: NDArray[np.float64],
    axis: str,
    center_mpc_h: float,
    thickness_mpc_h: float,
    box_size_mpc_h: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Select halo centers inside a periodic slab."""

    if thickness_mpc_h <= 0.0 or not np.isfinite(thickness_mpc_h):
        raise ValueError("slice thickness must be positive and finite")
    distances = periodic_axis_distance(
        positions_mpc_h[:, AXIS_INDEX[axis]],
        center_mpc_h=center_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    return distances <= 0.5 * thickness_mpc_h, distances


def void_slice_mask(
    *,
    positions_mpc_h: NDArray[np.float64],
    radii_mpc_h: NDArray[np.float64],
    axis: str,
    center_mpc_h: float,
    thickness_mpc_h: float,
    box_size_mpc_h: float,
    selection: str,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Select voids for a slab using center-only or sphere-intersection logic."""

    if thickness_mpc_h <= 0.0 or not np.isfinite(thickness_mpc_h):
        raise ValueError("slice thickness must be positive and finite")
    distances = periodic_axis_distance(
        positions_mpc_h[:, AXIS_INDEX[axis]],
        center_mpc_h=center_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    half_thickness = 0.5 * thickness_mpc_h
    if selection == "centers":
        return distances <= half_thickness, distances
    if selection == "intersections":
        return distances <= half_thickness + radii_mpc_h, distances
    raise ValueError(f"Unknown void selection mode: {selection}")


def display_radii_for_slice(
    *,
    radii_mpc_h: NDArray[np.float64],
    distance_to_slice_mpc_h: NDArray[np.float64],
    thickness_mpc_h: float,
    radius_mode: str,
) -> NDArray[np.float64]:
    """Return the circle radius to draw for each selected 3D void sphere."""

    radii = np.asarray(radii_mpc_h, dtype=np.float64)
    distances = np.asarray(distance_to_slice_mpc_h, dtype=np.float64)
    if radius_mode == "reff":
        return radii.copy()
    if radius_mode != "cross-section":
        raise ValueError(f"Unknown circle radius mode: {radius_mode}")
    if thickness_mpc_h <= 0.0 or not np.isfinite(thickness_mpc_h):
        raise ValueError("slice thickness must be positive and finite")
    outside_slab_distance = np.maximum(0.0, distances - 0.5 * thickness_mpc_h)
    display_radii = np.sqrt(np.maximum(0.0, radii**2 - outside_slab_distance**2))
    return display_radii


def select_void_rows_for_halo_slice(
    rows: VoidSliceRows,
    *,
    args: argparse.Namespace,
) -> tuple[VoidSliceRows, NDArray[np.float64]]:
    """Select void rows and display radii for the halo-background plot."""

    mask, distances = void_slice_mask(
        positions_mpc_h=rows.positions_mpc_h,
        radii_mpc_h=rows.radii_mpc_h,
        axis=args.slice_axis,
        center_mpc_h=args.slice_center,
        thickness_mpc_h=args.slice_thickness,
        box_size_mpc_h=args.box_size,
        selection=args.void_selection,
    )
    selected = VoidSliceRows(
        method=rows.method,
        target=rows.target,
        position_mode=rows.position_mode,
        positions_mpc_h=rows.positions_mpc_h[mask],
        radii_mpc_h=rows.radii_mpc_h[mask],
        void_ids=rows.void_ids[mask],
        file_void_ids=rows.file_void_ids[mask],
        distance_to_slice_mpc_h=distances[mask],
        center_kind=rows.center_kind,
    )
    display_radii = display_radii_for_slice(
        radii_mpc_h=selected.radii_mpc_h,
        distance_to_slice_mpc_h=selected.distance_to_slice_mpc_h,
        thickness_mpc_h=args.slice_thickness,
        radius_mode=args.circle_radius_mode,
    )
    return selected, display_radii


def _target_paths(args: argparse.Namespace, target: str) -> tuple[Path, Path, Path]:
    if target == "A":
        return args.vide_desc_a, args.vide_centers_a, args.vide_macrocenters_a
    return args.vide_desc_b, args.vide_centers_b, args.vide_macrocenters_b


def _target_member_paths(args: argparse.Namespace, target: str) -> tuple[Path, Path, Path]:
    if target == "A":
        return args.vide_input_a, args.vide_void_zones_a, args.vide_zone_particles_a
    return args.vide_input_b, args.vide_void_zones_b, args.vide_zone_particles_b


def vide_member_positions_for_rows(
    *,
    rows: VoidSliceRows,
    input_path: Path,
    void_zones_path: Path,
    zone_particles_path: Path,
    args: argparse.Namespace,
) -> NDArray[np.float64]:
    """Return VIDE member tracer positions for selected ``FileVoid#`` rows."""

    if len(rows.file_void_ids) == 0:
        return np.empty((0, 3), dtype=np.float64)
    tracers = read_vide_input_tracers(input_path)
    void_zones = read_vide_void_zones(void_zones_path)
    zone_particles = read_vide_zone_particles(zone_particles_path)
    particle_chunks: list[NDArray[np.int64]] = []
    for file_void_id in rows.file_void_ids:
        if int(file_void_id) < 0:
            continue
        particle_chunks.append(
            vide_particle_ids_for_file_void_id(
                file_void_id=int(file_void_id),
                void_zones=void_zones,
                zone_particles=zone_particles,
            )
        )
    if not particle_chunks:
        return np.empty((0, 3), dtype=np.float64)
    particle_ids = np.unique(np.concatenate(particle_chunks).astype(np.int64, copy=False))
    if len(particle_ids) and int(np.max(particle_ids)) >= len(tracers):
        raise SystemExit(
            f"VIDE member particle ID exceeds tracer count: {zone_particles_path}"
        )
    positions = tracers.positions_mpc_h[particle_ids]
    mask, _ = halo_slice_mask(
        positions_mpc_h=positions,
        axis=args.slice_axis,
        center_mpc_h=args.slice_center,
        thickness_mpc_h=args.slice_thickness,
        box_size_mpc_h=args.box_size,
    )
    return positions[mask]


def _all_finder_rows(result, *, target: str, position_mode: str) -> VoidSliceRows:
    positions = np.asarray(
        [void.center_mpc_h for void in result.voids],
        dtype=np.float64,
    ).reshape((-1, 3))
    radii = np.asarray([void.effective_radius_mpc_h for void in result.voids], dtype=np.float64)
    void_ids = np.asarray([void.id for void in result.voids], dtype=np.int64)
    return VoidSliceRows(
        method="finder",
        target=target,
        position_mode=position_mode,
        positions_mpc_h=positions,
        radii_mpc_h=radii,
        void_ids=void_ids,
        file_void_ids=np.full(void_ids.shape, -1, dtype=np.int64),
        distance_to_slice_mpc_h=np.zeros(void_ids.shape, dtype=np.float64),
    )


def _all_vide_rows(
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
    return VoidSliceRows(
        method="vide",
        target=target,
        position_mode=args.position_mode,
        positions_mpc_h=spatial.positions_mpc_h,
        radii_mpc_h=spatial.radii_mpc_h,
        void_ids=spatial.void_ids,
        file_void_ids=spatial.file_void_ids,
        distance_to_slice_mpc_h=np.zeros(spatial.radii_mpc_h.shape, dtype=np.float64),
        center_kind=spatial.center_kind,
    )


def _output_path(args: argparse.Namespace, *, target: str, method: str) -> Path:
    if args.target == "both":
        return args.output_dir / f"{args.output_prefix}_target_{target.lower()}_{method}.png"
    return args.output_dir / f"{args.output_prefix}_{method}.png"


def _csv_output_path(args: argparse.Namespace, *, target: str, method: str) -> Path:
    return _output_path(args, target=target, method=method).with_suffix(".csv")


def write_void_csv(
    path: Path,
    *,
    rows: VoidSliceRows,
    display_radii_mpc_h: NDArray[np.float64],
) -> None:
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
                "display_radius_mpc_h",
                "distance_to_slice_mpc_h",
                "center_kind",
            ],
        )
        writer.writeheader()
        for position, radius, display_radius, void_id, file_void_id, distance in zip(
            rows.positions_mpc_h,
            rows.radii_mpc_h,
            display_radii_mpc_h,
            rows.void_ids,
            rows.file_void_ids,
            rows.distance_to_slice_mpc_h,
            strict=True,
        ):
            writer.writerow(
                {
                    "method": rows.method,
                    "target": rows.target,
                    "position_mode": rows.position_mode,
                    "void_id": int(void_id),
                    "file_void_id": "" if int(file_void_id) < 0 else int(file_void_id),
                    "x_mpc_h": float(position[0]),
                    "y_mpc_h": float(position[1]),
                    "z_mpc_h": float(position[2]),
                    "reff_mpc_h": float(radius),
                    "display_radius_mpc_h": float(display_radius),
                    "distance_to_slice_mpc_h": float(distance),
                    "center_kind": rows.center_kind,
                }
            )


def write_halo_void_plot(
    path: Path,
    *,
    halo_positions_mpc_h: NDArray[np.float64],
    void_rows: VoidSliceRows,
    display_radii_mpc_h: NDArray[np.float64],
    vide_member_positions_mpc_h: NDArray[np.float64] | None = None,
    target: str,
    method: str,
    args: argparse.Namespace,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch

    plane_x, plane_y = PLANE_AXES[args.slice_axis]
    fig, axis = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)
    halo_x, halo_y = _project(halo_positions_mpc_h, args.slice_axis)
    axis.scatter(
        halo_x,
        halo_y,
        s=3.0,
        color="black",
        alpha=0.22,
        linewidths=0.0,
        label="halos",
        zorder=1,
    )
    color = "tab:blue" if method == "finder" else "tab:orange"
    linestyle = "-" if method == "finder" else "--"
    void_x, void_y = _project(void_rows.positions_mpc_h, args.slice_axis)
    draw_circles = method != "vide" or args.vide_overlay in ("spheres", "both")
    if draw_circles:
        for x_value, y_value, radius in zip(void_x, void_y, display_radii_mpc_h, strict=True):
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
                        edgecolor=color,
                        linestyle=linestyle,
                        alpha=0.8,
                        zorder=2,
                    )
                )
    if (
        method == "vide"
        and args.vide_overlay in ("members", "both")
        and vide_member_positions_mpc_h is not None
        and len(vide_member_positions_mpc_h)
    ):
        member_x, member_y = _project(vide_member_positions_mpc_h, args.slice_axis)
        axis.scatter(
            member_x,
            member_y,
            s=9.0,
            color=color,
            alpha=0.55,
            linewidths=0.0,
            label="VIDE member tracers",
            zorder=2,
        )
    axis.scatter(void_x, void_y, s=13, color=color, alpha=0.9, zorder=3)
    axis.set_xlim(0.0, args.box_size)
    axis.set_ylim(0.0, args.box_size)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"{AXIS_LABELS[plane_x]} [Mpc/h]")
    axis.set_ylabel(f"{AXIS_LABELS[plane_y]} [Mpc/h]")
    axis.set_title(
        f"Target {target}: {method} voids over halo {args.slice_axis}-slice\n"
        f"{args.slice_center:g} +/- {0.5 * args.slice_thickness:g} Mpc/h, "
        f"halos {len(halo_positions_mpc_h)} ({args.position_mode}), "
        f"voids {len(void_rows.radii_mpc_h)}"
    )
    handles = [Patch(facecolor="none", edgecolor="black", label="halos")]
    if draw_circles:
        handles.append(Patch(facecolor="none", edgecolor=color, linestyle=linestyle, label=method))
    if method == "vide" and args.vide_overlay in ("members", "both"):
        handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.55, label="VIDE members"))
    axis.legend(handles=handles, loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suffix = (
        f"{vide_catalog_variant_output_suffix(args.vide_variant)}"
        f"{pinocchio_position_mode_output_suffix(args.position_mode)}"
    )
    if suffix and args.output_prefix == DEFAULT_OUTPUT_PREFIX:
        args.output_prefix = f"{args.output_prefix}{suffix}"
    summary = read_calibration_best_fit(args.calibration_summary)
    parameters = resolve_finder_parameters(args)
    paired = read_paired_pinocchio_halo_catalogs(
        args.catalog_a,
        args.catalog_b,
        box_size_mpc_h=args.box_size,
        position_mode=args.position_mode,
    )
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
    requested_targets = ("A", "B") if args.target == "both" else (args.target,)

    for target in requested_targets:
        target_catalog = paired.catalog_a if target == "A" else paired.catalog_b
        target_result = result.voids_a if target == "A" else result.voids_b
        mask, _ = halo_slice_mask(
            positions_mpc_h=target_catalog.positions_mpc_h,
            axis=args.slice_axis,
            center_mpc_h=args.slice_center,
            thickness_mpc_h=args.slice_thickness,
            box_size_mpc_h=args.box_size,
        )
        halo_positions = target_catalog.positions_mpc_h[mask]
        finder_all_rows = _all_finder_rows(
            target_result,
            target=target,
            position_mode=args.position_mode,
        )
        vide_desc, vide_centers, vide_macrocenters = _target_paths(args, target)
        vide_desc, vide_centers, vide_macrocenters = resolve_target_vide_paths(
            desc_path=vide_desc,
            centers_path=vide_centers,
            macrocenters_path=vide_macrocenters,
            variant=args.vide_variant,
        )
        vide_all_rows = _all_vide_rows(
            desc_path=vide_desc,
            centers_path=vide_centers,
            macrocenters_path=vide_macrocenters,
            target=target,
            args=args,
        )
        finder_rows, finder_display_radii = select_void_rows_for_halo_slice(
            finder_all_rows,
            args=args,
        )
        vide_rows, vide_display_radii = select_void_rows_for_halo_slice(
            vide_all_rows,
            args=args,
        )
        vide_member_positions = np.empty((0, 3), dtype=np.float64)
        if args.vide_overlay in ("members", "both"):
            vide_input, vide_void_zones, vide_zone_particles = _target_member_paths(args, target)
            vide_member_positions = vide_member_positions_for_rows(
                rows=vide_rows,
                input_path=vide_input,
                void_zones_path=vide_void_zones,
                zone_particles_path=vide_zone_particles,
                args=args,
            )
        for method, rows, display_radii in (
            ("finder", finder_rows, finder_display_radii),
            ("vide", vide_rows, vide_display_radii),
        ):
            output = _output_path(args, target=target, method=method)
            csv_output = _csv_output_path(args, target=target, method=method)
            write_halo_void_plot(
                output,
                halo_positions_mpc_h=halo_positions,
                void_rows=rows,
                display_radii_mpc_h=display_radii,
                vide_member_positions_mpc_h=(
                    vide_member_positions if method == "vide" else None
                ),
                target=target,
                method=method,
                args=args,
            )
            write_void_csv(csv_output, rows=rows, display_radii_mpc_h=display_radii)
            print(
                f"Target {target} {method}: halos={len(halo_positions)} "
                f"voids={len(rows.radii_mpc_h)}"
                f"{' VIDE_members=' + str(len(vide_member_positions)) if method == 'vide' and args.vide_overlay in ('members', 'both') else ''} "
                f"wrote {output} and {csv_output}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
