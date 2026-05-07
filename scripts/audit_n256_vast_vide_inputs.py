#!/usr/bin/env python
"""Audit whether VAST and VIDE used the same n256 tracer catalogs."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.exists():
        sys.path.insert(0, str(import_path))

from pinocchio_voids.geometry import periodic_distance
from pinocchio_voids.io import read_vide_input_tracers

try:
    from scripts.plot_n256_void_slice import DEFAULT_VIDE_INPUT_A, DEFAULT_VIDE_INPUT_B, N256_RUN
    from scripts.run_n256_vast_voidfinder import load_final_halo_positions
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from plot_n256_void_slice import DEFAULT_VIDE_INPUT_A, DEFAULT_VIDE_INPUT_B, N256_RUN
    from run_n256_vast_voidfinder import load_final_halo_positions


DEFAULT_OUTPUT = Path("runs/void-statistics/n256_vast_vide_input_audit.csv")
TARGETS = ("A", "B")


@dataclass(frozen=True)
class TracerSnapshot:
    """Minimal common tracer table used for input audits."""

    source_name: str
    ids: NDArray[np.int64]
    positions_mpc_h: NDArray[np.float64]
    velocities_km_s: NDArray[np.float64]
    masses_msun_h: NDArray[np.float64]
    path: Path
    box_size_mpc_h: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare n256 PINOCCHIO, VAST-input, and VIDE-input tracers."
    )
    parser.add_argument("--catalog-a", type=Path, default=Path(N256_RUN["catalog_a"]))
    parser.add_argument("--catalog-b", type=Path, default=Path(N256_RUN["catalog_b"]))
    parser.add_argument("--vide-input-a", type=Path, default=DEFAULT_VIDE_INPUT_A)
    parser.add_argument("--vide-input-b", type=Path, default=DEFAULT_VIDE_INPUT_B)
    parser.add_argument("--vast-root", type=Path, default=Path("runs/vast-voidfinder"))
    parser.add_argument("--vast-run-suffix", default="")
    parser.add_argument("--box-size", type=float, default=256.0)
    parser.add_argument("--target", choices=("A", "B", "both"), default="both")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def requested_targets(args: argparse.Namespace) -> tuple[str, ...]:
    return TARGETS if args.target == "both" else (args.target,)


def target_name(target: str) -> str:
    return "n256" if target == "A" else "n256_paired"


def vast_input_path(root: Path, target: str, suffix: str) -> Path:
    return root / f"{target_name(target)}{suffix}" / "vast_input_tracers.csv"


def read_pinocchio_snapshot(path: Path, *, target: str, box_size_mpc_h: float) -> TracerSnapshot:
    ids, positions, velocities, masses = load_final_halo_positions(
        path,
        box_size_mpc_h=box_size_mpc_h,
    )
    return TracerSnapshot(
        source_name=f"pinocchio_final_{target}",
        ids=ids,
        positions_mpc_h=positions,
        velocities_km_s=velocities,
        masses_msun_h=masses,
        path=Path(path),
        box_size_mpc_h=box_size_mpc_h,
    )


def read_vide_snapshot(path: Path, *, target: str) -> TracerSnapshot:
    catalog = read_vide_input_tracers(path)
    return TracerSnapshot(
        source_name=f"vide_input_{target}",
        ids=catalog.ids,
        positions_mpc_h=np.asarray(catalog.positions_mpc_h, dtype=np.float64)
        % catalog.box_size_mpc_h,
        velocities_km_s=catalog.velocities_km_s,
        masses_msun_h=catalog.masses_msun_h,
        path=Path(path),
        box_size_mpc_h=float(catalog.box_size_mpc_h),
    )


def read_vast_snapshot(path: Path, *, target: str, box_size_mpc_h: float) -> TracerSnapshot | None:
    if not path.exists():
        return None
    rows = read_csv(path)
    return TracerSnapshot(
        source_name=f"vast_input_{target}",
        ids=np.asarray([int(float(row["id"])) for row in rows], dtype=np.int64),
        positions_mpc_h=np.asarray(
            [[float(row["x_mpc_h"]), float(row["y_mpc_h"]), float(row["z_mpc_h"])] for row in rows],
            dtype=np.float64,
        )
        % box_size_mpc_h,
        velocities_km_s=np.asarray(
            [[float(row["vx_km_s"]), float(row["vy_km_s"]), float(row["vz_km_s"])] for row in rows],
            dtype=np.float64,
        ),
        masses_msun_h=np.asarray([float(row["mass_msun_h"]) for row in rows], dtype=np.float64),
        path=Path(path),
        box_size_mpc_h=box_size_mpc_h,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"Cannot read CSV: {path}") from exc


def nearest_neighbor_distances(
    positions_mpc_h: NDArray[np.float64],
    *,
    box_size_mpc_h: float,
    neighbor_index: int,
) -> NDArray[np.float64]:
    positions = np.asarray(positions_mpc_h, dtype=np.float64) % box_size_mpc_h
    if len(positions) <= neighbor_index:
        return np.empty(0, dtype=np.float64)
    tree = cKDTree(positions, boxsize=box_size_mpc_h)
    distances, _ = tree.query(positions, k=neighbor_index + 1)
    return np.asarray(distances[:, neighbor_index], dtype=np.float64)


def duplicate_position_count(positions_mpc_h: NDArray[np.float64], *, decimals: int = 8) -> int:
    rounded = np.round(np.asarray(positions_mpc_h, dtype=np.float64), decimals=decimals)
    return int(len(rounded) - len(np.unique(rounded, axis=0)))


def finite_float(value: float | int | np.floating) -> float | str:
    value = float(value)
    return value if np.isfinite(value) else ""


def catalog_summary_row(target: str, snapshot: TracerSnapshot) -> dict[str, float | int | str]:
    positions = snapshot.positions_mpc_h
    nn1 = nearest_neighbor_distances(
        positions,
        box_size_mpc_h=snapshot.box_size_mpc_h,
        neighbor_index=1,
    )
    nn3 = nearest_neighbor_distances(
        positions,
        box_size_mpc_h=snapshot.box_size_mpc_h,
        neighbor_index=3,
    )
    row: dict[str, float | int | str] = {
        "row_type": "catalog_summary",
        "target": target,
        "source": snapshot.source_name,
        "comparison": "",
        "path": str(snapshot.path),
        "count": int(len(snapshot.ids)),
        "box_size_mpc_h": float(snapshot.box_size_mpc_h),
        "id_min": int(np.min(snapshot.ids)) if len(snapshot.ids) else "",
        "id_max": int(np.max(snapshot.ids)) if len(snapshot.ids) else "",
        "duplicate_id_count": int(len(snapshot.ids) - len(np.unique(snapshot.ids))),
        "duplicate_position_count": duplicate_position_count(positions),
        "out_of_bounds_count": int(
            np.count_nonzero((positions < 0.0) | (positions >= snapshot.box_size_mpc_h))
        ),
        "x_min_mpc_h": finite_float(np.min(positions[:, 0])) if len(positions) else "",
        "x_max_mpc_h": finite_float(np.max(positions[:, 0])) if len(positions) else "",
        "y_min_mpc_h": finite_float(np.min(positions[:, 1])) if len(positions) else "",
        "y_max_mpc_h": finite_float(np.max(positions[:, 1])) if len(positions) else "",
        "z_min_mpc_h": finite_float(np.min(positions[:, 2])) if len(positions) else "",
        "z_max_mpc_h": finite_float(np.max(positions[:, 2])) if len(positions) else "",
        "mass_min_msun_h": finite_float(np.min(snapshot.masses_msun_h))
        if len(snapshot.masses_msun_h)
        else "",
        "mass_max_msun_h": finite_float(np.max(snapshot.masses_msun_h))
        if len(snapshot.masses_msun_h)
        else "",
        "nn1_median_mpc_h": finite_float(np.median(nn1)) if len(nn1) else "",
        "nn3_mean_mpc_h": finite_float(np.mean(nn3)) if len(nn3) else "",
        "nn3_std_mpc_h": finite_float(np.std(nn3)) if len(nn3) else "",
    }
    return row


def sorted_common_by_id(
    left: TracerSnapshot,
    right: TracerSnapshot,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    common = np.intersect1d(left.ids, right.ids)
    left_order = np.argsort(left.ids)
    right_order = np.argsort(right.ids)
    left_indices = left_order[np.searchsorted(left.ids[left_order], common)]
    right_indices = right_order[np.searchsorted(right.ids[right_order], common)]
    return common, left_indices.astype(np.int64), right_indices.astype(np.int64)


def compare_snapshots(
    *,
    target: str,
    left: TracerSnapshot,
    right: TracerSnapshot,
) -> dict[str, float | int | str]:
    common_ids, left_indices, right_indices = sorted_common_by_id(left, right)
    row: dict[str, float | int | str] = {
        "row_type": "catalog_comparison",
        "target": target,
        "source": f"{left.source_name}|{right.source_name}",
        "comparison": f"{left.source_name}_vs_{right.source_name}",
        "path": f"{left.path}|{right.path}",
        "count_left": int(len(left.ids)),
        "count_right": int(len(right.ids)),
        "count_delta_left_minus_right": int(len(left.ids) - len(right.ids)),
        "common_id_count": int(len(common_ids)),
        "left_only_id_count": int(len(np.setdiff1d(left.ids, right.ids))),
        "right_only_id_count": int(len(np.setdiff1d(right.ids, left.ids))),
        "same_id_order": bool_to_int(len(left.ids) == len(right.ids) and np.array_equal(left.ids, right.ids)),
    }
    if len(common_ids):
        distances = periodic_distance(
            left.positions_mpc_h[left_indices],
            right.positions_mpc_h[right_indices],
            min(float(left.box_size_mpc_h), float(right.box_size_mpc_h)),
        )
        mass_rel = np.abs(left.masses_msun_h[left_indices] - right.masses_msun_h[right_indices])
        mass_rel /= np.maximum(np.abs(left.masses_msun_h[left_indices]), 1.0)
        row.update(
            {
                "position_delta_max_mpc_h": finite_float(np.max(distances)),
                "position_delta_median_mpc_h": finite_float(np.median(distances)),
                "position_delta_p99_mpc_h": finite_float(np.percentile(distances, 99.0)),
                "position_match_count_1e-6": int(np.count_nonzero(distances <= 1.0e-6)),
                "mass_relative_delta_max": finite_float(np.max(mass_rel)),
                "mass_relative_delta_median": finite_float(np.median(mass_rel)),
            }
        )
    else:
        row.update(
            {
                "position_delta_max_mpc_h": "",
                "position_delta_median_mpc_h": "",
                "position_delta_p99_mpc_h": "",
                "position_match_count_1e-6": 0,
                "mass_relative_delta_max": "",
                "mass_relative_delta_median": "",
            }
        )
    return row


def bool_to_int(value: bool) -> int:
    return 1 if value else 0


def write_rows(path: Path, rows: Sequence[Mapping[str, float | int | str]]) -> None:
    if not rows:
        raise SystemExit("No audit rows to write")
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "row_type",
        "target",
        "source",
        "comparison",
        "count",
        "count_left",
        "count_right",
        "common_id_count",
        "position_delta_max_mpc_h",
        "position_delta_median_mpc_h",
        "path",
    ]
    fieldnames = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def target_paths(args: argparse.Namespace, target: str) -> tuple[Path, Path]:
    if target == "A":
        return args.catalog_a, args.vide_input_a
    return args.catalog_b, args.vide_input_b


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, float | int | str]] = []
    for target in requested_targets(args):
        pinocchio_path, vide_path = target_paths(args, target)
        pinocchio = read_pinocchio_snapshot(
            pinocchio_path,
            target=target,
            box_size_mpc_h=float(args.box_size),
        )
        vide = read_vide_snapshot(vide_path, target=target)
        vast = read_vast_snapshot(
            vast_input_path(args.vast_root, target, str(args.vast_run_suffix)),
            target=target,
            box_size_mpc_h=float(args.box_size),
        )
        snapshots = [pinocchio, vide] + ([vast] if vast is not None else [])
        rows.extend(catalog_summary_row(target, snapshot) for snapshot in snapshots)
        rows.append(compare_snapshots(target=target, left=pinocchio, right=vide))
        if vast is not None:
            rows.append(compare_snapshots(target=target, left=pinocchio, right=vast))
            rows.append(compare_snapshots(target=target, left=vide, right=vast))

    write_rows(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
