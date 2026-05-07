#!/usr/bin/env python
"""Run VAST.VoidFinder on the active n256 PINOCCHIO halo catalogs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT / "src", REPO_ROOT / "external" / "VAST" / "python"):
    if import_path.exists():
        sys.path.insert(0, str(import_path))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from pinocchio_voids.io.pinocchio import read_pinocchio_halo_catalog


N256_TARGETS = {
    "A": {
        "name": "n256",
        "catalog": Path("runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out"),
    },
    "B": {
        "name": "n256_paired",
        "catalog": Path(
            "runs/pinocchio-lowres/n256_paired/"
            "pinocchio.0.0000.lowres_n256_paired.catalog.out"
        ),
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VAST.VoidFinder on the local n256 paired PINOCCHIO catalogs."
    )
    parser.add_argument(
        "--target",
        choices=("A", "B", "both"),
        default="both",
        help="Run realization A, B, or both.",
    )
    parser.add_argument("--catalog-a", type=Path, default=N256_TARGETS["A"]["catalog"])
    parser.add_argument("--catalog-b", type=Path, default=N256_TARGETS["B"]["catalog"])
    parser.add_argument("--box-size", type=float, default=256.0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/vast-voidfinder"))
    parser.add_argument(
        "--hole-grid-edge-length",
        type=float,
        default=5.0,
        help="VAST empty-cell search grid spacing in Mpc/h.",
    )
    parser.add_argument(
        "--galaxy-map-grid-edge-length",
        type=float,
        default=16.0,
        help="VAST nearest-neighbor grid spacing. Must divide the periodic box.",
    )
    parser.add_argument(
        "--min-maximal-radius",
        type=float,
        action="append",
        default=None,
        help="Minimum VAST maximal sphere radius in Mpc/h.",
    )
    parser.add_argument(
        "--wall-field-separation",
        action="store_true",
        help="Run VAST wall/field isolated-tracer removal before find_voids.",
    )
    parser.add_argument(
        "--sep-neighbor",
        type=int,
        default=3,
        help="Nth neighbor used by VAST wall/field separation.",
    )
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument(
        "--pts-per-unit-volume",
        type=float,
        default=0.01,
        help="VAST Monte-Carlo density for boundary checks.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args(argv)


def requested_targets(args: argparse.Namespace) -> tuple[str, ...]:
    return ("A", "B") if args.target == "both" else (args.target,)


def requested_min_maximal_radii(args: argparse.Namespace) -> tuple[float, ...]:
    values = args.min_maximal_radius if args.min_maximal_radius is not None else [10.0]
    if isinstance(values, (float, int)):
        values = [values]
    return tuple(float(value) for value in values)


def format_radius_token(radius_mpc_h: float) -> str:
    return f"{float(radius_mpc_h):g}".replace(".", "p").replace("-", "m")


def run_label(base_name: str, *, wall_field_separation: bool, min_maximal_radius: float) -> str:
    parts = []
    if wall_field_separation:
        parts.append("wall")
    if not np.isclose(float(min_maximal_radius), 10.0):
        parts.append(f"rmin{format_radius_token(min_maximal_radius)}")
    if not parts:
        return base_name
    return f"{base_name}_{'_'.join(parts)}"


def _target_catalog_path(args: argparse.Namespace, target: str) -> Path:
    return args.catalog_a if target == "A" else args.catalog_b


def load_final_halo_positions(
    path: Path,
    *,
    box_size_mpc_h: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return IDs, final positions, velocities, and masses for VAST."""

    catalog = read_pinocchio_halo_catalog(path, box_size_mpc_h=box_size_mpc_h)
    ids = catalog.group_ids
    positions = np.asarray(catalog.final_positions_mpc_h, dtype=np.float64) % box_size_mpc_h
    velocities = np.asarray(catalog.velocities_km_s, dtype=np.float64)
    masses = np.asarray(catalog.masses_msun_h, dtype=np.float64)
    return ids, positions, velocities, masses


def write_input_audit_csv(
    path: Path,
    *,
    ids: NDArray[np.int64],
    positions_mpc_h: NDArray[np.float64],
    velocities_km_s: NDArray[np.float64],
    masses_msun_h: NDArray[np.float64],
) -> None:
    """Write the exact final-position tracer catalog passed to VAST."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "x_mpc_h",
                "y_mpc_h",
                "z_mpc_h",
                "vx_km_s",
                "vy_km_s",
                "vz_km_s",
                "mass_msun_h",
            ],
        )
        writer.writeheader()
        for halo_id, position, velocity, mass in zip(
            ids, positions_mpc_h, velocities_km_s, masses_msun_h, strict=True
        ):
            writer.writerow(
                {
                    "id": int(halo_id),
                    "x_mpc_h": float(position[0]),
                    "y_mpc_h": float(position[1]),
                    "z_mpc_h": float(position[2]),
                    "vx_km_s": float(velocity[0]),
                    "vy_km_s": float(velocity[1]),
                    "vz_km_s": float(velocity[2]),
                    "mass_msun_h": float(mass),
                }
            )


def _match_selected_indices(
    original_positions_mpc_h: NDArray[np.float64],
    selected_positions_mpc_h: NDArray[np.float64],
    *,
    tolerance_mpc_h: float = 1.0e-8,
) -> NDArray[np.int64]:
    """Return original row indices for selected positions returned by VAST."""

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(original_positions_mpc_h, dtype=np.float64))
    distances, indices = tree.query(np.asarray(selected_positions_mpc_h, dtype=np.float64), k=1)
    if len(distances) and float(np.max(distances)) > tolerance_mpc_h:
        raise ValueError(
            "Cannot map VAST wall tracers back to input rows; "
            f"max nearest distance is {float(np.max(distances))}"
        )
    return np.asarray(indices, dtype=np.int64)


def prepare_vast_input_positions(
    *,
    positions_mpc_h: NDArray[np.float64],
    output_dir: Path,
    survey_name: str,
    args: argparse.Namespace,
) -> NDArray[np.int64]:
    """Return input row indices passed into VAST find_voids."""

    if not args.wall_field_separation:
        return np.arange(len(positions_mpc_h), dtype=np.int64)

    try:
        from vast.voidfinder import wall_field_separation
    except ModuleNotFoundError as exc:
        raise SystemExit("VAST is not installed; cannot run wall-field separation") from exc

    wall_positions, field_positions = wall_field_separation(
        np.asarray(positions_mpc_h, dtype=np.float64),
        sep_neighbor=int(args.sep_neighbor),
        verbose=int(args.verbose),
        survey_name=survey_name,
        out_directory=str(output_dir),
        write_galaxies=True,
        capitalize_colnames=False,
    )
    selected_indices = _match_selected_indices(positions_mpc_h, wall_positions)
    print(
        f"VAST wall-field separation kept {len(wall_positions)} wall tracers "
        f"and removed {len(field_positions)} field tracers"
    )
    return selected_indices


def _table_columns(table: Any) -> Mapping[str, Any]:
    if hasattr(table, "colnames"):
        return {str(name).lower(): table[name] for name in table.colnames}
    if isinstance(table, Mapping):
        return {str(name).lower(): values for name, values in table.items()}
    raise TypeError("VAST output tables must be Astropy tables or mappings")


def _column(columns: Mapping[str, Any], aliases: Sequence[str]) -> NDArray[np.float64] | None:
    for alias in aliases:
        value = columns.get(alias.lower())
        if value is not None:
            return np.asarray(value, dtype=np.float64)
    return None


def _required_column(columns: Mapping[str, Any], aliases: Sequence[str], name: str) -> NDArray[np.float64]:
    values = _column(columns, aliases)
    if values is None:
        available = ", ".join(sorted(columns))
        raise ValueError(f"Cannot find VAST {name} column in table with columns: {available}")
    return values


def _void_ids(columns: Mapping[str, Any], size: int, *, fallback_sequential: bool) -> NDArray[np.int64]:
    values = _column(
        columns,
        (
            "void_id",
            "void",
            "voidnum",
            "void_number",
            "void_number_",
            "void#",
        ),
    )
    if values is None:
        if fallback_sequential:
            return np.arange(size, dtype=np.int64)
        return np.full(size, -1, dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def _positions_and_radii(table: Any) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    columns = _table_columns(table)
    x = _required_column(columns, ("x", "x_mpc_h", "x_void"), "x")
    y = _required_column(columns, ("y", "y_mpc_h", "y_void"), "y")
    z = _required_column(columns, ("z", "z_mpc_h", "z_void"), "z")
    radius = _required_column(columns, ("radius", "r", "radius_mpc_h"), "radius")
    positions = np.column_stack([x, y, z]).astype(np.float64, copy=False)
    return positions, np.asarray(radius, dtype=np.float64), _void_ids(
        columns, len(radius), fallback_sequential=True
    )


def normalize_vast_tables(
    maximal_table: Any,
    holes_table: Any,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    """Convert VAST returned tables to stable maximal/holes CSV rows."""

    maximal_positions, maximal_radii, maximal_ids = _positions_and_radii(maximal_table)
    hole_positions, hole_radii, hole_ids = _positions_and_radii(holes_table)
    holes_by_void: dict[int, int] = {}
    for void_id in hole_ids:
        if int(void_id) >= 0:
            holes_by_void[int(void_id)] = holes_by_void.get(int(void_id), 0) + 1

    maximal_rows: list[dict[str, float | int]] = []
    for row_id, position, radius, void_id in zip(
        range(len(maximal_radii)),
        maximal_positions,
        maximal_radii,
        maximal_ids,
        strict=True,
    ):
        normalized_id = int(void_id)
        maximal_rows.append(
            {
                "void_id": normalized_id,
                "x_mpc_h": float(position[0]),
                "y_mpc_h": float(position[1]),
                "z_mpc_h": float(position[2]),
                "maximal_radius_mpc_h": float(radius),
                "n_holes": holes_by_void.get(normalized_id, 0),
                "source_row": row_id,
            }
        )

    hole_rows: list[dict[str, float | int]] = []
    for hole_id, position, radius, void_id in zip(
        range(len(hole_radii)),
        hole_positions,
        hole_radii,
        hole_ids,
        strict=True,
    ):
        hole_rows.append(
            {
                "hole_id": int(hole_id),
                "void_id": int(void_id),
                "x_mpc_h": float(position[0]),
                "y_mpc_h": float(position[1]),
                "z_mpc_h": float(position[2]),
                "radius_mpc_h": float(radius),
            }
        )
    return maximal_rows, hole_rows


def write_rows(path: Path, rows: Sequence[Mapping[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["void_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_vast(
    *,
    positions_mpc_h: NDArray[np.float64],
    output_dir: Path,
    survey_name: str,
    box_size_mpc_h: float,
    min_maximal_radius_mpc_h: float,
    args: argparse.Namespace,
):
    """Run VAST.VoidFinder and return its maximal and holes tables."""

    try:
        from vast.voidfinder import find_voids
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "VAST is not installed. Install it in the VAST environment, for example:\n"
            "  /home/tcastro/miniforge3/bin/mamba create -n .conda-vast "
            "python=3.10 numpy scipy cython astropy scikit-learn matplotlib h5py pip -y\n"
            "  /home/tcastro/miniforge3/bin/mamba install -n .conda-vast psutil healpy -y\n"
            "  git clone https://github.com/DESI-UR/VAST external/VAST\n"
            "  cd external/VAST\n"
            "  /home/tcastro/miniforge3/envs/.conda-vast/bin/python setup.py build_ext --inplace"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    xyz_limits = np.asarray(
        [[0.0, 0.0, 0.0], [box_size_mpc_h, box_size_mpc_h, box_size_mpc_h]],
        dtype=np.float64,
    )
    return find_voids(
        positions_mpc_h.astype(np.float64, copy=False),
        survey_name=survey_name,
        out_directory=str(output_dir),
        mask_type="periodic",
        xyz_limits=xyz_limits,
        hole_grid_edge_length=float(args.hole_grid_edge_length),
        galaxy_map_grid_edge_length=float(args.galaxy_map_grid_edge_length),
        min_maximal_radius=float(min_maximal_radius_mpc_h),
        pts_per_unit_volume=float(args.pts_per_unit_volume),
        num_cpus=int(args.num_cpus),
        batch_size=int(args.batch_size),
        verbose=int(args.verbose),
        capitalize_colnames=False,
        SOCKET_PATH=str(output_dir / f"{survey_name}_voidfinder.sock"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for target in requested_targets(args):
        base_run_name = str(N256_TARGETS[target]["name"])
        ids, positions, velocities, masses = load_final_halo_positions(
            _target_catalog_path(args, target),
            box_size_mpc_h=args.box_size,
        )
        for min_maximal_radius in requested_min_maximal_radii(args):
            run_name = run_label(
                base_run_name,
                wall_field_separation=bool(args.wall_field_separation),
                min_maximal_radius=float(min_maximal_radius),
            )
            output_dir = args.output_root / run_name
            output_dir.mkdir(parents=True, exist_ok=True)
            selected_indices = prepare_vast_input_positions(
                positions_mpc_h=positions,
                output_dir=output_dir,
                survey_name=run_name,
                args=args,
            )
            write_input_audit_csv(
                output_dir / "vast_input_tracers.csv",
                ids=ids[selected_indices],
                positions_mpc_h=positions[selected_indices],
                velocities_km_s=velocities[selected_indices],
                masses_msun_h=masses[selected_indices],
            )
            print(
                f"Running VAST target {target} ({run_name}) with "
                f"{len(selected_indices)} final-position halos, "
                f"min_maximal_radius={float(min_maximal_radius):g}"
            )
            maximal_table, holes_table = run_vast(
                positions_mpc_h=positions[selected_indices],
                output_dir=output_dir,
                survey_name=run_name,
                box_size_mpc_h=float(args.box_size),
                min_maximal_radius_mpc_h=float(min_maximal_radius),
                args=args,
            )
            maximal_rows, hole_rows = normalize_vast_tables(maximal_table, holes_table)
            if not maximal_rows:
                raise SystemExit(f"VAST returned no maximal spheres for target {target}")
            write_rows(output_dir / "vast_voids_maximal.csv", maximal_rows)
            write_rows(output_dir / "vast_voids_holes.csv", hole_rows)
            print(
                f"Target {target}: wrote {len(maximal_rows)} maximal spheres and "
                f"{len(hole_rows)} holes under {output_dir}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
