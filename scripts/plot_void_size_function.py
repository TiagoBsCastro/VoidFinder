#!/usr/bin/env python
"""Plot VIDE void size functions from voidDesc catalog files."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinocchio_voids.io import read_vide_void_desc
from pinocchio_voids.theory import compute_vdn_svdw_size_function, read_pinocchio_cosmology


DEFAULT_ROOT = Path("runs/vide-lowres")
DEFAULT_PATTERN = "n256*/outputs/*/sample_*/voidDesc_all_*.out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot differential VIDE void size functions, dN/dlnR/V."
    )
    parser.add_argument(
        "void_desc",
        nargs="*",
        type=Path,
        help="VIDE voidDesc files. If omitted, files are discovered under --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root used for automatic discovery. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern under --root. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Labels matching the input files. Defaults to run directory names.",
    )
    parser.add_argument("--box-size", type=float, default=256.0, help="Box size in Mpc/h.")
    parser.add_argument("--bins", type=int, default=12, help="Number of log-radius bins.")
    parser.add_argument(
        "--binning",
        choices=("log", "linear"),
        default="log",
        help="Radius bin spacing. Without --bin-min/--bin-max, log uses data bounds.",
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
        "--summary-csv",
        type=Path,
        help="Optional radius-summary CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/void-statistics/n256_vide_target_vsf.png"),
        help="Output plot path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("runs/void-statistics/n256_vide_target_vsf.csv"),
        help="Output table path.",
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
    return parser.parse_args()


def load_void_radii(path: Path) -> np.ndarray:
    return read_vide_void_desc(path).effective_radii_mpc_h


def label_for(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except ValueError:
        return path.stem


def discover_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def resolve_edges(args: argparse.Namespace, all_radii: np.ndarray) -> np.ndarray:
    if args.bins < 1:
        raise SystemExit("--bins must be at least 1")
    has_min = args.bin_min is not None
    has_max = args.bin_max is not None
    if has_min != has_max:
        raise SystemExit("--bin-min and --bin-max must be provided together")
    if has_min:
        lower = float(args.bin_min)
        upper = float(args.bin_max)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
            raise SystemExit("--bin-min/--bin-max must be positive finite increasing edges")
        if args.binning == "linear":
            return np.linspace(lower, upper, args.bins + 1)
        return np.geomspace(lower, upper, args.bins + 1)
    if args.binning == "linear":
        return np.linspace(all_radii.min(), all_radii.max(), args.bins + 1)
    return np.geomspace(all_radii.min(), all_radii.max(), args.bins + 1)


def write_radius_summary(path: Path, radii_by_label: list[tuple[str, np.ndarray]]) -> None:
    rows = []
    for label, radii in radii_by_label:
        row = {
            "label": label,
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
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    paths = args.void_desc or discover_files(args.root, args.pattern)
    if not paths:
        raise SystemExit(f"No voidDesc files found under {args.root} with {args.pattern}")

    labels = args.labels or [label_for(path, args.root) for path in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels must match the number of input files")

    radii_by_label = [(label, load_void_radii(path)) for label, path in zip(labels, paths)]
    non_empty = [radii for _, radii in radii_by_label if radii.size]
    if not non_empty:
        raise SystemExit("No positive void radii found")

    all_radii = np.concatenate(non_empty)
    edges = resolve_edges(args, all_radii)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dlnr = np.diff(np.log(edges))
    volume = args.box_size**3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    rows = []
    for label, radii in radii_by_label:
        counts, _ = np.histogram(radii, bins=edges)
        size_function = counts / (volume * dlnr)
        ax.step(centers, size_function, where="mid", marker="o", label=f"{label} ({radii.size})")
        for r_min, r_max, r_mid, count, dn_dlnr_dv in zip(
            edges[:-1], edges[1:], centers, counts, size_function
        ):
            rows.append(
                {
                    "label": label,
                    "r_min_mpc_h": r_min,
                    "r_max_mpc_h": r_max,
                    "r_mid_mpc_h": r_mid,
                    "count": int(count),
                    "dn_dlnr_dv_h3_mpc-3": dn_dlnr_dv,
                }
            )

    if args.theory is not None:
        if args.cosmology_file is None:
            raise SystemExit("--cosmology-file is required with --theory")
        cosmology = read_pinocchio_cosmology(args.cosmology_file)
        theory = compute_vdn_svdw_size_function(
            edges,
            cosmology,
            delta_v_linear=args.delta_v_linear,
            delta_c_linear=args.delta_c_linear,
            delta_v_nonlinear=args.delta_v_nonlinear,
        )
        valid = theory.density_dndlnr_per_mpc_h3 > 0.0
        ax.plot(
            theory.bin_centers_mpc_h[valid],
            theory.density_dndlnr_per_mpc_h3[valid],
            linestyle="--",
            color="black",
            label="Vdn/SVdW theory",
        )
        for r_min, r_max, r_mid, dn_dlnr_dv in zip(
            theory.bin_edges_mpc_h[:-1],
            theory.bin_edges_mpc_h[1:],
            theory.bin_centers_mpc_h,
            theory.density_dndlnr_per_mpc_h3,
        ):
            rows.append(
                {
                    "label": theory.model,
                    "r_min_mpc_h": r_min,
                    "r_max_mpc_h": r_max,
                    "r_mid_mpc_h": r_mid,
                    "count": "",
                    "dn_dlnr_dv_h3_mpc-3": dn_dlnr_dv,
                }
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R_\mathrm{eff}$ [$h^{-1}\,\mathrm{Mpc}$]")
    ax.set_ylabel(r"$dN / d\ln R / V$ [$(h^{-1}\,\mathrm{Mpc})^{-3}$]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("VIDE void size function")
    fig.savefig(args.output, dpi=180)

    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.summary_csv is not None:
        write_radius_summary(args.summary_csv, radii_by_label)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")
    if args.summary_csv is not None:
        print(f"Wrote {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
