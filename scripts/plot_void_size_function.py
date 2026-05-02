#!/usr/bin/env python
"""Plot VIDE void size functions from voidDesc catalog files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("runs/vide-lowres/n128_pair")
DEFAULT_PATTERN = "*/outputs/*/sample_*/voidDesc_all_*.out"


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
        "--output",
        type=Path,
        default=Path("runs/vide-lowres/n128_pair/plots/void_size_function.png"),
        help="Output plot path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("runs/vide-lowres/n128_pair/plots/void_size_function.csv"),
        help="Output table path.",
    )
    return parser.parse_args()


def load_void_radii(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 3:
        return np.array([], dtype=float)

    header = lines[1].split()
    try:
        volume_index = header.index("VoidVol")
    except ValueError as exc:
        raise ValueError(f"{path} does not contain a VoidVol column") from exc

    volumes = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) <= volume_index:
            continue
        volume = float(parts[volume_index])
        if volume > 0:
            volumes.append(volume)

    volumes_array = np.asarray(volumes, dtype=float)
    return np.power(3.0 * volumes_array / (4.0 * np.pi), 1.0 / 3.0)


def label_for(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except ValueError:
        return path.stem


def discover_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


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
        raise SystemExit("No positive VoidVol entries found")

    all_radii = np.concatenate(non_empty)
    edges = np.geomspace(all_radii.min(), all_radii.max(), args.bins + 1)
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

    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
