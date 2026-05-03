#!/usr/bin/env python
"""Debug theoretical void size-function amplitudes against CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pinocchio_voids.theory import (
    PinocchioCosmologyTable,
    compute_vdn_svdw_factors,
    read_pinocchio_cosmology,
)


@dataclass(frozen=True)
class VsfCsvRow:
    """One row from a finder/VIDE/theory VSF CSV."""

    label: str
    source: str
    target: str
    bin_min_mpc_h: float
    bin_max_mpc_h: float
    bin_center_mpc_h: float
    count: int | None
    density_dndlnr_per_mpc_h3: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose theory-vs-observed void size-function amplitude differences."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="VSF CSV files written by compare_void_size_functions.py.",
    )
    parser.add_argument("--box-size", type=float, default=256.0, help="Box size in Mpc/h.")
    parser.add_argument(
        "--cosmology-file",
        type=Path,
        help="PINOCCHIO cosmology table used for intermediate Vdn/SVdW factors.",
    )
    parser.add_argument(
        "--theory-source",
        default="vdn-svdw",
        help="CSV source label for the theoretical curve.",
    )
    parser.add_argument(
        "--observed-source",
        action="append",
        choices=("vide", "finder"),
        help="Observed source to compare against. Defaults to both VIDE and finder.",
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
    parser.add_argument("--terms", type=int, default=4, help="Number of SVdW series terms.")
    parser.add_argument(
        "--no-h-conversion",
        action="store_true",
        help="Diagnostic only: treat cosmology smoothing radii as Mpc/h instead of true Mpc.",
    )
    parser.add_argument(
        "--volume-denominator",
        choices=("eulerian", "lagrangian"),
        default="eulerian",
        help="Diagnostic volume denominator used in decomposition.",
    )
    return parser.parse_args(argv)


def read_vsf_csv(path: str | Path) -> list[VsfCsvRow]:
    """Read rows from a VSF comparison CSV."""

    rows: list[VsfCsvRow] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            count_text = raw.get("count", "")
            rows.append(
                VsfCsvRow(
                    label=raw["label"],
                    source=raw["source"],
                    target=raw["target"],
                    bin_min_mpc_h=float(raw["bin_min_mpc_h"]),
                    bin_max_mpc_h=float(raw["bin_max_mpc_h"]),
                    bin_center_mpc_h=float(raw["bin_center_mpc_h"]),
                    count=None if count_text == "" else int(count_text),
                    density_dndlnr_per_mpc_h3=float(raw["density_dndlnr_per_mpc_h3"]),
                )
            )
    return rows


def theory_implied_count(
    *,
    density_dndlnr_per_mpc_h3: float,
    bin_min_mpc_h: float,
    bin_max_mpc_h: float,
    box_size_mpc_h: float,
) -> float:
    """Convert ``dn/dlnR/V`` into the count implied in one log-radius bin."""

    if bin_min_mpc_h <= 0.0 or bin_max_mpc_h <= bin_min_mpc_h:
        raise ValueError("bin edges must be positive and increasing")
    if box_size_mpc_h <= 0.0:
        raise ValueError("box_size_mpc_h must be positive")
    return density_dndlnr_per_mpc_h3 * box_size_mpc_h**3 * math.log(
        bin_max_mpc_h / bin_min_mpc_h
    )


def _bin_key(row: VsfCsvRow) -> tuple[str, float, float]:
    return (row.target, row.bin_min_mpc_h, row.bin_max_mpc_h)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    return numerator / denominator


def _rows_by_source(rows: Iterable[VsfCsvRow]) -> dict[str, dict[tuple[str, float, float], VsfCsvRow]]:
    grouped: dict[str, dict[tuple[str, float, float], VsfCsvRow]] = {}
    for row in rows:
        grouped.setdefault(row.source, {})[_bin_key(row)] = row
    return grouped


def write_ratio_report(
    rows: Sequence[VsfCsvRow],
    *,
    box_size_mpc_h: float,
    theory_source: str,
    observed_sources: Sequence[str],
) -> None:
    """Print theory-to-observed ratios and theory-implied counts."""

    grouped = _rows_by_source(rows)
    theory_rows = grouped.get(theory_source, {})
    print(
        "ratio,target,bin_center_mpc_h,comparison,theory_density,"
        "observed_density,theory_observed_ratio,theory_implied_count,observed_count"
    )
    for key, theory in sorted(theory_rows.items()):
        for observed_source in observed_sources:
            observed = grouped.get(observed_source, {}).get(key)
            if observed is None:
                continue
            implied_count = theory_implied_count(
                density_dndlnr_per_mpc_h3=theory.density_dndlnr_per_mpc_h3,
                bin_min_mpc_h=theory.bin_min_mpc_h,
                bin_max_mpc_h=theory.bin_max_mpc_h,
                box_size_mpc_h=box_size_mpc_h,
            )
            ratio = _safe_ratio(
                theory.density_dndlnr_per_mpc_h3,
                observed.density_dndlnr_per_mpc_h3,
            )
            observed_count = "" if observed.count is None else observed.count
            print(
                f"ratio,{theory.target},{theory.bin_center_mpc_h:.8g},"
                f"theory/{observed_source},{theory.density_dndlnr_per_mpc_h3:.8e},"
                f"{observed.density_dndlnr_per_mpc_h3:.8e},{ratio:.8e},"
                f"{implied_count:.8e},{observed_count}"
            )


def write_decomposition_report(
    rows: Sequence[VsfCsvRow],
    *,
    cosmology: PinocchioCosmologyTable,
    theory_source: str,
    delta_v_linear: float,
    delta_c_linear: float,
    delta_v_nonlinear: float,
    terms: int,
    apply_h_conversion: bool,
    volume_denominator: str,
) -> None:
    """Print Vdn/SVdW intermediate factors for theory bins."""

    theory_rows = [
        row for row in rows if row.source == theory_source
    ]
    print(
        "decomposition,target,bin_center_mpc_h,eulerian_radius_mpc_h,"
        "lagrangian_radius_mpc_h,lagrangian_radius_mpc,sigma,"
        "dlog_sigma_inv_dlog_r,first_crossing_fraction,"
        "denominator_volume_mpc_h3,density_dndlnr_per_mpc_h3,valid"
    )
    if not theory_rows:
        return
    radii = np.asarray([row.bin_center_mpc_h for row in theory_rows], dtype=np.float64)
    factors = compute_vdn_svdw_factors(
        radii,
        cosmology,
        delta_v_linear=delta_v_linear,
        delta_c_linear=delta_c_linear,
        delta_v_nonlinear=delta_v_nonlinear,
        terms=terms,
        apply_h_conversion=apply_h_conversion,
        volume_denominator=volume_denominator,
    )
    for index, row in enumerate(theory_rows):
        print(
            f"decomposition,{row.target},{row.bin_center_mpc_h:.8g},"
            f"{factors.eulerian_radii_mpc_h[index]:.8e},"
            f"{factors.lagrangian_radii_mpc_h[index]:.8e},"
            f"{factors.lagrangian_radii_mpc[index]:.8e},"
            f"{factors.sigma[index]:.8e},"
            f"{factors.dlog_sigma_inv_dlog_r[index]:.8e},"
            f"{factors.first_crossing_fraction[index]:.8e},"
            f"{factors.denominator_volume_mpc_h3[index]:.8e},"
            f"{factors.density_dndlnr_per_mpc_h3[index]:.8e},"
            f"{bool(factors.valid[index])}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    observed_sources = args.observed_source or ["vide", "finder"]
    cosmology = read_pinocchio_cosmology(args.cosmology_file) if args.cosmology_file else None
    for path in args.csv_files:
        print(f"# file: {path}")
        rows = read_vsf_csv(path)
        write_ratio_report(
            rows,
            box_size_mpc_h=args.box_size,
            theory_source=args.theory_source,
            observed_sources=observed_sources,
        )
        if cosmology is not None:
            write_decomposition_report(
                rows,
                cosmology=cosmology,
                theory_source=args.theory_source,
                delta_v_linear=args.delta_v_linear,
                delta_c_linear=args.delta_c_linear,
                delta_v_nonlinear=args.delta_v_nonlinear,
                terms=args.terms,
                apply_h_conversion=not args.no_h_conversion,
                volume_denominator=args.volume_denominator,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
