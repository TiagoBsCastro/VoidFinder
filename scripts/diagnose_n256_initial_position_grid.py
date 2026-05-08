#!/usr/bin/env python
"""Coarse posterior grid for debugging n256 initial-position calibration."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np

from pinocchio_voids.io import (
    PINOCCHIO_POSITION_MODES,
    VIDE_CATALOG_VARIANTS,
    pinocchio_position_mode_output_suffix,
    resolve_vide_catalog_variant_path,
    vide_catalog_variant_output_suffix,
)

try:
    from scripts.optimize_n256_full_algorithm_mcmc import (
        BLOB_NAMES,
        DEFAULT_BOUNDS,
        DEFAULT_CATALOG_A,
        DEFAULT_CATALOG_B,
        DEFAULT_INITIAL_CENTER,
        DEFAULT_VIDE_A,
        DEFAULT_VIDE_B,
        FullMcmcSettings,
        N256FullLogPosterior,
        N256FullMcmcPaths,
        PARAMETER_NAMES,
        format_diagnostic_summary,
        posterior_diagnostic_summary,
    )
    from scripts.plot_n256_void_slice import (
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from optimize_n256_full_algorithm_mcmc import (
        BLOB_NAMES,
        DEFAULT_BOUNDS,
        DEFAULT_CATALOG_A,
        DEFAULT_CATALOG_B,
        DEFAULT_INITIAL_CENTER,
        DEFAULT_VIDE_A,
        DEFAULT_VIDE_B,
        FullMcmcSettings,
        N256FullLogPosterior,
        N256FullMcmcPaths,
        PARAMETER_NAMES,
        format_diagnostic_summary,
        posterior_diagnostic_summary,
    )
    from plot_n256_void_slice import (
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
    )


DEFAULT_OUTPUT_CSV = Path("runs/void-statistics/n256_initial_position_grid.csv")
SWEEP_PARAMETER_INDICES = {
    "linking_factor": 0,
    "radius_a0": 1,
    "adjacency_factor": 3,
    "merge_threshold": 4,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a coarse n256 full-algorithm posterior grid to find "
            "non-degenerate initial-position parameter regions."
        )
    )
    parser.add_argument("--catalog-a", type=Path, default=DEFAULT_CATALOG_A)
    parser.add_argument("--catalog-b", type=Path, default=DEFAULT_CATALOG_B)
    parser.add_argument("--vide-a", type=Path, default=DEFAULT_VIDE_A)
    parser.add_argument("--vide-b", type=Path, default=DEFAULT_VIDE_B)
    parser.add_argument("--vide-centers-a", type=Path, default=DEFAULT_VIDE_CENTERS_A)
    parser.add_argument("--vide-centers-b", type=Path, default=DEFAULT_VIDE_CENTERS_B)
    parser.add_argument("--vide-macrocenters-a", type=Path, default=DEFAULT_VIDE_MACROCENTERS_A)
    parser.add_argument("--vide-macrocenters-b", type=Path, default=DEFAULT_VIDE_MACROCENTERS_B)
    parser.add_argument(
        "--position-mode",
        choices=PINOCCHIO_POSITION_MODES,
        default="initial",
        help="PINOCCHIO coordinate columns used by the finder.",
    )
    parser.add_argument(
        "--vide-variant",
        choices=VIDE_CATALOG_VARIANTS,
        default="untrimmed",
        help="VIDE catalog variant used for calibration diagnostics.",
    )
    parser.add_argument(
        "--vide-center-kind",
        choices=("center", "macrocenter"),
        default="center",
    )
    parser.add_argument("--linking-factor-values", type=float, nargs="+", default=(0.10, 0.13, 0.16))
    parser.add_argument("--radius-a0-values", type=float, nargs="+", default=(4.0, 6.0, 8.0))
    parser.add_argument(
        "--adjacency-factor-values",
        type=float,
        nargs="+",
        default=(0.15, 0.40, 0.65),
    )
    parser.add_argument(
        "--merge-threshold-values",
        type=float,
        nargs="+",
        default=(0.25, 1.0, 2.0),
    )
    parser.add_argument("--radius-alpha", type=float, default=float(DEFAULT_INITIAL_CENTER[2]))
    parser.add_argument(
        "--bridge-radius-factor",
        type=float,
        default=float(DEFAULT_INITIAL_CENTER[5]),
    )
    parser.add_argument("--bridge-weight", type=float, default=float(DEFAULT_INITIAL_CENTER[6]))
    parser.add_argument(
        "--compatibility-weight",
        type=float,
        default=float(DEFAULT_INITIAL_CENTER[7]),
    )
    parser.add_argument("--vsf-weight", type=float, default=1.0)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--center-sigma", type=float, default=1.0)
    parser.add_argument("--center-nu", type=float, default=3.0)
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Keep degenerate samples finite when scoring the grid.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args(argv)


def _append_default_suffix(path: Path, *, vide_variant: str, position_mode: str) -> Path:
    suffix = (
        f"{vide_catalog_variant_output_suffix(vide_variant)}"
        f"{pinocchio_position_mode_output_suffix(position_mode)}"
    )
    if not suffix or path != DEFAULT_OUTPUT_CSV:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _base_theta(args: argparse.Namespace) -> np.ndarray:
    theta = DEFAULT_INITIAL_CENTER.copy()
    theta[2] = args.radius_alpha
    theta[5] = args.bridge_radius_factor
    theta[6] = args.bridge_weight
    theta[7] = args.compatibility_weight
    return theta


def _score_count_fields(score) -> dict[str, float | int]:
    if score is None:
        return {
            "source_a_cluster_count": 0,
            "source_b_cluster_count": 0,
            "predicted_a_void_count": 0,
            "predicted_b_void_count": 0,
            "reference_a_void_count": 0,
            "reference_b_void_count": 0,
            "predicted_a_reference_fraction": np.nan,
            "predicted_b_reference_fraction": np.nan,
        }
    radius_result = score.vsf_score.radius_result
    return {
        "source_a_cluster_count": radius_result.source_a_cluster_count,
        "source_b_cluster_count": radius_result.source_b_cluster_count,
        "predicted_a_void_count": radius_result.score_a.size_score.predicted_void_count,
        "predicted_b_void_count": radius_result.score_b.size_score.predicted_void_count,
        "reference_a_void_count": radius_result.score_a.size_score.reference_void_count,
        "reference_b_void_count": radius_result.score_b.size_score.reference_void_count,
        "predicted_a_reference_fraction": (
            radius_result.score_a.size_score.predicted_reference_fraction
        ),
        "predicted_b_reference_fraction": (
            radius_result.score_b.size_score.predicted_reference_fraction
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_csv = _append_default_suffix(
        args.output_csv,
        vide_variant=args.vide_variant,
        position_mode=args.position_mode,
    )
    paths = N256FullMcmcPaths(
        catalog_a=args.catalog_a,
        catalog_b=args.catalog_b,
        vide_a=resolve_vide_catalog_variant_path(args.vide_a, args.vide_variant),
        vide_b=resolve_vide_catalog_variant_path(args.vide_b, args.vide_variant),
        vide_centers_a=resolve_vide_catalog_variant_path(args.vide_centers_a, args.vide_variant),
        vide_centers_b=resolve_vide_catalog_variant_path(args.vide_centers_b, args.vide_variant),
        vide_macrocenters_a=resolve_vide_catalog_variant_path(
            args.vide_macrocenters_a,
            args.vide_variant,
        ),
        vide_macrocenters_b=resolve_vide_catalog_variant_path(
            args.vide_macrocenters_b,
            args.vide_variant,
        ),
        vide_variant=args.vide_variant,
        position_mode=args.position_mode,
    )
    settings = FullMcmcSettings(
        reject_degenerate=not args.allow_degenerate,
        vsf_weight=args.vsf_weight,
        center_weight=args.center_weight,
        center_sigma=args.center_sigma,
        center_nu=args.center_nu,
        vide_center_kind=args.vide_center_kind,
    )
    posterior = N256FullLogPosterior(
        paths=paths,
        settings=settings,
        bounds=DEFAULT_BOUNDS,
    )
    sweep_values = (
        args.linking_factor_values,
        args.radius_a0_values,
        args.adjacency_factor_values,
        args.merge_threshold_values,
    )
    labels = tuple(SWEEP_PARAMETER_INDICES)
    base_theta = _base_theta(args)
    rows = []
    log_probabilities = []
    blob_rows = []
    for values in product(*sweep_values):
        theta = base_theta.copy()
        for label, value in zip(labels, values, strict=True):
            theta[SWEEP_PARAMETER_INDICES[label]] = float(value)
        log_probability, blob, score = posterior.evaluate_with_score(theta)
        row: dict[str, object] = {
            name: float(value) for name, value in zip(PARAMETER_NAMES, theta, strict=True)
        }
        row.update(
            {
                "position_mode": args.position_mode,
                "vide_variant": args.vide_variant,
                "log_probability": float(log_probability),
            }
        )
        row.update({name: float(blob[index]) for index, name in enumerate(BLOB_NAMES)})
        row.update(_score_count_fields(score))
        rows.append(row)
        log_probabilities.append(log_probability)
        blob_rows.append(blob)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    log_probability_array = np.asarray(log_probabilities, dtype=np.float64)
    blob_array = np.asarray(blob_rows, dtype=[(name, "f8") for name in BLOB_NAMES])
    summary = posterior_diagnostic_summary(
        log_probability_array,
        blob_array,
        reject_degenerate=settings.reject_degenerate,
    )
    print(format_diagnostic_summary("Initial-position grid diagnostic", summary))
    if np.any(np.isfinite(log_probability_array)):
        best = rows[int(np.nanargmax(log_probability_array))]
        print("Best finite grid point:")
        for name in PARAMETER_NAMES:
            print(f"  {name}: {float(best[name]):.8g}")
        print(f"  log_probability: {float(best['log_probability']):.8g}")
    print(f"Wrote grid diagnostics to {output_csv}")
    return 0 if summary["finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
