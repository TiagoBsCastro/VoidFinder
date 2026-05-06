#!/usr/bin/env python
"""Sample the n256 joint VSF plus center-match posterior with emcee."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pinocchio_voids.calibration import (
    mean_halo_spacing_mpc_h,
    score_paired_joint_calibration,
)
from pinocchio_voids.io import (
    VIDE_CATALOG_VARIANTS,
    read_paired_pinocchio_halo_catalogs,
    read_vide_void_desc,
    resolve_vide_catalog_variant_path,
    vide_catalog_variant_output_suffix,
)
from pinocchio_voids.voidfinder import PairedVoidFinderConfig, run_paired_halo_void_finder

try:
    from scripts.optimize_n256_vsf_mcmc import (
        DEFAULT_BOUNDS,
        DEFAULT_CATALOG_A,
        DEFAULT_CATALOG_B,
        DEFAULT_INITIAL_CENTER,
        DEFAULT_VIDE_A,
        DEFAULT_VIDE_B,
        PARAMETER_NAMES,
        VsfMcmcSettings,
        fixed_linear_edges,
        flatten_chain,
        initial_walker_positions,
        log_uniform_prior,
        summarize_samples,
        write_contour_plot,
        write_trace_plot,
    )
    from scripts.plot_n256_void_slice import (
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        load_vide_spatial_catalog,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from optimize_n256_vsf_mcmc import (
        DEFAULT_BOUNDS,
        DEFAULT_CATALOG_A,
        DEFAULT_CATALOG_B,
        DEFAULT_INITIAL_CENTER,
        DEFAULT_VIDE_A,
        DEFAULT_VIDE_B,
        PARAMETER_NAMES,
        VsfMcmcSettings,
        fixed_linear_edges,
        flatten_chain,
        initial_walker_positions,
        log_uniform_prior,
        summarize_samples,
        write_contour_plot,
        write_trace_plot,
    )
    from plot_n256_void_slice import (
        DEFAULT_VIDE_CENTERS_A,
        DEFAULT_VIDE_CENTERS_B,
        DEFAULT_VIDE_MACROCENTERS_A,
        DEFAULT_VIDE_MACROCENTERS_B,
        load_vide_spatial_catalog,
    )


BLOB_NAMES = (
    "log_prior",
    "vsf_log_likelihood",
    "center_log_likelihood",
    "weighted_vsf_log_likelihood",
    "weighted_center_log_likelihood",
    "target_a_median_distance_over_min_reff",
    "target_b_median_distance_over_min_reff",
    "target_a_fraction_distance_lt_min_reff",
    "target_b_fraction_distance_lt_min_reff",
    "is_degenerate",
)
BLOB_DTYPE = [(name, "f8") for name in BLOB_NAMES]


@dataclass(frozen=True)
class N256JointMcmcPaths:
    """Input catalogs for the n256 joint optimizer."""

    catalog_a: Path = DEFAULT_CATALOG_A
    catalog_b: Path = DEFAULT_CATALOG_B
    vide_a: Path = DEFAULT_VIDE_A
    vide_b: Path = DEFAULT_VIDE_B
    vide_centers_a: Path = DEFAULT_VIDE_CENTERS_A
    vide_centers_b: Path = DEFAULT_VIDE_CENTERS_B
    vide_macrocenters_a: Path = DEFAULT_VIDE_MACROCENTERS_A
    vide_macrocenters_b: Path = DEFAULT_VIDE_MACROCENTERS_B
    vide_variant: str = "all"


DEFAULT_OUTPUT_PREFIX = Path("runs/void-statistics/n256_joint_mcmc")


@dataclass(frozen=True)
class JointMcmcSettings(VsfMcmcSettings):
    """Fixed joint calibration settings for one MCMC run."""

    vsf_weight: float = 1.0
    center_weight: float = 1.0
    center_sigma: float = 1.0
    center_nu: float = 3.0
    center_radius_min_mpc_h: float | None = None
    center_radius_max_mpc_h: float | None = None
    vide_center_kind: str = "center"


def _invalid_blob(log_prior: float = -np.inf) -> tuple[float, ...]:
    return (
        float(log_prior),
        -np.inf,
        -np.inf,
        -np.inf,
        -np.inf,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        1.0,
    )


def _blob_from_score(*, log_prior: float, score) -> tuple[float, ...]:
    return (
        float(log_prior),
        float(score.vsf_score.total_log_likelihood),
        float(score.center_score.total_log_likelihood),
        float(score.weighted_vsf_log_likelihood),
        float(score.weighted_center_log_likelihood),
        float(score.center_score.score_a.median_distance_over_min_reff),
        float(score.center_score.score_b.median_distance_over_min_reff),
        float(score.center_score.score_a.fraction_distance_lt_min_reff),
        float(score.center_score.score_b.fraction_distance_lt_min_reff),
        1.0 if score.is_degenerate else 0.0,
    )


class N256JointLogPosterior:
    """Callable posterior for joint n256 paired VSF and center calibration."""

    def __init__(
        self,
        *,
        paths: N256JointMcmcPaths,
        settings: JointMcmcSettings,
        bounds: NDArray[np.float64],
    ) -> None:
        self.paths = paths
        self.settings = settings
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.paired = read_paired_pinocchio_halo_catalogs(
            paths.catalog_a,
            paths.catalog_b,
            box_size_mpc_h=settings.box_size_mpc_h,
        )
        self.reference_a = read_vide_void_desc(paths.vide_a)
        self.reference_b = read_vide_void_desc(paths.vide_b)
        self.spatial_a = load_vide_spatial_catalog(
            desc_path=paths.vide_a,
            centers_path=paths.vide_centers_a,
            macrocenters_path=paths.vide_macrocenters_a,
            center_kind=settings.vide_center_kind,
        )
        self.spatial_b = load_vide_spatial_catalog(
            desc_path=paths.vide_b,
            centers_path=paths.vide_centers_b,
            macrocenters_path=paths.vide_macrocenters_b,
            center_kind=settings.vide_center_kind,
        )
        self.bin_edges = fixed_linear_edges(
            bins=settings.bins,
            lower=settings.bin_min_mpc_h,
            upper=settings.bin_max_mpc_h,
        )
        self.mean_spacing_a = mean_halo_spacing_mpc_h(self.paired.catalog_a)
        self.mean_spacing_b = mean_halo_spacing_mpc_h(self.paired.catalog_b)

    def evaluate(self, theta: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        prior = log_uniform_prior(theta, self.bounds)
        if not np.isfinite(prior):
            return -np.inf, _invalid_blob(prior)

        linking_factor, radius_a0, radius_alpha, adjacency_factor = (
            float(value) for value in theta
        )
        config = PairedVoidFinderConfig(
            linking_length_mpc_h=linking_factor * self.mean_spacing_a,
            source_b_linking_length_mpc_h=linking_factor * self.mean_spacing_b,
            min_cluster_members=self.settings.min_cluster_members,
            min_cluster_mass_msun_h=self.settings.min_cluster_mass_msun_h,
            reference_rho_bar_msun_h_mpc3=self.settings.reference_rho_bar_msun_h_mpc3,
            radius_a0=radius_a0,
            radius_alpha=radius_alpha,
            adjacency_factor=adjacency_factor,
        )
        result = run_paired_halo_void_finder(
            self.paired.catalog_a,
            self.paired.catalog_b,
            config=config,
        )
        center_min = (
            self.settings.bin_min_mpc_h
            if self.settings.center_radius_min_mpc_h is None
            else self.settings.center_radius_min_mpc_h
        )
        center_max = (
            self.settings.bin_max_mpc_h
            if self.settings.center_radius_max_mpc_h is None
            else self.settings.center_radius_max_mpc_h
        )
        score = score_paired_joint_calibration(
            result,
            reference_a=self.reference_a,
            reference_b=self.reference_b,
            reference_positions_a_mpc_h=self.spatial_a.positions_mpc_h,
            reference_radii_a_mpc_h=self.spatial_a.radii_mpc_h,
            reference_positions_b_mpc_h=self.spatial_b.positions_mpc_h,
            reference_radii_b_mpc_h=self.spatial_b.radii_mpc_h,
            box_size_mpc_h=self.settings.box_size_mpc_h,
            bins=self.bin_edges,
            radius_min_mpc_h=self.settings.bin_min_mpc_h,
            radius_max_mpc_h=self.settings.bin_max_mpc_h,
            config=config,
            linking_mode="mean_spacing",
            linking_value=linking_factor,
            source_a_linking_length_mpc_h=config.linking_length_mpc_h,
            source_b_linking_length_mpc_h=config.source_b_linking_length_mpc_h,
            count_floor=self.settings.count_floor,
            min_predicted_fraction=self.settings.min_predicted_fraction,
            center_radius_min_mpc_h=center_min,
            center_radius_max_mpc_h=center_max,
            center_sigma=self.settings.center_sigma,
            center_nu=self.settings.center_nu,
            vsf_weight=self.settings.vsf_weight,
            center_weight=self.settings.center_weight,
        )
        blob = _blob_from_score(log_prior=prior, score=score)
        if self.settings.reject_degenerate and score.is_degenerate:
            return -np.inf, blob
        return prior + score.total_log_likelihood, blob

    def __call__(self, theta: Sequence[float]) -> tuple[float, ...]:
        value, blob = self.evaluate(theta)
        return (value, *blob)


def flatten_blobs(
    blobs: NDArray,
    *,
    burn_in: int,
    thin: int,
) -> NDArray:
    if thin < 1:
        raise ValueError("thin must be at least 1")
    if burn_in < 0 or burn_in >= blobs.shape[0]:
        raise ValueError("burn_in must be non-negative and smaller than the chain length")
    return blobs[burn_in::thin].reshape(-1)


def write_joint_samples_csv(
    path: Path,
    samples: NDArray[np.float64],
    log_probability: NDArray[np.float64],
    blobs: NDArray,
) -> None:
    rows = []
    for index, (sample, log_prob, blob) in enumerate(
        zip(samples, log_probability, blobs, strict=True)
    ):
        row: dict[str, object] = {"sample": index, "log_probability": float(log_prob)}
        row.update(
            {name: float(value) for name, value in zip(PARAMETER_NAMES, sample, strict=True)}
        )
        row.update({name: float(blob[name]) for name in BLOB_NAMES})
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_joint_summary_csv(
    path: Path,
    *,
    best_fit: NDArray[np.float64],
    best_log_probability: float,
    best_blob,
    percentiles: NDArray[np.float64],
) -> None:
    rows = []
    for column, name in enumerate(PARAMETER_NAMES):
        row = {
            "parameter": name,
            "best_fit": float(best_fit[column]),
            "p16": float(percentiles[0, column]),
            "p50": float(percentiles[1, column]),
            "p84": float(percentiles[2, column]),
            "best_log_probability": float(best_log_probability),
        }
        row.update({f"best_{blob_name}": float(best_blob[blob_name]) for blob_name in BLOB_NAMES})
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_joint_best_fit_command(
    path: Path,
    *,
    best_fit: NDArray[np.float64],
    vide_center_kind: str,
    paths: N256JointMcmcPaths,
) -> None:
    linking_factor, radius_a0, radius_alpha, adjacency_factor = (
        float(value) for value in best_fit
    )
    suffix = vide_catalog_variant_output_suffix(paths.vide_variant)
    command = f"""#!/usr/bin/env bash
set -euo pipefail

/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/compare_void_size_functions.py \\
  {paths.catalog_a} \\
  {paths.catalog_b} \\
  {paths.vide_a} \\
  {paths.vide_b} \\
  --box-size 256 \\
  --rho-bar 8.63025e10 \\
  --linking-factor {linking_factor:.8g} \\
  --min-cluster-members 2 \\
  --min-cluster-mass 0 \\
  --radius-a0 {radius_a0:.8g} \\
  --radius-alpha {radius_alpha:.8g} \\
  --adjacency-factor {adjacency_factor:.8g} \\
  --bins 17 \\
  --binning linear \\
  --bin-min 10 \\
  --bin-max 80 \\
  --output-csv runs/void-statistics/n256_joint_best_fit_paper_bins_vsf{suffix}.csv \\
  --summary-csv runs/void-statistics/n256_joint_best_fit_paper_bins_vsf{suffix}_summary.csv \\
  --output-plot runs/void-statistics/n256_joint_best_fit_paper_bins_vsf{suffix}.png

/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/match_n256_void_centers.py \\
  --vide-desc-a {paths.vide_a} \\
  --vide-desc-b {paths.vide_b} \\
  --vide-centers-a {paths.vide_centers_a} \\
  --vide-centers-b {paths.vide_centers_b} \\
  --vide-macrocenters-a {paths.vide_macrocenters_a} \\
  --vide-macrocenters-b {paths.vide_macrocenters_b} \\
  --vide-variant {paths.vide_variant} \\
  --linking-factor {linking_factor:.8g} \\
  --radius-a0 {radius_a0:.8g} \\
  --radius-alpha {radius_alpha:.8g} \\
  --adjacency-factor {adjacency_factor:.8g} \\
  --vide-center-kind {vide_center_kind} \\
  --output-csv runs/void-statistics/n256_joint_best_fit_center_matches{suffix}.csv \\
  --summary-csv runs/void-statistics/n256_joint_best_fit_center_match_summary{suffix}.csv
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(command, encoding="utf-8")
    path.chmod(0o755)


def _parse_vector(values: Sequence[float] | None, default: NDArray[np.float64]) -> NDArray[np.float64]:
    if values is None:
        return default.copy()
    array = np.asarray(values, dtype=np.float64)
    if array.shape != default.shape:
        raise SystemExit(f"Expected {default.size} values")
    return array


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emcee sampling for joint n256 VSF plus center calibration."
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
        "--vide-variant",
        choices=VIDE_CATALOG_VARIANTS,
        default="all",
        help="VIDE catalog variant used for calibration.",
    )
    parser.add_argument(
        "--vide-center-kind",
        choices=("center", "macrocenter"),
        default="center",
    )
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--vsf-weight", type=float, default=1.0)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--center-sigma", type=float, default=1.0)
    parser.add_argument("--center-nu", type=float, default=3.0)
    parser.add_argument("--center-radius-min", type=float)
    parser.add_argument("--center-radius-max", type=float)
    parser.add_argument(
        "--initial-center",
        type=float,
        nargs=len(PARAMETER_NAMES),
        metavar=tuple(name.upper() for name in PARAMETER_NAMES),
        help="Initial center: linking_factor radius_a0 radius_alpha adjacency_factor.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
    )
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Keep degenerate underprediction samples instead of rejecting them.",
    )
    return parser.parse_args(argv)


def run_sampler(
    *,
    log_probability,
    initial_positions: NDArray[np.float64],
    steps: int,
    processes: int,
):
    try:
        import emcee
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "emcee is required. Run: "
            "/home/tcastro/miniforge3/envs/voidfinder/bin/python -m pip install -e \".[dev]\""
        ) from exc

    if steps < 1:
        raise ValueError("steps must be at least 1")
    walkers, ndim = initial_positions.shape
    if processes > 1:
        with Pool(processes=processes) as pool:
            sampler = emcee.EnsembleSampler(
                walkers,
                ndim,
                log_probability,
                pool=pool,
                blobs_dtype=BLOB_DTYPE,
            )
            sampler.run_mcmc(initial_positions, steps, progress=True)
            return sampler.get_chain(), sampler.get_log_prob(), sampler.get_blobs()
    sampler = emcee.EnsembleSampler(
        walkers,
        ndim,
        log_probability,
        blobs_dtype=BLOB_DTYPE,
    )
    sampler.run_mcmc(initial_positions, steps, progress=True)
    return sampler.get_chain(), sampler.get_log_prob(), sampler.get_blobs()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps <= args.burn_in:
        raise SystemExit("--steps must be larger than --burn-in")
    suffix = vide_catalog_variant_output_suffix(args.vide_variant)
    if suffix and args.output_prefix == DEFAULT_OUTPUT_PREFIX:
        args.output_prefix = args.output_prefix.with_name(f"{args.output_prefix.name}{suffix}")
    rng = np.random.default_rng(args.seed)
    center = _parse_vector(args.initial_center, DEFAULT_INITIAL_CENTER)
    paths = N256JointMcmcPaths(
        catalog_a=args.catalog_a,
        catalog_b=args.catalog_b,
        vide_a=resolve_vide_catalog_variant_path(args.vide_a, args.vide_variant),
        vide_b=resolve_vide_catalog_variant_path(args.vide_b, args.vide_variant),
        vide_centers_a=resolve_vide_catalog_variant_path(
            args.vide_centers_a,
            args.vide_variant,
        ),
        vide_centers_b=resolve_vide_catalog_variant_path(
            args.vide_centers_b,
            args.vide_variant,
        ),
        vide_macrocenters_a=resolve_vide_catalog_variant_path(
            args.vide_macrocenters_a,
            args.vide_variant,
        ),
        vide_macrocenters_b=resolve_vide_catalog_variant_path(
            args.vide_macrocenters_b,
            args.vide_variant,
        ),
        vide_variant=args.vide_variant,
    )
    settings = JointMcmcSettings(
        reject_degenerate=not args.allow_degenerate,
        vsf_weight=args.vsf_weight,
        center_weight=args.center_weight,
        center_sigma=args.center_sigma,
        center_nu=args.center_nu,
        center_radius_min_mpc_h=args.center_radius_min,
        center_radius_max_mpc_h=args.center_radius_max,
        vide_center_kind=args.vide_center_kind,
    )
    initial_positions = initial_walker_positions(
        center=center,
        bounds=DEFAULT_BOUNDS,
        walkers=args.walkers,
        rng=rng,
    )
    log_probability = N256JointLogPosterior(
        paths=paths,
        settings=settings,
        bounds=DEFAULT_BOUNDS,
    )
    chain, log_prob, blobs = run_sampler(
        log_probability=log_probability,
        initial_positions=initial_positions,
        steps=args.steps,
        processes=args.processes,
    )
    samples, sample_log_prob = flatten_chain(
        chain,
        log_prob,
        burn_in=args.burn_in,
        thin=args.thin,
    )
    sample_blobs = flatten_blobs(blobs, burn_in=args.burn_in, thin=args.thin)
    best_fit, best_log_probability, percentiles = summarize_samples(samples, sample_log_prob)
    finite = np.isfinite(sample_log_prob)
    best_index = int(np.argmax(sample_log_prob[finite]))
    best_blob = sample_blobs[finite][best_index]

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prefix.with_name(prefix.name + "_chain.npz"),
        chain=chain,
        log_probability=log_prob,
        blobs=blobs,
        blob_names=np.asarray(BLOB_NAMES),
        parameter_names=np.asarray(PARAMETER_NAMES),
        bounds=DEFAULT_BOUNDS,
        initial_center=center,
        initial_positions=initial_positions,
        seed=args.seed,
        burn_in=args.burn_in,
        thin=args.thin,
        vsf_weight=args.vsf_weight,
        center_weight=args.center_weight,
        center_sigma=args.center_sigma,
        center_nu=args.center_nu,
        vide_center_kind=args.vide_center_kind,
        vide_variant=args.vide_variant,
        vide_a=str(paths.vide_a),
        vide_b=str(paths.vide_b),
        vide_centers_a=str(paths.vide_centers_a),
        vide_centers_b=str(paths.vide_centers_b),
        vide_macrocenters_a=str(paths.vide_macrocenters_a),
        vide_macrocenters_b=str(paths.vide_macrocenters_b),
    )
    write_joint_samples_csv(
        prefix.with_name(prefix.name + "_samples.csv"),
        samples,
        sample_log_prob,
        sample_blobs,
    )
    write_joint_summary_csv(
        prefix.with_name(prefix.name + "_summary.csv"),
        best_fit=best_fit,
        best_log_probability=best_log_probability,
        best_blob=best_blob,
        percentiles=percentiles,
    )
    write_trace_plot(prefix.with_name(prefix.name + "_trace.png"), chain)
    write_contour_plot(
        prefix.with_name(prefix.name + "_contours.png"),
        samples[np.isfinite(sample_log_prob)],
        best_fit=best_fit,
    )
    write_joint_best_fit_command(
        prefix.with_name(prefix.name + "_best_fit_command.sh"),
        best_fit=best_fit,
        vide_center_kind=args.vide_center_kind,
        paths=paths,
    )

    print("Best fit:")
    for name, value in zip(PARAMETER_NAMES, best_fit, strict=True):
        print(f"  {name}: {value:.8g}")
    print(f"  log_probability: {best_log_probability:.8g}")
    print(f"  vsf_log_likelihood: {float(best_blob['vsf_log_likelihood']):.8g}")
    print(f"  center_log_likelihood: {float(best_blob['center_log_likelihood']):.8g}")
    print(f"  vide_variant: {args.vide_variant}")
    print(f"Wrote products with prefix {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
