#!/usr/bin/env python
"""Sample the n256 VSF calibration posterior with emcee."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pinocchio_voids.calibration import (
    mean_halo_spacing_mpc_h,
    score_paired_vsf_likelihood,
)
from pinocchio_voids.io import (
    PINOCCHIO_POSITION_MODES,
    VIDE_CATALOG_VARIANTS,
    pinocchio_position_mode_output_suffix,
    read_paired_pinocchio_halo_catalogs,
    read_vide_void_desc,
    resolve_vide_catalog_variant_path,
    vide_catalog_variant_output_suffix,
)
from pinocchio_voids.voidfinder import PairedVoidFinderConfig, run_paired_halo_void_finder


PARAMETER_NAMES = ("linking_factor", "radius_a0", "radius_alpha", "adjacency_factor")
DEFAULT_BOUNDS = np.array(
    [
        [0.10, 0.17],
        [4.0, 8.0],
        [0.85, 1.20],
        [0.15, 0.70],
    ],
    dtype=np.float64,
)
DEFAULT_INITIAL_CENTER = np.array([0.13, 5.5, 1.05, 0.40], dtype=np.float64)
DEFAULT_CATALOG_A = Path(
    "runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out"
)
DEFAULT_CATALOG_B = Path(
    "runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out"
)
DEFAULT_VIDE_A = Path(
    "runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/"
    "sample_pinocchio_n256_ss1.0_z0.00_d00/"
    "voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out"
)
DEFAULT_VIDE_B = Path(
    "runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/"
    "sample_pinocchio_n256_paired_ss1.0_z0.00_d00/"
    "voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out"
)
DEFAULT_OUTPUT_PREFIX = Path("runs/void-statistics/n256_vsf_mcmc")


@dataclass(frozen=True)
class N256McmcPaths:
    """Input catalogs for the n256 optimizer."""

    catalog_a: Path = DEFAULT_CATALOG_A
    catalog_b: Path = DEFAULT_CATALOG_B
    vide_a: Path = DEFAULT_VIDE_A
    vide_b: Path = DEFAULT_VIDE_B
    vide_variant: str = "all"
    position_mode: str = "final"


@dataclass(frozen=True)
class VsfMcmcSettings:
    """Fixed VSF calibration settings for one MCMC run."""

    box_size_mpc_h: float = 256.0
    reference_rho_bar_msun_h_mpc3: float = 8.63025e10
    bins: int = 17
    bin_min_mpc_h: float = 10.0
    bin_max_mpc_h: float = 80.0
    count_floor: float = 0.5
    min_predicted_fraction: float = 0.10
    min_cluster_members: int = 2
    min_cluster_mass_msun_h: float = 0.0
    reject_degenerate: bool = True


def fixed_linear_edges(*, bins: int, lower: float, upper: float) -> NDArray[np.float64]:
    """Return fixed linear radius-bin edges."""

    if bins < 1:
        raise ValueError("bins must be at least 1")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise ValueError("radius bin edges must be positive, finite, and increasing")
    return np.linspace(float(lower), float(upper), int(bins) + 1)


def log_uniform_prior(theta: Sequence[float], bounds: NDArray[np.float64]) -> float:
    """Uniform prior over finite parameter bounds."""

    values = np.asarray(theta, dtype=np.float64)
    limits = np.asarray(bounds, dtype=np.float64)
    if values.shape != (len(PARAMETER_NAMES),):
        return -np.inf
    if limits.shape != (len(PARAMETER_NAMES), 2):
        raise ValueError("bounds must have shape (4, 2)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(limits)):
        return -np.inf
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("bounds must be increasing")
    inside = np.all((values >= limits[:, 0]) & (values <= limits[:, 1]))
    return 0.0 if inside else -np.inf


def initial_walker_positions(
    *,
    center: Sequence[float],
    bounds: NDArray[np.float64],
    walkers: int,
    rng: np.random.Generator,
    width_fraction: float = 0.02,
) -> NDArray[np.float64]:
    """Initialize walkers near the requested center while respecting bounds."""

    center_values = np.asarray(center, dtype=np.float64)
    limits = np.asarray(bounds, dtype=np.float64)
    ndim = len(PARAMETER_NAMES)
    if walkers < 2 * ndim:
        raise ValueError("walkers must be at least twice the parameter dimension")
    if log_uniform_prior(center_values, limits) != 0.0:
        raise ValueError("initial center must lie inside the parameter bounds")
    if not np.isfinite(width_fraction) or width_fraction <= 0.0:
        raise ValueError("width_fraction must be positive and finite")

    widths = width_fraction * (limits[:, 1] - limits[:, 0])
    positions = np.empty((walkers, ndim), dtype=np.float64)
    for walker in range(walkers):
        for _ in range(1000):
            candidate = center_values + rng.normal(scale=widths, size=ndim)
            if log_uniform_prior(candidate, limits) == 0.0:
                positions[walker] = candidate
                break
        else:
            raise RuntimeError("could not initialize walkers inside bounds")
    return positions


class N256VsfLogPosterior:
    """Callable posterior for the n256 paired VSF calibration."""

    def __init__(
        self,
        *,
        paths: N256McmcPaths,
        settings: VsfMcmcSettings,
        bounds: NDArray[np.float64],
    ) -> None:
        self.paths = paths
        self.settings = settings
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.paired = read_paired_pinocchio_halo_catalogs(
            paths.catalog_a,
            paths.catalog_b,
            box_size_mpc_h=settings.box_size_mpc_h,
            position_mode=paths.position_mode,
        )
        self.reference_a = read_vide_void_desc(paths.vide_a)
        self.reference_b = read_vide_void_desc(paths.vide_b)
        self.bin_edges = fixed_linear_edges(
            bins=settings.bins,
            lower=settings.bin_min_mpc_h,
            upper=settings.bin_max_mpc_h,
        )
        self.mean_spacing_a = mean_halo_spacing_mpc_h(self.paired.catalog_a)
        self.mean_spacing_b = mean_halo_spacing_mpc_h(self.paired.catalog_b)

    def __call__(self, theta: Sequence[float]) -> float:
        prior = log_uniform_prior(theta, self.bounds)
        if not np.isfinite(prior):
            return -np.inf

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
        score = score_paired_vsf_likelihood(
            result,
            reference_a=self.reference_a,
            reference_b=self.reference_b,
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
        )
        if self.settings.reject_degenerate and score.radius_result.is_degenerate:
            return -np.inf
        return prior + score.total_log_likelihood


def flatten_chain(
    chain: NDArray[np.float64],
    log_probability: NDArray[np.float64],
    *,
    burn_in: int,
    thin: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return post-burn-in flattened samples and log probabilities."""

    if thin < 1:
        raise ValueError("thin must be at least 1")
    if burn_in < 0 or burn_in >= chain.shape[0]:
        raise ValueError("burn_in must be non-negative and smaller than the chain length")
    thinned_chain = chain[burn_in::thin]
    thinned_log_probability = log_probability[burn_in::thin]
    return (
        thinned_chain.reshape((-1, chain.shape[-1])),
        thinned_log_probability.reshape(-1),
    )


def summarize_samples(
    samples: NDArray[np.float64],
    log_probability: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Return best-fit vector, best log probability, and 16/50/84 percentiles."""

    if samples.ndim != 2 or samples.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("samples must have shape (n, 4)")
    if log_probability.ndim != 1 or log_probability.shape[0] != samples.shape[0]:
        raise ValueError("log_probability must match sample count")
    finite = np.isfinite(log_probability)
    if not np.any(finite):
        raise ValueError("at least one finite sample is required")
    finite_samples = samples[finite]
    finite_log_probability = log_probability[finite]
    best_index = int(np.argmax(finite_log_probability))
    percentiles = np.percentile(finite_samples, [16.0, 50.0, 84.0], axis=0)
    return (
        finite_samples[best_index],
        float(finite_log_probability[best_index]),
        percentiles,
    )


def credible_density_levels(
    density: NDArray[np.float64],
    masses: Sequence[float] = (0.95, 0.68),
) -> list[float]:
    """Return ascending density thresholds enclosing the requested masses."""

    values = np.asarray(density, dtype=np.float64)
    if values.ndim != 2 or not np.any(values > 0.0):
        return []
    flattened = np.sort(values.ravel())[::-1]
    cumulative = np.cumsum(flattened)
    cumulative /= cumulative[-1]
    levels = []
    for mass in masses:
        if not np.isfinite(mass) or mass <= 0.0 or mass >= 1.0:
            raise ValueError("credible masses must be between 0 and 1")
        index = min(int(np.searchsorted(cumulative, mass, side="left")), flattened.size - 1)
        levels.append(float(flattened[index]))
    return sorted(set(levels))


def write_samples_csv(
    path: Path,
    samples: NDArray[np.float64],
    log_probability: NDArray[np.float64],
    *,
    position_mode: str,
) -> None:
    rows = []
    for index, (sample, log_prob) in enumerate(zip(samples, log_probability, strict=True)):
        row: dict[str, object] = {
            "sample": index,
            "position_mode": position_mode,
            "log_probability": float(log_prob),
        }
        row.update(
            {name: float(value) for name, value in zip(PARAMETER_NAMES, sample, strict=True)}
        )
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    path: Path,
    *,
    best_fit: NDArray[np.float64],
    best_log_probability: float,
    percentiles: NDArray[np.float64],
    position_mode: str,
) -> None:
    rows = []
    for column, name in enumerate(PARAMETER_NAMES):
        rows.append(
            {
                "parameter": name,
                "position_mode": position_mode,
                "best_fit": float(best_fit[column]),
                "p16": float(percentiles[0, column]),
                "p50": float(percentiles[1, column]),
                "p84": float(percentiles[2, column]),
                "best_log_probability": float(best_log_probability),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_trace_plot(
    path: Path,
    chain: NDArray[np.float64],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(chain.shape[0])
    fig, axes = plt.subplots(
        len(PARAMETER_NAMES),
        1,
        figsize=(8.0, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    for axis, name, values in zip(axes, PARAMETER_NAMES, np.moveaxis(chain, -1, 0), strict=True):
        axis.plot(steps, values, color="tab:blue", alpha=0.25, linewidth=0.8)
        axis.set_ylabel(name)
    axes[-1].set_xlabel("step")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_contour_plot(
    path: Path,
    samples: NDArray[np.float64],
    *,
    best_fit: NDArray[np.float64],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ndim = len(PARAMETER_NAMES)
    fig, axes = plt.subplots(
        ndim,
        ndim,
        figsize=(9.0, 9.0),
        constrained_layout=True,
    )
    for row in range(ndim):
        for column in range(ndim):
            axis = axes[row, column]
            if row < column:
                axis.axis("off")
                continue
            if row == column:
                axis.hist(samples[:, column], bins=40, histtype="step", color="black")
                axis.axvline(best_fit[column], color="tab:red", linewidth=1.2)
            else:
                hist, x_edges, y_edges = np.histogram2d(
                    samples[:, column],
                    samples[:, row],
                    bins=40,
                )
                levels = credible_density_levels(hist)
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                axis.pcolormesh(
                    x_edges,
                    y_edges,
                    hist.T,
                    cmap="Blues",
                    shading="auto",
                    alpha=0.55,
                )
                if len(levels) >= 2:
                    axis.contour(
                        x_centers,
                        y_centers,
                        hist.T,
                        levels=levels,
                        colors=("tab:blue", "black")[: len(levels)],
                        linewidths=1.0,
                    )
                axis.plot(best_fit[column], best_fit[row], "o", color="tab:red", markersize=3)
            if row == ndim - 1:
                axis.set_xlabel(PARAMETER_NAMES[column])
            else:
                axis.set_xticklabels([])
            if column == 0 and row > 0:
                axis.set_ylabel(PARAMETER_NAMES[row])
            elif column != 0:
                axis.set_yticklabels([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_best_fit_command(
    path: Path,
    *,
    best_fit: NDArray[np.float64],
    paths: N256McmcPaths,
) -> None:
    linking_factor, radius_a0, radius_alpha, adjacency_factor = (
        float(value) for value in best_fit
    )
    suffix = (
        f"{vide_catalog_variant_output_suffix(paths.vide_variant)}"
        f"{pinocchio_position_mode_output_suffix(paths.position_mode)}"
    )
    command = f"""#!/usr/bin/env bash
set -euo pipefail

/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/compare_void_size_functions.py \\
  {paths.catalog_a} \\
  {paths.catalog_b} \\
  {paths.vide_a} \\
  {paths.vide_b} \\
  --box-size 256 \\
  --position-mode {paths.position_mode} \\
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
  --output-csv runs/void-statistics/n256_mcmc_best_fit_paper_bins_vsf{suffix}.csv \\
  --summary-csv runs/void-statistics/n256_mcmc_best_fit_paper_bins_vsf{suffix}_summary.csv \\
  --output-plot runs/void-statistics/n256_mcmc_best_fit_paper_bins_vsf{suffix}.png
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
        description="Run emcee posterior sampling for the n256 paired VSF calibration."
    )
    parser.add_argument("--catalog-a", type=Path, default=DEFAULT_CATALOG_A)
    parser.add_argument("--catalog-b", type=Path, default=DEFAULT_CATALOG_B)
    parser.add_argument("--vide-a", type=Path, default=DEFAULT_VIDE_A)
    parser.add_argument("--vide-b", type=Path, default=DEFAULT_VIDE_B)
    parser.add_argument(
        "--position-mode",
        choices=PINOCCHIO_POSITION_MODES,
        default="final",
        help="PINOCCHIO coordinate columns used by the finder.",
    )
    parser.add_argument(
        "--vide-variant",
        choices=VIDE_CATALOG_VARIANTS,
        default="all",
        help="VIDE catalog variant used for calibration.",
    )
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
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
            sampler = emcee.EnsembleSampler(walkers, ndim, log_probability, pool=pool)
            sampler.run_mcmc(initial_positions, steps, progress=True)
            return sampler.get_chain(), sampler.get_log_prob()
    sampler = emcee.EnsembleSampler(walkers, ndim, log_probability)
    sampler.run_mcmc(initial_positions, steps, progress=True)
    return sampler.get_chain(), sampler.get_log_prob()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps <= args.burn_in:
        raise SystemExit("--steps must be larger than --burn-in")
    suffix = (
        f"{vide_catalog_variant_output_suffix(args.vide_variant)}"
        f"{pinocchio_position_mode_output_suffix(args.position_mode)}"
    )
    if suffix and args.output_prefix == DEFAULT_OUTPUT_PREFIX:
        args.output_prefix = args.output_prefix.with_name(f"{args.output_prefix.name}{suffix}")
    rng = np.random.default_rng(args.seed)
    center = _parse_vector(args.initial_center, DEFAULT_INITIAL_CENTER)
    paths = N256McmcPaths(
        catalog_a=args.catalog_a,
        catalog_b=args.catalog_b,
        vide_a=resolve_vide_catalog_variant_path(args.vide_a, args.vide_variant),
        vide_b=resolve_vide_catalog_variant_path(args.vide_b, args.vide_variant),
        vide_variant=args.vide_variant,
        position_mode=args.position_mode,
    )
    settings = VsfMcmcSettings(reject_degenerate=not args.allow_degenerate)
    initial_positions = initial_walker_positions(
        center=center,
        bounds=DEFAULT_BOUNDS,
        walkers=args.walkers,
        rng=rng,
    )
    log_probability = N256VsfLogPosterior(
        paths=paths,
        settings=settings,
        bounds=DEFAULT_BOUNDS,
    )
    chain, log_prob = run_sampler(
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
    best_fit, best_log_probability, percentiles = summarize_samples(samples, sample_log_prob)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prefix.with_name(prefix.name + "_chain.npz"),
        chain=chain,
        log_probability=log_prob,
        parameter_names=np.asarray(PARAMETER_NAMES),
        bounds=DEFAULT_BOUNDS,
        initial_center=center,
        initial_positions=initial_positions,
        seed=args.seed,
        burn_in=args.burn_in,
        thin=args.thin,
        vide_variant=args.vide_variant,
        position_mode=args.position_mode,
        vide_a=str(paths.vide_a),
        vide_b=str(paths.vide_b),
    )
    write_samples_csv(
        prefix.with_name(prefix.name + "_samples.csv"),
        samples,
        sample_log_prob,
        position_mode=args.position_mode,
    )
    write_summary_csv(
        prefix.with_name(prefix.name + "_summary.csv"),
        best_fit=best_fit,
        best_log_probability=best_log_probability,
        percentiles=percentiles,
        position_mode=args.position_mode,
    )
    write_trace_plot(prefix.with_name(prefix.name + "_trace.png"), chain)
    write_contour_plot(
        prefix.with_name(prefix.name + "_contours.png"),
        samples[np.isfinite(sample_log_prob)],
        best_fit=best_fit,
    )
    write_best_fit_command(
        prefix.with_name(prefix.name + "_best_fit_command.sh"),
        best_fit=best_fit,
        paths=paths,
    )

    print("Best fit:")
    for name, value in zip(PARAMETER_NAMES, best_fit, strict=True):
        print(f"  {name}: {value:.8g}")
    print(f"  log_probability: {best_log_probability:.8g}")
    print(f"  vide_variant: {args.vide_variant}")
    print(f"  position_mode: {args.position_mode}")
    print(f"Wrote products with prefix {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
