"""Calibration helpers for the geometry-only paired-halo prototype."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike

from pinocchio_voids.catalog import HaloCatalog
from pinocchio_voids.evaluation import (
    VoidSizeFunctionComparison,
    compare_void_size_functions,
)
from pinocchio_voids.io.vide import VideVoidCatalog
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    MergedRadiusMode,
    PairedVoidFinderConfig,
    PairedVoidFinderResult,
    SourceClusterCatalog,
    build_protovoid_adjacency,
    find_source_clusters,
    merge_protovoids,
    run_paired_halo_void_finder,
    source_clusters_to_protovoids,
)


class CalibrationError(ValueError):
    """Raised when calibration inputs or parameters are invalid."""


@dataclass(frozen=True)
class DirectionalGeometryScore:
    """Size-function score for one target direction."""

    target_label: str
    predicted_void_count: int
    reference_void_count: int
    predicted_reference_fraction: float
    count_l1_difference: int
    guarded_count_l1_difference: int
    density_l1_difference: float
    is_degenerate_underprediction: bool
    size_function: VoidSizeFunctionComparison


@dataclass(frozen=True)
class PairedGeometrySweepResult:
    """One geometry-only parameter combination and its paired score."""

    config: PairedVoidFinderConfig
    linking_mode: Literal["fixed", "mean_spacing"]
    linking_value: float
    source_a_linking_length_mpc_h: float
    source_b_linking_length_mpc_h: float
    score_a: DirectionalGeometryScore
    score_b: DirectionalGeometryScore
    total_count_l1_difference: int
    total_guarded_count_l1_difference: int
    total_density_l1_difference: float
    is_degenerate_underprediction: bool


@dataclass(frozen=True)
class RadiusSummary:
    """Summary statistics for one set of effective void radii."""

    count: int
    count_in_range: int
    min_mpc_h: float
    p10_mpc_h: float
    median_mpc_h: float
    p90_mpc_h: float
    max_mpc_h: float


@dataclass(frozen=True)
class DirectionalRadiusCalibrationScore:
    """Paper-bin radius-scale score for one target direction."""

    target_label: str
    size_score: DirectionalGeometryScore
    finder_summary: RadiusSummary
    reference_summary: RadiusSummary
    median_radius_abs_error_mpc_h: float
    in_bin_count_abs_error: int
    is_zero_in_bin: bool
    is_degenerate: bool


@dataclass(frozen=True)
class PairedRadiusCalibrationResult:
    """One cached radius-scale calibration row."""

    config: PairedVoidFinderConfig
    linking_mode: Literal["fixed", "mean_spacing"]
    linking_value: float
    source_a_linking_length_mpc_h: float
    source_b_linking_length_mpc_h: float
    source_a_cluster_count: int
    source_b_cluster_count: int
    protovoid_a_count: int
    protovoid_b_count: int
    edge_a_count: int
    edge_b_count: int
    score_a: DirectionalRadiusCalibrationScore
    score_b: DirectionalRadiusCalibrationScore
    total_count_l1_difference: int
    total_density_l1_difference: float
    total_median_radius_abs_error_mpc_h: float
    total_in_bin_count_abs_error: int
    is_degenerate: bool


@dataclass(frozen=True)
class _LinkingLengthCandidate:
    mode: Literal["fixed", "mean_spacing"]
    value: float
    source_a_length_mpc_h: float
    source_b_length_mpc_h: float


def _positive_values(
    name: str,
    values: Iterable[float],
    *,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    numbers = tuple(float(value) for value in values)
    if not numbers and not allow_empty:
        raise CalibrationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(numbers)) or any(value <= 0.0 for value in numbers):
        raise CalibrationError(f"{name} values must be positive and finite")
    return numbers


def _min_predicted_fraction(value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise CalibrationError("min_predicted_fraction must be between 0 and 1")
    return number


def _non_negative_values(
    name: str,
    values: Iterable[float],
    *,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    numbers = tuple(float(value) for value in values)
    if not numbers and not allow_empty:
        raise CalibrationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(numbers)) or any(value < 0.0 for value in numbers):
        raise CalibrationError(f"{name} values must be non-negative and finite")
    return numbers


def _positive_int_values(name: str, values: Iterable[int]) -> tuple[int, ...]:
    numbers = tuple(int(value) for value in values)
    if not numbers:
        raise CalibrationError(f"{name} must contain at least one value")
    if any(value < 1 for value in numbers):
        raise CalibrationError(f"{name} values must be at least 1")
    return numbers


def _void_radii(result: DirectionalVoidFinderResult) -> list[float]:
    return [void.effective_radius_mpc_h for void in result.voids]


def mean_halo_spacing_mpc_h(catalog: HaloCatalog) -> float:
    """Return the source-catalog mean halo separation in Mpc/h."""

    halo_count = len(catalog)
    if halo_count < 1:
        raise CalibrationError("mean halo spacing requires at least one halo")
    return float((catalog.box_size_mpc_h**3 / halo_count) ** (1.0 / 3.0))


def _linking_length_candidates(
    catalog_a: HaloCatalog,
    catalog_b: HaloCatalog,
    *,
    linking_lengths_mpc_h: Iterable[float],
    linking_length_mean_spacing_factors: Iterable[float],
) -> tuple[_LinkingLengthCandidate, ...]:
    fixed_lengths = _positive_values(
        "linking_lengths_mpc_h",
        linking_lengths_mpc_h,
        allow_empty=True,
    )
    factors = _positive_values(
        "linking_length_mean_spacing_factors",
        linking_length_mean_spacing_factors,
        allow_empty=True,
    )
    candidates = [
        _LinkingLengthCandidate(
            mode="fixed",
            value=linking_length,
            source_a_length_mpc_h=linking_length,
            source_b_length_mpc_h=linking_length,
        )
        for linking_length in fixed_lengths
    ]
    if factors:
        mean_spacing_a = mean_halo_spacing_mpc_h(catalog_a)
        mean_spacing_b = mean_halo_spacing_mpc_h(catalog_b)
        candidates.extend(
            _LinkingLengthCandidate(
                mode="mean_spacing",
                value=factor,
                source_a_length_mpc_h=factor * mean_spacing_a,
                source_b_length_mpc_h=factor * mean_spacing_b,
            )
            for factor in factors
        )
    if not candidates:
        raise CalibrationError(
            "At least one fixed linking length or mean-spacing factor is required"
        )
    return tuple(candidates)


def score_direction_against_vide(
    result: DirectionalVoidFinderResult,
    reference: VideVoidCatalog,
    *,
    box_size_mpc_h: float,
    bins: int | ArrayLike,
    min_predicted_fraction: float = 0.25,
) -> DirectionalGeometryScore:
    """Score one predicted target catalog against a VIDE reference."""

    min_fraction = _min_predicted_fraction(min_predicted_fraction)
    comparison = compare_void_size_functions(
        _void_radii(result),
        reference.effective_radii_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
    )
    predicted_count = int(comparison.predicted.counts.sum())
    reference_count = int(comparison.reference.counts.sum())
    if reference_count == 0:
        predicted_reference_fraction = 1.0
    else:
        predicted_reference_fraction = predicted_count / reference_count
    is_degenerate = predicted_reference_fraction < min_fraction
    guard_penalty = reference_count if is_degenerate else 0
    return DirectionalGeometryScore(
        target_label=result.target_label,
        predicted_void_count=predicted_count,
        reference_void_count=reference_count,
        predicted_reference_fraction=predicted_reference_fraction,
        count_l1_difference=comparison.count_l1_difference,
        guarded_count_l1_difference=comparison.count_l1_difference + guard_penalty,
        density_l1_difference=comparison.density_l1_difference,
        is_degenerate_underprediction=is_degenerate,
        size_function=comparison,
    )


def summarize_radii_for_range(
    radii_mpc_h: ArrayLike,
    *,
    radius_min_mpc_h: float,
    radius_max_mpc_h: float,
) -> RadiusSummary:
    """Summarize radii and count objects in a fixed radius interval."""

    lower = float(radius_min_mpc_h)
    upper = float(radius_max_mpc_h)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise CalibrationError("radius range must be positive, finite, and increasing")
    radii = np.asarray(radii_mpc_h, dtype=np.float64)
    if radii.ndim != 1:
        raise CalibrationError("radii_mpc_h must be one-dimensional")
    if radii.size and (not np.all(np.isfinite(radii)) or np.any(radii <= 0.0)):
        raise CalibrationError("radii_mpc_h must contain positive finite values")
    count_in_range = int(np.count_nonzero((radii >= lower) & (radii <= upper)))
    if radii.size == 0:
        return RadiusSummary(
            count=0,
            count_in_range=count_in_range,
            min_mpc_h=np.nan,
            p10_mpc_h=np.nan,
            median_mpc_h=np.nan,
            p90_mpc_h=np.nan,
            max_mpc_h=np.nan,
        )
    return RadiusSummary(
        count=int(radii.size),
        count_in_range=count_in_range,
        min_mpc_h=float(np.min(radii)),
        p10_mpc_h=float(np.percentile(radii, 10.0)),
        median_mpc_h=float(np.median(radii)),
        p90_mpc_h=float(np.percentile(radii, 90.0)),
        max_mpc_h=float(np.max(radii)),
    )


def score_radius_calibration_radii(
    *,
    target_label: str,
    predicted_radii_mpc_h: ArrayLike,
    reference_radii_mpc_h: ArrayLike,
    box_size_mpc_h: float,
    bins: int | ArrayLike,
    radius_min_mpc_h: float,
    radius_max_mpc_h: float,
    min_predicted_fraction: float = 0.25,
) -> DirectionalRadiusCalibrationScore:
    """Score radius-scale calibration from predicted and reference radii."""

    min_fraction = _min_predicted_fraction(min_predicted_fraction)
    comparison = compare_void_size_functions(
        predicted_radii_mpc_h,
        reference_radii_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
    )
    predicted_count = int(comparison.predicted.counts.sum())
    reference_count = int(comparison.reference.counts.sum())
    if reference_count == 0:
        predicted_reference_fraction = 1.0
    else:
        predicted_reference_fraction = predicted_count / reference_count
    finder_summary = summarize_radii_for_range(
        predicted_radii_mpc_h,
        radius_min_mpc_h=radius_min_mpc_h,
        radius_max_mpc_h=radius_max_mpc_h,
    )
    reference_summary = summarize_radii_for_range(
        reference_radii_mpc_h,
        radius_min_mpc_h=radius_min_mpc_h,
        radius_max_mpc_h=radius_max_mpc_h,
    )
    median_error = (
        np.inf
        if not np.isfinite(finder_summary.median_mpc_h)
        or not np.isfinite(reference_summary.median_mpc_h)
        else abs(finder_summary.median_mpc_h - reference_summary.median_mpc_h)
    )
    is_zero_in_bin = reference_count > 0 and predicted_count == 0
    is_degenerate = is_zero_in_bin or predicted_reference_fraction < min_fraction
    guard_penalty = reference_count if is_degenerate else 0
    size_score = DirectionalGeometryScore(
        target_label=target_label,
        predicted_void_count=predicted_count,
        reference_void_count=reference_count,
        predicted_reference_fraction=predicted_reference_fraction,
        count_l1_difference=comparison.count_l1_difference,
        guarded_count_l1_difference=comparison.count_l1_difference + guard_penalty,
        density_l1_difference=comparison.density_l1_difference,
        is_degenerate_underprediction=is_degenerate,
        size_function=comparison,
    )
    return DirectionalRadiusCalibrationScore(
        target_label=target_label,
        size_score=size_score,
        finder_summary=finder_summary,
        reference_summary=reference_summary,
        median_radius_abs_error_mpc_h=float(median_error),
        in_bin_count_abs_error=abs(predicted_count - reference_count),
        is_zero_in_bin=is_zero_in_bin,
        is_degenerate=is_degenerate,
    )


def score_direction_radius_calibration(
    result: DirectionalVoidFinderResult,
    reference: VideVoidCatalog,
    *,
    box_size_mpc_h: float,
    bins: int | ArrayLike,
    radius_min_mpc_h: float,
    radius_max_mpc_h: float,
    min_predicted_fraction: float = 0.25,
) -> DirectionalRadiusCalibrationScore:
    """Score one predicted target catalog for paper-bin radius calibration."""

    return score_radius_calibration_radii(
        target_label=result.target_label,
        predicted_radii_mpc_h=_void_radii(result),
        reference_radii_mpc_h=reference.effective_radii_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
        radius_min_mpc_h=radius_min_mpc_h,
        radius_max_mpc_h=radius_max_mpc_h,
        min_predicted_fraction=min_predicted_fraction,
    )


def score_paired_radius_calibration(
    result: PairedVoidFinderResult,
    *,
    reference_a: VideVoidCatalog,
    reference_b: VideVoidCatalog,
    box_size_mpc_h: float,
    bins: int | ArrayLike,
    radius_min_mpc_h: float,
    radius_max_mpc_h: float,
    config: PairedVoidFinderConfig,
    linking_mode: Literal["fixed", "mean_spacing"] = "fixed",
    linking_value: float | None = None,
    source_a_linking_length_mpc_h: float | None = None,
    source_b_linking_length_mpc_h: float | None = None,
    min_predicted_fraction: float = 0.25,
) -> PairedRadiusCalibrationResult:
    """Score a paired result for paper-bin radius calibration."""

    score_a = score_direction_radius_calibration(
        result.voids_a,
        reference_a,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
        radius_min_mpc_h=radius_min_mpc_h,
        radius_max_mpc_h=radius_max_mpc_h,
        min_predicted_fraction=min_predicted_fraction,
    )
    score_b = score_direction_radius_calibration(
        result.voids_b,
        reference_b,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
        radius_min_mpc_h=radius_min_mpc_h,
        radius_max_mpc_h=radius_max_mpc_h,
        min_predicted_fraction=min_predicted_fraction,
    )
    resolved_linking_value = (
        config.linking_length_mpc_h if linking_value is None else float(linking_value)
    )
    source_a_linking_length = (
        config.linking_length_mpc_h
        if source_a_linking_length_mpc_h is None
        else float(source_a_linking_length_mpc_h)
    )
    source_b_linking_length = (
        (config.source_b_linking_length_mpc_h or config.linking_length_mpc_h)
        if source_b_linking_length_mpc_h is None
        else float(source_b_linking_length_mpc_h)
    )
    return PairedRadiusCalibrationResult(
        config=config,
        linking_mode=linking_mode,
        linking_value=resolved_linking_value,
        source_a_linking_length_mpc_h=source_a_linking_length,
        source_b_linking_length_mpc_h=source_b_linking_length,
        source_a_cluster_count=len(result.voids_b.source_clusters),
        source_b_cluster_count=len(result.voids_a.source_clusters),
        protovoid_a_count=len(result.voids_a.protovoids),
        protovoid_b_count=len(result.voids_b.protovoids),
        edge_a_count=len(result.voids_a.adjacency_edges),
        edge_b_count=len(result.voids_b.adjacency_edges),
        score_a=score_a,
        score_b=score_b,
        total_count_l1_difference=(
            score_a.size_score.count_l1_difference
            + score_b.size_score.count_l1_difference
        ),
        total_density_l1_difference=(
            score_a.size_score.density_l1_difference
            + score_b.size_score.density_l1_difference
        ),
        total_median_radius_abs_error_mpc_h=(
            score_a.median_radius_abs_error_mpc_h
            + score_b.median_radius_abs_error_mpc_h
        ),
        total_in_bin_count_abs_error=(
            score_a.in_bin_count_abs_error + score_b.in_bin_count_abs_error
        ),
        is_degenerate=score_a.is_degenerate or score_b.is_degenerate,
    )


def score_paired_result_against_vide(
    result: PairedVoidFinderResult,
    *,
    reference_a: VideVoidCatalog,
    reference_b: VideVoidCatalog,
    box_size_mpc_h: float,
    bins: int | ArrayLike,
    config: PairedVoidFinderConfig,
    linking_mode: Literal["fixed", "mean_spacing"] = "fixed",
    linking_value: float | None = None,
    source_a_linking_length_mpc_h: float | None = None,
    source_b_linking_length_mpc_h: float | None = None,
    min_predicted_fraction: float = 0.25,
) -> PairedGeometrySweepResult:
    """Score a paired result against target-A and target-B VIDE references."""

    score_a = score_direction_against_vide(
        result.voids_a,
        reference_a,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
        min_predicted_fraction=min_predicted_fraction,
    )
    score_b = score_direction_against_vide(
        result.voids_b,
        reference_b,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
        min_predicted_fraction=min_predicted_fraction,
    )
    total_guarded_count_l1_difference = (
        score_a.guarded_count_l1_difference + score_b.guarded_count_l1_difference
    )
    is_degenerate = (
        score_a.is_degenerate_underprediction or score_b.is_degenerate_underprediction
    )
    resolved_linking_value = (
        config.linking_length_mpc_h if linking_value is None else float(linking_value)
    )
    source_a_linking_length = (
        config.linking_length_mpc_h
        if source_a_linking_length_mpc_h is None
        else float(source_a_linking_length_mpc_h)
    )
    source_b_linking_length = (
        (config.source_b_linking_length_mpc_h or config.linking_length_mpc_h)
        if source_b_linking_length_mpc_h is None
        else float(source_b_linking_length_mpc_h)
    )
    return PairedGeometrySweepResult(
        config=config,
        linking_mode=linking_mode,
        linking_value=resolved_linking_value,
        source_a_linking_length_mpc_h=source_a_linking_length,
        source_b_linking_length_mpc_h=source_b_linking_length,
        score_a=score_a,
        score_b=score_b,
        total_count_l1_difference=score_a.count_l1_difference + score_b.count_l1_difference,
        total_guarded_count_l1_difference=total_guarded_count_l1_difference,
        total_density_l1_difference=score_a.density_l1_difference + score_b.density_l1_difference,
        is_degenerate_underprediction=is_degenerate,
    )


def _direction_from_cached_clusters(
    source_clusters: SourceClusterCatalog,
    *,
    source_label: str,
    target_label: str,
    radius_a0: float,
    radius_alpha: float,
    reference_rho_bar_msun_h_mpc3: float,
    adjacency_factor: float,
    merged_radius_mode: MergedRadiusMode,
    min_void_radius_mpc_h: float,
) -> DirectionalVoidFinderResult:
    protovoids = source_clusters_to_protovoids(
        source_clusters,
        radius_a0=radius_a0,
        radius_alpha=radius_alpha,
        reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
        target_label=target_label,
    )
    adjacency_edges = build_protovoid_adjacency(
        protovoids,
        adjacency_factor=adjacency_factor,
    )
    voids = merge_protovoids(
        protovoids,
        adjacency_edges,
        radius_mode=merged_radius_mode,
        min_void_radius_mpc_h=min_void_radius_mpc_h,
        radius_a0=radius_a0,
        radius_alpha=radius_alpha,
        reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
    )
    return DirectionalVoidFinderResult(
        source_label=source_label,
        target_label=target_label,
        source_clusters=source_clusters,
        protovoids=protovoids,
        adjacency_edges=adjacency_edges,
        voids=voids,
    )


def _sort_radius_calibration_results(
    results: Iterable[PairedRadiusCalibrationResult],
) -> tuple[PairedRadiusCalibrationResult, ...]:
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item.is_degenerate,
                item.total_density_l1_difference,
                item.total_median_radius_abs_error_mpc_h,
                item.total_in_bin_count_abs_error,
                item.total_count_l1_difference,
                item.linking_mode,
                item.linking_value,
                item.config.min_cluster_members,
                item.config.min_cluster_mass_msun_h,
                item.config.radius_a0,
                item.config.radius_alpha,
                item.config.adjacency_factor,
            ),
        )
    )


def sweep_radius_scale_parameters(
    catalog_a: HaloCatalog,
    catalog_b: HaloCatalog,
    *,
    reference_a: VideVoidCatalog,
    reference_b: VideVoidCatalog,
    reference_rho_bar_msun_h_mpc3: float,
    linking_lengths_mpc_h: Iterable[float],
    radius_a0_values: Iterable[float],
    radius_alpha_values: Iterable[float],
    adjacency_factors: Iterable[float],
    bins: int | ArrayLike,
    radius_min_mpc_h: float,
    radius_max_mpc_h: float,
    linking_length_mean_spacing_factors: Iterable[float] = (),
    min_cluster_members_values: Iterable[int] = (2,),
    min_cluster_mass_values_msun_h: Iterable[float] = (0.0,),
    min_predicted_fraction: float = 0.25,
    max_source_clusters: int | None = None,
) -> tuple[PairedRadiusCalibrationResult, ...]:
    """Run a cached radius-scale calibration grid against VIDE references."""

    linking_candidates = _linking_length_candidates(
        catalog_a,
        catalog_b,
        linking_lengths_mpc_h=linking_lengths_mpc_h,
        linking_length_mean_spacing_factors=linking_length_mean_spacing_factors,
    )
    radius_a0s = _positive_values("radius_a0_values", radius_a0_values)
    radius_alphas = _positive_values("radius_alpha_values", radius_alpha_values)
    adjacency_factor_values = _positive_values("adjacency_factors", adjacency_factors)
    min_members_values = _positive_int_values(
        "min_cluster_members_values",
        min_cluster_members_values,
    )
    min_mass_values = _non_negative_values(
        "min_cluster_mass_values_msun_h",
        min_cluster_mass_values_msun_h,
    )
    min_fraction = _min_predicted_fraction(min_predicted_fraction)
    if max_source_clusters is not None and int(max_source_clusters) < 1:
        raise CalibrationError("max_source_clusters must be at least 1")
    max_clusters = None if max_source_clusters is None else int(max_source_clusters)

    scored_results: list[PairedRadiusCalibrationResult] = []
    for linking_candidate, min_members, min_mass in product(
        linking_candidates,
        min_members_values,
        min_mass_values,
    ):
        source_clusters_a = find_source_clusters(
            catalog_a,
            linking_length_mpc_h=linking_candidate.source_a_length_mpc_h,
            min_cluster_members=min_members,
            min_cluster_mass_msun_h=min_mass,
            source_label="A",
        )
        source_clusters_b = find_source_clusters(
            catalog_b,
            linking_length_mpc_h=linking_candidate.source_b_length_mpc_h,
            min_cluster_members=min_members,
            min_cluster_mass_msun_h=min_mass,
            source_label="B",
        )
        if max_clusters is not None and (
            len(source_clusters_a) > max_clusters or len(source_clusters_b) > max_clusters
        ):
            continue

        for radius_a0, radius_alpha, adjacency_factor in product(
            radius_a0s,
            radius_alphas,
            adjacency_factor_values,
        ):
            config = PairedVoidFinderConfig(
                linking_length_mpc_h=linking_candidate.source_a_length_mpc_h,
                source_b_linking_length_mpc_h=linking_candidate.source_b_length_mpc_h,
                min_cluster_members=min_members,
                min_cluster_mass_msun_h=min_mass,
                reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
                radius_a0=radius_a0,
                radius_alpha=radius_alpha,
                adjacency_factor=adjacency_factor,
            )
            voids_a = _direction_from_cached_clusters(
                source_clusters_b,
                source_label="B",
                target_label="A",
                radius_a0=radius_a0,
                radius_alpha=radius_alpha,
                reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
                adjacency_factor=adjacency_factor,
                merged_radius_mode=config.merged_radius_mode,
                min_void_radius_mpc_h=config.min_void_radius_mpc_h,
            )
            voids_b = _direction_from_cached_clusters(
                source_clusters_a,
                source_label="A",
                target_label="B",
                radius_a0=radius_a0,
                radius_alpha=radius_alpha,
                reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
                adjacency_factor=adjacency_factor,
                merged_radius_mode=config.merged_radius_mode,
                min_void_radius_mpc_h=config.min_void_radius_mpc_h,
            )
            result = PairedVoidFinderResult(voids_a=voids_a, voids_b=voids_b)
            scored_results.append(
                score_paired_radius_calibration(
                    result,
                    reference_a=reference_a,
                    reference_b=reference_b,
                    box_size_mpc_h=catalog_a.box_size_mpc_h,
                    bins=bins,
                    radius_min_mpc_h=radius_min_mpc_h,
                    radius_max_mpc_h=radius_max_mpc_h,
                    config=config,
                    linking_mode=linking_candidate.mode,
                    linking_value=linking_candidate.value,
                    source_a_linking_length_mpc_h=linking_candidate.source_a_length_mpc_h,
                    source_b_linking_length_mpc_h=linking_candidate.source_b_length_mpc_h,
                    min_predicted_fraction=min_fraction,
                )
            )

    return _sort_radius_calibration_results(scored_results)


def sweep_geometry_parameters(
    catalog_a: HaloCatalog,
    catalog_b: HaloCatalog,
    *,
    reference_a: VideVoidCatalog,
    reference_b: VideVoidCatalog,
    reference_rho_bar_msun_h_mpc3: float,
    linking_lengths_mpc_h: Iterable[float],
    radius_a0_values: Iterable[float],
    radius_alpha_values: Iterable[float],
    adjacency_factors: Iterable[float],
    linking_length_mean_spacing_factors: Iterable[float] = (),
    min_cluster_members: int = 2,
    min_cluster_mass_msun_h: float = 0.0,
    bins: int = 6,
    min_predicted_fraction: float = 0.25,
) -> tuple[PairedGeometrySweepResult, ...]:
    """Run a deterministic geometry-only grid sweep against VIDE references."""

    linking_candidates = _linking_length_candidates(
        catalog_a,
        catalog_b,
        linking_lengths_mpc_h=linking_lengths_mpc_h,
        linking_length_mean_spacing_factors=linking_length_mean_spacing_factors,
    )
    radius_a0s = _positive_values("radius_a0_values", radius_a0_values)
    radius_alphas = _positive_values("radius_alpha_values", radius_alpha_values)
    adjacency_factor_values = _positive_values("adjacency_factors", adjacency_factors)
    if bins < 1:
        raise CalibrationError("bins must be at least 1")
    min_fraction = _min_predicted_fraction(min_predicted_fraction)

    scored_results: list[PairedGeometrySweepResult] = []
    for linking_candidate, radius_a0, radius_alpha, adjacency_factor in product(
        linking_candidates,
        radius_a0s,
        radius_alphas,
        adjacency_factor_values,
    ):
        config = PairedVoidFinderConfig(
            linking_length_mpc_h=linking_candidate.source_a_length_mpc_h,
            source_b_linking_length_mpc_h=linking_candidate.source_b_length_mpc_h,
            min_cluster_members=min_cluster_members,
            min_cluster_mass_msun_h=min_cluster_mass_msun_h,
            reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
            radius_a0=radius_a0,
            radius_alpha=radius_alpha,
            adjacency_factor=adjacency_factor,
        )
        result = run_paired_halo_void_finder(catalog_a, catalog_b, config=config)
        scored_results.append(
            score_paired_result_against_vide(
                result,
                reference_a=reference_a,
                reference_b=reference_b,
                box_size_mpc_h=catalog_a.box_size_mpc_h,
                bins=bins,
                config=config,
                linking_mode=linking_candidate.mode,
                linking_value=linking_candidate.value,
                source_a_linking_length_mpc_h=linking_candidate.source_a_length_mpc_h,
                source_b_linking_length_mpc_h=linking_candidate.source_b_length_mpc_h,
                min_predicted_fraction=min_fraction,
            )
        )

    return tuple(
        sorted(
            scored_results,
            key=lambda item: (
                item.is_degenerate_underprediction,
                item.total_guarded_count_l1_difference,
                item.total_count_l1_difference,
                item.total_density_l1_difference,
                item.linking_mode,
                item.linking_value,
                item.source_a_linking_length_mpc_h,
                item.source_b_linking_length_mpc_h,
                item.config.radius_a0,
                item.config.radius_alpha,
                item.config.adjacency_factor,
            ),
        )
    )
