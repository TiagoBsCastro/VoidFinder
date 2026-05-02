"""Calibration helpers for the geometry-only paired-halo prototype."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Literal

import numpy as np

from pinocchio_voids.catalog import HaloCatalog
from pinocchio_voids.evaluation import (
    VoidSizeFunctionComparison,
    compare_void_size_functions,
)
from pinocchio_voids.io.vide import VideVoidCatalog
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    PairedVoidFinderConfig,
    PairedVoidFinderResult,
    run_paired_halo_void_finder,
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
    bins: int,
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


def score_paired_result_against_vide(
    result: PairedVoidFinderResult,
    *,
    reference_a: VideVoidCatalog,
    reference_b: VideVoidCatalog,
    box_size_mpc_h: float,
    bins: int,
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
