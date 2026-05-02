"""Calibration helpers for the geometry-only paired-halo prototype."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

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
    count_l1_difference: int
    density_l1_difference: float
    size_function: VoidSizeFunctionComparison


@dataclass(frozen=True)
class PairedGeometrySweepResult:
    """One geometry-only parameter combination and its paired score."""

    config: PairedVoidFinderConfig
    score_a: DirectionalGeometryScore
    score_b: DirectionalGeometryScore
    total_count_l1_difference: int
    total_density_l1_difference: float


def _positive_values(name: str, values: Iterable[float]) -> tuple[float, ...]:
    numbers = tuple(float(value) for value in values)
    if not numbers:
        raise CalibrationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(numbers)) or any(value <= 0.0 for value in numbers):
        raise CalibrationError(f"{name} values must be positive and finite")
    return numbers


def _void_radii(result: DirectionalVoidFinderResult) -> list[float]:
    return [void.effective_radius_mpc_h for void in result.voids]


def score_direction_against_vide(
    result: DirectionalVoidFinderResult,
    reference: VideVoidCatalog,
    *,
    box_size_mpc_h: float,
    bins: int,
) -> DirectionalGeometryScore:
    """Score one predicted target catalog against a VIDE reference."""

    comparison = compare_void_size_functions(
        _void_radii(result),
        reference.effective_radii_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
    )
    return DirectionalGeometryScore(
        target_label=result.target_label,
        predicted_void_count=int(comparison.predicted.counts.sum()),
        reference_void_count=int(comparison.reference.counts.sum()),
        count_l1_difference=comparison.count_l1_difference,
        density_l1_difference=comparison.density_l1_difference,
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
) -> PairedGeometrySweepResult:
    """Score a paired result against target-A and target-B VIDE references."""

    score_a = score_direction_against_vide(
        result.voids_a,
        reference_a,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
    )
    score_b = score_direction_against_vide(
        result.voids_b,
        reference_b,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
    )
    return PairedGeometrySweepResult(
        config=config,
        score_a=score_a,
        score_b=score_b,
        total_count_l1_difference=score_a.count_l1_difference + score_b.count_l1_difference,
        total_density_l1_difference=score_a.density_l1_difference + score_b.density_l1_difference,
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
    min_cluster_members: int = 2,
    min_cluster_mass_msun_h: float = 0.0,
    bins: int = 6,
) -> tuple[PairedGeometrySweepResult, ...]:
    """Run a deterministic geometry-only grid sweep against VIDE references."""

    linking_lengths = _positive_values("linking_lengths_mpc_h", linking_lengths_mpc_h)
    radius_a0s = _positive_values("radius_a0_values", radius_a0_values)
    radius_alphas = _positive_values("radius_alpha_values", radius_alpha_values)
    adjacency_factor_values = _positive_values("adjacency_factors", adjacency_factors)
    if bins < 1:
        raise CalibrationError("bins must be at least 1")

    scored_results: list[PairedGeometrySweepResult] = []
    for linking_length, radius_a0, radius_alpha, adjacency_factor in product(
        linking_lengths,
        radius_a0s,
        radius_alphas,
        adjacency_factor_values,
    ):
        config = PairedVoidFinderConfig(
            linking_length_mpc_h=linking_length,
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
            )
        )

    return tuple(
        sorted(
            scored_results,
            key=lambda item: (
                item.total_count_l1_difference,
                item.total_density_l1_difference,
                item.config.linking_length_mpc_h,
                item.config.radius_a0,
                item.config.radius_alpha,
                item.config.adjacency_factor,
            ),
        )
    )
