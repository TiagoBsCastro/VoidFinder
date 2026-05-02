"""Evaluation helpers for predicted and reference void catalogs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


class EvaluationError(ValueError):
    """Raised when evaluation inputs or parameters are invalid."""


@dataclass(frozen=True)
class VoidSizeFunction:
    """Differential void size function, ``dN / dlnR / V``."""

    bin_edges_mpc_h: NDArray[np.float64]
    bin_centers_mpc_h: NDArray[np.float64]
    counts: NDArray[np.int64]
    density_dndlnr_per_mpc_h3: NDArray[np.float64]
    box_volume_mpc_h3: float

    def __post_init__(self) -> None:
        for name in ("bin_edges_mpc_h", "bin_centers_mpc_h", "density_dndlnr_per_mpc_h3"):
            values = np.asarray(getattr(self, name))
            readonly = values.copy()
            readonly.setflags(write=False)
            object.__setattr__(self, name, readonly)
        counts = np.asarray(self.counts, dtype=np.int64).copy()
        counts.setflags(write=False)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "box_volume_mpc_h3", _validate_positive("box_volume_mpc_h3", self.box_volume_mpc_h3))


@dataclass(frozen=True)
class VoidSizeFunctionComparison:
    """Predicted and reference void size functions on shared radius bins."""

    predicted: VoidSizeFunction
    reference: VoidSizeFunction
    count_l1_difference: int
    density_l1_difference: float


def _validate_positive(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise EvaluationError(f"{name} must be positive and finite")
    return number


def _positive_radii(radii_mpc_h: ArrayLike) -> NDArray[np.float64]:
    radii = np.asarray(radii_mpc_h, dtype=np.float64)
    if radii.ndim != 1:
        raise EvaluationError("radii_mpc_h must be one-dimensional")
    if radii.size == 0:
        raise EvaluationError("radii_mpc_h must contain at least one radius")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise EvaluationError("radii_mpc_h must contain positive finite radii")
    return radii


def _bin_edges(radii: NDArray[np.float64], bins: int | ArrayLike) -> NDArray[np.float64]:
    if isinstance(bins, int):
        if bins < 1:
            raise EvaluationError("bins must be at least 1")
        if radii.min() == radii.max():
            lower = radii.min() / np.sqrt(2.0)
            upper = radii.max() * np.sqrt(2.0)
            return np.geomspace(lower, upper, bins + 1)
        return np.geomspace(radii.min(), radii.max(), bins + 1)

    edges = np.asarray(bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise EvaluationError("bin edges must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(edges)) or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise EvaluationError("bin edges must be positive, finite, and strictly increasing")
    return edges


def compute_void_size_function(
    radii_mpc_h: ArrayLike,
    *,
    box_size_mpc_h: float,
    bins: int | ArrayLike = 10,
) -> VoidSizeFunction:
    """Compute ``dN / dlnR / V`` from effective void radii."""

    radii = _positive_radii(radii_mpc_h)
    box_size = _validate_positive("box_size_mpc_h", box_size_mpc_h)
    edges = _bin_edges(radii, bins)
    counts, _ = np.histogram(radii, bins=edges)
    dlnr = np.diff(np.log(edges))
    volume = box_size**3
    density = counts / (volume * dlnr)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return VoidSizeFunction(
        bin_edges_mpc_h=edges,
        bin_centers_mpc_h=centers,
        counts=counts,
        density_dndlnr_per_mpc_h3=density,
        box_volume_mpc_h3=volume,
    )


def compare_void_size_functions(
    predicted_radii_mpc_h: ArrayLike,
    reference_radii_mpc_h: ArrayLike,
    *,
    box_size_mpc_h: float,
    bins: int | ArrayLike = 10,
) -> VoidSizeFunctionComparison:
    """Compare predicted and reference void size functions on shared bins."""

    predicted_radii = _positive_radii(predicted_radii_mpc_h)
    reference_radii = _positive_radii(reference_radii_mpc_h)
    combined_radii = np.concatenate([predicted_radii, reference_radii])
    edges = _bin_edges(combined_radii, bins)
    predicted = compute_void_size_function(
        predicted_radii,
        box_size_mpc_h=box_size_mpc_h,
        bins=edges,
    )
    reference = compute_void_size_function(
        reference_radii,
        box_size_mpc_h=box_size_mpc_h,
        bins=edges,
    )
    density_delta = np.abs(
        predicted.density_dndlnr_per_mpc_h3 - reference.density_dndlnr_per_mpc_h3
    )
    return VoidSizeFunctionComparison(
        predicted=predicted,
        reference=reference,
        count_l1_difference=int(np.sum(np.abs(predicted.counts - reference.counts))),
        density_l1_difference=float(np.sum(density_delta)),
    )
