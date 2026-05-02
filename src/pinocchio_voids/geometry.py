"""Periodic-box geometry helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class PeriodicGeometryError(ValueError):
    """Raised when periodic geometry inputs are invalid."""


def _validate_box_size(box_size_mpc_h: float) -> float:
    box_size = float(box_size_mpc_h)
    if not np.isfinite(box_size) or box_size <= 0:
        raise PeriodicGeometryError("box_size_mpc_h must be positive and finite")
    return box_size


def _as_xyz_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1:] != (3,):
        raise PeriodicGeometryError(f"{name} must have a final dimension of length 3")
    if not np.all(np.isfinite(array)):
        raise PeriodicGeometryError(f"{name} contains non-finite values")
    return array


def minimum_image_displacement(
    start_mpc_h: ArrayLike,
    end_mpc_h: ArrayLike,
    box_size_mpc_h: float,
) -> NDArray[np.float64]:
    """Return the shortest periodic displacement from ``start`` to ``end``.

    Inputs may be single 3-vectors or broadcastable arrays whose final
    dimension is length 3.
    """

    box_size = _validate_box_size(box_size_mpc_h)
    start = _as_xyz_array("start_mpc_h", start_mpc_h)
    end = _as_xyz_array("end_mpc_h", end_mpc_h)
    return (end - start + 0.5 * box_size) % box_size - 0.5 * box_size


def periodic_distance(
    start_mpc_h: ArrayLike,
    end_mpc_h: ArrayLike,
    box_size_mpc_h: float,
) -> NDArray[np.float64]:
    """Return the minimum-image periodic distance between points."""

    displacement = minimum_image_displacement(start_mpc_h, end_mpc_h, box_size_mpc_h)
    return np.linalg.norm(displacement, axis=-1)


def periodic_center_of_mass(
    positions_mpc_h: ArrayLike,
    *,
    box_size_mpc_h: float,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return a weighted center in a periodic box using circular means."""

    box_size = _validate_box_size(box_size_mpc_h)
    positions = _as_xyz_array("positions_mpc_h", positions_mpc_h)
    if positions.ndim != 2:
        raise PeriodicGeometryError("positions_mpc_h must have shape (n, 3)")
    if positions.shape[0] == 0:
        raise PeriodicGeometryError("positions_mpc_h must contain at least one point")

    if weights is None:
        weights_array = None
    else:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.ndim != 1 or weights_array.shape[0] != positions.shape[0]:
            raise PeriodicGeometryError("weights must be one-dimensional with one row per point")
        if not np.all(np.isfinite(weights_array)) or np.any(weights_array <= 0):
            raise PeriodicGeometryError("weights must be positive and finite")

    angles = (positions % box_size) * (2.0 * np.pi / box_size)
    sin_mean = np.average(np.sin(angles), axis=0, weights=weights_array)
    cos_mean = np.average(np.cos(angles), axis=0, weights=weights_array)
    resultant = np.hypot(sin_mean, cos_mean)
    if np.any(resultant < 1.0e-12):
        raise PeriodicGeometryError("periodic center is ambiguous for the supplied positions")

    angle = np.arctan2(sin_mean, cos_mean)
    angle = np.where(angle < 0.0, angle + 2.0 * np.pi, angle)
    center = angle * (box_size / (2.0 * np.pi))
    center %= box_size
    center = np.where(np.isclose(center, box_size), 0.0, center)
    center.setflags(write=False)
    return center
