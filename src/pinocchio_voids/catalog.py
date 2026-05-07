"""Canonical catalog data structures used inside pinocchio_voids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


class CatalogValidationError(ValueError):
    """Raised when a canonical catalog cannot be validated."""


def _readonly_1d(name: str, values: ArrayLike, dtype: type) -> NDArray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise CatalogValidationError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise CatalogValidationError(f"{name} contains non-finite values")

    readonly = array.copy()
    readonly.setflags(write=False)
    return readonly


def _readonly_nx3(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise CatalogValidationError(f"{name} must have shape (n, 3)")
    if not np.all(np.isfinite(array)):
        raise CatalogValidationError(f"{name} contains non-finite values")

    readonly = array.copy()
    readonly.setflags(write=False)
    return readonly


def _validate_box_size(box_size_mpc_h: float) -> float:
    box_size = float(box_size_mpc_h)
    if not np.isfinite(box_size) or box_size <= 0:
        raise CatalogValidationError("box_size_mpc_h must be positive and finite")
    return box_size


def _validate_row_count(name: str, row_count: int, expected: int) -> None:
    if row_count != expected:
        raise CatalogValidationError(
            f"{name} has {row_count} rows; expected {expected}"
        )


@dataclass(frozen=True)
class TracerCatalog:
    """Canonical point-tracer catalog for future void-finding algorithms."""

    ids: ArrayLike
    positions_mpc_h: ArrayLike
    box_size_mpc_h: float
    velocities_km_s: ArrayLike | None = None
    weights: ArrayLike | None = None
    position_mode: str = "final"
    position_unit: str = "Mpc/h"
    velocity_unit: str = "km/s"
    weight_unit: str = "dimensionless"

    def __post_init__(self) -> None:
        ids = _readonly_1d("ids", self.ids, np.int64)
        positions = _readonly_nx3("positions_mpc_h", self.positions_mpc_h)
        _validate_row_count("positions_mpc_h", positions.shape[0], len(ids))

        if self.velocities_km_s is None:
            velocities = None
        else:
            velocities = _readonly_nx3("velocities_km_s", self.velocities_km_s)
            _validate_row_count("velocities_km_s", velocities.shape[0], len(ids))

        if self.weights is None:
            weights = np.ones(len(ids), dtype=np.float64)
            weights.setflags(write=False)
        else:
            weights = _readonly_1d("weights", self.weights, np.float64)
            _validate_row_count("weights", weights.shape[0], len(ids))
            if np.any(weights <= 0):
                raise CatalogValidationError("weights must be positive")

        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "positions_mpc_h", positions)
        object.__setattr__(self, "velocities_km_s", velocities)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "position_mode", str(self.position_mode))
        object.__setattr__(self, "box_size_mpc_h", _validate_box_size(self.box_size_mpc_h))

    def __len__(self) -> int:
        return int(self.ids.shape[0])


@dataclass(frozen=True)
class HaloCatalog:
    """Canonical halo catalog with the fields needed for tracer construction."""

    ids: ArrayLike
    masses_msun_h: ArrayLike
    positions_mpc_h: ArrayLike
    velocities_km_s: ArrayLike
    particle_counts: ArrayLike
    box_size_mpc_h: float
    position_mode: str = "final"
    position_unit: str = "Mpc/h"
    mass_unit: str = "Msun/h"
    velocity_unit: str = "km/s"

    def __post_init__(self) -> None:
        ids = _readonly_1d("ids", self.ids, np.int64)
        masses = _readonly_1d("masses_msun_h", self.masses_msun_h, np.float64)
        positions = _readonly_nx3("positions_mpc_h", self.positions_mpc_h)
        velocities = _readonly_nx3("velocities_km_s", self.velocities_km_s)
        particle_counts = _readonly_1d("particle_counts", self.particle_counts, np.int64)

        expected = len(ids)
        _validate_row_count("masses_msun_h", masses.shape[0], expected)
        _validate_row_count("positions_mpc_h", positions.shape[0], expected)
        _validate_row_count("velocities_km_s", velocities.shape[0], expected)
        _validate_row_count("particle_counts", particle_counts.shape[0], expected)

        if np.any(masses <= 0):
            raise CatalogValidationError("masses_msun_h must be positive")
        if np.any(particle_counts <= 0):
            raise CatalogValidationError("particle_counts must be positive")

        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "masses_msun_h", masses)
        object.__setattr__(self, "positions_mpc_h", positions)
        object.__setattr__(self, "velocities_km_s", velocities)
        object.__setattr__(self, "particle_counts", particle_counts)
        object.__setattr__(self, "position_mode", str(self.position_mode))
        object.__setattr__(self, "box_size_mpc_h", _validate_box_size(self.box_size_mpc_h))

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def to_tracer_catalog(self, *, weight_by_mass: bool = True) -> TracerCatalog:
        """Represent halos as point tracers using their canonical positions."""

        weights = self.masses_msun_h if weight_by_mass else None
        weight_unit = self.mass_unit if weight_by_mass else "dimensionless"
        return TracerCatalog(
            ids=self.ids,
            positions_mpc_h=self.positions_mpc_h,
            velocities_km_s=self.velocities_km_s,
            weights=weights,
            box_size_mpc_h=self.box_size_mpc_h,
            position_mode=self.position_mode,
            position_unit=self.position_unit,
            velocity_unit=self.velocity_unit,
            weight_unit=weight_unit,
        )
