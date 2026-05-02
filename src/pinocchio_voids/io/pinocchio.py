"""Readers for PINOCCHIO output catalogs.

Only lightweight ASCII halo catalogs are supported at this stage. This module
does not execute PINOCCHIO or implement void finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PINOCCHIO_HALO_COLUMN_NAMES: tuple[str, ...] = (
    "group_id",
    "mass_msun_h",
    "initial_x_mpc_h",
    "initial_y_mpc_h",
    "initial_z_mpc_h",
    "final_x_mpc_h",
    "final_y_mpc_h",
    "final_z_mpc_h",
    "velocity_x_km_s",
    "velocity_y_km_s",
    "velocity_z_km_s",
    "n_particles",
)

_COLUMN_INDEX = {name: index for index, name in enumerate(PINOCCHIO_HALO_COLUMN_NAMES)}


class PinocchioCatalogError(ValueError):
    """Raised when a PINOCCHIO catalog cannot be parsed or validated."""


@dataclass(frozen=True)
class PinocchioHaloCatalog:
    """In-memory representation of a PINOCCHIO ASCII halo catalog."""

    data: NDArray[np.float64]
    source: Path
    box_size_mpc_h: float | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        return PINOCCHIO_HALO_COLUMN_NAMES

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def column(self, name: str) -> NDArray[np.float64]:
        """Return a named catalog column."""

        try:
            index = _COLUMN_INDEX[name]
        except KeyError as exc:
            raise KeyError(f"Unknown PINOCCHIO halo catalog column: {name}") from exc
        return self.data[:, index]

    @property
    def group_ids(self) -> NDArray[np.int64]:
        return self.column("group_id").astype(np.int64)

    @property
    def masses_msun_h(self) -> NDArray[np.float64]:
        return self.column("mass_msun_h")

    @property
    def initial_positions_mpc_h(self) -> NDArray[np.float64]:
        return self.data[:, 2:5]

    @property
    def final_positions_mpc_h(self) -> NDArray[np.float64]:
        return self.data[:, 5:8]

    @property
    def velocities_km_s(self) -> NDArray[np.float64]:
        return self.data[:, 8:11]

    @property
    def n_particles(self) -> NDArray[np.int64]:
        return self.column("n_particles").astype(np.int64)


def read_pinocchio_halo_catalog(
    path: str | Path,
    *,
    box_size_mpc_h: float | None = None,
    wrap_positions: bool = False,
) -> PinocchioHaloCatalog:
    """Read a PINOCCHIO ASCII halo catalog.

    The expected format is the 12-column halo catalog written by PINOCCHIO:
    group ID, mass, initial position, final position, velocity, and particle
    count. Header lines beginning with ``#`` are ignored.
    """

    catalog_path = Path(path)
    try:
        raw = np.loadtxt(catalog_path, comments="#", dtype=np.float64)
    except OSError as exc:
        raise PinocchioCatalogError(f"Cannot read catalog: {catalog_path}") from exc
    except ValueError as exc:
        raise PinocchioCatalogError(f"Invalid numeric catalog: {catalog_path}") from exc

    if raw.size == 0:
        raise PinocchioCatalogError(f"Catalog contains no halo rows: {catalog_path}")

    data = np.atleast_2d(raw).astype(np.float64, copy=True)
    if data.shape[1] != len(PINOCCHIO_HALO_COLUMN_NAMES):
        raise PinocchioCatalogError(
            "PINOCCHIO halo catalogs must have "
            f"{len(PINOCCHIO_HALO_COLUMN_NAMES)} columns; got {data.shape[1]}"
        )

    if not np.all(np.isfinite(data)):
        raise PinocchioCatalogError(f"Catalog contains non-finite values: {catalog_path}")

    if np.any(data[:, _COLUMN_INDEX["mass_msun_h"]] <= 0):
        raise PinocchioCatalogError(f"Catalog contains non-positive halo masses: {catalog_path}")

    if np.any(data[:, _COLUMN_INDEX["n_particles"]] <= 0):
        raise PinocchioCatalogError(
            f"Catalog contains non-positive particle counts: {catalog_path}"
        )

    if wrap_positions:
        if box_size_mpc_h is None or box_size_mpc_h <= 0:
            raise PinocchioCatalogError("A positive box_size_mpc_h is required to wrap positions")
        data[:, 5:8] %= box_size_mpc_h

    return PinocchioHaloCatalog(
        data=data,
        source=catalog_path,
        box_size_mpc_h=box_size_mpc_h,
    )
