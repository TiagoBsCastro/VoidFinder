"""Readers for paired PINOCCHIO halo catalog inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinocchio_voids.catalog import HaloCatalog
from pinocchio_voids.io.pinocchio import (
    PinocchioPositionMode,
    normalize_pinocchio_position_mode,
    read_pinocchio_halo_catalog,
)


class PairedCatalogError(ValueError):
    """Raised when paired catalog inputs are invalid."""


@dataclass(frozen=True)
class PairedHaloCatalogs:
    """Canonical halo catalogs for one paired realization."""

    catalog_a: HaloCatalog
    catalog_b: HaloCatalog
    path_a: Path
    path_b: Path
    box_size_mpc_h: float
    position_mode: str = "final"


def read_paired_pinocchio_halo_catalogs(
    path_a: str | Path,
    path_b: str | Path,
    *,
    box_size_mpc_h: float,
    wrap_positions: bool = True,
    position_mode: PinocchioPositionMode = "final",
) -> PairedHaloCatalogs:
    """Read two PINOCCHIO ASCII catalogs as canonical paired halo catalogs."""

    mode = normalize_pinocchio_position_mode(position_mode)
    box_size = float(box_size_mpc_h)
    if not np.isfinite(box_size) or box_size <= 0:
        raise PairedCatalogError("box_size_mpc_h must be positive and finite")

    source_a = Path(path_a)
    source_b = Path(path_b)
    catalog_a = read_pinocchio_halo_catalog(source_a).to_halo_catalog(
        box_size_mpc_h=box_size,
        wrap_positions=wrap_positions,
        position_mode=mode,
    )
    catalog_b = read_pinocchio_halo_catalog(source_b).to_halo_catalog(
        box_size_mpc_h=box_size,
        wrap_positions=wrap_positions,
        position_mode=mode,
    )

    return PairedHaloCatalogs(
        catalog_a=catalog_a,
        catalog_b=catalog_b,
        path_a=source_a,
        path_b=source_b,
        box_size_mpc_h=box_size,
        position_mode=mode,
    )
