"""Readers for VIDE reference void catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pinocchio_voids.geometry import spherical_equivalent_radius_from_volume


class VideCatalogError(ValueError):
    """Raised when a VIDE catalog cannot be parsed or validated."""


@dataclass(frozen=True)
class VideVoidCatalog:
    """In-memory representation of a VIDE ``voidDesc`` catalog."""

    data: NDArray[np.float64]
    source: Path
    columns: tuple[str, ...]
    summary: str = ""
    volume_scale_mpc_h3: float = 1.0

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim != 2:
            raise VideCatalogError("data must be a two-dimensional array")
        if data.shape[1] != len(self.columns):
            raise VideCatalogError("data column count does not match columns")
        if not np.all(np.isfinite(data)):
            raise VideCatalogError(f"Catalog contains non-finite values: {self.source}")
        readonly = data.copy()
        readonly.setflags(write=False)
        object.__setattr__(self, "data", readonly)
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "columns", tuple(self.columns))
        volume_scale = float(self.volume_scale_mpc_h3)
        if not np.isfinite(volume_scale) or volume_scale <= 0.0:
            raise VideCatalogError("volume_scale_mpc_h3 must be positive and finite")
        object.__setattr__(self, "volume_scale_mpc_h3", volume_scale)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def column(self, name: str) -> NDArray[np.float64]:
        """Return a named VIDE catalog column."""

        try:
            index = self.columns.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown VIDE void catalog column: {name}") from exc
        return self.data[:, index]

    @property
    def void_ids(self) -> NDArray[np.int64]:
        return self.column("Void#").astype(np.int64)

    @property
    def raw_void_volumes(self) -> NDArray[np.float64]:
        return self.column("VoidVol")

    @property
    def void_volumes_mpc_h3(self) -> NDArray[np.float64]:
        return self.raw_void_volumes * self.volume_scale_mpc_h3

    @property
    def effective_radii_mpc_h(self) -> NDArray[np.float64]:
        """VIDE ``R_eff`` as the spherical-equivalent radius of void volume."""

        return np.asarray(
            spherical_equivalent_radius_from_volume(self.void_volumes_mpc_h3),
            dtype=np.float64,
        )


def _read_volume_scale_mpc_h3(catalog_path: Path) -> float:
    sample_info_path = catalog_path.parent / "sample_info.txt"
    if not sample_info_path.exists():
        return 1.0
    try:
        lines = sample_info_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VideCatalogError(f"Cannot read VIDE sample info: {sample_info_path}") from exc

    prefix = "Estimated mean tracer separation (Mpc/h):"
    for line in lines:
        if line.startswith(prefix):
            try:
                mean_separation = float(line.split(":", maxsplit=1)[1])
            except ValueError as exc:
                raise VideCatalogError(
                    f"Invalid mean tracer separation in {sample_info_path}"
                ) from exc
            if not np.isfinite(mean_separation) or mean_separation <= 0.0:
                raise VideCatalogError(
                    f"Invalid mean tracer separation in {sample_info_path}"
                )
            return mean_separation**3
    return 1.0


def read_vide_void_desc(
    path: str | Path,
    *,
    volume_scale_mpc_h3: float | None = None,
) -> VideVoidCatalog:
    """Read a VIDE ``voidDesc`` ASCII catalog."""

    catalog_path = Path(path)
    try:
        lines = [line.strip() for line in catalog_path.read_text().splitlines() if line.strip()]
    except OSError as exc:
        raise VideCatalogError(f"Cannot read VIDE catalog: {catalog_path}") from exc

    if len(lines) < 2:
        raise VideCatalogError(f"VIDE catalog is missing header rows: {catalog_path}")

    summary = lines[0]
    columns = tuple(lines[1].split())
    if "VoidVol" not in columns:
        raise VideCatalogError(f"VIDE catalog is missing VoidVol column: {catalog_path}")

    rows: list[list[float]] = []
    for line_number, line in enumerate(lines[2:], start=3):
        parts = line.split()
        if len(parts) != len(columns):
            raise VideCatalogError(
                f"VIDE row {line_number} has {len(parts)} columns; expected {len(columns)}"
            )
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise VideCatalogError(f"VIDE row {line_number} contains non-numeric values") from exc

    data = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.empty((0, len(columns)), dtype=np.float64)
    )
    scale = (
        _read_volume_scale_mpc_h3(catalog_path)
        if volume_scale_mpc_h3 is None
        else float(volume_scale_mpc_h3)
    )
    catalog = VideVoidCatalog(
        data=data,
        source=catalog_path,
        columns=columns,
        summary=summary,
        volume_scale_mpc_h3=scale,
    )
    if len(catalog) and np.any(catalog.raw_void_volumes <= 0):
        raise VideCatalogError(f"VIDE catalog contains non-positive VoidVol values: {catalog_path}")
    return catalog
