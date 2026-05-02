"""Readers for VIDE reference void catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class VideCatalogError(ValueError):
    """Raised when a VIDE catalog cannot be parsed or validated."""


@dataclass(frozen=True)
class VideVoidCatalog:
    """In-memory representation of a VIDE ``voidDesc`` catalog."""

    data: NDArray[np.float64]
    source: Path
    columns: tuple[str, ...]
    summary: str = ""

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
    def void_volumes_mpc_h3(self) -> NDArray[np.float64]:
        return self.column("VoidVol")

    @property
    def effective_radii_mpc_h(self) -> NDArray[np.float64]:
        volumes = self.void_volumes_mpc_h3
        return np.power(3.0 * volumes / (4.0 * np.pi), 1.0 / 3.0)


def read_vide_void_desc(path: str | Path) -> VideVoidCatalog:
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
    catalog = VideVoidCatalog(data=data, source=catalog_path, columns=columns, summary=summary)
    if len(catalog) and np.any(catalog.void_volumes_mpc_h3 <= 0):
        raise VideCatalogError(f"VIDE catalog contains non-positive VoidVol values: {catalog_path}")
    return catalog
