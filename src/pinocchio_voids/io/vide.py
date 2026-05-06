"""Readers for VIDE reference void catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pinocchio_voids.geometry import spherical_equivalent_radius_from_volume


VIDE_CATALOG_VARIANTS: tuple[str, ...] = (
    "all",
    "trimmed_nodencut",
    "untrimmed",
    "untrimmed_dencut",
)
_VIDE_CATALOG_VARIANT_ALIASES = {"default": "all"}
_VIDE_CATALOG_VARIANT_PREFIXES = (
    "trimmed_nodencut_",
    "untrimmed_dencut_",
    "untrimmed_",
)


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


@dataclass(frozen=True)
class VideVoidCenterCatalog:
    """In-memory representation of a VIDE ``centers_all`` catalog."""

    data: NDArray[np.float64]
    source: Path

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 14:
            raise VideCatalogError("VIDE centers data must have shape (n, 14)")
        if not np.all(np.isfinite(data)):
            raise VideCatalogError(f"Centers catalog contains non-finite values: {self.source}")
        readonly = data.copy()
        readonly.setflags(write=False)
        object.__setattr__(self, "data", readonly)
        object.__setattr__(self, "source", Path(self.source))

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def positions_mpc_h(self) -> NDArray[np.float64]:
        return self.data[:, :3]

    @property
    def file_void_ids(self) -> NDArray[np.int64]:
        return self.data[:, 7].astype(np.int64)

    @property
    def radii_mpc_h(self) -> NDArray[np.float64]:
        return self.data[:, 4]


@dataclass(frozen=True)
class VideVoidMacrocenterCatalog:
    """In-memory representation of a VIDE ``macrocenters_all`` catalog."""

    data: NDArray[np.float64]
    source: Path

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 4:
            raise VideCatalogError("VIDE macrocenters data must have shape (n, 4)")
        if not np.all(np.isfinite(data)):
            raise VideCatalogError(
                f"Macrocenters catalog contains non-finite values: {self.source}"
            )
        readonly = data.copy()
        readonly.setflags(write=False)
        object.__setattr__(self, "data", readonly)
        object.__setattr__(self, "source", Path(self.source))

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def file_void_ids(self) -> NDArray[np.int64]:
        return self.data[:, 0].astype(np.int64)

    @property
    def positions_mpc_h(self) -> NDArray[np.float64]:
        return self.data[:, 1:4]


@dataclass(frozen=True)
class VideTracerCatalog:
    """Tracer rows from the ASCII catalog passed to VIDE."""

    ids: NDArray[np.int64]
    positions_mpc_h: NDArray[np.float64]
    velocities_km_s: NDArray[np.float64]
    masses_msun_h: NDArray[np.float64]
    source: Path
    box_size_mpc_h: float
    omega_m: float
    omega_lambda: float
    redshift: float

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids, dtype=np.int64)
        positions = np.asarray(self.positions_mpc_h, dtype=np.float64)
        velocities = np.asarray(self.velocities_km_s, dtype=np.float64)
        masses = np.asarray(self.masses_msun_h, dtype=np.float64)
        if ids.ndim != 1:
            raise VideCatalogError("VIDE tracer ids must be one-dimensional")
        if positions.shape != (len(ids), 3):
            raise VideCatalogError("VIDE tracer positions must have shape (n, 3)")
        if velocities.shape != (len(ids), 3):
            raise VideCatalogError("VIDE tracer velocities must have shape (n, 3)")
        if masses.shape != (len(ids),):
            raise VideCatalogError("VIDE tracer masses must have shape (n,)")
        if not np.all(np.isfinite(positions)):
            raise VideCatalogError(f"Tracer catalog contains non-finite positions: {self.source}")
        if not np.all(np.isfinite(velocities)):
            raise VideCatalogError(f"Tracer catalog contains non-finite velocities: {self.source}")
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise VideCatalogError(f"Tracer catalog contains invalid masses: {self.source}")
        for name in ("box_size_mpc_h", "omega_m", "omega_lambda"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise VideCatalogError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        redshift = float(self.redshift)
        if not np.isfinite(redshift):
            raise VideCatalogError("redshift must be finite")
        object.__setattr__(self, "redshift", redshift)
        ids_readonly = ids.copy()
        positions_readonly = positions.copy()
        velocities_readonly = velocities.copy()
        masses_readonly = masses.copy()
        ids_readonly.setflags(write=False)
        positions_readonly.setflags(write=False)
        velocities_readonly.setflags(write=False)
        masses_readonly.setflags(write=False)
        object.__setattr__(self, "ids", ids_readonly)
        object.__setattr__(self, "positions_mpc_h", positions_readonly)
        object.__setattr__(self, "velocities_km_s", velocities_readonly)
        object.__setattr__(self, "masses_msun_h", masses_readonly)
        object.__setattr__(self, "source", Path(self.source))

    def __len__(self) -> int:
        return int(self.ids.shape[0])


@dataclass(frozen=True)
class VideParticleVolumeCatalog:
    """Voronoi volumes from VIDE ``vol_*.dat`` binary files."""

    volumes: NDArray[np.float64]
    source: Path
    header_count: int

    def __post_init__(self) -> None:
        volumes = np.asarray(self.volumes, dtype=np.float64)
        if volumes.ndim != 1:
            raise VideCatalogError("VIDE particle volumes must be one-dimensional")
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise VideCatalogError(f"VIDE particle volumes must be positive: {self.source}")
        header_count = int(self.header_count)
        if header_count != len(volumes):
            raise VideCatalogError(
                f"VIDE volume header count {header_count} does not match "
                f"{len(volumes)} values: {self.source}"
            )
        readonly = volumes.copy()
        readonly.setflags(write=False)
        object.__setattr__(self, "volumes", readonly)
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "header_count", header_count)

    def __len__(self) -> int:
        return int(self.volumes.shape[0])


@dataclass(frozen=True)
class VideVoidZoneCatalog:
    """Void-to-zone membership from VIDE ``voidZone_*.dat`` binary files."""

    zone_ids_by_file_void_id: tuple[NDArray[np.int64], ...]
    source: Path

    def __post_init__(self) -> None:
        zone_ids: list[NDArray[np.int64]] = []
        for zones in self.zone_ids_by_file_void_id:
            array = np.asarray(zones, dtype=np.int64)
            if array.ndim != 1:
                raise VideCatalogError("VIDE void zone ids must be one-dimensional")
            if np.any(array < 0):
                raise VideCatalogError(f"VIDE void zone ids must be non-negative: {self.source}")
            readonly = array.copy()
            readonly.setflags(write=False)
            zone_ids.append(readonly)
        object.__setattr__(self, "zone_ids_by_file_void_id", tuple(zone_ids))
        object.__setattr__(self, "source", Path(self.source))

    def __len__(self) -> int:
        return len(self.zone_ids_by_file_void_id)

    def zones_for_file_void_id(self, file_void_id: int) -> NDArray[np.int64]:
        try:
            return self.zone_ids_by_file_void_id[int(file_void_id)]
        except IndexError as exc:
            raise VideCatalogError(f"Unknown VIDE FileVoid# {file_void_id}") from exc


@dataclass(frozen=True)
class VideZoneParticleCatalog:
    """Zone-to-particle membership from VIDE ``voidPart_*.dat`` binary files."""

    particle_ids_by_zone_id: tuple[NDArray[np.int64], ...]
    source: Path
    header_count: int

    def __post_init__(self) -> None:
        particle_ids: list[NDArray[np.int64]] = []
        for particles in self.particle_ids_by_zone_id:
            array = np.asarray(particles, dtype=np.int64)
            if array.ndim != 1:
                raise VideCatalogError("VIDE zone particle ids must be one-dimensional")
            if np.any(array < 0):
                raise VideCatalogError(
                    f"VIDE zone particle ids must be non-negative: {self.source}"
                )
            readonly = array.copy()
            readonly.setflags(write=False)
            particle_ids.append(readonly)
        header_count = int(self.header_count)
        if header_count <= 0:
            raise VideCatalogError("VIDE voidPart header count must be positive")
        object.__setattr__(self, "particle_ids_by_zone_id", tuple(particle_ids))
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "header_count", header_count)

    def __len__(self) -> int:
        return len(self.particle_ids_by_zone_id)

    def particles_for_zone_id(self, zone_id: int) -> NDArray[np.int64]:
        try:
            return self.particle_ids_by_zone_id[int(zone_id)]
        except IndexError as exc:
            raise VideCatalogError(f"Unknown VIDE zone ID {zone_id}") from exc


def vide_particle_ids_for_file_void_id(
    *,
    file_void_id: int,
    void_zones: VideVoidZoneCatalog,
    zone_particles: VideZoneParticleCatalog,
) -> NDArray[np.int64]:
    """Return unique VIDE input tracer row indices belonging to ``FileVoid#``."""

    particle_chunks = [
        zone_particles.particles_for_zone_id(int(zone_id))
        for zone_id in void_zones.zones_for_file_void_id(file_void_id)
    ]
    if not particle_chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(particle_chunks).astype(np.int64, copy=False))


def normalize_vide_catalog_variant(variant: str) -> str:
    """Return a canonical VIDE catalog variant name."""

    normalized = _VIDE_CATALOG_VARIANT_ALIASES.get(str(variant), str(variant))
    if normalized not in VIDE_CATALOG_VARIANTS:
        choices = ", ".join(VIDE_CATALOG_VARIANTS)
        raise VideCatalogError(f"Unknown VIDE catalog variant {variant!r}; choose one of {choices}")
    return normalized


def strip_vide_catalog_variant_prefix(filename: str) -> str:
    """Remove any known VIDE catalog variant prefix from a filename."""

    for prefix in _VIDE_CATALOG_VARIANT_PREFIXES:
        if filename.startswith(prefix):
            return filename[len(prefix) :]
    return filename


def resolve_vide_catalog_variant_path(path: str | Path, variant: str) -> Path:
    """Resolve a VIDE catalog path for a requested output variant.

    ``all`` maps to the unprefixed VIDE catalog. Other variants prepend the
    variant name to the unprefixed catalog filename. Existing known prefixes are
    stripped first so repeated variant switching is deterministic.
    """

    catalog_path = Path(path)
    canonical = normalize_vide_catalog_variant(variant)
    base_name = strip_vide_catalog_variant_prefix(catalog_path.name)
    if canonical == "all":
        return catalog_path.with_name(base_name)
    return catalog_path.with_name(f"{canonical}_{base_name}")


def vide_catalog_variant_output_suffix(variant: str) -> str:
    """Return a filename suffix for non-default VIDE catalog variants."""

    canonical = normalize_vide_catalog_variant(variant)
    return "" if canonical == "all" else f"_{canonical}"


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


def read_vide_void_centers(path: str | Path) -> VideVoidCenterCatalog:
    """Read a VIDE ``centers_all`` ASCII catalog."""

    catalog_path = Path(path)
    rows: list[list[float]] = []
    try:
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VideCatalogError(f"Cannot read VIDE centers catalog: {catalog_path}") from exc

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 14:
            raise VideCatalogError(
                f"VIDE centers row {line_number} has {len(parts)} columns; expected 14"
            )
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise VideCatalogError(
                f"VIDE centers row {line_number} contains non-numeric values"
            ) from exc

    data = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.empty((0, 14), dtype=np.float64)
    )
    catalog = VideVoidCenterCatalog(data=data, source=catalog_path)
    if len(catalog):
        if np.any(catalog.radii_mpc_h <= 0.0):
            raise VideCatalogError(
                f"VIDE centers catalog contains non-positive radii: {catalog_path}"
            )
    return catalog


def read_vide_void_macrocenters(path: str | Path) -> VideVoidMacrocenterCatalog:
    """Read a VIDE ``macrocenters_all`` ASCII catalog."""

    catalog_path = Path(path)
    rows: list[list[float]] = []
    try:
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VideCatalogError(
            f"Cannot read VIDE macrocenters catalog: {catalog_path}"
        ) from exc

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 4:
            raise VideCatalogError(
                f"VIDE macrocenters row {line_number} has {len(parts)} columns; expected 4"
            )
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise VideCatalogError(
                f"VIDE macrocenters row {line_number} contains non-numeric values"
            ) from exc

    data = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.empty((0, 4), dtype=np.float64)
    )
    return VideVoidMacrocenterCatalog(data=data, source=catalog_path)


def _read_int32(handle, *, count: int, path: Path, label: str) -> NDArray[np.int32]:
    values = np.fromfile(handle, dtype=np.int32, count=count)
    if values.size != count:
        raise VideCatalogError(f"Unexpected end of file while reading {label}: {path}")
    return values


def read_vide_particle_volumes(path: str | Path) -> VideParticleVolumeCatalog:
    """Read VIDE ``vol_*.dat`` binary Voronoi volumes."""

    catalog_path = Path(path)
    try:
        with catalog_path.open("rb") as handle:
            header_count = int(_read_int32(handle, count=1, path=catalog_path, label="volume count")[0])
            volumes = np.fromfile(handle, dtype=np.float32)
    except OSError as exc:
        raise VideCatalogError(f"Cannot read VIDE particle volumes: {catalog_path}") from exc
    return VideParticleVolumeCatalog(
        volumes=np.asarray(volumes, dtype=np.float64),
        source=catalog_path,
        header_count=header_count,
    )


def read_vide_void_zones(path: str | Path) -> VideVoidZoneCatalog:
    """Read VIDE ``voidZone_*.dat`` binary void-to-zone membership."""

    catalog_path = Path(path)
    zones_by_void: list[NDArray[np.int64]] = []
    try:
        with catalog_path.open("rb") as handle:
            void_count = int(_read_int32(handle, count=1, path=catalog_path, label="void count")[0])
            for _ in range(void_count):
                zone_count = int(
                    _read_int32(handle, count=1, path=catalog_path, label="void zone count")[0]
                )
                if zone_count < 0:
                    raise VideCatalogError(f"Negative VIDE void zone count: {catalog_path}")
                zones = _read_int32(
                    handle,
                    count=zone_count,
                    path=catalog_path,
                    label="void zone IDs",
                ).astype(np.int64)
                zones_by_void.append(zones)
            trailing = np.fromfile(handle, dtype=np.uint8, count=1)
    except OSError as exc:
        raise VideCatalogError(f"Cannot read VIDE void zones: {catalog_path}") from exc
    if trailing.size:
        raise VideCatalogError(f"VIDE voidZone file has trailing bytes: {catalog_path}")
    return VideVoidZoneCatalog(zone_ids_by_file_void_id=tuple(zones_by_void), source=catalog_path)


def read_vide_zone_particles(path: str | Path) -> VideZoneParticleCatalog:
    """Read VIDE ``voidPart_*.dat`` binary zone-to-particle membership."""

    catalog_path = Path(path)
    particles_by_zone: list[NDArray[np.int64]] = []
    try:
        with catalog_path.open("rb") as handle:
            header_count = int(
                _read_int32(handle, count=1, path=catalog_path, label="voidPart header")[0]
            )
            zone_count = int(
                _read_int32(handle, count=1, path=catalog_path, label="zone count")[0]
            )
            for _ in range(zone_count):
                particle_count = int(
                    _read_int32(handle, count=1, path=catalog_path, label="zone particle count")[0]
                )
                if particle_count < 0:
                    raise VideCatalogError(f"Negative VIDE zone particle count: {catalog_path}")
                particles = _read_int32(
                    handle,
                    count=particle_count,
                    path=catalog_path,
                    label="zone particle IDs",
                ).astype(np.int64)
                particles_by_zone.append(particles)
            trailing = np.fromfile(handle, dtype=np.uint8, count=1)
    except OSError as exc:
        raise VideCatalogError(f"Cannot read VIDE zone particles: {catalog_path}") from exc
    if trailing.size:
        raise VideCatalogError(f"VIDE voidPart file has trailing bytes: {catalog_path}")
    return VideZoneParticleCatalog(
        particle_ids_by_zone_id=tuple(particles_by_zone),
        source=catalog_path,
        header_count=header_count,
    )


def read_vide_input_tracers(path: str | Path) -> VideTracerCatalog:
    """Read the ASCII tracer catalog generated for VIDE."""

    catalog_path = Path(path)
    try:
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VideCatalogError(f"Cannot read VIDE input tracer catalog: {catalog_path}") from exc
    if len(lines) < 5:
        raise VideCatalogError(f"VIDE input tracer catalog is missing header rows: {catalog_path}")
    try:
        box_size = float(lines[0].strip())
        omega_m = float(lines[1].strip())
        omega_lambda = float(lines[2].strip())
        redshift = float(lines[3].strip())
        tracer_count = int(lines[4].strip())
    except ValueError as exc:
        raise VideCatalogError(f"Invalid VIDE input tracer header: {catalog_path}") from exc
    try:
        raw = np.loadtxt(catalog_path, skiprows=5, dtype=np.float64)
    except ValueError as exc:
        raise VideCatalogError(f"Invalid VIDE input tracer rows: {catalog_path}") from exc
    data = (
        np.atleast_2d(raw).astype(np.float64, copy=False)
        if np.asarray(raw).size
        else np.empty((0, 8), dtype=np.float64)
    )
    if data.shape[0] != tracer_count:
        raise VideCatalogError(
            f"VIDE tracer header count {tracer_count} does not match "
            f"{data.shape[0]} rows: {catalog_path}"
        )
    if data.shape[1] != 8:
        raise VideCatalogError(f"VIDE input tracer rows must have 8 columns: {catalog_path}")
    return VideTracerCatalog(
        ids=data[:, 0].astype(np.int64),
        positions_mpc_h=data[:, 1:4],
        velocities_km_s=data[:, 4:7],
        masses_msun_h=data[:, 7],
        source=catalog_path,
        box_size_mpc_h=box_size,
        omega_m=omega_m,
        omega_lambda=omega_lambda,
        redshift=redshift,
    )
