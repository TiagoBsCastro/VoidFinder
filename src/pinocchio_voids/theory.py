"""Analytic void size-function predictions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TheoryError(ValueError):
    """Raised when theoretical size-function inputs are invalid."""


@dataclass(frozen=True)
class PinocchioCosmologyTable:
    """Scale-dependent PINOCCHIO cosmology quantities."""

    smoothing_radii_mpc: NDArray[np.float64]
    sigma2: NDArray[np.float64]
    dlog_sigma2_dlog_r: NDArray[np.float64]
    h: float
    source: Path

    def __post_init__(self) -> None:
        radii = _readonly_array("smoothing_radii_mpc", self.smoothing_radii_mpc)
        sigma2 = _readonly_array("sigma2", self.sigma2)
        derivative = _readonly_array("dlog_sigma2_dlog_r", self.dlog_sigma2_dlog_r)
        if radii.ndim != 1 or sigma2.ndim != 1 or derivative.ndim != 1:
            raise TheoryError("cosmology arrays must be one-dimensional")
        if not (radii.size == sigma2.size == derivative.size):
            raise TheoryError("cosmology arrays must have matching lengths")
        if radii.size < 2:
            raise TheoryError("cosmology table must contain at least two scale rows")
        if (
            not np.all(np.isfinite(radii))
            or not np.all(np.isfinite(sigma2))
            or not np.all(np.isfinite(derivative))
        ):
            raise TheoryError("cosmology table contains non-finite values")
        if np.any(radii <= 0.0) or np.any(sigma2 <= 0.0):
            raise TheoryError("cosmology radii and variances must be positive")
        if np.any(np.diff(radii) <= 0.0):
            raise TheoryError("cosmology smoothing radii must be strictly increasing")
        h = float(self.h)
        if not np.isfinite(h) or h <= 0.0:
            raise TheoryError("h must be positive and finite")
        object.__setattr__(self, "smoothing_radii_mpc", radii)
        object.__setattr__(self, "sigma2", sigma2)
        object.__setattr__(self, "dlog_sigma2_dlog_r", derivative)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "source", Path(self.source))


@dataclass(frozen=True)
class TheoreticalVoidSizeFunction:
    """Theoretical differential void size function, ``dn / dlnR``."""

    model: str
    bin_edges_mpc_h: NDArray[np.float64]
    bin_centers_mpc_h: NDArray[np.float64]
    density_dndlnr_per_mpc_h3: NDArray[np.float64]

    def __post_init__(self) -> None:
        edges = _readonly_array("bin_edges_mpc_h", self.bin_edges_mpc_h)
        centers = _readonly_array("bin_centers_mpc_h", self.bin_centers_mpc_h)
        density = _readonly_array(
            "density_dndlnr_per_mpc_h3",
            self.density_dndlnr_per_mpc_h3,
            allow_nan=True,
        )
        if edges.ndim != 1 or centers.ndim != 1 or density.ndim != 1:
            raise TheoryError("theory size-function arrays must be one-dimensional")
        if edges.size != centers.size + 1 or centers.size != density.size:
            raise TheoryError("theory size-function arrays have inconsistent lengths")
        if not np.all(np.isfinite(edges)) or np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
            raise TheoryError("theory bin edges must be positive, finite, and increasing")
        if not np.all(np.isfinite(centers)) or np.any(centers <= 0.0):
            raise TheoryError("theory bin centers must be positive and finite")
        object.__setattr__(self, "bin_edges_mpc_h", edges)
        object.__setattr__(self, "bin_centers_mpc_h", centers)
        object.__setattr__(self, "density_dndlnr_per_mpc_h3", density)


@dataclass(frozen=True)
class VdnSvdwFactors:
    """Intermediate factors in the Vdn/SVdW abundance calculation."""

    eulerian_radii_mpc_h: NDArray[np.float64]
    lagrangian_radii_mpc_h: NDArray[np.float64]
    lagrangian_radii_mpc: NDArray[np.float64]
    sigma: NDArray[np.float64]
    dlog_sigma_inv_dlog_r: NDArray[np.float64]
    first_crossing_fraction: NDArray[np.float64]
    denominator_volume_mpc_h3: NDArray[np.float64]
    density_dndlnr_per_mpc_h3: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        numeric_names = (
            "eulerian_radii_mpc_h",
            "lagrangian_radii_mpc_h",
            "lagrangian_radii_mpc",
            "sigma",
            "dlog_sigma_inv_dlog_r",
            "first_crossing_fraction",
            "denominator_volume_mpc_h3",
            "density_dndlnr_per_mpc_h3",
        )
        arrays = [_readonly_array(name, getattr(self, name), allow_nan=True) for name in numeric_names]
        valid = np.asarray(self.valid, dtype=np.bool_).copy()
        if valid.ndim != 1:
            raise TheoryError("valid mask must be one-dimensional")
        if any(array.ndim != 1 for array in arrays) or any(array.size != valid.size for array in arrays):
            raise TheoryError("Vdn/SVdW factor arrays must be one-dimensional with matching lengths")
        for name, array in zip(numeric_names, arrays):
            object.__setattr__(self, name, array)
        valid.setflags(write=False)
        object.__setattr__(self, "valid", valid)


def _readonly_array(
    name: str,
    values: ArrayLike,
    *,
    allow_nan: bool = False,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).copy()
    finite_or_nan = np.isfinite(array) | np.isnan(array) if allow_nan else np.isfinite(array)
    if not np.all(finite_or_nan):
        raise TheoryError(f"{name} contains invalid numeric values")
    array.setflags(write=False)
    return array


def _parse_h_from_header(path: Path) -> float:
    pattern = re.compile(r"\bh\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)")
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("#"):
                    break
                match = pattern.search(line)
                if match is not None:
                    h = float(match.group(1))
                    if h > 0.0 and np.isfinite(h):
                        return h
    except OSError as exc:
        raise TheoryError(f"Cannot read cosmology file: {path}") from exc
    raise TheoryError(f"Cannot find positive h value in cosmology header: {path}")


def read_pinocchio_cosmology(path: str | Path) -> PinocchioCosmologyTable:
    """Read scale-dependent quantities from a PINOCCHIO cosmology table."""

    cosmology_path = Path(path)
    h = _parse_h_from_header(cosmology_path)
    try:
        raw = np.loadtxt(cosmology_path, comments="#", dtype=np.float64)
    except OSError as exc:
        raise TheoryError(f"Cannot read cosmology file: {cosmology_path}") from exc
    except ValueError as exc:
        raise TheoryError(f"Invalid numeric cosmology table: {cosmology_path}") from exc

    data = np.atleast_2d(raw)
    if data.shape[1] < 18:
        raise TheoryError(
            "PINOCCHIO cosmology tables must include columns 15, 16, and 18"
        )

    radii = data[:, 14]
    sigma2 = data[:, 15]
    derivative = data[:, 17]
    valid = (
        np.isfinite(radii)
        & np.isfinite(sigma2)
        & np.isfinite(derivative)
        & (radii > 0.0)
        & (sigma2 > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise TheoryError("cosmology table contains fewer than two valid scale rows")

    order = np.argsort(radii[valid])
    return PinocchioCosmologyTable(
        smoothing_radii_mpc=radii[valid][order],
        sigma2=sigma2[valid][order],
        dlog_sigma2_dlog_r=derivative[valid][order],
        h=h,
        source=cosmology_path,
    )


def svdw_first_crossing_fraction(
    sigma: ArrayLike,
    *,
    delta_v_linear: float = -2.7,
    delta_c_linear: float = 1.686,
    terms: int = 4,
) -> NDArray[np.float64]:
    """Return the SVdW two-barrier ``f_ln_sigma`` approximation."""

    sigma_values = np.asarray(sigma, dtype=np.float64)
    if not np.all(np.isfinite(sigma_values)) or np.any(sigma_values <= 0.0):
        raise TheoryError("sigma must contain positive finite values")
    abs_delta_v = abs(float(delta_v_linear))
    delta_c = float(delta_c_linear)
    if not np.isfinite(abs_delta_v) or abs_delta_v <= 0.0 or float(delta_v_linear) >= 0.0:
        raise TheoryError("delta_v_linear must be negative and finite")
    if not np.isfinite(delta_c) or delta_c <= 0.0:
        raise TheoryError("delta_c_linear must be positive and finite")
    if terms < 1:
        raise TheoryError("terms must be at least 1")

    barrier_ratio = abs_delta_v / (delta_c + abs_delta_v)
    x = barrier_ratio * sigma_values / abs_delta_v
    single_barrier = (
        np.sqrt(2.0 / np.pi)
        * abs_delta_v
        / sigma_values
        * np.exp(-(abs_delta_v**2) / (2.0 * sigma_values**2))
    )

    series = np.zeros_like(sigma_values, dtype=np.float64)
    for index in range(1, terms + 1):
        j = float(index)
        series += (
            np.exp(-((j * np.pi * x) ** 2) / 2.0)
            * j
            * np.pi
            * x**2
            * np.sin(j * np.pi * barrier_ratio)
        )
    values = np.where(x <= 0.276, single_barrier, 2.0 * series)
    return np.maximum(values, 0.0)


def compute_vdn_svdw_factors(
    radii_mpc_h: ArrayLike,
    cosmology: PinocchioCosmologyTable,
    *,
    delta_v_linear: float = -2.7,
    delta_c_linear: float = 1.686,
    delta_v_nonlinear: float = -0.8,
    terms: int = 4,
    apply_h_conversion: bool = True,
    volume_denominator: str = "eulerian",
) -> VdnSvdwFactors:
    """Compute Vdn/SVdW factors at Eulerian radius samples."""

    radii = np.asarray(radii_mpc_h, dtype=np.float64)
    if radii.ndim != 1:
        raise TheoryError("radii_mpc_h must be one-dimensional")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise TheoryError("radii_mpc_h must contain positive finite radii")
    delta_nl = float(delta_v_nonlinear)
    if not np.isfinite(delta_nl) or delta_nl <= -1.0:
        raise TheoryError("delta_v_nonlinear must be finite and greater than -1")
    if volume_denominator not in {"eulerian", "lagrangian"}:
        raise TheoryError("volume_denominator must be 'eulerian' or 'lagrangian'")

    lagrangian_factor = (1.0 + delta_nl) ** (1.0 / 3.0)
    lagrangian_radii_mpc_h = radii * lagrangian_factor
    if apply_h_conversion:
        lagrangian_radii_mpc = lagrangian_radii_mpc_h / cosmology.h
    else:
        lagrangian_radii_mpc = lagrangian_radii_mpc_h.copy()

    sigma = np.full_like(radii, np.nan, dtype=np.float64)
    dlog_sigma_inv = np.full_like(radii, np.nan, dtype=np.float64)
    first_crossing = np.full_like(radii, np.nan, dtype=np.float64)
    density = np.full_like(radii, np.nan, dtype=np.float64)
    denominator_radii = radii if volume_denominator == "eulerian" else lagrangian_radii_mpc_h
    denominator_volumes = (4.0 * np.pi / 3.0) * denominator_radii**3

    valid = (
        (lagrangian_radii_mpc >= cosmology.smoothing_radii_mpc[0])
        & (lagrangian_radii_mpc <= cosmology.smoothing_radii_mpc[-1])
    )
    if np.any(valid):
        log_radii = np.log(cosmology.smoothing_radii_mpc)
        log_sigma2 = np.log(cosmology.sigma2)
        query = np.log(lagrangian_radii_mpc[valid])
        sigma2 = np.exp(np.interp(query, log_radii, log_sigma2))
        derivative = np.interp(query, log_radii, cosmology.dlog_sigma2_dlog_r)
        sigma[valid] = np.sqrt(sigma2)
        dlog_sigma_inv[valid] = np.maximum(-0.5 * derivative, 0.0)
        first_crossing[valid] = svdw_first_crossing_fraction(
            sigma[valid],
            delta_v_linear=delta_v_linear,
            delta_c_linear=delta_c_linear,
            terms=terms,
        )
        density[valid] = (
            first_crossing[valid]
            * dlog_sigma_inv[valid]
            / denominator_volumes[valid]
        )

    return VdnSvdwFactors(
        eulerian_radii_mpc_h=radii,
        lagrangian_radii_mpc_h=lagrangian_radii_mpc_h,
        lagrangian_radii_mpc=lagrangian_radii_mpc,
        sigma=sigma,
        dlog_sigma_inv_dlog_r=dlog_sigma_inv,
        first_crossing_fraction=first_crossing,
        denominator_volume_mpc_h3=denominator_volumes,
        density_dndlnr_per_mpc_h3=density,
        valid=valid,
    )


def compute_vdn_svdw_size_function(
    bin_edges_mpc_h: ArrayLike,
    cosmology: PinocchioCosmologyTable,
    *,
    delta_v_linear: float = -2.7,
    delta_c_linear: float = 1.686,
    delta_v_nonlinear: float = -0.8,
    terms: int = 4,
    apply_h_conversion: bool = True,
    volume_denominator: str = "eulerian",
) -> TheoreticalVoidSizeFunction:
    """Compute the Vdn/SVdW differential abundance on radius bins."""

    edges = np.asarray(bin_edges_mpc_h, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise TheoryError("bin_edges_mpc_h must be a one-dimensional array with at least two edges")
    if not np.all(np.isfinite(edges)) or np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise TheoryError("bin_edges_mpc_h must be positive, finite, and increasing")
    centers = np.sqrt(edges[:-1] * edges[1:])
    factors = compute_vdn_svdw_factors(
        centers,
        cosmology,
        delta_v_linear=delta_v_linear,
        delta_c_linear=delta_c_linear,
        delta_v_nonlinear=delta_v_nonlinear,
        terms=terms,
        apply_h_conversion=apply_h_conversion,
        volume_denominator=volume_denominator,
    )

    return TheoreticalVoidSizeFunction(
        model="vdn-svdw",
        bin_edges_mpc_h=edges,
        bin_centers_mpc_h=centers,
        density_dndlnr_per_mpc_h3=factors.density_dndlnr_per_mpc_h3,
    )
