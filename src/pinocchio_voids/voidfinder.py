"""Paired-halo void finder with geometry-only and scored-merge modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from pinocchio_voids.catalog import HaloCatalog
from pinocchio_voids.geometry import (
    minimum_image_displacement,
    periodic_center_of_mass,
    periodic_distance,
    sphere_volume_from_radius,
    spherical_equivalent_radius_from_volume,
)

BridgeDensityMode = Literal["number", "mass", "both"]
BridgeNormalization = Literal["global_mean"]
MergeScoreMode = Literal["geometry_only", "weighted"]
MergedRadiusMode = Literal["volume_sum", "mass_sum"]


class VoidFinderError(ValueError):
    """Raised when void-finder inputs or parameters are invalid."""


def _readonly_1d_int(name: str, values: ArrayLike) -> NDArray[np.int64]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise VoidFinderError(f"{name} must be one-dimensional")
    readonly = array.copy()
    readonly.setflags(write=False)
    return readonly


def _readonly_3_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise VoidFinderError(f"{name} must be a finite 3-vector")
    readonly = array.copy()
    readonly.setflags(write=False)
    return readonly


def _readonly_3x3_matrix(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise VoidFinderError(f"{name} must be a finite 3x3 matrix")
    readonly = array.copy()
    readonly.setflags(write=False)
    return readonly


def _validate_positive(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise VoidFinderError(f"{name} must be positive and finite")
    return number


def _validate_optional_positive(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _validate_positive(name, value)


def _validate_at_least(name: str, value: float, minimum: float) -> float:
    number = _validate_positive(name, value)
    if number < minimum:
        raise VoidFinderError(f"{name} must be at least {minimum:g}")
    return number


def _validate_unit_interval(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise VoidFinderError(f"{name} must be between 0 and 1")
    return number


def _validate_non_negative(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise VoidFinderError(f"{name} must be non-negative and finite")
    return number


@dataclass(frozen=True)
class SourceCluster:
    """Summary of a compact halo source cluster.

    ``effective_radius_mpc_h`` is the source-cluster RMS size, not the final
    void ``R_eff`` used in size-function comparisons.
    """

    id: int
    member_indices: ArrayLike
    total_mass_msun_h: float
    center_mpc_h: ArrayLike
    richness: int
    effective_radius_mpc_h: float
    shape_tensor_mpc_h2: ArrayLike | None = None
    axis_ratio: float = 1.0
    max_member_distance_mpc_h: float = 0.0
    rms_radius_over_linking_length: float = 0.0
    mass_concentration_proxy: float = 1.0

    def __post_init__(self) -> None:
        member_indices = _readonly_1d_int("member_indices", self.member_indices)
        if member_indices.size == 0:
            raise VoidFinderError("member_indices must not be empty")
        total_mass = _validate_positive("total_mass_msun_h", self.total_mass_msun_h)
        center = _readonly_3_vector("center_mpc_h", self.center_mpc_h)
        richness = int(self.richness)
        if richness != member_indices.size:
            raise VoidFinderError("richness must match the number of member indices")
        effective_radius = _validate_non_negative(
            "effective_radius_mpc_h",
            self.effective_radius_mpc_h,
        )
        if self.shape_tensor_mpc_h2 is None:
            shape_tensor = np.zeros((3, 3), dtype=np.float64)
            shape_tensor.setflags(write=False)
        else:
            shape_tensor = _readonly_3x3_matrix(
                "shape_tensor_mpc_h2",
                self.shape_tensor_mpc_h2,
            )
        axis_ratio = _validate_at_least("axis_ratio", self.axis_ratio, 1.0)
        max_member_distance = _validate_non_negative(
            "max_member_distance_mpc_h",
            self.max_member_distance_mpc_h,
        )
        rms_radius_over_linking_length = _validate_non_negative(
            "rms_radius_over_linking_length",
            self.rms_radius_over_linking_length,
        )
        mass_concentration_proxy = _validate_at_least(
            "mass_concentration_proxy",
            self.mass_concentration_proxy,
            1.0,
        )

        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "member_indices", member_indices)
        object.__setattr__(self, "total_mass_msun_h", total_mass)
        object.__setattr__(self, "center_mpc_h", center)
        object.__setattr__(self, "richness", richness)
        object.__setattr__(self, "effective_radius_mpc_h", effective_radius)
        object.__setattr__(self, "shape_tensor_mpc_h2", shape_tensor)
        object.__setattr__(self, "axis_ratio", axis_ratio)
        object.__setattr__(self, "max_member_distance_mpc_h", max_member_distance)
        object.__setattr__(
            self,
            "rms_radius_over_linking_length",
            rms_radius_over_linking_length,
        )
        object.__setattr__(
            self,
            "mass_concentration_proxy",
            mass_concentration_proxy,
        )


@dataclass(frozen=True)
class SourceClusterCatalog:
    """Collection of source-cluster summaries."""

    clusters: tuple[SourceCluster, ...]
    box_size_mpc_h: float
    source_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(self, "box_size_mpc_h", _validate_positive("box_size_mpc_h", self.box_size_mpc_h))

    def __iter__(self) -> Iterable[SourceCluster]:
        return iter(self.clusters)

    def __len__(self) -> int:
        return len(self.clusters)


@dataclass(frozen=True)
class Protovoid:
    """Spherical protovoid produced from one source cluster."""

    id: int
    source_cluster_id: int
    center_mpc_h: ArrayLike
    radius_mpc_h: float
    source_mass_msun_h: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "source_cluster_id", int(self.source_cluster_id))
        object.__setattr__(self, "center_mpc_h", _readonly_3_vector("center_mpc_h", self.center_mpc_h))
        object.__setattr__(self, "radius_mpc_h", _validate_positive("radius_mpc_h", self.radius_mpc_h))
        object.__setattr__(self, "source_mass_msun_h", _validate_positive("source_mass_msun_h", self.source_mass_msun_h))


@dataclass(frozen=True)
class ProtovoidCatalog:
    """Collection of spherical protovoids for one target catalog."""

    protovoids: tuple[Protovoid, ...]
    box_size_mpc_h: float
    target_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "protovoids", tuple(self.protovoids))
        object.__setattr__(self, "box_size_mpc_h", _validate_positive("box_size_mpc_h", self.box_size_mpc_h))

    def __iter__(self) -> Iterable[Protovoid]:
        return iter(self.protovoids)

    def __len__(self) -> int:
        return len(self.protovoids)


@dataclass(frozen=True)
class ProtovoidEdge:
    """Merge candidate between two protovoids."""

    protovoid_i: int
    protovoid_j: int
    distance_mpc_h: float
    geometric_score: float
    bridge_score: float = 0.0
    compatibility_score: float = 0.0
    merge_score: float | None = None
    passes_merge_threshold: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "protovoid_i", int(self.protovoid_i))
        object.__setattr__(self, "protovoid_j", int(self.protovoid_j))
        object.__setattr__(self, "distance_mpc_h", _validate_non_negative("distance_mpc_h", self.distance_mpc_h))
        geometric_score = _validate_unit_interval(
            "geometric_score",
            self.geometric_score,
        )
        object.__setattr__(self, "geometric_score", geometric_score)
        object.__setattr__(
            self,
            "bridge_score",
            _validate_unit_interval("bridge_score", self.bridge_score),
        )
        object.__setattr__(
            self,
            "compatibility_score",
            _validate_unit_interval("compatibility_score", self.compatibility_score),
        )
        merge_score = (
            geometric_score
            if self.merge_score is None
            else _validate_non_negative("merge_score", self.merge_score)
        )
        object.__setattr__(self, "merge_score", merge_score)
        object.__setattr__(
            self,
            "passes_merge_threshold",
            bool(self.passes_merge_threshold),
        )


@dataclass(frozen=True)
class FinalVoid:
    """Merged geometry-only void catalog entry.

    ``effective_radius_mpc_h`` is the spherical-equivalent radius of the
    modeled final void volume.
    """

    id: int
    center_mpc_h: ArrayLike
    effective_radius_mpc_h: float
    member_protovoid_ids: ArrayLike
    source_cluster_ids: ArrayLike
    total_source_mass_msun_h: float
    mean_merge_score: float = 0.0
    max_merge_score: float = 0.0
    total_source_richness: int = 0
    mean_source_compactness: float = 0.0
    max_source_axis_ratio: float = 0.0

    def __post_init__(self) -> None:
        member_ids = _readonly_1d_int("member_protovoid_ids", self.member_protovoid_ids)
        source_ids = _readonly_1d_int("source_cluster_ids", self.source_cluster_ids)
        if member_ids.size == 0 or source_ids.size == 0:
            raise VoidFinderError("void membership arrays must not be empty")

        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "center_mpc_h", _readonly_3_vector("center_mpc_h", self.center_mpc_h))
        object.__setattr__(
            self,
            "effective_radius_mpc_h",
            _validate_positive("effective_radius_mpc_h", self.effective_radius_mpc_h),
        )
        object.__setattr__(self, "member_protovoid_ids", member_ids)
        object.__setattr__(self, "source_cluster_ids", source_ids)
        object.__setattr__(
            self,
            "total_source_mass_msun_h",
            _validate_positive("total_source_mass_msun_h", self.total_source_mass_msun_h),
        )
        object.__setattr__(
            self,
            "mean_merge_score",
            _validate_non_negative("mean_merge_score", self.mean_merge_score),
        )
        object.__setattr__(
            self,
            "max_merge_score",
            _validate_non_negative("max_merge_score", self.max_merge_score),
        )
        total_source_richness = int(self.total_source_richness)
        if total_source_richness < 0:
            raise VoidFinderError("total_source_richness must be non-negative")
        object.__setattr__(self, "total_source_richness", total_source_richness)
        object.__setattr__(
            self,
            "mean_source_compactness",
            _validate_non_negative(
                "mean_source_compactness",
                self.mean_source_compactness,
            ),
        )
        object.__setattr__(
            self,
            "max_source_axis_ratio",
            _validate_non_negative("max_source_axis_ratio", self.max_source_axis_ratio),
        )


@dataclass(frozen=True)
class FinalVoidCatalog:
    """Collection of merged voids for one target catalog."""

    voids: tuple[FinalVoid, ...]
    box_size_mpc_h: float
    target_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "voids", tuple(self.voids))
        object.__setattr__(self, "box_size_mpc_h", _validate_positive("box_size_mpc_h", self.box_size_mpc_h))

    def __iter__(self) -> Iterable[FinalVoid]:
        return iter(self.voids)

    def __len__(self) -> int:
        return len(self.voids)


@dataclass(frozen=True)
class PairedVoidFinderConfig:
    """Free parameters for the paired-halo void finder."""

    linking_length_mpc_h: float
    reference_rho_bar_msun_h_mpc3: float
    source_b_linking_length_mpc_h: float | None = None
    min_cluster_members: int = 1
    min_cluster_mass_msun_h: float = 0.0
    radius_a0: float = 1.0
    radius_alpha: float = 1.0
    adjacency_factor: float = 1.0
    merged_radius_mode: MergedRadiusMode = "volume_sum"
    min_void_radius_mpc_h: float = 0.0
    geom_weight: float = 1.0
    bridge_weight: float = 0.0
    compatibility_weight: float = 0.0
    merge_threshold: float = 0.0
    bridge_radius_factor: float = 0.5
    bridge_min_radius_mpc_h: float = 0.0
    bridge_delta_scale: float = 1.0
    bridge_density_mode: BridgeDensityMode = "mass"
    bridge_normalization: BridgeNormalization = "global_mean"
    merge_score_mode: MergeScoreMode = "geometry_only"
    max_cluster_effective_radius_mpc_h: float | None = None
    max_cluster_axis_ratio: float | None = None
    max_cluster_rms_over_linking_length: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "linking_length_mpc_h",
            _validate_positive("linking_length_mpc_h", self.linking_length_mpc_h),
        )
        if self.source_b_linking_length_mpc_h is not None:
            object.__setattr__(
                self,
                "source_b_linking_length_mpc_h",
                _validate_positive(
                    "source_b_linking_length_mpc_h",
                    self.source_b_linking_length_mpc_h,
                ),
            )
        object.__setattr__(
            self,
            "reference_rho_bar_msun_h_mpc3",
            _validate_positive(
                "reference_rho_bar_msun_h_mpc3",
                self.reference_rho_bar_msun_h_mpc3,
            ),
        )
        min_cluster_members = int(self.min_cluster_members)
        if min_cluster_members < 1:
            raise VoidFinderError("min_cluster_members must be at least 1")
        object.__setattr__(self, "min_cluster_members", min_cluster_members)
        object.__setattr__(
            self,
            "min_cluster_mass_msun_h",
            _validate_non_negative("min_cluster_mass_msun_h", self.min_cluster_mass_msun_h),
        )
        object.__setattr__(self, "radius_a0", _validate_positive("radius_a0", self.radius_a0))
        object.__setattr__(self, "radius_alpha", _validate_positive("radius_alpha", self.radius_alpha))
        object.__setattr__(
            self,
            "adjacency_factor",
            _validate_positive("adjacency_factor", self.adjacency_factor),
        )
        if self.merged_radius_mode not in ("volume_sum", "mass_sum"):
            raise VoidFinderError("merged_radius_mode must be 'volume_sum' or 'mass_sum'")
        object.__setattr__(
            self,
            "min_void_radius_mpc_h",
            _validate_non_negative("min_void_radius_mpc_h", self.min_void_radius_mpc_h),
        )
        object.__setattr__(
            self,
            "geom_weight",
            _validate_non_negative("geom_weight", self.geom_weight),
        )
        object.__setattr__(
            self,
            "bridge_weight",
            _validate_non_negative("bridge_weight", self.bridge_weight),
        )
        object.__setattr__(
            self,
            "compatibility_weight",
            _validate_non_negative("compatibility_weight", self.compatibility_weight),
        )
        object.__setattr__(
            self,
            "merge_threshold",
            _validate_non_negative("merge_threshold", self.merge_threshold),
        )
        object.__setattr__(
            self,
            "bridge_radius_factor",
            _validate_positive("bridge_radius_factor", self.bridge_radius_factor),
        )
        object.__setattr__(
            self,
            "bridge_min_radius_mpc_h",
            _validate_non_negative(
                "bridge_min_radius_mpc_h",
                self.bridge_min_radius_mpc_h,
            ),
        )
        object.__setattr__(
            self,
            "bridge_delta_scale",
            _validate_positive("bridge_delta_scale", self.bridge_delta_scale),
        )
        if self.bridge_density_mode not in ("number", "mass", "both"):
            raise VoidFinderError("bridge_density_mode must be 'number', 'mass', or 'both'")
        if self.bridge_normalization != "global_mean":
            raise VoidFinderError("bridge_normalization must be 'global_mean'")
        if self.merge_score_mode not in ("geometry_only", "weighted"):
            raise VoidFinderError("merge_score_mode must be 'geometry_only' or 'weighted'")
        object.__setattr__(
            self,
            "max_cluster_effective_radius_mpc_h",
            _validate_optional_positive(
                "max_cluster_effective_radius_mpc_h",
                self.max_cluster_effective_radius_mpc_h,
            ),
        )
        object.__setattr__(
            self,
            "max_cluster_axis_ratio",
            (
                None
                if self.max_cluster_axis_ratio is None
                else _validate_at_least(
                    "max_cluster_axis_ratio",
                    self.max_cluster_axis_ratio,
                    1.0,
                )
            ),
        )
        object.__setattr__(
            self,
            "max_cluster_rms_over_linking_length",
            _validate_optional_positive(
                "max_cluster_rms_over_linking_length",
                self.max_cluster_rms_over_linking_length,
            ),
        )


@dataclass(frozen=True)
class DirectionalVoidFinderResult:
    """Void-finder result for one source-to-target direction."""

    source_label: str
    target_label: str
    source_clusters: SourceClusterCatalog
    protovoids: ProtovoidCatalog
    adjacency_edges: tuple[ProtovoidEdge, ...]
    voids: FinalVoidCatalog
    merge_edges: tuple[ProtovoidEdge, ...] = ()


@dataclass(frozen=True)
class PairedVoidFinderResult:
    """Symmetric A-to-B and B-to-A results for a paired realization."""

    voids_a: DirectionalVoidFinderResult
    voids_b: DirectionalVoidFinderResult


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def lagrangian_radius_from_mass(
    mass_msun_h: float,
    *,
    reference_rho_bar_msun_h_mpc3: float,
) -> float:
    """Return ``R_L = (3M / 4 pi rho_bar)^(1/3)``."""

    mass = _validate_positive("mass_msun_h", mass_msun_h)
    rho_bar = _validate_positive(
        "reference_rho_bar_msun_h_mpc3",
        reference_rho_bar_msun_h_mpc3,
    )
    return float(spherical_equivalent_radius_from_volume(mass / rho_bar))


def protovoid_radius_from_mass(
    mass_msun_h: float,
    *,
    radius_a0: float,
    radius_alpha: float,
    reference_rho_bar_msun_h_mpc3: float,
) -> float:
    """Map source-cluster mass to spherical protovoid radius."""

    radius_a0 = _validate_positive("radius_a0", radius_a0)
    radius_alpha = _validate_positive("radius_alpha", radius_alpha)
    lagrangian_radius = lagrangian_radius_from_mass(
        mass_msun_h,
        reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
    )
    return float(radius_a0 * lagrangian_radius**radius_alpha)


def _source_cluster_shape_metrics(
    offsets_mpc_h: NDArray[np.float64],
    masses_msun_h: NDArray[np.float64],
    *,
    linking_length_mpc_h: float,
) -> tuple[NDArray[np.float64], float, float, float, float]:
    total_mass = float(np.sum(masses_msun_h))
    if total_mass <= 0.0:
        raise VoidFinderError("cluster member masses must sum to a positive value")

    shape_tensor = (offsets_mpc_h.T * masses_msun_h) @ offsets_mpc_h / total_mass
    shape_tensor = np.asarray(shape_tensor, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(shape_tensor)
    largest = float(np.max(eigenvalues))
    if largest <= 0.0:
        axis_ratio = 1.0
    else:
        floor = max(largest * 1.0e-12, 1.0e-30)
        smallest = max(float(np.min(eigenvalues)), floor)
        axis_ratio = float(np.sqrt(largest / smallest))
    distances = np.linalg.norm(offsets_mpc_h, axis=1)
    max_distance = float(np.max(distances)) if distances.size else 0.0
    rms_radius = float(np.sqrt(np.average(distances * distances, weights=masses_msun_h)))
    rms_over_linking = rms_radius / linking_length_mpc_h
    mass_concentration = float(np.max(masses_msun_h) / np.mean(masses_msun_h))
    shape_tensor.setflags(write=False)
    return (
        shape_tensor,
        axis_ratio,
        max_distance,
        rms_over_linking,
        mass_concentration,
    )


def find_source_clusters(
    catalog: HaloCatalog,
    *,
    linking_length_mpc_h: float,
    min_cluster_members: int = 1,
    min_cluster_mass_msun_h: float = 0.0,
    max_cluster_effective_radius_mpc_h: float | None = None,
    max_cluster_axis_ratio: float | None = None,
    max_cluster_rms_over_linking_length: float | None = None,
    source_label: str = "",
) -> SourceClusterCatalog:
    """Cluster halos with a periodic FoF-like linking length."""

    linking_length = _validate_positive("linking_length_mpc_h", linking_length_mpc_h)
    min_members = int(min_cluster_members)
    if min_members < 1:
        raise VoidFinderError("min_cluster_members must be at least 1")
    min_mass = _validate_non_negative("min_cluster_mass_msun_h", min_cluster_mass_msun_h)
    max_effective_radius = _validate_optional_positive(
        "max_cluster_effective_radius_mpc_h",
        max_cluster_effective_radius_mpc_h,
    )
    max_axis_ratio = (
        None
        if max_cluster_axis_ratio is None
        else _validate_at_least("max_cluster_axis_ratio", max_cluster_axis_ratio, 1.0)
    )
    max_rms_over_linking = _validate_optional_positive(
        "max_cluster_rms_over_linking_length",
        max_cluster_rms_over_linking_length,
    )

    box_size = catalog.box_size_mpc_h
    positions = np.asarray(catalog.positions_mpc_h, dtype=np.float64) % box_size
    masses = np.asarray(catalog.masses_msun_h, dtype=np.float64)
    halo_count = len(catalog)
    if halo_count == 0:
        return SourceClusterCatalog((), box_size_mpc_h=box_size, source_label=source_label)

    disjoint_set = _DisjointSet(halo_count)
    tree = cKDTree(positions, boxsize=box_size)
    for left, right in tree.query_pairs(linking_length):
        disjoint_set.union(int(left), int(right))

    groups: dict[int, list[int]] = {}
    for index in range(halo_count):
        groups.setdefault(disjoint_set.find(index), []).append(index)

    clusters: list[SourceCluster] = []
    for member_list in sorted(groups.values(), key=lambda members: min(members)):
        member_indices = np.asarray(member_list, dtype=np.int64)
        richness = int(member_indices.size)
        total_mass = float(np.sum(masses[member_indices]))
        if richness < min_members or total_mass < min_mass:
            continue

        member_positions = positions[member_indices]
        member_masses = masses[member_indices]
        center = periodic_center_of_mass(
            member_positions,
            box_size_mpc_h=box_size,
            weights=member_masses,
        )
        offsets = minimum_image_displacement(center, member_positions, box_size)
        squared_distances = np.sum(offsets * offsets, axis=1)
        effective_radius = float(np.sqrt(np.average(squared_distances, weights=member_masses)))
        (
            shape_tensor,
            axis_ratio,
            max_member_distance,
            rms_radius_over_linking,
            mass_concentration,
        ) = _source_cluster_shape_metrics(
            offsets,
            member_masses,
            linking_length_mpc_h=linking_length,
        )
        if max_effective_radius is not None and effective_radius > max_effective_radius:
            continue
        if max_axis_ratio is not None and axis_ratio > max_axis_ratio:
            continue
        if max_rms_over_linking is not None and rms_radius_over_linking > max_rms_over_linking:
            continue
        clusters.append(
            SourceCluster(
                id=len(clusters),
                member_indices=member_indices,
                total_mass_msun_h=total_mass,
                center_mpc_h=center,
                richness=richness,
                effective_radius_mpc_h=effective_radius,
                shape_tensor_mpc_h2=shape_tensor,
                axis_ratio=axis_ratio,
                max_member_distance_mpc_h=max_member_distance,
                rms_radius_over_linking_length=rms_radius_over_linking,
                mass_concentration_proxy=mass_concentration,
            )
        )

    return SourceClusterCatalog(
        tuple(clusters),
        box_size_mpc_h=box_size,
        source_label=source_label,
    )


def source_clusters_to_protovoids(
    source_clusters: SourceClusterCatalog,
    *,
    radius_a0: float,
    radius_alpha: float,
    reference_rho_bar_msun_h_mpc3: float,
    target_label: str = "",
) -> ProtovoidCatalog:
    """Map source clusters to spherical protovoids in the paired target."""

    protovoids = [
        Protovoid(
            id=index,
            source_cluster_id=cluster.id,
            center_mpc_h=cluster.center_mpc_h,
            radius_mpc_h=protovoid_radius_from_mass(
                cluster.total_mass_msun_h,
                radius_a0=radius_a0,
                radius_alpha=radius_alpha,
                reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
            ),
            source_mass_msun_h=cluster.total_mass_msun_h,
        )
        for index, cluster in enumerate(source_clusters)
    ]
    return ProtovoidCatalog(
        tuple(protovoids),
        box_size_mpc_h=source_clusters.box_size_mpc_h,
        target_label=target_label,
    )


def compatibility_score(
    protovoid_i: Protovoid,
    protovoid_j: Protovoid,
    source_cluster_i: SourceCluster,
    source_cluster_j: SourceCluster,
) -> float:
    """Return a bounded similarity score for two protovoid merge candidates."""

    radius_ratio = min(protovoid_i.radius_mpc_h, protovoid_j.radius_mpc_h) / max(
        protovoid_i.radius_mpc_h,
        protovoid_j.radius_mpc_h,
    )
    richness_ratio = min(source_cluster_i.richness, source_cluster_j.richness) / max(
        source_cluster_i.richness,
        source_cluster_j.richness,
    )
    return float(np.clip(0.7 * radius_ratio + 0.3 * richness_ratio, 0.0, 1.0))


def _capsule_membership_mask(
    positions_mpc_h: NDArray[np.float64],
    *,
    start_mpc_h: NDArray[np.float64],
    end_mpc_h: NDArray[np.float64],
    radius_mpc_h: float,
    box_size_mpc_h: float,
) -> NDArray[np.bool_]:
    segment = minimum_image_displacement(start_mpc_h, end_mpc_h, box_size_mpc_h)
    segment_length_squared = float(np.dot(segment, segment))
    offsets = minimum_image_displacement(start_mpc_h, positions_mpc_h, box_size_mpc_h)
    if segment_length_squared <= 0.0:
        distances = np.linalg.norm(offsets, axis=1)
        return distances <= radius_mpc_h

    t = np.clip((offsets @ segment) / segment_length_squared, 0.0, 1.0)
    closest = t[:, np.newaxis] * segment
    transverse = offsets - closest
    distances = np.linalg.norm(transverse, axis=1)
    return distances <= radius_mpc_h


def _bridge_capsule_volume(radius_mpc_h: float, length_mpc_h: float) -> float:
    cylinder_volume = np.pi * radius_mpc_h**2 * length_mpc_h
    cap_volume = 4.0 * np.pi * radius_mpc_h**3 / 3.0
    return float(cylinder_volume + cap_volume)


def bridge_density_score(
    source_catalog: HaloCatalog,
    source_cluster_i: SourceCluster,
    source_cluster_j: SourceCluster,
    *,
    bridge_radius_factor: float,
    bridge_min_radius_mpc_h: float = 0.0,
    bridge_delta_scale: float = 1.0,
    bridge_density_mode: BridgeDensityMode = "mass",
    bridge_normalization: BridgeNormalization = "global_mean",
    source_tree: cKDTree | None = None,
) -> float:
    """Measure source-halo overdensity inside a periodic bridge capsule."""

    bridge_radius_factor = _validate_positive(
        "bridge_radius_factor",
        bridge_radius_factor,
    )
    bridge_min_radius = _validate_non_negative(
        "bridge_min_radius_mpc_h",
        bridge_min_radius_mpc_h,
    )
    bridge_delta_scale = _validate_positive("bridge_delta_scale", bridge_delta_scale)
    if bridge_density_mode not in ("number", "mass", "both"):
        raise VoidFinderError("bridge_density_mode must be 'number', 'mass', or 'both'")
    if bridge_normalization != "global_mean":
        raise VoidFinderError("bridge_normalization must be 'global_mean'")

    box_size = source_catalog.box_size_mpc_h
    start = np.asarray(source_cluster_i.center_mpc_h, dtype=np.float64)
    end = np.asarray(source_cluster_j.center_mpc_h, dtype=np.float64)
    segment = minimum_image_displacement(start, end, box_size)
    length = float(np.linalg.norm(segment))
    base_radius = max(
        source_cluster_i.effective_radius_mpc_h,
        source_cluster_j.effective_radius_mpc_h,
        0.05 * length,
    )
    radius = max(bridge_min_radius, bridge_radius_factor * base_radius)
    if radius <= 0.0:
        return 0.0

    midpoint = (start + 0.5 * segment) % box_size
    query_radius = 0.5 * length + radius
    tree = (
        cKDTree(source_catalog.positions_mpc_h % box_size, boxsize=box_size)
        if source_tree is None
        else source_tree
    )
    candidate_indices = tree.query_ball_point(midpoint, query_radius)
    if not candidate_indices:
        return 0.0

    candidates = np.asarray(candidate_indices, dtype=np.int64)
    positions = np.asarray(source_catalog.positions_mpc_h[candidates], dtype=np.float64)
    inside = _capsule_membership_mask(
        positions,
        start_mpc_h=start,
        end_mpc_h=end,
        radius_mpc_h=radius,
        box_size_mpc_h=box_size,
    )
    if not np.any(inside):
        return 0.0

    excluded = np.union1d(
        source_cluster_i.member_indices,
        source_cluster_j.member_indices,
    )
    bridge_indices = candidates[inside]
    if excluded.size:
        bridge_indices = bridge_indices[~np.isin(bridge_indices, excluded)]
    if bridge_indices.size == 0:
        return 0.0

    volume = _bridge_capsule_volume(radius, length)
    box_volume = box_size**3
    scores: list[float] = []
    if bridge_density_mode in ("number", "both"):
        bridge_density = bridge_indices.size / volume
        mean_density = len(source_catalog) / box_volume
        ratio = 0.0 if mean_density <= 0.0 else bridge_density / mean_density
        scores.append(float(np.clip((ratio - 1.0) / bridge_delta_scale, 0.0, 1.0)))
    if bridge_density_mode in ("mass", "both"):
        bridge_density = float(np.sum(source_catalog.masses_msun_h[bridge_indices])) / volume
        mean_density = float(np.sum(source_catalog.masses_msun_h)) / box_volume
        ratio = 0.0 if mean_density <= 0.0 else bridge_density / mean_density
        scores.append(float(np.clip((ratio - 1.0) / bridge_delta_scale, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0


def merge_score(
    *,
    geometric_score: float,
    bridge_score: float,
    compatibility_score: float,
    geom_weight: float,
    bridge_weight: float,
    compatibility_weight: float,
) -> float:
    """Return the weighted protovoid edge merge score."""

    return float(
        _validate_non_negative("geom_weight", geom_weight)
        * _validate_unit_interval("geometric_score", geometric_score)
        + _validate_non_negative("bridge_weight", bridge_weight)
        * _validate_unit_interval("bridge_score", bridge_score)
        + _validate_non_negative("compatibility_weight", compatibility_weight)
        * _validate_unit_interval("compatibility_score", compatibility_score)
    )


def _clusters_by_id(source_clusters: SourceClusterCatalog | None) -> dict[int, SourceCluster]:
    if source_clusters is None:
        return {}
    return {cluster.id: cluster for cluster in source_clusters}


def build_protovoid_adjacency(
    protovoids: ProtovoidCatalog,
    *,
    adjacency_factor: float,
    source_catalog: HaloCatalog | None = None,
    source_clusters: SourceClusterCatalog | None = None,
    geom_weight: float = 1.0,
    bridge_weight: float = 0.0,
    compatibility_weight: float = 0.0,
    merge_threshold: float = 0.0,
    bridge_radius_factor: float = 0.5,
    bridge_min_radius_mpc_h: float = 0.0,
    bridge_delta_scale: float = 1.0,
    bridge_density_mode: BridgeDensityMode = "mass",
    bridge_normalization: BridgeNormalization = "global_mean",
    merge_score_mode: MergeScoreMode = "geometry_only",
) -> tuple[ProtovoidEdge, ...]:
    """Build candidate protovoid edges and score them for merging."""

    adjacency_factor = _validate_positive("adjacency_factor", adjacency_factor)
    geom_weight = _validate_non_negative("geom_weight", geom_weight)
    bridge_weight = _validate_non_negative("bridge_weight", bridge_weight)
    compatibility_weight = _validate_non_negative(
        "compatibility_weight",
        compatibility_weight,
    )
    merge_threshold = _validate_non_negative("merge_threshold", merge_threshold)
    if merge_score_mode not in ("geometry_only", "weighted"):
        raise VoidFinderError("merge_score_mode must be 'geometry_only' or 'weighted'")
    clusters_by_id = _clusters_by_id(source_clusters)
    source_tree = (
        cKDTree(
            source_catalog.positions_mpc_h % source_catalog.box_size_mpc_h,
            boxsize=source_catalog.box_size_mpc_h,
        )
        if (
            merge_score_mode == "weighted"
            and source_catalog is not None
            and bridge_weight > 0.0
        )
        else None
    )
    needs_source_clusters = merge_score_mode == "weighted" and (
        bridge_weight > 0.0 or compatibility_weight > 0.0
    )
    if needs_source_clusters and source_clusters is None:
        raise VoidFinderError("weighted bridge or compatibility scoring requires source_clusters")
    if merge_score_mode == "weighted" and bridge_weight > 0.0 and source_catalog is None:
        raise VoidFinderError("weighted bridge scoring requires source_catalog")
    if len(protovoids) < 2:
        return ()

    centers = np.asarray([protovoid.center_mpc_h for protovoid in protovoids], dtype=np.float64)
    radii = np.asarray([protovoid.radius_mpc_h for protovoid in protovoids], dtype=np.float64)
    max_search_radius = float(2.0 * adjacency_factor * np.max(radii))
    tree = cKDTree(centers % protovoids.box_size_mpc_h, boxsize=protovoids.box_size_mpc_h)

    edges: list[ProtovoidEdge] = []
    for left, right in sorted(tree.query_pairs(max_search_radius)):
        distance = float(periodic_distance(centers[left], centers[right], protovoids.box_size_mpc_h))
        threshold = float(adjacency_factor * (radii[left] + radii[right]))
        if distance < threshold:
            left_protovoid = protovoids.protovoids[left]
            right_protovoid = protovoids.protovoids[right]
            geometric_score = max(0.0, 1.0 - distance / threshold)
            bridge = 0.0
            compatibility = 0.0
            if merge_score_mode == "weighted":
                left_cluster = clusters_by_id.get(left_protovoid.source_cluster_id)
                right_cluster = clusters_by_id.get(right_protovoid.source_cluster_id)
                if (bridge_weight > 0.0 or compatibility_weight > 0.0) and (
                    left_cluster is None or right_cluster is None
                ):
                    raise VoidFinderError("protovoid source_cluster_id was not found")
                if bridge_weight > 0.0:
                    if source_catalog is None or left_cluster is None or right_cluster is None:
                        raise VoidFinderError("bridge scoring requires source catalog and clusters")
                    bridge = bridge_density_score(
                        source_catalog,
                        left_cluster,
                        right_cluster,
                        bridge_radius_factor=bridge_radius_factor,
                        bridge_min_radius_mpc_h=bridge_min_radius_mpc_h,
                        bridge_delta_scale=bridge_delta_scale,
                        bridge_density_mode=bridge_density_mode,
                        bridge_normalization=bridge_normalization,
                        source_tree=source_tree,
                    )
                if compatibility_weight > 0.0:
                    if left_cluster is None or right_cluster is None:
                        raise VoidFinderError("compatibility scoring requires source clusters")
                    compatibility = compatibility_score(
                        left_protovoid,
                        right_protovoid,
                        left_cluster,
                        right_cluster,
                    )
                score = merge_score(
                    geometric_score=geometric_score,
                    bridge_score=bridge,
                    compatibility_score=compatibility,
                    geom_weight=geom_weight,
                    bridge_weight=bridge_weight,
                    compatibility_weight=compatibility_weight,
                )
                passes = score >= merge_threshold
            else:
                score = geometric_score
                passes = True
            edges.append(
                ProtovoidEdge(
                    protovoid_i=int(left),
                    protovoid_j=int(right),
                    distance_mpc_h=distance,
                    geometric_score=geometric_score,
                    bridge_score=bridge,
                    compatibility_score=compatibility,
                    merge_score=score,
                    passes_merge_threshold=passes,
                )
            )
    return tuple(edges)


def _connected_components(node_count: int, edges: Iterable[ProtovoidEdge]) -> list[list[int]]:
    disjoint_set = _DisjointSet(node_count)
    for edge in edges:
        if not edge.passes_merge_threshold:
            continue
        disjoint_set.union(edge.protovoid_i, edge.protovoid_j)

    components: dict[int, list[int]] = {}
    for index in range(node_count):
        components.setdefault(disjoint_set.find(index), []).append(index)
    return sorted(components.values(), key=lambda component: min(component))


def merge_protovoids(
    protovoids: ProtovoidCatalog,
    adjacency_edges: Iterable[ProtovoidEdge],
    *,
    radius_mode: MergedRadiusMode = "volume_sum",
    min_void_radius_mpc_h: float = 0.0,
    radius_a0: float | None = None,
    radius_alpha: float | None = None,
    reference_rho_bar_msun_h_mpc3: float | None = None,
    source_clusters: SourceClusterCatalog | None = None,
) -> FinalVoidCatalog:
    """Merge accepted graph components into final voids."""

    if radius_mode not in ("volume_sum", "mass_sum"):
        raise VoidFinderError("radius_mode must be 'volume_sum' or 'mass_sum'")
    min_radius = _validate_non_negative("min_void_radius_mpc_h", min_void_radius_mpc_h)
    edges = tuple(edge for edge in adjacency_edges if edge.passes_merge_threshold)
    components = _connected_components(len(protovoids), edges)
    edge_scores_by_pair = {
        tuple(sorted((edge.protovoid_i, edge.protovoid_j))): edge.merge_score
        for edge in edges
    }
    clusters_by_id = _clusters_by_id(source_clusters)

    final_voids: list[FinalVoid] = []
    for component in components:
        members = [protovoids.protovoids[index] for index in component]
        centers = np.asarray([member.center_mpc_h for member in members], dtype=np.float64)
        radii = np.asarray([member.radius_mpc_h for member in members], dtype=np.float64)
        source_masses = np.asarray([member.source_mass_msun_h for member in members], dtype=np.float64)
        weights = radii**3
        center = periodic_center_of_mass(
            centers,
            box_size_mpc_h=protovoids.box_size_mpc_h,
            weights=weights,
        )
        total_mass = float(np.sum(source_masses))
        if radius_mode == "volume_sum":
            effective_radius = float(
                spherical_equivalent_radius_from_volume(
                    np.sum(sphere_volume_from_radius(radii))
                )
            )
        else:
            if (
                radius_a0 is None
                or radius_alpha is None
                or reference_rho_bar_msun_h_mpc3 is None
            ):
                raise VoidFinderError(
                    "mass_sum radius mode requires radius_a0, radius_alpha, "
                    "and reference_rho_bar_msun_h_mpc3"
                )
            effective_radius = protovoid_radius_from_mass(
                total_mass,
                radius_a0=radius_a0,
                radius_alpha=radius_alpha,
                reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
            )

        if effective_radius < min_radius:
            continue
        component_set = set(component)
        component_scores = [
            score
            for (left, right), score in edge_scores_by_pair.items()
            if left in component_set and right in component_set
        ]
        mean_merge_score = float(np.mean(component_scores)) if component_scores else 0.0
        max_merge_score = float(np.max(component_scores)) if component_scores else 0.0
        member_clusters = [
            clusters_by_id[member.source_cluster_id]
            for member in members
            if member.source_cluster_id in clusters_by_id
        ]
        total_source_richness = int(sum(cluster.richness for cluster in member_clusters))
        mean_source_compactness = (
            float(np.mean([cluster.mass_concentration_proxy for cluster in member_clusters]))
            if member_clusters
            else 0.0
        )
        max_source_axis_ratio = (
            float(np.max([cluster.axis_ratio for cluster in member_clusters]))
            if member_clusters
            else 0.0
        )

        final_voids.append(
            FinalVoid(
                id=len(final_voids),
                center_mpc_h=center,
                effective_radius_mpc_h=effective_radius,
                member_protovoid_ids=[member.id for member in members],
                source_cluster_ids=[member.source_cluster_id for member in members],
                total_source_mass_msun_h=total_mass,
                mean_merge_score=mean_merge_score,
                max_merge_score=max_merge_score,
                total_source_richness=total_source_richness,
                mean_source_compactness=mean_source_compactness,
                max_source_axis_ratio=max_source_axis_ratio,
            )
        )

    return FinalVoidCatalog(
        tuple(final_voids),
        box_size_mpc_h=protovoids.box_size_mpc_h,
        target_label=protovoids.target_label,
    )


def run_directional_void_finder(
    source_catalog: HaloCatalog,
    *,
    source_label: str,
    target_label: str,
    config: PairedVoidFinderConfig,
) -> DirectionalVoidFinderResult:
    """Run the source-to-target half of the paired-halo prototype."""

    source_clusters = find_source_clusters(
        source_catalog,
        linking_length_mpc_h=config.linking_length_mpc_h,
        min_cluster_members=config.min_cluster_members,
        min_cluster_mass_msun_h=config.min_cluster_mass_msun_h,
        max_cluster_effective_radius_mpc_h=config.max_cluster_effective_radius_mpc_h,
        max_cluster_axis_ratio=config.max_cluster_axis_ratio,
        max_cluster_rms_over_linking_length=config.max_cluster_rms_over_linking_length,
        source_label=source_label,
    )
    protovoids = source_clusters_to_protovoids(
        source_clusters,
        radius_a0=config.radius_a0,
        radius_alpha=config.radius_alpha,
        reference_rho_bar_msun_h_mpc3=config.reference_rho_bar_msun_h_mpc3,
        target_label=target_label,
    )
    adjacency_edges = build_protovoid_adjacency(
        protovoids,
        adjacency_factor=config.adjacency_factor,
        source_catalog=source_catalog,
        source_clusters=source_clusters,
        geom_weight=config.geom_weight,
        bridge_weight=config.bridge_weight,
        compatibility_weight=config.compatibility_weight,
        merge_threshold=config.merge_threshold,
        bridge_radius_factor=config.bridge_radius_factor,
        bridge_min_radius_mpc_h=config.bridge_min_radius_mpc_h,
        bridge_delta_scale=config.bridge_delta_scale,
        bridge_density_mode=config.bridge_density_mode,
        bridge_normalization=config.bridge_normalization,
        merge_score_mode=config.merge_score_mode,
    )
    merge_edges = tuple(edge for edge in adjacency_edges if edge.passes_merge_threshold)
    final_voids = merge_protovoids(
        protovoids,
        merge_edges,
        radius_mode=config.merged_radius_mode,
        min_void_radius_mpc_h=config.min_void_radius_mpc_h,
        radius_a0=config.radius_a0,
        radius_alpha=config.radius_alpha,
        reference_rho_bar_msun_h_mpc3=config.reference_rho_bar_msun_h_mpc3,
        source_clusters=source_clusters,
    )
    return DirectionalVoidFinderResult(
        source_label=source_label,
        target_label=target_label,
        source_clusters=source_clusters,
        protovoids=protovoids,
        adjacency_edges=adjacency_edges,
        merge_edges=merge_edges,
        voids=final_voids,
    )


def run_paired_halo_void_finder(
    catalog_a: HaloCatalog,
    catalog_b: HaloCatalog,
    *,
    config: PairedVoidFinderConfig,
    label_a: str = "A",
    label_b: str = "B",
) -> PairedVoidFinderResult:
    """Run the symmetric paired-halo prototype in both directions."""

    if not np.isclose(catalog_a.box_size_mpc_h, catalog_b.box_size_mpc_h):
        raise VoidFinderError("paired catalogs must use the same box_size_mpc_h")

    voids_b = run_directional_void_finder(
        catalog_a,
        source_label=label_a,
        target_label=label_b,
        config=config,
    )
    config_b = config
    if config.source_b_linking_length_mpc_h is not None:
        config_b = replace(
            config,
            linking_length_mpc_h=config.source_b_linking_length_mpc_h,
            source_b_linking_length_mpc_h=None,
        )
    voids_a = run_directional_void_finder(
        catalog_b,
        source_label=label_b,
        target_label=label_a,
        config=config_b,
    )
    return PairedVoidFinderResult(voids_a=voids_a, voids_b=voids_b)
