"""Phase 1 paired-halo void-finder prototype."""

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
)

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


def _validate_positive(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise VoidFinderError(f"{name} must be positive and finite")
    return number


def _validate_non_negative(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise VoidFinderError(f"{name} must be non-negative and finite")
    return number


@dataclass(frozen=True)
class SourceCluster:
    """Summary of a compact halo source cluster."""

    id: int
    member_indices: ArrayLike
    total_mass_msun_h: float
    center_mpc_h: ArrayLike
    richness: int
    effective_radius_mpc_h: float

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

        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "member_indices", member_indices)
        object.__setattr__(self, "total_mass_msun_h", total_mass)
        object.__setattr__(self, "center_mpc_h", center)
        object.__setattr__(self, "richness", richness)
        object.__setattr__(self, "effective_radius_mpc_h", effective_radius)


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
    """Geometry-only merge candidate between two protovoids."""

    protovoid_i: int
    protovoid_j: int
    distance_mpc_h: float
    geometric_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "protovoid_i", int(self.protovoid_i))
        object.__setattr__(self, "protovoid_j", int(self.protovoid_j))
        object.__setattr__(self, "distance_mpc_h", _validate_non_negative("distance_mpc_h", self.distance_mpc_h))
        object.__setattr__(self, "geometric_score", _validate_non_negative("geometric_score", self.geometric_score))


@dataclass(frozen=True)
class FinalVoid:
    """Merged geometry-only void catalog entry."""

    id: int
    center_mpc_h: ArrayLike
    effective_radius_mpc_h: float
    member_protovoid_ids: ArrayLike
    source_cluster_ids: ArrayLike
    total_source_mass_msun_h: float

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
    """Free parameters for the initial paired-halo prototype."""

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


@dataclass(frozen=True)
class DirectionalVoidFinderResult:
    """Void-finder result for one source-to-target direction."""

    source_label: str
    target_label: str
    source_clusters: SourceClusterCatalog
    protovoids: ProtovoidCatalog
    adjacency_edges: tuple[ProtovoidEdge, ...]
    voids: FinalVoidCatalog


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
    return float((3.0 * mass / (4.0 * np.pi * rho_bar)) ** (1.0 / 3.0))


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


def find_source_clusters(
    catalog: HaloCatalog,
    *,
    linking_length_mpc_h: float,
    min_cluster_members: int = 1,
    min_cluster_mass_msun_h: float = 0.0,
    source_label: str = "",
) -> SourceClusterCatalog:
    """Cluster halos with a periodic FoF-like linking length."""

    linking_length = _validate_positive("linking_length_mpc_h", linking_length_mpc_h)
    min_members = int(min_cluster_members)
    if min_members < 1:
        raise VoidFinderError("min_cluster_members must be at least 1")
    min_mass = _validate_non_negative("min_cluster_mass_msun_h", min_cluster_mass_msun_h)

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
        clusters.append(
            SourceCluster(
                id=len(clusters),
                member_indices=member_indices,
                total_mass_msun_h=total_mass,
                center_mpc_h=center,
                richness=richness,
                effective_radius_mpc_h=effective_radius,
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


def build_protovoid_adjacency(
    protovoids: ProtovoidCatalog,
    *,
    adjacency_factor: float,
) -> tuple[ProtovoidEdge, ...]:
    """Build a geometry-only adjacency graph for protovoid merging."""

    adjacency_factor = _validate_positive("adjacency_factor", adjacency_factor)
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
            edges.append(
                ProtovoidEdge(
                    protovoid_i=int(left),
                    protovoid_j=int(right),
                    distance_mpc_h=distance,
                    geometric_score=max(0.0, 1.0 - distance / threshold),
                )
            )
    return tuple(edges)


def _connected_components(node_count: int, edges: Iterable[ProtovoidEdge]) -> list[list[int]]:
    disjoint_set = _DisjointSet(node_count)
    for edge in edges:
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
) -> FinalVoidCatalog:
    """Merge thresholded graph components into final geometry-only voids."""

    if radius_mode not in ("volume_sum", "mass_sum"):
        raise VoidFinderError("radius_mode must be 'volume_sum' or 'mass_sum'")
    min_radius = _validate_non_negative("min_void_radius_mpc_h", min_void_radius_mpc_h)
    edges = tuple(adjacency_edges)
    components = _connected_components(len(protovoids), edges)

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
            effective_radius = float(np.sum(radii**3) ** (1.0 / 3.0))
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

        final_voids.append(
            FinalVoid(
                id=len(final_voids),
                center_mpc_h=center,
                effective_radius_mpc_h=effective_radius,
                member_protovoid_ids=[member.id for member in members],
                source_cluster_ids=[member.source_cluster_id for member in members],
                total_source_mass_msun_h=total_mass,
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
    )
    final_voids = merge_protovoids(
        protovoids,
        adjacency_edges,
        radius_mode=config.merged_radius_mode,
        min_void_radius_mpc_h=config.min_void_radius_mpc_h,
        radius_a0=config.radius_a0,
        radius_alpha=config.radius_alpha,
        reference_rho_bar_msun_h_mpc3=config.reference_rho_bar_msun_h_mpc3,
    )
    return DirectionalVoidFinderResult(
        source_label=source_label,
        target_label=target_label,
        source_clusters=source_clusters,
        protovoids=protovoids,
        adjacency_edges=adjacency_edges,
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
