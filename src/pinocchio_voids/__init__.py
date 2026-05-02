"""Tools for PINOCCHIO-based cosmic void finding."""

from pinocchio_voids.catalog import CatalogValidationError, HaloCatalog, TracerCatalog
from pinocchio_voids.evaluation import (
    EvaluationError,
    VoidSizeFunction,
    VoidSizeFunctionComparison,
    compare_void_size_functions,
    compute_void_size_function,
)
from pinocchio_voids.geometry import (
    PeriodicGeometryError,
    minimum_image_displacement,
    periodic_center_of_mass,
    periodic_distance,
)
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    FinalVoid,
    FinalVoidCatalog,
    PairedVoidFinderConfig,
    PairedVoidFinderResult,
    Protovoid,
    ProtovoidCatalog,
    ProtovoidEdge,
    SourceCluster,
    SourceClusterCatalog,
    VoidFinderError,
    build_protovoid_adjacency,
    find_source_clusters,
    lagrangian_radius_from_mass,
    merge_protovoids,
    protovoid_radius_from_mass,
    run_directional_void_finder,
    run_paired_halo_void_finder,
    source_clusters_to_protovoids,
)

__all__ = [
    "CatalogValidationError",
    "DirectionalVoidFinderResult",
    "EvaluationError",
    "FinalVoid",
    "FinalVoidCatalog",
    "HaloCatalog",
    "PairedVoidFinderConfig",
    "PairedVoidFinderResult",
    "PeriodicGeometryError",
    "Protovoid",
    "ProtovoidCatalog",
    "ProtovoidEdge",
    "SourceCluster",
    "SourceClusterCatalog",
    "TracerCatalog",
    "VoidFinderError",
    "VoidSizeFunction",
    "VoidSizeFunctionComparison",
    "__version__",
    "build_protovoid_adjacency",
    "compare_void_size_functions",
    "compute_void_size_function",
    "find_source_clusters",
    "lagrangian_radius_from_mass",
    "merge_protovoids",
    "minimum_image_displacement",
    "periodic_center_of_mass",
    "periodic_distance",
    "protovoid_radius_from_mass",
    "run_directional_void_finder",
    "run_paired_halo_void_finder",
    "source_clusters_to_protovoids",
]

__version__ = "0.1.0"
