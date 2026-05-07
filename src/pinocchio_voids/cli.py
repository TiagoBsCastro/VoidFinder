"""Command-line interface for pinocchio_voids."""

from pathlib import Path
from typing import Literal, Optional

import typer
from rich.console import Console
from rich.table import Table

from pinocchio_voids.calibration import mean_halo_spacing_mpc_h, sweep_geometry_parameters
from pinocchio_voids.config import load_run_config
from pinocchio_voids.evaluation import compare_void_size_functions
from pinocchio_voids.io import read_paired_pinocchio_halo_catalogs, read_vide_void_desc
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    PairedVoidFinderConfig,
    run_paired_halo_void_finder,
)

app = typer.Typer(help="PINOCCHIO-based cosmic void finder utilities.")
console = Console(width=120)


@app.callback()
def main() -> None:
    """PINOCCHIO-based cosmic void finder utilities."""


@app.command("smoke-test")
def smoke_test() -> None:
    """Run a minimal installation smoke test."""
    console.print("pinocchio_voids smoke test succeeded.")


@app.command("validate-config")
def validate_config(config_path: Path) -> None:
    """Validate a YAML run configuration."""
    config = load_run_config(config_path)
    console.print(f"Valid configuration: {config.name}")


def _reference_count(reference_path: Optional[Path]) -> str:
    if reference_path is None:
        return "-"
    return str(len(read_vide_void_desc(reference_path)))


def _void_radii(result: DirectionalVoidFinderResult) -> list[float]:
    return [void.effective_radius_mpc_h for void in result.voids]


def _add_direction_row(
    table: Table,
    *,
    target_label: str,
    source_halo_count: int,
    result: DirectionalVoidFinderResult,
    reference_path: Optional[Path],
) -> None:
    table.add_row(
        target_label,
        result.source_label,
        str(source_halo_count),
        str(len(result.source_clusters)),
        str(len(result.protovoids)),
        str(len(result.adjacency_edges)),
        str(len(result.merge_edges)),
        str(len(result.voids)),
        _reference_count(reference_path),
    )


def _add_size_function_row(
    table: Table,
    *,
    target_label: str,
    result: DirectionalVoidFinderResult,
    reference_path: Optional[Path],
    box_size_mpc_h: float,
    bins: int,
) -> bool:
    if reference_path is None:
        return False

    reference = read_vide_void_desc(reference_path)
    if len(reference) == 0:
        return False

    comparison = compare_void_size_functions(
        _void_radii(result),
        reference.effective_radii_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
        bins=bins,
    )
    table.add_row(
        target_label,
        str(int(comparison.predicted.counts.sum())),
        str(int(comparison.reference.counts.sum())),
        str(comparison.count_l1_difference),
    )
    return True


@app.command("paired-prototype")
def paired_prototype(
    catalog_a: Path,
    catalog_b: Path,
    box_size_mpc_h: float = typer.Option(..., "--box-size", help="Periodic box size in Mpc/h."),
    position_mode: Literal["final", "initial"] = typer.Option(
        "final",
        "--position-mode",
        help="PINOCCHIO coordinate columns used by the finder.",
    ),
    reference_rho_bar_msun_h_mpc3: float = typer.Option(
        ...,
        "--rho-bar",
        help="Mean matter density used by the protovoid radius mapping.",
    ),
    linking_length_mpc_h: float = typer.Option(
        8.0,
        "--linking-length",
        help="Periodic FoF-like source-cluster linking length in Mpc/h.",
    ),
    linking_factor: Optional[float] = typer.Option(
        None,
        "--linking-factor",
        help="Use this source mean-spacing factor instead of a fixed linking length.",
    ),
    min_cluster_members: int = typer.Option(
        2,
        "--min-cluster-members",
        help="Minimum halo count for source clusters.",
    ),
    min_cluster_mass_msun_h: float = typer.Option(
        0.0,
        "--min-cluster-mass",
        help="Minimum source-cluster mass in Msun/h.",
    ),
    radius_a0: float = typer.Option(
        1.0,
        "--radius-a0",
        help="Protovoid radius normalization.",
    ),
    radius_alpha: float = typer.Option(
        1.0,
        "--radius-alpha",
        help="Protovoid radius power-law slope.",
    ),
    adjacency_factor: float = typer.Option(
        1.0,
        "--adjacency-factor",
        help="Adjacency threshold multiplier for protovoid merging.",
    ),
    merge_score_mode: Literal["geometry_only", "weighted"] = typer.Option(
        "geometry_only",
        "--merge-score-mode",
        help="Use all adjacency edges or threshold weighted merge scores.",
    ),
    merge_threshold: float = typer.Option(
        0.0,
        "--merge-threshold",
        help="Minimum weighted merge score required to merge an adjacency edge.",
    ),
    geom_weight: float = typer.Option(
        1.0,
        "--geom-weight",
        help="Weight applied to the geometric edge score.",
    ),
    bridge_weight: float = typer.Option(
        0.0,
        "--bridge-weight",
        help="Weight applied to the source-catalog bridge score.",
    ),
    compatibility_weight: float = typer.Option(
        0.0,
        "--compatibility-weight",
        help="Weight applied to protovoid/source-cluster compatibility.",
    ),
    bridge_radius_factor: float = typer.Option(
        0.5,
        "--bridge-radius-factor",
        help="Bridge capsule radius factor for weighted merging.",
    ),
    bridge_min_radius_mpc_h: float = typer.Option(
        0.0,
        "--bridge-min-radius",
        help="Minimum bridge capsule radius in Mpc/h.",
    ),
    bridge_delta_scale: float = typer.Option(
        1.0,
        "--bridge-delta-scale",
        help="Overdensity scale used to map bridge density to a 0..1 score.",
    ),
    bridge_density_mode: Literal["number", "mass", "both"] = typer.Option(
        "mass",
        "--bridge-density-mode",
        help="Halo density field used for bridge scoring.",
    ),
    vide_a: Optional[Path] = typer.Option(
        None,
        "--vide-a",
        help="Optional VIDE voidDesc reference for target catalog A.",
    ),
    vide_b: Optional[Path] = typer.Option(
        None,
        "--vide-b",
        help="Optional VIDE voidDesc reference for target catalog B.",
    ),
    size_bins: int = typer.Option(
        0,
        "--size-bins",
        help="If positive, compare predicted and VIDE void size functions with this many bins.",
    ),
) -> None:
    """Run the paired-halo void finder on existing catalogs."""

    paired = read_paired_pinocchio_halo_catalogs(
        catalog_a,
        catalog_b,
        box_size_mpc_h=box_size_mpc_h,
        position_mode=position_mode,
    )
    if linking_factor is None:
        source_a_linking_length = linking_length_mpc_h
        source_b_linking_length = None
    else:
        source_a_linking_length = linking_factor * mean_halo_spacing_mpc_h(paired.catalog_a)
        source_b_linking_length = linking_factor * mean_halo_spacing_mpc_h(paired.catalog_b)
    config = PairedVoidFinderConfig(
        linking_length_mpc_h=source_a_linking_length,
        source_b_linking_length_mpc_h=source_b_linking_length,
        min_cluster_members=min_cluster_members,
        min_cluster_mass_msun_h=min_cluster_mass_msun_h,
        reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
        radius_a0=radius_a0,
        radius_alpha=radius_alpha,
        adjacency_factor=adjacency_factor,
        merge_score_mode=merge_score_mode,
        merge_threshold=merge_threshold,
        geom_weight=geom_weight,
        bridge_weight=bridge_weight,
        compatibility_weight=compatibility_weight,
        bridge_radius_factor=bridge_radius_factor,
        bridge_min_radius_mpc_h=bridge_min_radius_mpc_h,
        bridge_delta_scale=bridge_delta_scale,
        bridge_density_mode=bridge_density_mode,
    )
    result = run_paired_halo_void_finder(
        paired.catalog_a,
        paired.catalog_b,
        config=config,
    )

    table = Table(title=f"Paired Prototype Summary ({position_mode} positions)")
    table.add_column("Target")
    table.add_column("Source")
    table.add_column("Halos", justify="right")
    table.add_column("Clusters", justify="right")
    table.add_column("Protos", justify="right")
    table.add_column("Edges", justify="right")
    table.add_column("Merge Edges", justify="right")
    table.add_column("Voids", justify="right")
    table.add_column("VIDE", justify="right")

    _add_direction_row(
        table,
        target_label="A",
        source_halo_count=len(paired.catalog_b),
        result=result.voids_a,
        reference_path=vide_a,
    )
    _add_direction_row(
        table,
        target_label="B",
        source_halo_count=len(paired.catalog_a),
        result=result.voids_b,
        reference_path=vide_b,
    )
    console.print(table)

    if size_bins > 0 and (vide_a is not None or vide_b is not None):
        size_table = Table(title="VSF Count Difference")
        size_table.add_column("Target")
        size_table.add_column("Pred", justify="right")
        size_table.add_column("VIDE", justify="right")
        size_table.add_column("L1", justify="right")
        rows_added = 0
        rows_added += int(
            _add_size_function_row(
                size_table,
                target_label="A",
                result=result.voids_a,
                reference_path=vide_a,
                box_size_mpc_h=box_size_mpc_h,
                bins=size_bins,
            )
        )
        rows_added += int(
            _add_size_function_row(
                size_table,
                target_label="B",
                result=result.voids_b,
                reference_path=vide_b,
                box_size_mpc_h=box_size_mpc_h,
                bins=size_bins,
            )
        )
        if rows_added:
            console.print(size_table)


@app.command("paired-sweep")
def paired_sweep(
    catalog_a: Path,
    catalog_b: Path,
    vide_a: Path,
    vide_b: Path,
    box_size_mpc_h: float = typer.Option(..., "--box-size", help="Periodic box size in Mpc/h."),
    position_mode: Literal["final", "initial"] = typer.Option(
        "final",
        "--position-mode",
        help="PINOCCHIO coordinate columns used by the finder.",
    ),
    reference_rho_bar_msun_h_mpc3: float = typer.Option(
        ...,
        "--rho-bar",
        help="Mean matter density used by the protovoid radius mapping.",
    ),
    linking_lengths_mpc_h: list[float] = typer.Option(
        [],
        "--linking-length",
        help="FoF linking length to test. Repeat for multiple values.",
    ),
    linking_length_mean_spacing_factors: list[float] = typer.Option(
        [],
        "--linking-factor",
        help="Source halo mean-spacing factor to test. Repeat for multiple values.",
    ),
    radius_a0_values: list[float] = typer.Option(
        [1.0],
        "--radius-a0",
        help="Protovoid radius normalization to test. Repeat for multiple values.",
    ),
    radius_alpha_values: list[float] = typer.Option(
        [1.0],
        "--radius-alpha",
        help="Protovoid radius slope to test. Repeat for multiple values.",
    ),
    adjacency_factors: list[float] = typer.Option(
        [1.0],
        "--adjacency-factor",
        help="Adjacency threshold multiplier to test. Repeat for multiple values.",
    ),
    min_cluster_members: int = typer.Option(
        2,
        "--min-cluster-members",
        help="Minimum halo count for source clusters.",
    ),
    min_cluster_mass_msun_h: float = typer.Option(
        0.0,
        "--min-cluster-mass",
        help="Minimum source-cluster mass in Msun/h.",
    ),
    size_bins: int = typer.Option(
        6,
        "--size-bins",
        help="Number of shared log-radius bins used for VIDE size-function scoring.",
    ),
    min_predicted_fraction: float = typer.Option(
        0.25,
        "--min-predicted-fraction",
        help="Penalize sweep rows below this predicted/VIDE count fraction.",
    ),
    top: int = typer.Option(
        10,
        "--top",
        help="Maximum number of sorted sweep rows to print.",
    ),
) -> None:
    """Sweep geometry-only paired-prototype parameters against VIDE catalogs."""

    paired = read_paired_pinocchio_halo_catalogs(
        catalog_a,
        catalog_b,
        box_size_mpc_h=box_size_mpc_h,
        position_mode=position_mode,
    )
    reference_a = read_vide_void_desc(vide_a)
    reference_b = read_vide_void_desc(vide_b)
    fixed_linking_lengths = linking_lengths_mpc_h
    if not fixed_linking_lengths and not linking_length_mean_spacing_factors:
        fixed_linking_lengths = [8.0]
    results = sweep_geometry_parameters(
        paired.catalog_a,
        paired.catalog_b,
        reference_a=reference_a,
        reference_b=reference_b,
        reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
        linking_lengths_mpc_h=fixed_linking_lengths,
        linking_length_mean_spacing_factors=linking_length_mean_spacing_factors,
        radius_a0_values=radius_a0_values,
        radius_alpha_values=radius_alpha_values,
        adjacency_factors=adjacency_factors,
        min_cluster_members=min_cluster_members,
        min_cluster_mass_msun_h=min_cluster_mass_msun_h,
        bins=size_bins,
        min_predicted_fraction=min_predicted_fraction,
    )

    table = Table(title=f"Geometry Sweep ({position_mode} positions)")
    table.add_column("Rank", justify="right")
    table.add_column("Mode")
    table.add_column("Link", justify="right")
    table.add_column("Src A/B Link", justify="right")
    table.add_column("a0", justify="right")
    table.add_column("alpha", justify="right")
    table.add_column("adj", justify="right")
    table.add_column("A Pred/VIDE", justify="right")
    table.add_column("B Pred/VIDE", justify="right")
    table.add_column("Raw", justify="right")
    table.add_column("Guard", justify="right")
    table.add_column("Deg", justify="right")

    for rank, result in enumerate(results[:top], start=1):
        config = result.config
        table.add_row(
            str(rank),
            "factor" if result.linking_mode == "mean_spacing" else "fixed",
            f"{result.linking_value:g}",
            (
                f"{result.source_a_linking_length_mpc_h:g}/"
                f"{result.source_b_linking_length_mpc_h:g}"
            ),
            f"{config.radius_a0:g}",
            f"{config.radius_alpha:g}",
            f"{config.adjacency_factor:g}",
            f"{result.score_a.predicted_void_count}/{result.score_a.reference_void_count}",
            f"{result.score_b.predicted_void_count}/{result.score_b.reference_void_count}",
            str(result.total_count_l1_difference),
            str(result.total_guarded_count_l1_difference),
            "yes" if result.is_degenerate_underprediction else "no",
        )
    console.print(table)
