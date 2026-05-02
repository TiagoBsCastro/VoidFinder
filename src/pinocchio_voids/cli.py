"""Command-line interface for pinocchio_voids."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pinocchio_voids.config import load_run_config
from pinocchio_voids.evaluation import compare_void_size_functions
from pinocchio_voids.io import read_paired_pinocchio_halo_catalogs, read_vide_void_desc
from pinocchio_voids.voidfinder import (
    DirectionalVoidFinderResult,
    PairedVoidFinderConfig,
    run_paired_halo_void_finder,
)

app = typer.Typer(help="PINOCCHIO-based cosmic void finder utilities.")
console = Console()


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
    if reference_path is None or len(result.voids) == 0:
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
    """Run the geometry-only paired-halo prototype on existing catalogs."""

    paired = read_paired_pinocchio_halo_catalogs(
        catalog_a,
        catalog_b,
        box_size_mpc_h=box_size_mpc_h,
    )
    config = PairedVoidFinderConfig(
        linking_length_mpc_h=linking_length_mpc_h,
        min_cluster_members=min_cluster_members,
        min_cluster_mass_msun_h=min_cluster_mass_msun_h,
        reference_rho_bar_msun_h_mpc3=reference_rho_bar_msun_h_mpc3,
        radius_a0=radius_a0,
        radius_alpha=radius_alpha,
        adjacency_factor=adjacency_factor,
    )
    result = run_paired_halo_void_finder(
        paired.catalog_a,
        paired.catalog_b,
        config=config,
    )

    table = Table(title="Paired Prototype Summary")
    table.add_column("Target")
    table.add_column("Source")
    table.add_column("Halos", justify="right")
    table.add_column("Clusters", justify="right")
    table.add_column("Protos", justify="right")
    table.add_column("Edges", justify="right")
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
