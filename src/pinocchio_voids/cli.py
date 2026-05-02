"""Command-line interface for pinocchio_voids."""

from pathlib import Path

import typer
from rich.console import Console

from pinocchio_voids.config import load_run_config

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
