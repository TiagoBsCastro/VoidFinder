"""Command-line interface for pinocchio_voids."""

import typer
from rich.console import Console

app = typer.Typer(help="PINOCCHIO-based cosmic void finder utilities.")
console = Console()


@app.callback()
def main() -> None:
    """PINOCCHIO-based cosmic void finder utilities."""


@app.command("smoke-test")
def smoke_test() -> None:
    """Run a minimal installation smoke test."""
    console.print("pinocchio_voids smoke test succeeded.")
