"""Validated configuration models for pinocchio_voids workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class PinocchioCatalogConfig(BaseModel):
    """Configuration for one PINOCCHIO halo catalog input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path = Field(description="Path to a PINOCCHIO halo catalog.")
    format: Literal["pinocchio_halo_ascii"] = Field(
        default="pinocchio_halo_ascii",
        description="Supported catalog format identifier.",
    )
    redshift: float = Field(default=0.0, description="Catalog redshift.")
    box_size_mpc_h: PositiveFloat = Field(
        description="Periodic box size in comoving Mpc/h."
    )
    mass_unit: str = Field(default="Msun/h")
    position_unit: str = Field(default="Mpc/h")
    velocity_unit: str = Field(default="km/s")
    position_mode: Literal["final", "initial"] = Field(
        default="final",
        description="PINOCCHIO coordinate columns used as canonical halo positions.",
    )
    wrap_positions: bool = Field(
        default=True,
        description="Whether selected positions should be wrapped into the periodic box.",
    )


class RunConfig(BaseModel):
    """Top-level configuration for a local void-finder run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, description="Human-readable run name.")
    catalog: PinocchioCatalogConfig
    output_dir: Path = Field(
        default=Path("runs/pinocchio_voids"),
        description="Directory for generated package outputs.",
    )


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run configuration from a YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)

    if raw_config is None:
        raw_config = {}

    return RunConfig.model_validate(raw_config)
