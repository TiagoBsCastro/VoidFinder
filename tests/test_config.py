from pathlib import Path

from pinocchio_voids.config import RunConfig, load_run_config


def test_run_config_validates_catalog_settings() -> None:
    config = RunConfig.model_validate(
        {
            "name": "small-test",
            "catalog": {
                "path": "tests/fixtures/pinocchio_catalog_small.out",
                "box_size_mpc_h": 256.0,
            },
        }
    )

    assert config.name == "small-test"
    assert config.catalog.format == "pinocchio_halo_ascii"
    assert config.catalog.position_mode == "final"
    assert config.catalog.wrap_positions is True


def test_load_run_config_from_yaml() -> None:
    config_path = Path("tests/fixtures/run_config_small.yaml")

    config = load_run_config(config_path)

    assert config.name == "fixture-test"
    assert config.catalog.box_size_mpc_h == 256.0
    assert config.output_dir == Path("runs/test")
