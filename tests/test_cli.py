from typer.testing import CliRunner

from pinocchio_voids.cli import app


def test_paired_prototype_command_reports_direction_summaries() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "paired-prototype",
            "tests/fixtures/pinocchio_pair_a.out",
            "tests/fixtures/pinocchio_pair_b.out",
            "--box-size",
            "10.0",
            "--rho-bar",
            "1.0",
            "--linking-length",
            "0.3",
            "--min-cluster-members",
            "2",
            "--vide-a",
            "tests/fixtures/vide_voidDesc_all_small.out",
            "--vide-b",
            "tests/fixtures/vide_voidDesc_all_small.out",
            "--size-bins",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Paired Prototype Summary" in result.output
    assert "Voids" in result.output
    assert "VIDE" in result.output
    assert "VSF Count Difference" in result.output
    assert "A" in result.output
    assert "B" in result.output
