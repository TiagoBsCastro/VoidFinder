import csv

import numpy as np

from scripts.optimize_n256_full_algorithm_mcmc import (
    BLOB_NAMES,
    N256FullMcmcPaths,
    PARAMETER_NAMES,
    main,
    write_best_fit_command,
)


def test_full_optimizer_script_writes_mcmc_products(tmp_path) -> None:
    output_prefix = tmp_path / "n256_full_mcmc"

    exit_code = main(
        [
            "--catalog-a",
            "tests/fixtures/pinocchio_pair_a.out",
            "--catalog-b",
            "tests/fixtures/pinocchio_pair_b.out",
            "--vide-a",
            "tests/fixtures/vide_voidDesc_all_small.out",
            "--vide-b",
            "tests/fixtures/vide_voidDesc_all_small.out",
            "--vide-centers-a",
            "tests/fixtures/vide_centers_all_small.out",
            "--vide-centers-b",
            "tests/fixtures/vide_centers_all_small.out",
            "--vide-macrocenters-a",
            "tests/fixtures/vide_macrocenters_all_small.out",
            "--vide-macrocenters-b",
            "tests/fixtures/vide_macrocenters_all_small.out",
            "--walkers",
            "16",
            "--steps",
            "8",
            "--burn-in",
            "2",
            "--thin",
            "1",
            "--seed",
            "123",
            "--center-weight",
            "0",
            "--allow-degenerate",
            "--output-prefix",
            str(output_prefix),
        ]
    )

    assert exit_code == 0
    expected_paths = [
        output_prefix.with_name(output_prefix.name + "_chain.npz"),
        output_prefix.with_name(output_prefix.name + "_samples.csv"),
        output_prefix.with_name(output_prefix.name + "_summary.csv"),
        output_prefix.with_name(output_prefix.name + "_trace.png"),
        output_prefix.with_name(output_prefix.name + "_contours.png"),
        output_prefix.with_name(output_prefix.name + "_best_fit_command.sh"),
    ]
    for path in expected_paths:
        assert path.exists()

    with output_prefix.with_name(output_prefix.name + "_samples.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        sample_rows = list(csv.DictReader(handle))
    with output_prefix.with_name(output_prefix.name + "_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        summary_rows = list(csv.DictReader(handle))

    assert sample_rows
    assert all(name in sample_rows[0] for name in BLOB_NAMES)
    assert [row["parameter"] for row in summary_rows] == list(PARAMETER_NAMES)
    assert all(row["best_vsf_log_likelihood"] for row in summary_rows)


def test_full_best_fit_command_includes_scored_merge_parameters(tmp_path) -> None:
    command_path = tmp_path / "full_best_fit.sh"

    write_best_fit_command(
        command_path,
        best_fit=np.array([0.12, 5.0, 1.0, 0.3, 0.7, 0.8, 1.2, 0.4]),
        vide_center_kind="center",
        paths=N256FullMcmcPaths(
            vide_a=tmp_path / "untrimmed_voidDesc_all_a.out",
            vide_b=tmp_path / "untrimmed_voidDesc_all_b.out",
            vide_centers_a=tmp_path / "untrimmed_centers_all_a.out",
            vide_centers_b=tmp_path / "untrimmed_centers_all_b.out",
            vide_macrocenters_a=tmp_path / "untrimmed_macrocenters_all_a.out",
            vide_macrocenters_b=tmp_path / "untrimmed_macrocenters_all_b.out",
            vide_variant="untrimmed",
            position_mode="initial",
        ),
    )

    command = command_path.read_text(encoding="utf-8")
    assert "untrimmed_voidDesc_all_a.out" in command
    assert "--merge-score-mode weighted" in command
    assert "--merge-threshold 0.7" in command
    assert "--bridge-radius-factor 0.8" in command
    assert "--bridge-weight 1.2" in command
    assert "--compatibility-weight 0.4" in command
    assert "--position-mode initial" in command
    assert "n256_full_best_fit_paper_bins_vsf_untrimmed_initial.csv" in command
