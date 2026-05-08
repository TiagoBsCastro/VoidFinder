import csv

import numpy as np

import scripts.optimize_n256_full_algorithm_mcmc as full_mcmc
from scripts.optimize_n256_full_algorithm_mcmc import (
    BLOB_NAMES,
    BLOB_DTYPE,
    N256FullMcmcPaths,
    PARAMETER_NAMES,
    format_diagnostic_summary,
    main,
    posterior_diagnostic_summary,
    write_best_fit_command,
)
from scripts.diagnose_n256_initial_position_grid import main as grid_main


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
    assert output_prefix.with_name(output_prefix.name + "_preflight_positions.csv").exists()
    assert output_prefix.with_name(output_prefix.name + "_preflight_summary.csv").exists()

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


def test_posterior_diagnostic_summary_classifies_rejection_reasons() -> None:
    log_probability = np.array([-1.0, -np.inf, -np.inf, -np.inf, -np.inf])
    blobs = np.array(
        [
            (0.0, -10.0, -5.0, -10.0, -5.0, 1.0, 1.0, 0.5, 0.5, 0.0),
            (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, np.nan, np.nan, np.nan, np.nan, 1.0),
            (0.0, -20.0, -6.0, -20.0, -6.0, 1.1, 1.2, 0.4, 0.3, 1.0),
            (0.0, -np.inf, -7.0, -np.inf, -7.0, 1.3, 1.4, 0.2, 0.1, 0.0),
            (0.0, -30.0, -np.inf, -30.0, -np.inf, 1.5, 1.6, 0.0, 0.0, 0.0),
        ],
        dtype=BLOB_DTYPE,
    )

    summary = posterior_diagnostic_summary(
        log_probability,
        blobs,
        reject_degenerate=True,
    )

    assert summary["finite"] == 1
    assert summary["prior_rejected"] == 1
    assert summary["degenerate_rejected"] == 1
    assert summary["nonfinite_vsf_likelihood"] == 1
    assert summary["nonfinite_center_likelihood"] == 1
    assert summary["other_nonfinite"] == 0
    assert summary["best_log_probability"] == -1.0
    assert "finite: 1 / 5" in format_diagnostic_summary("diagnostic", summary)


def test_full_optimizer_diagnose_initial_state_writes_preflight_products(tmp_path) -> None:
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
            "--seed",
            "123",
            "--center-weight",
            "0",
            "--allow-degenerate",
            "--diagnose-initial-state",
            "--output-prefix",
            str(output_prefix),
        ]
    )

    assert exit_code == 0
    assert output_prefix.with_name(output_prefix.name + "_preflight_positions.csv").exists()
    assert output_prefix.with_name(output_prefix.name + "_preflight_summary.csv").exists()
    assert not output_prefix.with_name(output_prefix.name + "_chain.npz").exists()


class _AllRejectedPosterior:
    def __init__(self, **_kwargs) -> None:
        pass

    def evaluate(self, _theta):
        return (
            -np.inf,
            (
                0.0,
                -18.0,
                -np.inf,
                -18.0,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.0,
            ),
        )

    def __call__(self, theta):
        value, blob = self.evaluate(theta)
        return (value, *blob)


def test_full_optimizer_fail_fast_preflight_stops_all_rejected_walkers(
    tmp_path,
    monkeypatch,
) -> None:
    output_prefix = tmp_path / "n256_full_mcmc"

    monkeypatch.setattr(full_mcmc, "N256FullLogPosterior", _AllRejectedPosterior)

    def fail_if_called(**_kwargs):
        raise AssertionError("run_sampler should not be called")

    monkeypatch.setattr(full_mcmc, "run_sampler", fail_if_called)

    exit_code = full_mcmc.main(
        [
            "--walkers",
            "16",
            "--steps",
            "8",
            "--burn-in",
            "2",
            "--thin",
            "1",
            "--fail-on-rejected-preflight",
            "--output-prefix",
            str(output_prefix),
        ]
    )

    assert exit_code == 2
    assert output_prefix.with_name(output_prefix.name + "_preflight_positions.csv").exists()
    assert output_prefix.with_name(output_prefix.name + "_preflight_summary.csv").exists()
    assert not output_prefix.with_name(output_prefix.name + "_chain.npz").exists()


def test_full_optimizer_default_runs_after_all_rejected_preflight(
    tmp_path,
    monkeypatch,
) -> None:
    output_prefix = tmp_path / "n256_full_mcmc"
    monkeypatch.setattr(full_mcmc, "N256FullLogPosterior", _AllRejectedPosterior)
    sampler_called = False

    def fake_run_sampler(*, initial_positions, steps, **_kwargs):
        nonlocal sampler_called
        sampler_called = True
        walkers, ndim = initial_positions.shape
        chain = np.broadcast_to(initial_positions, (steps, walkers, ndim)).copy()
        log_probability = np.full((steps, walkers), -np.inf)
        blobs = np.empty((steps, walkers), dtype=BLOB_DTYPE)
        blobs[...] = _AllRejectedPosterior().evaluate(initial_positions[0])[1]
        return chain, log_probability, blobs

    monkeypatch.setattr(full_mcmc, "run_sampler", fake_run_sampler)

    exit_code = full_mcmc.main(
        [
            "--walkers",
            "16",
            "--steps",
            "8",
            "--burn-in",
            "2",
            "--thin",
            "1",
            "--output-prefix",
            str(output_prefix),
        ]
    )

    assert exit_code == 2
    assert sampler_called
    assert output_prefix.with_name(output_prefix.name + "_preflight_positions.csv").exists()
    assert output_prefix.with_name(output_prefix.name + "_chain.npz").exists()
    assert output_prefix.with_name(output_prefix.name + "_samples.csv").exists()
    assert output_prefix.with_name(output_prefix.name + "_failure_summary.csv").exists()


def test_initial_position_grid_script_writes_void_count_diagnostics(tmp_path) -> None:
    output_csv = tmp_path / "initial_grid.csv"

    exit_code = grid_main(
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
            "--vide-variant",
            "all",
            "--position-mode",
            "initial",
            "--linking-factor-values",
            "0.13",
            "--radius-a0-values",
            "5.0",
            "--adjacency-factor-values",
            "0.4",
            "--merge-threshold-values",
            "1.0",
            "--center-weight",
            "0",
            "--allow-degenerate",
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["position_mode"] == "initial"
    assert "predicted_a_void_count" in rows[0]
    assert "source_a_cluster_count" in rows[0]
