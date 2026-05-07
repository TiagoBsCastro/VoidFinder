import csv

import numpy as np

from scripts.optimize_n256_vsf_mcmc import (
    DEFAULT_BOUNDS,
    N256McmcPaths,
    PARAMETER_NAMES,
    credible_density_levels,
    initial_walker_positions,
    log_uniform_prior,
    main,
    summarize_samples,
    write_best_fit_command,
)


def test_log_uniform_prior_accepts_only_bounded_vectors() -> None:
    assert log_uniform_prior([0.13, 5.5, 1.05, 0.4], DEFAULT_BOUNDS) == 0.0
    assert not np.isfinite(log_uniform_prior([0.09, 5.5, 1.05, 0.4], DEFAULT_BOUNDS))
    assert not np.isfinite(log_uniform_prior([0.13, 5.5, 1.05], DEFAULT_BOUNDS))


def test_initial_walker_positions_stay_inside_bounds() -> None:
    rng = np.random.default_rng(123)

    positions = initial_walker_positions(
        center=[0.13, 5.5, 1.05, 0.4],
        bounds=DEFAULT_BOUNDS,
        walkers=8,
        rng=rng,
    )

    assert positions.shape == (8, len(PARAMETER_NAMES))
    assert np.all(positions >= DEFAULT_BOUNDS[:, 0])
    assert np.all(positions <= DEFAULT_BOUNDS[:, 1])


def test_credible_density_levels_are_ordered() -> None:
    density = np.array([[0.0, 1.0], [3.0, 2.0]])

    levels = credible_density_levels(density)

    assert len(levels) == 2
    assert levels[0] < levels[1]


def test_summarize_samples_selects_max_probability_sample() -> None:
    samples = np.array(
        [
            [0.12, 5.0, 1.0, 0.3],
            [0.13, 5.5, 1.05, 0.4],
            [0.14, 6.0, 1.1, 0.5],
        ]
    )
    log_probability = np.array([-3.0, -1.0, -2.0])

    best_fit, best_log_probability, percentiles = summarize_samples(
        samples,
        log_probability,
    )

    np.testing.assert_allclose(best_fit, samples[1])
    assert best_log_probability == -1.0
    assert percentiles.shape == (3, len(PARAMETER_NAMES))


def test_optimizer_script_writes_mcmc_products(tmp_path) -> None:
    output_prefix = tmp_path / "n256_vsf_mcmc"

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
            "--walkers",
            "8",
            "--steps",
            "8",
            "--burn-in",
            "2",
            "--thin",
            "1",
            "--seed",
            "123",
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

    with output_prefix.with_name(output_prefix.name + "_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [row["parameter"] for row in rows] == list(PARAMETER_NAMES)
    assert all(row["best_fit"] for row in rows)


def test_best_fit_command_preserves_resolved_vide_paths(tmp_path) -> None:
    command_path = tmp_path / "best_fit.sh"

    write_best_fit_command(
        command_path,
        best_fit=np.array([0.12, 5.0, 1.0, 0.3]),
        paths=N256McmcPaths(
            vide_a=tmp_path / "untrimmed_voidDesc_all_a.out",
            vide_b=tmp_path / "untrimmed_voidDesc_all_b.out",
            vide_variant="untrimmed",
            position_mode="initial",
        ),
    )

    command = command_path.read_text(encoding="utf-8")
    assert "untrimmed_voidDesc_all_a.out" in command
    assert "--position-mode initial" in command
    assert "n256_mcmc_best_fit_paper_bins_vsf_untrimmed_initial.csv" in command
