import csv

from scripts.calibrate_vsf_likelihood import main


def test_calibrate_vsf_likelihood_script_writes_ranked_csv(tmp_path, capsys) -> None:
    output_csv = tmp_path / "vsf_likelihood_calibration.csv"

    exit_code = main(
        [
            "tests/fixtures/pinocchio_pair_a.out",
            "tests/fixtures/pinocchio_pair_b.out",
            "tests/fixtures/vide_voidDesc_all_small.out",
            "tests/fixtures/vide_voidDesc_all_small.out",
            "--box-size",
            "10.0",
            "--rho-bar",
            "1.0",
            "--linking-length",
            "0.3",
            "--min-cluster-members",
            "2",
            "--min-cluster-mass",
            "0.0",
            "--radius-a0",
            "1.0",
            "--radius-alpha",
            "1.0",
            "--adjacency-factor",
            "1.0",
            "--bins",
            "2",
            "--bin-min",
            "0.1",
            "--bin-max",
            "10.0",
            "--count-floor",
            "0.5",
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "rank,degenerate,log_likelihood" in captured.out
    assert f"Wrote {output_csv}" in captured.out

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["rank"] == "1"
    assert row["linking_mode"] == "fixed"
    assert row["min_cluster_members"] == "2"
    assert row["radius_a0"] == "1.0"
    assert row["count_floor"] == "0.5"
    assert row["log_likelihood"]
    assert row["negative_log_likelihood"]
    assert row["target_a_log_likelihood"]
    assert row["target_a_finder_total"] == "1"
    assert row["target_a_vide_total"] == "2"
