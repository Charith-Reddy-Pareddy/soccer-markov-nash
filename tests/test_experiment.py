"""The experiment row builder that feeds experiments/*.csv."""

import csv

import pytest

from soccer_nash.experiment import (
    FIELDS,
    run_config,
    standard_goal_rows,
    write_csv,
)


def test_standard_goal_rows():
    assert standard_goal_rows(3) == (1,)
    assert standard_goal_rows(5) == (1, 2, 3)
    assert standard_goal_rows(9) == (1, 2, 3, 4, 5, 6, 7)


def test_run_config_row_matches_fields_exactly():
    row = run_config(width=3, height=3, goal_rows=(1,), gamma=0.8)
    # write_csv uses a strict DictWriter -- any drift here breaks every CSV.
    assert set(row) == set(FIELDS)


def test_run_config_deterministic_board_needs_no_mixing():
    row = run_config(width=3, height=3, goal_rows=(1,), gamma=0.8)
    # Every deterministic stage game has a pure saddle (pure or degenerate),
    # so the hybrid solver never calls the LP.
    assert row["mixed_states"] == 0
    assert row["mixed_fraction"] == 0.0
    assert row["pure_states"] + row["degenerate_states"] == row["states"]
    assert row["lp_calls"] == 0
    assert row["duality_gap"] != row["duality_gap"]  # NaN unless measured


def test_run_config_measures_exploitability_when_asked():
    row = run_config(
        width=3, height=3, goal_rows=(1,), gamma=0.8, measure_exploitability=True
    )
    assert row["duality_gap"] == pytest.approx(0.0, abs=1e-6)


def test_write_csv_roundtrips(tmp_path):
    rows = [run_config(width=3, height=3, goal_rows=(1,), gamma=0.8)]
    out = tmp_path / "sub" / "run.csv"
    write_csv(out, rows)

    with out.open() as f:
        back = list(csv.DictReader(f))
    assert list(back[0]) == FIELDS
    assert int(back[0]["states"]) == rows[0]["states"]
