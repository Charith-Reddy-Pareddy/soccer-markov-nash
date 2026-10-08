import json
import pathlib

import numpy as np
import pytest

from soccer_nash import finite_horizon as fh
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

ROOT = pathlib.Path(__file__).resolve().parent.parent
GAMMA, HORIZON = 0.9, 5


@pytest.fixture(scope="module")
def small():
    game = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random", max_steps=HORIZON)
    solver = NashQIteration(game, gamma=GAMMA, mode="hybrid", tol=1e-10)
    values, row, col = fh.solve_finite_horizon(solver, GAMMA, HORIZON)
    mixed = fh.mixed_states(solver, values, GAMMA)
    return game, solver, values, row, col, mixed


def test_a_board_with_a_two_cell_goal_and_random_order_has_mixed_states(small):
    mixed = small[5]
    assert len(mixed) > 0 and len(set(mixed)) == len(mixed)


def test_the_exact_policies_are_at_zero_distance_and_zero_regret_from_themselves(small):
    game, solver, values, row, col, mixed = small
    row0 = {s: row[0][s] for s in solver._states}
    col0 = {s: col[0][s] for s in solver._states}
    m = fh.mixed_state_metrics(solver, values, GAMMA, mixed, row0, col0, row0, col0)
    assert m["mixed_states"] == len(mixed)
    assert m["tv_row"] == pytest.approx(0) and m["tv_col"] == pytest.approx(0)
    assert m["regret_max"] == pytest.approx(0, abs=1e-8)
    assert m["share_mixing_row"] > 0   # the exact policy really does mix there


def test_a_uniform_policy_is_a_worse_equilibrium_at_the_mixed_states_than_the_exact_one(small):
    game, solver, values, row, col, mixed = small
    uniform = {s: np.full(4, 0.25) for s in solver._states}
    row0 = {s: row[0][s] for s in solver._states}
    col0 = {s: col[0][s] for s in solver._states}
    m = fh.mixed_state_metrics(solver, values, GAMMA, mixed, uniform, uniform, row0, col0)
    assert m["regret_mean"] > 0 and m["tv_row"] > 0 and m["share_mixing_row"] == 1.0


def test_play_can_start_from_given_states_and_still_returns_rates_that_sum_to_one(small):
    game, _, _, row, col, mixed = small
    r = fh.play(game, fh.tables_to_policy(row), fh.uniform, 40, HORIZON, seed=2, starts=mixed)
    assert r["win"] + r["tie"] + r["loss"] == pytest.approx(1.0)


def test_play_cycles_through_the_starts_in_order(small):
    game, _, _, _, _, mixed = small
    first_step = []

    def recording_row(t, states):
        if t == 0:
            first_step.extend(states)
        return fh.uniform(t, states)

    fh.play(game, recording_row, fh.uniform, 7, HORIZON, seed=0, starts=mixed[:3])
    assert first_step == [mixed[k % 3] for k in range(7)]


# ---- the measured results ----------------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    return json.loads((ROOT / "experiments" / "pg_mixed_states.json").read_text())


def test_the_measured_file_covers_every_learner_and_seed_at_the_94_mixed_states(data):
    assert data["mixed_states"] == 94
    assert sum(data["exact"]["support_sizes"].values()) == 94
    keys = {(r["algo"], r["mode"], r["seed"]) for r in data["runs"]}
    assert len(data["runs"]) == 18 and len(keys) == 18


def test_every_rate_and_distance_in_the_measured_file_is_in_range(data):
    triples = [data["exact"]["wins_from_mixed_starts"][k] for k in ("random", "nash", "br")]
    for r in data["runs"]:
        triples += list(r["wins_from_mixed_starts"].values())
        assert 0 <= r["tv_row"] <= 1 and 0 <= r["tv_col"] <= 1
        assert r["regret_mean"] >= 0 and r["regret_max"] >= r["regret_mean"]
        assert 0 <= r["share_mixing_row"] <= 1
    for t in triples:
        assert sum(t) == pytest.approx(1.0, abs=1e-4)


def test_no_learner_matches_the_exact_mix_it_is_measurably_away_from_it(data):
    assert min(r["tv_row"] for r in data["runs"]) > 0.05
    assert min(r["regret_mean"] for r in data["runs"]) > 0


def test_the_examples_are_real_mixes_and_every_probability_vector_sums_to_one(data):
    assert len(data["examples"]) == 3
    assert sum(1 for e in data["examples"] if max(e["row"]) < 1 - 1e-6) >= 2
    for e in data["examples"]:
        assert sum(e["row"]) == pytest.approx(1.0, abs=1e-3)
        assert sum(e["col"]) == pytest.approx(1.0, abs=1e-3)
    for r in data["runs"]:
        for e in r["examples"]:
            assert sum(e["row"]) == pytest.approx(1.0, abs=1e-3)
