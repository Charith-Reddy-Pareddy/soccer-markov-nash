"""scripts/tournament.py and soccer_nash/evaluate.py -- the Littman Table 3
reproduction."""

import importlib.util
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from soccer_nash.evaluate import policy_value
from soccer_nash.exploit import best_response_to, uniform_policy
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.viz import P0, P1, grouped_bars_svg

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def game_and_nash():
    game = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random",
                      n_actions=5, scoring="rate")
    nash = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    return game, nash


def test_policy_value_agrees_with_the_solver(game_and_nash):
    game, nash = game_and_nash
    v = policy_value(game, nash.row_policy, nash.col_policy, gamma=0.9)
    for s in list(game.states())[:200]:
        assert v[s] == pytest.approx(nash.values[s], abs=1e-6)


def test_nash_policy_is_not_exploitable(game_and_nash):
    game, nash = game_and_nash
    k = game.initial_state()
    br = best_response_to(game, nash.row_policy, responder=1, gamma=0.9)
    # a challenger cannot beat the Nash value
    assert -br.values[k] == pytest.approx(nash.values[k], abs=1e-6)


@pytest.mark.slow
def test_greedy_policy_is_exploitable(game_and_nash):
    game, _nash = game_and_nash
    k = game.initial_state()
    gvr = best_response_to(game, uniform_policy(game), responder=0, gamma=0.9)
    from soccer_nash.exploit import onehot_policy

    greedy = onehot_policy(gvr.policy, game.n_actions)
    vs_random = policy_value(game, greedy, uniform_policy(game), gamma=0.9)[k]
    challenger = best_response_to(game, greedy, responder=1, gamma=0.9)
    vs_challenger = -challenger.values[k]
    assert vs_random > 0.5                    # crushes a random opponent
    assert vs_challenger < 0.0                # but loses to its challenger
    assert vs_random - vs_challenger > 1.0    # a large robustness gap


@pytest.mark.slow
def test_tournament_script_runs_and_ranks_minimax_first_on_robustness():
    t = _load("tournament")
    _game, table, _opps = t.run("rate")
    gaps = {n: table[n]["random"] - table[n]["challenger"] for n in table}
    # among policies that actually beat a random opponent, minimax has the
    # smallest robustness gap
    pressers = {n: g for n, g in gaps.items() if table[n]["random"] > 0.3}
    assert min(pressers, key=pressers.get) == "minimax"
    assert table["minimax"]["challenger"] > 0.0
    assert table["greedy/rand"]["challenger"] < 0.0


def test_grouped_bars_svg_is_valid():
    svg = grouped_bars_svg(
        ["a", "b"], [("x", P0, [0.5, -0.3]), ("y", P1, [0.2, 0.9])],
        y_label="v", title="t",
    )
    ET.fromstring(svg)
    assert "rect" in svg


def test_argmax_onehot_is_a_unit_vector():
    t = _load("tournament")
    v = t._argmax_onehot(np.array([0.1, 0.7, 0.2, 0.0, 0.0]))
    assert v.sum() == 1.0 and v[1] == 1.0
