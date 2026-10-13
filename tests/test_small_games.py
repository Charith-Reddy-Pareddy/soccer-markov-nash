# ruff: noqa: E501
import json
import pathlib

import pytest

from soccer_nash import small_games as sg

ROOT = pathlib.Path(__file__).resolve().parent.parent
BOS = sg.GAMES["Battle of the sexes (coordination)"]
DIFF = sg.GAMES["Different choices (anti-coordination)"]
PENNIES = sg.GAMES["Matching pennies (zero-sum)"]


def test_battle_of_the_sexes_has_two_pure_equilibria_and_the_mixed_one_from_the_notes():
    assert sg.pure_equilibria(BOS["A"], BOS["B"]) == [(0, 0), (1, 1)]
    row, col = sg.mixed_equilibrium_2x2(BOS["A"], BOS["B"])
    assert row == pytest.approx(2 / 3) and col == pytest.approx(1 / 3)


def test_coordination_best_responses_settle_on_one_of_the_options_or_keep_switching():
    for start, end in (((0, 0), (0, 0)), ((1, 1), (1, 1))):
        for alternating in (False, True):
            assert sg.best_response_dynamics(BOS["A"], BOS["B"], start, alternating=alternating)[-1] == end
    mismatched = sg.best_response_dynamics(BOS["A"], BOS["B"], (0, 1))
    assert not sg.settles(mismatched) and mismatched[0] == mismatched[2]
    assert sg.best_response_dynamics(BOS["A"], BOS["B"], (0, 1), alternating=True)[-1] == (1, 1)


def test_in_the_game_of_different_choices_simultaneous_responses_cycle_but_fictitious_play_averages_to_a_half():
    cycle = sg.best_response_dynamics(DIFF["A"], DIFF["B"], (0, 0))
    assert set(cycle) == {(0, 0), (1, 1)} and not sg.settles(cycle)
    row, col = sg.fictitious_play(DIFF["A"], DIFF["B"], (0, 0))
    assert row == pytest.approx([0.5, 0.5], abs=0.01) and col == pytest.approx([0.5, 0.5], abs=0.01)
    assert sg.mixed_equilibrium_2x2(DIFF["A"], DIFF["B"]) == (0.5, 0.5)


def test_matching_pennies_has_no_pure_equilibrium_and_only_fictitious_play_settles():
    assert sg.pure_equilibria(PENNIES["A"], PENNIES["B"]) == []
    for start in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for alternating in (False, True):
            assert not sg.settles(sg.best_response_dynamics(PENNIES["A"], PENNIES["B"], start, alternating=alternating))
        row, col = sg.fictitious_play(PENNIES["A"], PENNIES["B"], start)
        assert row == pytest.approx([0.5, 0.5], abs=0.02) and col == pytest.approx([0.5, 0.5], abs=0.02)


def test_the_saved_summary_matches_a_fresh_computation():
    saved = json.loads((ROOT / "experiments" / "small_games.json").read_text())
    assert [g["game"] for g in saved] == list(sg.GAMES)
    for g in saved:
        assert g == json.loads(json.dumps(sg.summarize(g["game"])))
