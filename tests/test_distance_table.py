"""scripts/distance_table.py -- Manhattan distance is computed correctly and
matches the count showcase.md already quotes (40 at distance 1, 54 at
distance 2, 0 elsewhere)."""

import importlib.util
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manhattan_distance():
    d = _load("distance_table")
    assert d.manhattan((0, 0, 1, 1, 0)) == 2
    assert d.manhattan((0, 0, 0, 1, 0)) == 1
    assert d.manhattan((3, 2, 3, 2, 0)) == 0
    assert d.manhattan((0, 0, 5, 4, 0)) == 9


def test_no_pure_saddle_states_are_within_distance_2():
    # cross-checks the exact figure showcase.md quotes by hand: 40 states at
    # distance 1, 54 at distance 2, 0 elsewhere -- run once here so a future
    # change to the game/board that breaks this gets caught by the suite,
    # not just noticed by inspection.
    d = _load("distance_table")
    from soccer_nash.game import SoccerGame
    from soccer_nash.nash_q import NashQIteration

    game = SoccerGame(**d.CANON)
    result = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10).run()
    no_saddle = set(result.no_saddle_states)

    by_distance = {}
    for s in no_saddle:
        by_distance[d.manhattan(s)] = by_distance.get(d.manhattan(s), 0) + 1

    assert by_distance == {1: 40, 2: 54}
    assert sum(by_distance.values()) == 94
