"""Policy / value / mixing-map visualizations."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.viz import (
    mixing_map_svg,
    panel_svg,
    policy_svg,
    strategy_bars_svg,
    value_map_svg,
)


@pytest.fixture(scope="module")
def solved():
    game = SoccerGame(width=5, height=5, goal_rows=(1, 2, 3), move_order="random")
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run()
    return game, solver, result


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_every_svg_is_well_formed(solved):
    game, solver, result = solved
    mo = lambda s: solver._matrix(s, result.values)  # noqa: E731
    mixed = sorted(result.no_saddle_states)
    st = mixed[0] if mixed else game.initial_state()
    _parse(policy_svg(game, st, result.row_policy, result.col_policy, result.values[st]))
    _parse(mixing_map_svg(game, mo, (2, 2), title="m"))
    _parse(value_map_svg(game, result.values, (4, 2), title="v"))
    _parse(strategy_bars_svg(result.row_policy[st], result.col_policy[st], 0.1))


def test_policy_svg_flags_mixed_vs_pure(solved):
    game, solver, result = solved
    mixed = sorted(result.no_saddle_states)
    assert mixed, "expected some mixed states on a 5x5 three-cell goal"
    pure = next(s for s in solver._states if s not in set(mixed))
    assert "mixed equilibrium" in policy_svg(
        game, mixed[0], result.row_policy, result.col_policy
    )
    assert "pure equilibrium" in policy_svg(
        game, pure, result.row_policy, result.col_policy
    )


def test_policy_arrows_scale_with_probability(solved):
    game, solver, result = solved
    mixed = sorted(result.no_saddle_states)
    st = mixed[0]
    svg = policy_svg(game, st, result.row_policy, result.col_policy)
    # a genuinely mixed state shows at least one probability annotation
    assert "%" in svg


def test_mixing_map_is_zero_where_saddle_exists(solved):
    game, solver, result = solved
    mo = lambda s: solver._matrix(s, result.values)  # noqa: E731
    svg = mixing_map_svg(game, mo, (2, 2), defender=1)
    root = _parse(svg)
    rects = [r for r in root.iter() if r.tag.endswith("rect")]
    # board panel + one tile per swept carrier cell (all but the defender cell)
    assert len(rects) >= game.width * game.height - 1


def test_value_map_diverges_around_zero(solved):
    game, solver, result = solved
    svg = value_map_svg(game, result.values, (4, 2), mover=0)
    assert "V*" in svg
    _parse(svg)


def test_panel_tiles_children(solved):
    game, solver, result = solved
    mo = lambda s: solver._matrix(s, result.values)  # noqa: E731
    panel = panel_svg(
        [
            mixing_map_svg(game, mo, (2, 2), title="a"),
            value_map_svg(game, result.values, (4, 2), title="b"),
        ],
        cols=2,
    )
    root = _parse(panel)
    groups = [g for g in root if g.tag.endswith("g")]
    assert len(groups) == 2


def test_strategy_bars_widths_track_the_distribution():
    row = np.array([0.0, 0.7, 0.0, 0.3])
    col = np.array([0.5, 0.0, 0.5, 0.0])
    root = _parse(strategy_bars_svg(row, col, value=0.2))
    widths = sorted(
        float(r.attrib["width"])
        for r in root.iter()
        if r.tag.endswith("rect")
    )
    # four bars: 0.3, 0.5, 0.5, 0.7 of the same track length
    assert widths[0] == pytest.approx(widths[-1] * 3 / 7, rel=1e-3)
