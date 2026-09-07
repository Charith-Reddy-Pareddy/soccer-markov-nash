"""SVG board renderer."""

import xml.etree.ElementTree as ET

import pytest

from soccer_nash.game import A10SoccerGame, SoccerGame
from soccer_nash.render import P0, P1, board_svg, trajectory_svg


def test_board_svg_is_well_formed_xml():
    game = A10SoccerGame()
    svg = board_svg(game, game.initial_state())
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")


def test_player_colours_present_and_distinct():
    svg = board_svg(A10SoccerGame(), (0, 2, 6, 2, 0))
    assert P0 in svg and P1 in svg
    assert P0 != P1


def test_carrier_ring_follows_the_ball():
    game = A10SoccerGame()
    with_p0 = board_svg(game, (0, 2, 6, 2, 0))
    with_p1 = board_svg(game, (0, 2, 6, 2, 1))
    assert with_p0.count("stroke-dasharray") == 1
    assert with_p1.count("stroke-dasharray") == 1
    assert with_p0 != with_p1


def test_row_zero_is_drawn_below_row_top():
    # y increases upward: player at y=0 sits lower on screen than at y=4.
    game = SoccerGame(width=5, height=5)
    low = _first_circle_cy(board_svg(game, (0, 0, 4, 4, 0)))
    high = _first_circle_cy(board_svg(game, (0, 4, 4, 0, 0)))
    assert low > high


def test_goal_rows_get_a_mouth_on_each_edge():
    game = SoccerGame(width=5, height=5, goal_rows=(2,))
    svg = board_svg(game, (0, 2, 4, 2, 0))
    assert svg.count("<rect") == 1 + 2  # board + two goal mouths


def test_terminal_state_rejected():
    with pytest.raises(ValueError):
        board_svg(A10SoccerGame(), (-1, -1, -1, -1, 0))


def test_trajectory_svg_skips_terminal_and_tiles():
    game = A10SoccerGame()
    states = [(0, 2, 6, 2, 0), (1, 2, 5, 2, 0), (-1, -1, -1, -1, 0)]
    svg = trajectory_svg(game, states, per_row=2)
    root = ET.fromstring(svg)
    groups = [g for g in root if g.tag.endswith("g")]
    assert len(groups) == 2


def _first_circle_cy(svg: str) -> float:
    root = ET.fromstring(svg)
    for el in root.iter():
        if el.tag.endswith("circle"):
            return float(el.attrib["cy"])
    raise AssertionError("no circle in svg")
