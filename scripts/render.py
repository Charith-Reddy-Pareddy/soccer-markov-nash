"""Render a soccer state (or the A10 Q8 trajectory) to an SVG file.

    python scripts/render.py 0,2,6,2,0            # one state -> board.svg
    python scripts/render.py 0,2,6,2,0 -o k.svg
    python scripts/render.py --trajectory         # the BR-vs-scripted win

Player 0 is blue, player 1 is green.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.best_response import BestResponse
from soccer_nash.game import A10SoccerGame
from soccer_nash.opponents import part2_opponent
from soccer_nash.render import board_svg, trajectory_svg
from soccer_nash.simulate import play_deterministic


def _parse_state(text: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(v) for v in text.replace(" ", "").split(","))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("state must be x0,y0,x1,y1,b")
    return parts  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", nargs="?", type=_parse_state)
    parser.add_argument("-o", "--out", type=pathlib.Path)
    parser.add_argument("--trajectory", action="store_true")
    args = parser.parse_args()

    game = A10SoccerGame()

    if args.trajectory:
        br = BestResponse(game, opponent=part2_opponent, me=0).solve()
        result = play_deterministic(game, br.policy, part2_opponent, me=0)
        svg = trajectory_svg(game, result.trajectory)
        out = args.out or pathlib.Path("trajectory.svg")
    else:
        state = args.state or game.initial_state()
        svg = board_svg(game, state, caption=f"state = {state}")
        out = args.out or pathlib.Path("board.svg")

    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
