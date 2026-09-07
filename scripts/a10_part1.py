"""Print the A10 Part 1 answers for a given state.

    python scripts/a10_part1.py 3,2,4,2,0
    python scripts/a10_part1.py 3,2,4,2,0 --goal-rows 1 2 3 --width 7 --height 5

Q1/Q3 is the block of 16 successor lines; Q2/Q4 is the single reward line.
"""

from __future__ import annotations

import argparse

from soccer_nash.a10 import ACTION_LABELS, format_rewards, successors
from soccer_nash.game import SoccerGame


def _parse_state(text: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(v) for v in text.split(","))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("state must be 'x0,y0,x1,y1,b'")
    return parts  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=_parse_state, help="x0,y0,x1,y1,b")
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--goal-rows", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--player",
        type=int,
        choices=[0, 1],
        default=0,
        help="perspective for the reward line",
    )
    args = parser.parse_args()

    game = SoccerGame(
        width=args.width,
        height=args.height,
        goal_rows=tuple(args.goal_rows),
        move_order="deterministic",
    )

    succ = successors(game, args.state)
    print(f"# state {','.join(map(str, args.state))}")
    print("# Q1/Q3 successors")
    for label, s in zip(ACTION_LABELS, succ):
        print(f"{','.join(map(str, s))}    # {label}")
    print("# Q2/Q4 rewards (player {})".format(args.player))
    print(format_rewards(game, args.state, args.player))


if __name__ == "__main__":
    main()
