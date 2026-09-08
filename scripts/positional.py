"""The deterministic soccer game has an explicit pure memoryless equilibrium.

For each player, ``soccer_nash.attractor`` computes the win attractor (states
from which the player forces a goal) and a strategy: the attractor move on that
set, a safety move (stay out of the opponent's attractor) elsewhere. This checks
that the resulting pure memoryless profile realizes the exact undiscounted value
at every state -- a constructive form of Claim B for the deterministic game.

    python scripts/positional.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.attractor import verify_positional_equilibrium
from soccer_nash.game import SoccerGame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-width", type=int, default=3)
    args = parser.parse_args()

    boards = [(w, h) for w in (3, 5, 7, 9) for h in (3, 5, 7) if w * h <= 45]
    print(f"{'board':>7}  {'|A0|':>6}  {'|A1|':>6}  {'draw':>6}  "
          f"{'mismatches':>10}")
    all_ok = True
    for w, h in boards:
        k = min(args.goal_width, h - 2) or 1
        start = (h - k) // 2
        game = SoccerGame(
            width=w, height=h, goal_rows=tuple(range(start, start + k)),
            move_order="deterministic",
        )
        c = verify_positional_equilibrium(game)
        all_ok &= c.ok
        print(f"{w}x{h:<4}  {c.a0:>6}  {c.a1:>6}  {c.draw:>6}  {c.mismatches:>10}"
              + ("" if c.ok else "   <-- FAIL"))

    print("\n" + ("the pure memoryless attractor/safety profile realizes V* "
                  "at every state" if all_ok else "SOME BOARDS FAILED"))


if __name__ == "__main__":
    main()
