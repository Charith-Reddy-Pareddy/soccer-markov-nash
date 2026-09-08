"""Certificate for "single goal cell => every stage game has a pure saddle".

Runs the two checks in ``soccer_nash.onecell`` over a grid of boards and
discounts. A zero in both columns is a machine-checked proof for that board.

    python scripts/onecell_proof.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.onecell import certify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.5, 0.7, 0.9, 0.95, 0.99])
    args = parser.parse_args()

    boards = [(w, h) for w in (3, 5, 7, 9, 11) for h in (3, 5, 7) if w * h <= 55]

    print(f"{'board':>7}  {'gamma':>5}  {'states':>7}  {'guard slack':>12}  "
          f"{'IEWDS fails':>11}  {'undisc. mixed':>13}  {'V(kickoff)':>10}")
    all_ok = True
    for w, h in boards:
        for gamma in args.gammas:
            c = certify(w, h, gamma)
            all_ok &= c.ok
            print(f"{w}x{h:<4}  {gamma:>5}  {c.states:>7}  {c.guard_slack:>12.2e}  "
                  f"{c.iewds_failures:>11}  {c.undiscounted_mixed_stage_games:>13}  "
                  f"{c.kickoff_value:>10.4f}"
                  + ("" if c.ok else "   <-- FAIL"))

    print("\n" + ("all boards certified: every single-cell stage game has a pure saddle"
                  if all_ok else "SOME BOARDS FAILED"))


if __name__ == "__main__":
    main()
