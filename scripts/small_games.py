"""Best-response dynamics and fictitious play on three small games.

    python scripts/small_games.py     # experiments/small_games.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import small_games

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"

if __name__ == "__main__":
    out = [small_games.summarize(name) for name in small_games.GAMES]
    (EXP / "small_games.json").write_text(json.dumps(out, indent=1) + "\n")
    for g in out:
        print(g["game"], "| pure", g["pure"], "| mixed", g["mixed"])
        for d in g["dynamics"]:
            print("   ", d)
        print("    fictitious play (row freq, col freq):", g["fictitious_play"])
