"""Win / tie / loss rates and mirror-symmetry gaps for learned soccer policies.

The step limit is part of the A10 game ("a tie if no one scores in 100 steps"),
so matches here are played for at most ``max_steps`` and a game with no goal is
a tie. This is a Monte-Carlo estimate of ``P(win)``, ``P(tie)``, ``P(loss)`` for
one player -- unlike the discounted goal difference the rest of the repo
reports, which cannot tell a tie from a win-and-loss that cancel.
"""

from __future__ import annotations

import numpy as np

from soccer_nash.game import MOVE_ACTIONS, SoccerGame
from soccer_nash.symmetry import flip_distribution, mirror_state


def play_matches(
    game: SoccerGame, row_policy: dict, col_policy: dict,
    n_games: int = 2000, max_steps: int = 100, seed: int = 0,
) -> dict[str, float]:
    """Play ``n_games`` from the kickoff; returns player 0's win/tie/loss rates."""
    rng = np.random.default_rng(seed)
    wins = losses = 0
    for _ in range(n_games):
        s = game.initial_state()
        for _t in range(max_steps):
            a0 = rng.choice(len(row_policy[s]), p=row_policy[s])
            a1 = rng.choice(len(col_policy[s]), p=col_policy[s])
            s, (r0, _r1), done = game.step(
                s, MOVE_ACTIONS[a0], MOVE_ACTIONS[a1], rng=rng)
            if done:
                wins += r0 > 0
                losses += r0 < 0
                break
    ties = n_games - wins - losses
    return {"win": wins / n_games, "tie": ties / n_games, "loss": losses / n_games}


def mirror_gap(game: SoccerGame, row_policy: dict, col_policy: dict) -> tuple[float, float]:
    """(mean, max) over states of ``max_a |row[s] - flip(col[mirror(s)])|`` --
    how far the two players' policies are from being mirror images (0 = exactly)."""
    devs = []
    for s in game.states():
        want = flip_distribution(col_policy[mirror_state(s, game.width)])
        devs.append(float(np.abs(np.asarray(row_policy[s]) - want).max()))
    return float(np.mean(devs)), float(np.max(devs))
