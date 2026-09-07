"""Simulate soccer games between two stationary policies.

A policy maps a state to a length-4 probability vector over actions. The
``row_policy`` controls player 0, the ``col_policy`` controls player 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_nash.game import Action, SoccerGame, State

Policy = dict[State, np.ndarray]


@dataclass
class GameResult:
    trajectory: list[State]
    joint_actions: list[tuple[int, int]]
    winner: int | None  # None on a tie
    steps: int

    @property
    def outcome_for_row(self) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == 0 else -1.0


def _sample(dist: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(dist), p=dist / dist.sum()))


def play_game(
    game: SoccerGame,
    row_policy: Policy,
    col_policy: Policy,
    rng: np.random.Generator | None = None,
    start: State | None = None,
) -> GameResult:
    rng = rng or np.random.default_rng()
    state = start or game.initial_state()

    traj: list[State] = [state]
    acts: list[tuple[int, int]] = []

    for _ in range(game.max_steps):
        a0 = _sample(row_policy[state], rng)
        a1 = _sample(col_policy[state], rng)
        state, _, done = game.step(state, Action(a0), Action(a1))
        traj.append(state)
        acts.append((a0, a1))
        if done:
            return GameResult(traj, acts, game.winner(state), len(acts))

    return GameResult(traj, acts, None, len(acts))


def win_rates(
    game: SoccerGame,
    row_policy: Policy,
    col_policy: Policy,
    n_games: int = 500,
    rng: np.random.Generator | None = None,
    start: State | None = None,
) -> dict[str, float]:
    rng = rng or np.random.default_rng(0)
    wins = ties = losses = 0
    for _ in range(n_games):
        r = play_game(game, row_policy, col_policy, rng, start)
        if r.winner is None:
            ties += 1
        elif r.winner == 0:
            wins += 1
        else:
            losses += 1
    return {
        "row_win": wins / n_games,
        "tie": ties / n_games,
        "row_loss": losses / n_games,
    }
