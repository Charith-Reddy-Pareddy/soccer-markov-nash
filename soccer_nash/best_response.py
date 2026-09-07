"""Optimal best response to a fixed opponent policy.

With the opponent's move pinned by a scripted policy, the soccer game collapses
to a single-agent MDP for ``me``. Deterministic dynamics make this a plain
value iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from soccer_nash.game import Action, SoccerGame, State

OpponentPolicy = Callable[[SoccerGame, State, int], Action]


@dataclass
class BestResponseResult:
    values: dict[State, float]
    policy: dict[State, Action]
    iterations: int
    me: int
    gamma: float


class BestResponse:
    def __init__(
        self,
        game: SoccerGame,
        opponent: OpponentPolicy,
        me: int = 0,
        gamma: float = 0.95,
        tol: float = 1e-10,
        max_iters: int = 5000,
        shaping=None,
    ):
        if game.move_order != "deterministic":
            raise ValueError("best response assumes deterministic dynamics")
        self.game = game
        self.me = me
        self.gamma = gamma
        self.tol = tol
        self.max_iters = max_iters
        self.shaping = shaping

        self._states = list(game.states())
        # state -> list over my actions of (action, next_state, my_reward),
        # with any shaping reward folded into my_reward.
        self._trans: dict[State, list[tuple[Action, State, float]]] = {}
        for s in self._states:
            a_opp = opponent(game, s, 1 - me)
            row = []
            for a_me in game.actions():
                a0, a1 = (a_me, a_opp) if me == 0 else (a_opp, a_me)
                ns, (r0, r1), _ = game.step(s, a0, a1)
                r = r0 if me == 0 else r1
                if shaping is not None:
                    delta = shaping.reward_delta(game, s, ns, gamma)
                    r = r + (delta if me == 0 else -delta)
                row.append((a_me, ns, r))
            self._trans[s] = row

    def solve(self) -> BestResponseResult:
        values: dict[State, float] = {s: 0.0 for s in self._states}

        iterations = 0
        for iterations in range(1, self.max_iters + 1):
            delta = 0.0
            updated: dict[State, float] = {}
            for s in self._states:
                best = max(
                    r + (0.0 if self.game.is_terminal(ns) else self.gamma * values[ns])
                    for _, ns, r in self._trans[s]
                )
                delta = max(delta, abs(best - values[s]))
                updated[s] = best
            values = updated
            if delta < self.tol:
                break

        policy: dict[State, Action] = {}
        for s in self._states:
            scored = [
                (
                    r
                    + (
                        0.0
                        if self.game.is_terminal(ns)
                        else self.gamma * values[ns]
                    ),
                    a,
                )
                for a, ns, r in self._trans[s]
            ]
            best_value = max(v for v, _ in scored)
            policy[s] = min(a for v, a in scored if v >= best_value - 1e-9)

        return BestResponseResult(values, policy, iterations, self.me, self.gamma)


def materialize(
    game: SoccerGame, opponent: OpponentPolicy, me: int
) -> dict[State, Action]:
    """Evaluate a scripted opponent policy at every state."""
    return {s: opponent(game, s, me) for s in game.states()}
