"""Intermediate-reward shaping for the soccer game.

The A10 write-up suggests "small positive rewards for possessing the ball and
moving closer to the goal" on top of the sparse win/lose signal. Two flavours:

* :class:`PotentialShaping` -- potential-based shaping
  ``F(s, s') = gamma * Phi(s') - Phi(s)`` (Ng, Harada & Russell 1999; Devlin &
  Kudenko 2011 for Markov games). Provably leaves every Nash equilibrium and
  the optimal policy unchanged; the value function shifts by exactly ``-Phi``.
* :class:`StepPossessionBonus` -- a plain per-step bonus for holding the ball.
  Not potential-based, so it *can* change the optimal policy (the carrier may
  prefer to keep the ball over scoring).

Both are zero-sum: whatever is added to player 0's reward is taken from
player 1's.
"""

from __future__ import annotations

from soccer_nash.game import SoccerGame, State


class PotentialShaping:
    def __init__(self, w_ball: float = 0.1, w_advance: float = 0.1):
        self.w_ball = w_ball
        self.w_advance = w_advance

    def phi(self, game: SoccerGame, state: State) -> float:
        """Player 0's potential: possession plus ball advancement, in [-1, 1]-ish."""
        if game.is_terminal(state):
            return 0.0
        x0, y0, x1, y1, b = state
        sign = 1.0 if b == 0 else -1.0
        if b == 0:  # player 0 carries; closer to the right edge is better
            advance = x0 / (game.width - 1)
        else:  # player 1 carries; closer to the left edge is worse for player 0
            advance = x1 / (game.width - 1) - 1.0
        return self.w_ball * sign + self.w_advance * advance

    def reward_delta(
        self, game: SoccerGame, s: State, s_next: State, gamma: float
    ) -> float:
        return gamma * self.phi(game, s_next) - self.phi(game, s)


class StepPossessionBonus:
    def __init__(self, bonus: float = 0.02):
        self.bonus = bonus

    def reward_delta(
        self, game: SoccerGame, s: State, s_next: State, gamma: float
    ) -> float:
        # Reward player 0 for having held the ball on this step.
        return self.bonus if s[4] == 0 else -self.bonus
