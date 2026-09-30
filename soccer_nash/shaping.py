"""Intermediate-reward shaping for the soccer game.

The A10 write-up suggests "small positive rewards for possessing the ball and
moving closer to the goal" on top of the sparse win/lose signal. Four flavours:

* :class:`PotentialShaping` -- potential-based shaping
  ``F(s, s') = gamma * Phi(s') - Phi(s)`` (Ng, Harada & Russell 1999; Devlin &
  Kudenko 2011 for Markov games). Provably leaves every Nash equilibrium and
  the optimal policy unchanged; the value function shifts by exactly ``-Phi``.
* :class:`StepPossessionBonus` -- a plain per-step bonus for holding the ball.
  Not potential-based, so it *can* change the optimal policy (the carrier may
  prefer to keep the ball over scoring).
* :class:`StepProgressionBonus` -- a plain per-step bonus proportional to
  player 0's raw ``x0``, unconditional on possession -- a legacy exploratory reward term.
  This does not reproduce Jae's current next-state, carrier-dependent
  viewer reward; see scripts/reward_q_audit.py. Also not
  potential-based.
* :class:`CombinedShaping` -- sums several shaping terms' deltas, so e.g.
  possession and progression can be applied together in one solve.

All are zero-sum: whatever is added to player 0's reward is taken from
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
        # Ng, Harada & Russell (1999): the current-state potential is NOT
        # discounted. ``gamma * (phi(s') - phi(s))`` would break invariance.
        return gamma * self.phi(game, s_next) - self.phi(game, s)


class StepPossessionBonus:
    def __init__(self, bonus: float = 0.02):
        self.bonus = bonus

    def reward_delta(
        self, game: SoccerGame, s: State, s_next: State, gamma: float
    ) -> float:
        # Reward player 0 for having held the ball on this step.
        return self.bonus if s[4] == 0 else -self.bonus


class StepProgressionBonus:
    """Legacy exploratory "progression" term: ``coeff * x0``, using player 0's raw
    (unnormalised) x-coordinate -- *not* gated on who has the ball, unlike
    this project's own ``territory_reward`` (which only pays the carrier).
    Mirrors :class:`StepPossessionBonus` in reading the *pre*-transition state
    ``s``, for the same reason: a reward for the position held *during* this
    step, not the one arrived at.
    """

    def __init__(self, coeff: float = 0.005):
        self.coeff = coeff

    def reward_delta(
        self, game: SoccerGame, s: State, s_next: State, gamma: float
    ) -> float:
        r = self.coeff * s[0]
        return r


class CombinedShaping:
    """Sums several shaping terms' ``reward_delta`` into one, so
    :class:`NashQIteration`'s single ``shaping=`` slot can carry more than one
    term (e.g. possession *and* progression together)."""

    def __init__(self, *terms):
        self.terms = terms

    def reward_delta(
        self, game: SoccerGame, s: State, s_next: State, gamma: float
    ) -> float:
        return sum(t.reward_delta(game, s, s_next, gamma) for t in self.terms)
