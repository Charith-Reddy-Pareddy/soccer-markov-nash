"""Reachability attractors for the deterministic soccer game.

A *win attractor* for a player is the set of states from which that player can
force a goal against any opponent play, together with the rank (minimum number
of forced moves). It is computed by the concurrent controllable-predecessor
fixpoint. Both the attractor strategy (on the attractor) and "stay in the draw
region" (off it) are pure and memoryless, so on games where the three sets
partition the state space the game is *positionally determined* -- see
`docs/discussion.md` and `scripts/onecell_proof.py --positional`.

Deterministic move resolution only; the concurrent fixpoint needs a
deterministic successor per joint action.
"""

from __future__ import annotations

from dataclasses import dataclass

from soccer_nash.game import Action, SoccerGame, State
from soccer_nash.nash_q import NashQIteration

_DELTAS = (Action.U, Action.D, Action.L, Action.R)


def _succ(game: SoccerGame, s: State, a_me: int, a_op: int, me: int) -> State:
    a0, a1 = (a_me, a_op) if me == 0 else (a_op, a_me)
    (_prob, ns, _r) = game.transitions(s, Action(a0), Action(a1))[0]
    return ns


def win_attractor(game: SoccerGame, player: int) -> dict[State, int]:
    """``{state: rank}`` -- states from which ``player`` forces a goal, and in
    how many forced moves. Deterministic games only."""
    if game.move_order != "deterministic":
        raise ValueError("win_attractor needs deterministic move resolution")
    if game.n_actions != 4:
        raise ValueError("win_attractor is only wired for the 4-action game")
    if game.scoring != "win":
        raise ValueError("win_attractor needs scoring='win' (a terminating game)")
    states = list(game.states())

    def me_forces_into(s: State, good: set[State], strict_goal: bool) -> bool:
        """Does ``player`` have a move that lands in ``good`` (or scores) for
        every opponent reply?"""
        for a_me in range(4):
            if all(
                _in_target(game, _succ(game, s, a_me, a_op, player), player, good)
                for a_op in range(4)
            ):
                return True
        return False

    won: dict[State, int] = {}
    rank = 0
    # rank 0: score outright this move
    for s in states:
        if me_forces_into(s, set(), True):
            won[s] = 0
    changed = True
    while changed:
        changed = False
        rank += 1
        good = set(won)
        for s in states:
            if s in won:
                continue
            if me_forces_into(s, good, False):
                won[s] = rank
                changed = True
    return won


def _in_target(game: SoccerGame, ns: State, player: int, good: set[State]) -> bool:
    if game.is_terminal(ns):
        return game.winner(ns) == player
    return ns in good


def attractor_strategy(
    game: SoccerGame, player: int, ranks: dict[State, int]
) -> dict[State, int]:
    """A pure action per attractor state: the move that scores now, or that
    forces every opponent reply to a strictly lower rank."""
    good = set(ranks)
    strat: dict[State, int] = {}
    for s, r in ranks.items():
        for a_me in range(4):
            outs = [_succ(game, s, a_me, a_op, player) for a_op in range(4)]
            if r == 0:
                if all(game.winner(ns) == player for ns in outs):
                    strat[s] = a_me
                    break
            elif all(
                _in_target(game, ns, player, good)
                and (game.is_terminal(ns) or ranks[ns] < r)
                for ns in outs
            ):
                strat[s] = a_me
                break
    return strat


def positional_profile(game: SoccerGame) -> tuple[dict, dict, dict, dict]:
    """``(policy0, policy1, A0, A1)``: a full pure memoryless strategy per
    player -- the attractor move on the player's own winning set, and a *safety*
    move (one keeping the game out of the opponent's attractor for every
    opponent reply) everywhere else."""
    a0 = win_attractor(game, 0)
    a1 = win_attractor(game, 1)
    strat0 = attractor_strategy(game, 0, a0)
    strat1 = attractor_strategy(game, 1, a1)

    def safety_move(s: State, me: int, avoid: set[State]) -> int:
        for a_me in range(4):
            if all(
                not _lands_in(game, _succ(game, s, a_me, a_op, me), 1 - me, avoid)
                for a_op in range(4)
            ):
                return a_me
        return 0  # unreachable on a game where the attractors leave a safe draw

    p0: dict[State, int] = {}
    p1: dict[State, int] = {}
    for s in game.states():
        p0[s] = strat0[s] if s in a0 else safety_move(s, 0, set(a1))
        p1[s] = strat1[s] if s in a1 else safety_move(s, 1, set(a0))
    return p0, p1, a0, a1


def _lands_in(game: SoccerGame, ns: State, opp: int, avoid: set[State]) -> bool:
    if game.is_terminal(ns):
        return game.winner(ns) == opp
    return ns in avoid


@dataclass
class PositionalCheck:
    states: int
    a0: int          # |win attractor of player 0|
    a1: int
    draw: int
    mismatches: int  # states where the positional profile does not realize V*

    @property
    def ok(self) -> bool:
        return self.mismatches == 0


def verify_positional_equilibrium(game: SoccerGame) -> PositionalCheck:
    """Check that the pure memoryless profile from :func:`positional_profile`
    realizes the exact undiscounted value ``V*`` at every state -- i.e. it is a
    pure memoryless Markov-perfect equilibrium. Deterministic games only."""
    p0, p1, a0, a1 = positional_profile(game)
    fh = NashQIteration(game, mode="hybrid").run_finite_horizon()
    v = fh.values

    mismatches = 0
    for s in game.states():
        state = s
        outcome: int | None = None
        for _ in range(game.max_steps):
            if game.is_terminal(state):
                outcome = game.winner(state)
                break
            (_p, state, _r) = game.transitions(
                state, Action(p0[state]), Action(p1[state])
            )[0]
        want = None if abs(v[s]) < 1e-9 else (0 if v[s] > 0 else 1)
        if outcome != want:
            mismatches += 1

    n = sum(1 for _ in game.states())
    return PositionalCheck(n, len(a0), len(a1), n - len(a0) - len(a1), mismatches)
