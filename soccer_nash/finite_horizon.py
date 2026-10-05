"""The discounted, finite-horizon soccer game: exact solution, exact best
response, and repeated play -- the ground truth the policy-gradient learners
are scored against once they are given the remaining step count.

A policy here is ``pol(t, states) -> (n, 4)`` action probabilities, where ``t``
is the number of steps already played. Rewards are ``+/-1`` at a goal,
discounted by ``gamma`` per step, and a game with no goal after ``horizon``
steps is a tie worth 0 (the A10 rule).
"""

from __future__ import annotations

import numpy as np

from soccer_nash.game import MOVE_ACTIONS, SoccerGame, State


def solve_finite_horizon(solver, gamma: float, horizon: int):
    """Exact backward induction. Returns ``(values, row, col)``, each a list
    indexed by ``t = 0 .. horizon - 1`` (``values`` also has ``t = horizon``, all
    zero): ``values[t][s]`` and the stage-game equilibrium strategies at step ``t``."""
    states = solver._states
    values = [dict.fromkeys(states, 0.0)]
    row, col = [], []
    for _ in range(horizon):
        nxt = values[0]
        v, p, q = {}, {}, {}
        for s in states:
            m = solver._matrix(s, nxt, gamma=gamma)
            p[s], q[s], _ = solver._stage_policy(m)
            v[s] = float(p[s] @ m @ q[s])
        values.insert(0, v)
        row.insert(0, p)
        col.insert(0, q)
    return values, row, col


def tables_to_policy(table: list[dict]):
    """A per-step ``{state: probs}`` table as a policy callable."""
    return lambda t, states: np.array([table[t][s] for s in states])


def best_response(solver, opponent, responder: int, gamma: float, horizon: int):
    """Exact best response of ``responder`` to a fixed opponent policy.

    Returns ``(value, table)``: the responder's discounted value from the
    kickoff, and the deterministic best-response ``{state: action}`` per step."""
    game = solver.game
    states = solver._states
    sign = 1.0 if responder == 0 else -1.0
    nxt = dict.fromkeys(states, 0.0)
    table: list[dict] = [{} for _ in range(horizon)]
    for t in reversed(range(horizon)):
        opp = opponent(t, states)
        cur = {}
        for k, s in enumerate(states):
            grid = solver._out[s]
            best, best_a = -np.inf, 0
            for a_r in range(solver._n):
                q = 0.0
                for a_o in range(solver._n):
                    w = opp[k][a_o]
                    if w == 0.0:
                        continue
                    i, j = (a_r, a_o) if responder == 0 else (a_o, a_r)
                    for prob, ns, r0 in grid[i, j]:
                        cont = 0.0 if game.is_terminal(ns) else gamma * nxt[ns]
                        q += w * prob * (sign * r0 + cont)
                if q > best + 1e-12:
                    best, best_a = q, a_r
            cur[s], table[t][s] = best, best_a
        nxt = cur
    return nxt[game.initial_state()], table


def exploitability(solver, row, col, gamma: float, horizon: int) -> float:
    """``V_br0 + V_br1`` from the kickoff: zero exactly at an equilibrium."""
    return (best_response(solver, col, 0, gamma, horizon)[0]
            + best_response(solver, row, 1, gamma, horizon)[0])


def br_policy(table: list[dict], n_actions: int = 4):
    def pol(t, states):
        out = np.zeros((len(states), n_actions))
        out[np.arange(len(states)), [table[t][s] for s in states]] = 1.0
        return out
    return pol


def uniform(t, states):
    return np.full((len(states), 4), 0.25)


def play(
    game: SoccerGame, row, col, n_games: int, horizon: int, seed: int = 0,
) -> dict[str, float]:
    """Repeated play from the kickoff; player 0's win / tie / loss rates."""
    rng = np.random.default_rng(seed)
    states: list[State] = [game.initial_state()] * n_games
    live = list(range(n_games))
    wins = losses = 0
    for t in range(horizon):
        if not live:
            break
        cur = [states[i] for i in live]
        p0, p1 = row(t, cur), col(t, cur)
        still = []
        for k, i in enumerate(live):
            a0 = rng.choice(4, p=p0[k] / p0[k].sum())
            a1 = rng.choice(4, p=p1[k] / p1[k].sum())
            ns, (r0, _r1), done = game.step(
                cur[k], MOVE_ACTIONS[a0], MOVE_ACTIONS[a1], rng=rng)
            if done:
                wins += r0 > 0
                losses += r0 < 0
            else:
                states[i] = ns
                still.append(i)
        live = still
    return {"win": wins / n_games, "tie": 1 - (wins + losses) / n_games,
            "loss": losses / n_games}
