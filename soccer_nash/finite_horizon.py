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
from soccer_nash.matrix_games import pure_bounds
from soccer_nash.numerics import epsilon_equilibrium
from soccer_nash.symmetry import flip_distribution, mirror_state


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
    starts: list[State] | None = None,
) -> dict[str, float]:
    """Repeated play from the kickoff, or cycling through ``starts``; player 0's
    win / tie / loss rates."""
    rng = np.random.default_rng(seed)
    first = starts or [game.initial_state()]
    states: list[State] = [first[k % len(first)] for k in range(n_games)]
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


def mirror_gap(game: SoccerGame, row, col, times=(0, 25, 50, 75, 99)) -> tuple[float, float]:
    """(mean, max) over all states and the given steps of
    ``max_a |row(t, s) - flip(col(t, mirror(s)))|``: how far the two players'
    policies are from being mirror images (0 for a mirror-symmetric pair)."""
    states = list(game.states())
    mirrored = [mirror_state(s, game.width) for s in states]
    gaps = []
    for t in times:
        want = np.array([flip_distribution(q) for q in col(t, mirrored)])
        gaps.append(np.abs(row(t, states) - want).max(axis=1))
    g = np.concatenate(gaps)
    return float(g.mean()), float(g.max())


def evaluate(game, solver, exact, row, col, gamma, horizon, n_games, seed) -> dict:
    """Exploitability plus repeated-play win / tie / loss counts for both players
    against a random player, the exact equilibrium and the exact best response."""
    e_row, e_col = exact
    v0, t0 = best_response(solver, col, 0, gamma, horizon)
    v1, t1 = best_response(solver, row, 1, gamma, horizon)
    out = {"exploitability": round(v0 + v1, 4)}
    opp_for_row = {"random": uniform, "nash": e_col, "br": br_policy(t1)}
    opp_for_col = {"random": uniform, "nash": e_row, "br": br_policy(t0)}
    for name, op in opp_for_row.items():
        r = play(game, row, op, n_games, horizon, seed)
        out.update({f"row_win_vs_{name}": r["win"], f"row_tie_vs_{name}": r["tie"],
                    f"row_loss_vs_{name}": r["loss"]})
    for name, op in opp_for_col.items():
        r = play(game, op, col, n_games, horizon, seed)
        out.update({f"col_win_vs_{name}": r["loss"], f"col_tie_vs_{name}": r["tie"],
                    f"col_loss_vs_{name}": r["win"]})
    out["mirror_gap_mean"], out["mirror_gap_max"] = mirror_gap(game, row, col)
    return {k: round(v, 3) for k, v in out.items()}


def mixed_states(solver, values, gamma: float, tol: float = 1e-9) -> list[State]:
    """States whose stage game at the first step has no pure saddle, i.e. where the
    exact equilibrium has to mix."""
    out = []
    for st in solver._states:
        lo, hi = pure_bounds(solver._matrix(st, values[1], gamma=gamma))
        if hi - lo > tol:
            out.append(st)
    return out


def mixed_state_metrics(solver, values, gamma, mixed, row0, col0, exact_row0, exact_col0) -> dict:
    """How close a learner's first-step policy is to the exact equilibrium at the mixed
    states: total-variation distance to the exact mix, equilibrium regret against the exact
    stage game (0 at any equilibrium, whichever one), and how often it actually mixes.

    Each of ``row0``, ``col0``, ``exact_row0`` and ``exact_col0`` maps a state to action
    probabilities."""
    tv_r, tv_c, regret, mixes_r = [], [], [], []
    for st in mixed:
        m = solver._matrix(st, values[1], gamma=gamma)
        tv_r.append(0.5 * np.abs(row0[st] - exact_row0[st]).sum())
        tv_c.append(0.5 * np.abs(col0[st] - exact_col0[st]).sum())
        regret.append(epsilon_equilibrium(m, row0[st], col0[st]))
        mixes_r.append(float(row0[st].max() < 0.9))
    return {"mixed_states": len(mixed), "tv_row": float(np.mean(tv_r)),
            "tv_col": float(np.mean(tv_c)),
            "regret_mean": float(np.mean(regret)), "regret_max": float(np.max(regret)),
            "share_mixing_row": float(np.mean(mixes_r))}
