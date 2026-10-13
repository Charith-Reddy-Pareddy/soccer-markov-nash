"""Best-response dynamics and fictitious play on small two-player games (payoff matrices ``A`` for
the row player and ``B`` for the column player): coordination, anti-coordination and zero-sum.

Best-response dynamics: each round the players switch to a best response to what the other just
played. Fictitious play: each round they best-respond to the *average* of everything the other has
played. Ties go to the lowest-numbered action.
"""

from __future__ import annotations

import numpy as np


def best_row(A: np.ndarray, col_action: int) -> int:
    return int(np.argmax(A[:, col_action]))


def best_col(B: np.ndarray, row_action: int) -> int:
    return int(np.argmax(B[row_action, :]))


def best_response_dynamics(A, B, start, rounds: int = 40, alternating: bool = False):
    """The path of joint actions. Simultaneous: both players respond to the other's last action.
    Alternating: the row player responds first, then the column player to the new action."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    r, c = start
    path = [(r, c)]
    for _ in range(rounds):
        if alternating:
            r = best_row(A, c)
            c = best_col(B, r)
        else:
            r, c = best_row(A, c), best_col(B, r)
        path.append((r, c))
    return path


def settles(path, tail: int = 6) -> bool:
    """The last ``tail`` joint actions are all the same."""
    return len(set(path[-tail:])) == 1


def fictitious_play(A, B, start=(0, 0), rounds: int = 2000):
    """The two empirical action frequencies after ``rounds`` rounds of fictitious play."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    cr = np.zeros(A.shape[0])
    cc = np.zeros(A.shape[1])
    cr[start[0]] += 1
    cc[start[1]] += 1
    for _ in range(rounds - 1):
        r = int(np.argmax(A @ (cc / cc.sum())))
        c = int(np.argmax((cr / cr.sum()) @ B))
        cr[r] += 1
        cc[c] += 1
    return cr / cr.sum(), cc / cc.sum()


def mixed_equilibrium_2x2(A, B):
    """The fully mixed equilibrium of a 2x2 game, as the (row's, column's) probability of action 0,
    or None."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    dc = (B[0, 0] - B[0, 1]) - (B[1, 0] - B[1, 1])   # column indifference gives the row's mix
    dr = (A[0, 0] - A[1, 0]) - (A[0, 1] - A[1, 1])   # row indifference gives the column's mix
    if abs(dc) < 1e-12 or abs(dr) < 1e-12:
        return None
    p = (B[1, 1] - B[1, 0]) / dc
    q = (A[1, 1] - A[0, 1]) / dr
    return (float(p), float(q)) if 0 < p < 1 and 0 < q < 1 else None


def pure_equilibria(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
            if A[i, j] >= A[:, j].max() - 1e-12 and B[i, j] >= B[i, :].max() - 1e-12]


GAMES = {
    # B(ach) / S(travinsky): both prefer to be together, each prefers a different place
    "Battle of the sexes (coordination)": {
        "actions": ("B", "S"), "A": [[2, 0], [0, 1]], "B": [[1, 0], [0, 2]]},
    # one point each for choosing different actions, nothing for the same one
    "Different choices (anti-coordination)": {
        "actions": ("0", "1"), "A": [[0, 1], [1, 0]], "B": [[0, 1], [1, 0]]},
    # the row player wants to match, the column player wants to differ: no pure equilibrium
    "Matching pennies (zero-sum)": {
        "actions": ("H", "T"), "A": [[1, -1], [-1, 1]], "B": [[-1, 1], [1, -1]]},
}


def summarize(name: str) -> dict:
    """Everything the report needs about one game: its equilibria, best-response dynamics from every
    start, and the frequencies fictitious play reaches."""
    g = GAMES[name]
    A, B, acts = np.asarray(g["A"], float), np.asarray(g["B"], float), g["actions"]
    label = lambda cell: acts[cell[0]] + acts[cell[1]]  # noqa: E731
    starts = [(i, j) for i in range(2) for j in range(2)]
    dyn = []
    for s in starts:
        sim = best_response_dynamics(A, B, s)
        alt = best_response_dynamics(A, B, s, alternating=True)
        dyn.append({"start": label(s),
                    "simultaneous": label(sim[-1]) if settles(sim) else "keeps switching",
                    "alternating": label(alt[-1]) if settles(alt) else "keeps switching"})
    fp = {label(s): [round(float(v), 3) for v in fictitious_play(A, B, s)[0]] +
          [round(float(v), 3) for v in fictitious_play(A, B, s)[1]] for s in starts}
    mixed = mixed_equilibrium_2x2(A, B)
    return {"game": name, "actions": list(acts),
            "pure": [label(e) for e in pure_equilibria(A, B)],
            "mixed": None if mixed is None else [round(mixed[0], 3), round(mixed[1], 3)],
            "dynamics": dyn, "fictitious_play": fp}
