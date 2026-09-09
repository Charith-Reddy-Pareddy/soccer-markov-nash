"""Independently checkable certificates for the pure / mixed classification of a
soccer stage game.

The Nash-Q solver labels every converged stage game ``M_s`` as *pure* or *mixed*
with an LP and a saddle tolerance. That is enough to run the solver, but it asks
a reader to trust the LP. This module re-derives the label from scratch and
emits a certificate that a skeptic checks with ``O(A^2)`` arithmetic:

* **pure** -- a saddle cell ``(i*, j*)``: ``M[i*, j*]`` is the largest entry in
  its column *and* the smallest in its row. Then it is simultaneously player 0's
  best reply to ``j*`` and player 1's best reply to ``i*``, and
  ``val(M) = M[i*, j*]``. Two length-``A`` comparisons settle it.

* **mixed** -- ``maximin(M) < minimax(M)``: *no* cell is both a row minimum and a
  column maximum. The certificate carries the row-minimum and column-maximum
  vectors (so ``max`` / ``min`` of them are recomputable), a best-reply cycle
  (player 0's best row to a column, player 1's best column to that row, ... back
  to the start -- the "someone always wants to deviate" witness), the shape of
  the game after iterated weak-dominance elimination, and the exact equilibrium
  ``(p*, q*, v)`` with its indifference residual. Where a ``2x2`` matching-pennies
  submatrix exists (it does for most soccer mixed states) it is included as the
  interpretable "each player must guess" core.

:func:`certify_game` builds a certificate from a matrix; :func:`verify` re-checks
every inequality in one and returns the worst violation; :func:`classify` is the
LP-free verdict (``"pure"`` / ``"mixed"``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from soccer_nash.dominance import iewds
from soccer_nash.matrix_games import solve_zero_sum
from soccer_nash.numerics import epsilon_equilibrium, mixing_entropy

_TOL = 1e-7


@dataclass(frozen=True)
class PureCertificate:
    kind: str  # "pure"
    saddle: tuple[int, int]
    value: float
    row_slack: float  # M[i*, j*] - min_j M[i*, j]  (0 => it is the row minimum)
    col_slack: float  # max_i M[i, j*] - M[i*, j*]  (0 => it is the column maximum)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MatchingPenniesCore:
    """A ``2x2`` submatrix whose four best replies cross -- matching pennies."""

    rows: tuple[int, int]
    cols: tuple[int, int]
    submatrix: list[list[float]]
    margin: float  # min slack across the four strict crossing inequalities

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MixedCertificate:
    kind: str  # "mixed"
    maximin: float
    minimax: float
    gap: float  # minimax - maximin > 0  <=>  no pure saddle
    row_minima: list[float]
    col_maxima: list[float]
    best_reply_cycle: list[tuple[int, int]]  # (row, col) legs of the cycle
    iewds_shape: tuple[int, int]  # residual shape after weak-dominance elimination
    iewds_gap: float  # its own maximin < minimax gap
    core: MatchingPenniesCore | None
    row_strategy: list[float]
    col_strategy: list[float]
    value: float
    equilibrium_residual: float  # epsilon_equilibrium(M, p*, q*)
    support: tuple[int, int]
    entropy: float  # bits; > 0 confirms genuine randomisation

    def as_dict(self) -> dict:
        d = asdict(self)
        d["core"] = self.core.as_dict() if self.core else None
        return d


# --------------------------------------------------------------------- builders


def _best_reply_cycle(M: np.ndarray) -> list[tuple[int, int]]:
    """A cycle in the best-reply correspondence of a saddle-free game.

    Start at player 1's minimax column and player 0's best row to it, then
    alternate: player 1 moves to its best column for the current row, player 0
    to its best row for the current column. Each leg changes exactly one index.
    A pure saddle would be a fixed point of this walk; its absence forces the
    walk to close into a cycle of length >= 2 distinct cells.
    """
    j = int(np.argmin(M.max(axis=0)))
    i = int(np.argmax(M[:, j]))
    path = [(i, j)]
    player1_moves = True
    while True:
        i, j = path[-1]
        nxt = (i, int(np.argmin(M[i, :]))) if player1_moves else (int(np.argmax(M[:, j])), j)
        if nxt in path:
            return path[path.index(nxt):] + [nxt]
        path.append(nxt)
        player1_moves = not player1_moves


def _matching_pennies_core(M: np.ndarray, tol: float = _TOL) -> MatchingPenniesCore | None:
    """The ``2x2`` submatrix with the widest crossing margin, if one exists.

    A submatrix ``A`` crosses when
    ``A[0,0] > A[1,0]``, ``A[1,1] > A[0,1]`` (player 0, the maximiser, wants row
    0 under column 0 and row 1 under column 1) and ``A[0,0] > A[0,1]``,
    ``A[1,1] > A[1,0]`` (player 1, the minimiser, wants column 1 under row 0 and
    column 0 under row 1). Such an ``A`` has no pure saddle. It is the
    interpretable "each player must guess" core; a few degenerate saddle-free
    games have no strict ``2x2`` crossing, and then this is ``None`` while the
    ``gap`` and the best-reply cycle still certify the label.
    """
    nr, nc = M.shape
    best: MatchingPenniesCore | None = None
    for i1 in range(nr):
        for i2 in range(i1 + 1, nr):
            for j1 in range(nc):
                for j2 in range(j1 + 1, nc):
                    for (a, b), (c, d) in (((i1, i2), (j1, j2)), ((i2, i1), (j1, j2)),
                                           ((i1, i2), (j2, j1)), ((i2, i1), (j2, j1))):
                        A = M[np.ix_([a, b], [c, d])]
                        m = min(
                            A[0, 0] - A[1, 0],
                            A[1, 1] - A[0, 1],
                            A[0, 0] - A[0, 1],
                            A[1, 1] - A[1, 0],
                        )
                        if m > tol and (best is None or m > best.margin):
                            best = MatchingPenniesCore(
                                rows=(int(a), int(b)),
                                cols=(int(c), int(d)),
                                submatrix=[[float(v) for v in row] for row in A],
                                margin=float(m),
                            )
    return best


def certify_game(
    M: np.ndarray, tol: float = _TOL
) -> PureCertificate | MixedCertificate:
    """Build the certificate for a zero-sum stage game (row player maximises)."""
    M = np.asarray(M, dtype=float)
    row_minima = M.min(axis=1)
    col_maxima = M.max(axis=0)
    maximin = float(row_minima.max())
    minimax = float(col_maxima.min())

    if minimax - maximin <= tol:
        i = int(np.argmax(row_minima))
        j = int(np.argmin(col_maxima))
        # a true saddle cell need not be (argmax row-min, argmin col-max) jointly
        # when there are ties; search for one that is both.
        for ii in range(M.shape[0]):
            for jj in range(M.shape[1]):
                if (
                    M[ii, jj] <= row_minima[ii] + tol
                    and M[ii, jj] >= col_maxima[jj] - tol
                ):
                    i, j = ii, jj
        return PureCertificate(
            kind="pure",
            saddle=(i, j),
            value=float(M[i, j]),
            row_slack=float(M[i, j] - row_minima[i]),
            col_slack=float(col_maxima[j] - M[i, j]),
        )

    ri, ci, residual = iewds(M, tol)
    r_lo, r_hi = float(residual.min(axis=1).max()), float(residual.max(axis=0).min())
    v, p, q = solve_zero_sum(M)
    return MixedCertificate(
        kind="mixed",
        maximin=maximin,
        minimax=minimax,
        gap=minimax - maximin,
        row_minima=[float(x) for x in row_minima],
        col_maxima=[float(x) for x in col_maxima],
        best_reply_cycle=[(int(a), int(b)) for a, b in _best_reply_cycle(M)],
        iewds_shape=(int(residual.shape[0]), int(residual.shape[1])),
        iewds_gap=r_hi - r_lo,
        core=_matching_pennies_core(M, tol),
        row_strategy=[float(x) for x in p],
        col_strategy=[float(x) for x in q],
        value=float(v),
        equilibrium_residual=float(epsilon_equilibrium(M, p, q)),
        support=(int((p > 1e-6).sum()), int((q > 1e-6).sum())),
        entropy=float(mixing_entropy(p)),
    )


def classify(M: np.ndarray, tol: float = _TOL) -> str:
    """LP-free verdict from the pure bounds alone: ``"pure"`` or ``"mixed"``."""
    M = np.asarray(M, dtype=float)
    maximin = M.min(axis=1).max()
    minimax = M.max(axis=0).min()
    return "mixed" if minimax - maximin > tol else "pure"


# ---------------------------------------------------------------------- verify


def verify(cert: PureCertificate | MixedCertificate, M: np.ndarray, tol: float = _TOL) -> float:
    """Re-check every claim in ``cert`` against ``M``; return the worst violation.

    ``0.0`` (to ``tol``) means the certificate is sound. Raises ``ValueError``
    only on a structural mismatch (wrong shape, claimed cycle that is not one).
    """
    M = np.asarray(M, dtype=float)
    worst = 0.0

    if isinstance(cert, PureCertificate):
        i, j = cert.saddle
        # M[i, j] is the row minimum and the column maximum.
        worst = max(worst, float(M[i, j] - M[i, :].min()))  # row-min violation
        worst = max(worst, float(M[:, j].max() - M[i, j]))  # col-max violation
        worst = max(worst, abs(M[i, j] - cert.value))
        return worst

    # mixed
    worst = max(worst, abs(float(M.min(axis=1).max()) - cert.maximin))
    worst = max(worst, abs(float(M.max(axis=0).min()) - cert.minimax))
    if cert.gap <= tol:
        raise ValueError("mixed certificate with a non-positive pure-bound gap")

    # the best-reply cycle: every leg is an actual best reply and it closes.
    legs = cert.best_reply_cycle
    if len(legs) < 3 or legs[0] != legs[-1]:
        raise ValueError("best_reply_cycle is not a closed walk")
    for k in range(len(legs) - 1):
        (i0, j0), (i1, j1) = legs[k], legs[k + 1]
        if i0 == i1:  # player 1 moved: j1 minimises row i0
            worst = max(worst, float(M[i0, j1] - M[i0, :].min()))
        elif j0 == j1:  # player 0 moved: i1 maximises column j0
            worst = max(worst, float(M[:, j0].max() - M[i1, j0]))
        else:
            raise ValueError("best_reply_cycle leg changes both indices")

    if cert.core is not None:
        A = np.array(cert.core.submatrix)
        crossings = [
            A[0, 0] - A[1, 0], A[1, 1] - A[0, 1],
            A[0, 0] - A[0, 1], A[1, 1] - A[1, 0],
        ]
        worst = max(worst, float(-min(crossings)))  # each should be > 0
        want = M[np.ix_(list(cert.core.rows), list(cert.core.cols))]
        worst = max(worst, float(np.abs(A - want).max()))

    p = np.array(cert.row_strategy)
    q = np.array(cert.col_strategy)
    worst = max(worst, abs(p.sum() - 1.0), abs(q.sum() - 1.0))
    worst = max(worst, float(epsilon_equilibrium(M, p, q)))
    return worst
