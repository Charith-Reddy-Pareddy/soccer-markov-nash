"""Best response in one continuous action -- the three methods on page 3 of the
group note, for a function ``f(a)`` on ``[lo, hi]`` that the caller wants
maximised (for a player holding ``Q(s, a1, a2)``, ``f = lambda a: Q(s, a, a2)``).

* :func:`bisect_br` -- bisect on the sign of the derivative ``dQ/da``.
* :func:`gradient_br` -- ascend a finite-difference gradient.
* :func:`quadratic_br` -- fit a parabola to a few samples, take its vertex
  (the LQR-style approximation), and refit around it.

All three assume ``f`` is smooth and single-peaked on the interval. A circular
action (an angle) is handled by picking ``[lo, hi]`` as one lap, so a peak at
the seam is not found.
"""

from __future__ import annotations

import numpy as np


def _slope(f, a: float, lo: float, hi: float, eps: float) -> float:
    left, right = max(a - eps, lo), min(a + eps, hi)
    return (f(right) - f(left)) / (right - left)


def bisect_br(f, lo: float, hi: float, eps: float = 1e-6, tol: float = 1e-8) -> float:
    a, b = lo, hi
    while b - a > tol:
        m = 0.5 * (a + b)
        if _slope(f, m, lo, hi, eps) > 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def gradient_br(
    f, lo: float, hi: float, start: float | None = None,
    lr: float = 0.5, eps: float = 1e-5, steps: int = 500,
) -> float:
    a = 0.5 * (lo + hi) if start is None else start
    for k in range(steps):
        a = float(np.clip(a + lr / (1 + k) ** 0.5 * _slope(f, a, lo, hi, eps), lo, hi))
    return a


def quadratic_br(f, lo: float, hi: float, n: int = 5, rounds: int = 6) -> float:
    a, b = lo, hi
    best = 0.5 * (lo + hi)
    for _ in range(rounds):
        xs = np.linspace(a, b, n)
        ys = np.array([f(x) for x in xs])
        c2, c1, _ = np.polyfit(xs, ys, 2)
        best = float(xs[np.argmax(ys)]) if c2 >= 0 else float(np.clip(-c1 / (2 * c2), lo, hi))
        half = (b - a) / 4
        a, b = max(lo, best - half), min(hi, best + half)
    return best
