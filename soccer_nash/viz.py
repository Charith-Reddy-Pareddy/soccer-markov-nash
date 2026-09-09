"""Policy and value visualizations for the soccer Markov game.

Everything is inline SVG (no plotting library), themed via ``var(--x, #hex)`` so
the same string renders in ``docs/report.html`` and as a standalone file.

- :func:`policy_svg` -- a board with each player's action distribution drawn as
  probability-weighted arrows: a pure policy is one bold arrow, a mixed policy
  fans out, so mixing is visible at a glance.
- :func:`mixing_map_svg` -- fix the defender, sweep the carrier over the board,
  shade each cell by how badly its stage game needs mixing. This is the "where
  does the carrier have to guess" picture.
- :func:`value_map_svg` -- fix one player, sweep the other, shade by ``V*`` on a
  blue (player 0 ahead) to orange (player 1 ahead) diverging scale.
- :func:`strategy_bars_svg` -- a compact two-bar chart of a mixed equilibrium.
- :func:`panel_svg` -- tile several of the above into one figure.
"""

from __future__ import annotations

import math

import numpy as np

from soccer_nash.game import SoccerGame, State
from soccer_nash.matrix_games import pure_bounds
from soccer_nash.render import (
    BALL,
    CELL,
    MARGIN,
    P0,
    P1,
    PAPER,
    board_frame,
    cell_center,
)

_INK = "var(--ink, #19211c)"
_FAINT = "var(--ink-faint, #77817a)"
_EMBER = (169, 78, 24)      # mixing intensity ramp target
_EMBER_HEX = f"#{_EMBER[0]:02x}{_EMBER[1]:02x}{_EMBER[2]:02x}"
_BLUE = (47, 107, 176)      # value ramp: player 0 ahead
_WARM = (194, 90, 42)       # value ramp: player 1 ahead
_ARROW = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}   # U D L R, y up


def _marker(colour: str) -> str:
    return (
        f'<marker id="ah-{colour[-7:-1]}" viewBox="0 0 8 8" refX="6" refY="4" '
        f'markerWidth="3.6" markerHeight="3.6" orient="auto">'
        f'<path d="M0 0.5 L8 4 L0 7.5 z" fill="{colour}"/></marker>'
    )


def _lerp(a: tuple, b: tuple, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    r, g, bl = (round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{r:02x}{g:02x}{bl:02x}"


def _svg(w: float, h: float, body: list[str], label: str, pad: float = 0) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{-pad:.0f} {-pad:.0f} {w + 2 * pad:.0f} {h + 2 * pad:.0f}" '
        f'role="img" aria-label="{label}">\n' + "\n".join(body) + "\n</svg>"
    )


# --------------------------------------------------------------- policy_svg

def _action_fan(cx: float, cy: float, dist: np.ndarray, colour: str) -> list[str]:
    out = []
    top = float(dist.max())
    for a, p in enumerate(dist):
        if p < 0.02:
            continue
        if a == 4:  # STAND -- a ring around the player, no arrow
            out.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="15" fill="none" '
                f'stroke="{colour}" stroke-width="{1.6 + 2.2 * (p / top):.1f}" '
                f'stroke-dasharray="2 2" '
                f'opacity="{0.4 + 0.55 * p:.2f}"/>'
            )
            if p < 0.985:
                out.append(
                    f'<text x="{cx:.1f}" y="{cy + 26:.1f}" text-anchor="middle" '
                    f'font-family="ui-monospace,monospace" font-size="9" '
                    f'fill="{colour}">hold {p * 100:.0f}%</text>'
                )
            continue
        dx, dy = _ARROW[a]
        start = 13                                  # clear the player disc
        length = start + 7 + 18 * p                 # 20..38 px
        sx, sy = cx + dx * start, cy - dy * start
        ex, ey = cx + dx * length, cy - dy * length  # screen y is inverted
        wgt = 1.8 + 2.4 * (p / top)
        out.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{colour}" stroke-width="{wgt:.1f}" stroke-linecap="round" '
            f'opacity="{0.4 + 0.55 * p:.2f}" marker-end="url(#ah-{colour[-7:-1]})"/>'
        )
        if p < 0.985:
            lx, ly = cx + dx * (length + 11), cy - dy * (length + 11)
            out.append(
                f'<text x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="middle" '
                f'font-family="ui-monospace,monospace" font-size="9" '
                f'fill="{colour}">{p * 100:.0f}%</text>'
            )
    return out


def policy_svg(
    game: SoccerGame,
    state: State,
    row_policy: dict[State, np.ndarray],
    col_policy: dict[State, np.ndarray],
    value: float | None = None,
    title: str | None = None,
) -> str:
    """Board with both players' action distributions as probability arrows."""
    w, h = game.width, game.height
    x0, y0, x1, y1, b = state
    tw = w * CELL + 2 * MARGIN
    th = h * CELL + 2 * MARGIN + 4

    p0 = np.asarray(row_policy[state], dtype=float)
    p1 = np.asarray(col_policy[state], dtype=float)
    mixed0 = float((p0 > 0.02).sum()) > 1
    mixed1 = float((p1 > 0.02).sum()) > 1

    body = [f'<defs>{_marker(P0)}{_marker(P1)}</defs>', *board_frame(game)]
    cx0, cy0 = cell_center(x0, y0, h)
    cx1, cy1 = cell_center(x1, y1, h)
    body.append(f'<circle cx="{cx0:.1f}" cy="{cy0:.1f}" r="11" fill="{P0}"/>')
    body.append(f'<circle cx="{cx1:.1f}" cy="{cy1:.1f}" r="11" fill="{P1}"/>')
    bx, by = (cx0, cy0) if b == 0 else (cx1, cy1)
    body.append(
        f'<circle cx="{bx + 10:.1f}" cy="{by - 10:.1f}" r="4.5" '
        f'fill="{BALL}" stroke="{PAPER}" stroke-width="1.2"/>'
    )
    body += _action_fan(cx0, cy0, p0, P0)
    body += _action_fan(cx1, cy1, p1, P1)

    tag = "mixed" if (mixed0 or mixed1) else "pure"
    sub = f"{tag} equilibrium" + (f" · V = {value:+.3f}" if value is not None else "")
    body.append(
        f'<text x="{MARGIN}" y="{th - 4}" font-family="ui-monospace,monospace" '
        f'font-size="10" fill="{_FAINT}">{sub}</text>'
    )
    if title:
        body.append(
            f'<text x="{MARGIN}" y="-6" font-family="Barlow Semi Condensed,'
            f'sans-serif" font-weight="600" font-size="13" fill="{_INK}">{title}</text>'
        )
    return _svg(tw, th, body, f"policy at state {state}: {tag}", pad=16)


# ----------------------------------------------------------- mixing_map_svg

def mixing_map_svg(
    game: SoccerGame,
    matrix_of,
    defender_cell: tuple[int, int],
    defender: int = 1,
    title: str | None = None,
) -> str:
    """Fix ``defender`` at ``defender_cell``; shade every carrier cell by
    ``minimax - maximin`` of its stage game (0 = pure saddle)."""
    w, h = game.width, game.height
    dx, dy = defender_cell
    carrier = 1 - defender
    tw = w * CELL + 2 * MARGIN
    th = h * CELL + 2 * MARGIN + 16

    gaps: dict[tuple[int, int], float] = {}
    for cx in range(w):
        for cy in range(h):
            if (cx, cy) == (dx, dy):
                continue
            s = (cx, cy, dx, dy, carrier) if carrier == 0 else (dx, dy, cx, cy, carrier)
            lo, hi = pure_bounds(np.asarray(matrix_of(s), dtype=float))
            gaps[(cx, cy)] = max(hi - lo, 0.0)
    hi_gap = max(gaps.values()) or 1.0

    body = [*board_frame(game)]
    for (cx, cy), gap in gaps.items():
        px = MARGIN + cx * CELL
        py = MARGIN + (h - 1 - cy) * CELL
        if gap < 1e-7:
            fill, txt = "#eef2ec", ""
        else:
            fill = _lerp((238, 242, 236), _EMBER, 0.25 + 0.75 * gap / hi_gap)
            txt = f"{gap:.2f}".lstrip("0")
        body.append(
            f'<rect x="{px + 1}" y="{py + 1}" width="{CELL - 2}" height="{CELL - 2}" '
            f'fill="{fill}"/>'
        )
        if txt:
            body.append(
                f'<text x="{px + CELL / 2:.0f}" y="{py + CELL / 2 + 3:.0f}" '
                f'text-anchor="middle" font-family="ui-monospace,monospace" '
                f'font-size="9" fill="{PAPER}">{txt}</text>'
            )
    ddx, ddy = cell_center(dx, dy, h)
    dcol = P1 if defender == 1 else P0
    body.append(f'<circle cx="{ddx:.1f}" cy="{ddy:.1f}" r="12" fill="{dcol}"/>')
    body.append(
        f'<text x="{MARGIN}" y="{th - 5}" font-family="ui-monospace,monospace" '
        f'font-size="10" fill="{_FAINT}">cells: minimax minus maximin of the '
        f"carrier's stage game</text>"
    )
    if title:
        body.append(
            f'<text x="{MARGIN}" y="-6" font-family="Barlow Semi Condensed,'
            f'sans-serif" font-weight="600" font-size="13" fill="{_INK}">{title}</text>'
        )
    return _svg(
        tw, th, body, "map of where the carrier's stage game is mixed",
        pad=20 if title else 6,
    )


# ------------------------------------------------------------ value_map_svg

def value_map_svg(
    game: SoccerGame,
    values: dict[State, float],
    other_cell: tuple[int, int],
    mover: int = 0,
    ball: int | None = None,
    title: str | None = None,
) -> str:
    """Fix the non-``mover`` player at ``other_cell``; shade every ``mover``
    cell by ``V*`` on a blue (player 0 ahead) / orange (player 1 ahead) scale."""
    w, h = game.width, game.height
    ox, oy = other_cell
    b = mover if ball is None else ball
    tw = w * CELL + 2 * MARGIN
    th = h * CELL + 2 * MARGIN + 16

    vs: dict[tuple[int, int], float] = {}
    for mx in range(w):
        for my in range(h):
            if (mx, my) == (ox, oy):
                continue
            s = (mx, my, ox, oy, b) if mover == 0 else (ox, oy, mx, my, b)
            if s in values:
                vs[(mx, my)] = values[s]
    span = max((abs(v) for v in vs.values()), default=1.0) or 1.0

    body = [*board_frame(game)]
    for (mx, my), v in vs.items():
        px = MARGIN + mx * CELL
        py = MARGIN + (h - 1 - my) * CELL
        t = abs(v) / span
        fill = _lerp((238, 242, 236), _BLUE if v > 0 else _WARM, 0.12 + 0.85 * t)
        body.append(
            f'<rect x="{px + 1}" y="{py + 1}" width="{CELL - 2}" '
            f'height="{CELL - 2}" fill="{fill}"/>'
        )
        if abs(v) > 1e-6:
            lab = f"{v:+.2f}".replace("0.", ".")
            ink = PAPER if t > 0.4 else _INK
            body.append(
                f'<text x="{px + CELL / 2:.0f}" y="{py + CELL / 2 + 3:.0f}" '
                f'text-anchor="middle" font-family="ui-monospace,monospace" '
                f'font-size="8.5" fill="{ink}">{lab}</text>'
            )
    ocx, ocy = cell_center(ox, oy, h)
    ocol = P1 if mover == 0 else P0
    body.append(f'<circle cx="{ocx:.1f}" cy="{ocy:.1f}" r="12" fill="{ocol}"/>')
    body.append(
        f'<text x="{MARGIN}" y="{th - 5}" font-family="ui-monospace,monospace" '
        f'font-size="10" fill="{_FAINT}">V* by player {mover} position '
        f'(blue = player 0 ahead)</text>'
    )
    if title:
        body.append(
            f'<text x="{MARGIN}" y="-6" font-family="Barlow Semi Condensed,'
            f'sans-serif" font-weight="600" font-size="13" fill="{_INK}">{title}</text>'
        )
    return _svg(tw, th, body, "value map", pad=20 if title else 6)


# --------------------------------------------------------- occupancy_map_svg

_VISIT = (32, 74, 122)   # occupancy ramp target (deep blue)


def occupancy_map_svg(
    game: SoccerGame,
    dist: dict[State, float],
    mixed_states: set[State] | None = None,
    ball: int = 0,
    title: str | None = None,
) -> str:
    """Marginalise ``dist`` over the carrier's cell (for ball owner ``ball``) and
    shade each cell by how much equilibrium time is spent there. Cells that hold
    a no-pure-saddle state carry an amber dot."""
    w, h = game.width, game.height
    mixed_states = mixed_states or set()
    tw = w * CELL + 2 * MARGIN
    th = h * CELL + 2 * MARGIN + 16

    carrier_cell = 0 if ball == 0 else 2
    occ: dict[tuple[int, int], float] = {}
    has_mixed: set[tuple[int, int]] = set()
    for s, m in dist.items():
        if s[4] != ball:
            continue
        cell = (s[carrier_cell], s[carrier_cell + 1])
        occ[cell] = occ.get(cell, 0.0) + m
        if s in mixed_states:
            has_mixed.add(cell)
    hi = max(occ.values(), default=1.0) or 1.0

    body = [*board_frame(game)]
    for cx in range(w):
        for cy in range(h):
            px = MARGIN + cx * CELL
            py = MARGIN + (h - 1 - cy) * CELL
            m = occ.get((cx, cy), 0.0)
            if m > 1e-9:
                fill = _lerp((238, 242, 236), _VISIT, 0.12 + 0.88 * (m / hi) ** 0.5)
                body.append(
                    f'<rect x="{px + 1}" y="{py + 1}" width="{CELL - 2}" '
                    f'height="{CELL - 2}" fill="{fill}"/>'
                )
                if m * 100 >= 0.5:
                    body.append(
                        f'<text x="{px + CELL / 2:.0f}" y="{py + CELL / 2 + 3:.0f}" '
                        f'text-anchor="middle" font-family="ui-monospace,monospace" '
                        f'font-size="8.5" fill="{PAPER if m / hi > 0.25 else _INK}">'
                        f'{m * 100:.0f}</text>'
                    )
            if (cx, cy) in has_mixed and occ.get((cx, cy), 0.0) * 100 >= 0.5:
                body.append(
                    f'<circle cx="{px + CELL - 7:.0f}" cy="{py + 7:.0f}" r="3.2" '
                    f'fill="{BALL}" stroke="{PAPER}" stroke-width="0.8"/>'
                )
    body.append(
        f'<text x="{MARGIN}" y="{th - 5}" font-family="ui-monospace,monospace" '
        f'font-size="10" fill="{_FAINT}">equilibrium time by carrier cell '
        f'(% ; amber = a mixed state sits here)</text>'
    )
    if title:
        body.append(
            f'<text x="{MARGIN}" y="-6" font-family="Barlow Semi Condensed,'
            f'sans-serif" font-weight="600" font-size="13" fill="{_INK}">{title}</text>'
        )
    return _svg(tw, th, body, "occupancy map", pad=20 if title else 6)


# -------------------------------------------------------- grouped_bars_svg

def grouped_bars_svg(
    categories: list[str],
    series: list[tuple[str, str, list[float]]],
    y_label: str = "",
    title: str | None = None,
    baseline: float = 0.0,
) -> str:
    """A grouped bar chart. ``categories`` label the x groups; ``series`` is
    ``(name, colour, values)`` with one value per category. Bars can go below
    ``baseline`` (drawn downward)."""
    n_cat = len(categories)
    n_ser = len(series)
    group_w, gap, bar_gap = 108, 26, 3
    bar_w = (group_w - (n_ser - 1) * bar_gap) / n_ser
    ph, ml, mt = 150, 46, 22
    pw = n_cat * group_w + (n_cat - 1) * gap
    allv = [v for _n, _c, vs in series for v in vs] + [baseline]
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1.0

    def sy(v: float) -> float:
        return mt + ph - (v - lo) / span * ph

    y_base = sy(baseline)
    body = [
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" '
        f'stroke="var(--rule-strong, #b7c4b9)" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{y_base:.1f}" x2="{ml + pw}" y2="{y_base:.1f}" '
        f'stroke="var(--rule-strong, #b7c4b9)" stroke-width="1"/>',
    ]
    for frac in (0.0, 0.5, 1.0):
        yv = lo + frac * span
        gy = sy(yv)
        body.append(
            f'<text x="{ml - 6}" y="{gy + 3:.1f}" text-anchor="end" '
            f'font-family="ui-monospace,monospace" font-size="8.5" '
            f'fill="{_FAINT}">{yv:+.1f}</text>'
        )
    for ci, cat in enumerate(categories):
        gx = ml + ci * (group_w + gap)
        body.append(
            f'<text x="{gx + group_w / 2:.0f}" y="{mt + ph + 14:.0f}" '
            f'text-anchor="middle" font-family="ui-monospace,monospace" '
            f'font-size="9" fill="{_FAINT}">{cat}</text>'
        )
        for si, (_name, colour, vals) in enumerate(series):
            v = vals[ci]
            bx = gx + si * (bar_w + bar_gap)
            top = min(sy(v), y_base)
            hgt = abs(sy(v) - y_base)
            body.append(
                f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
                f'height="{max(hgt, 0.5):.1f}" fill="{colour}" '
                f'opacity="0.9"/>'
            )
            body.append(
                f'<text x="{bx + bar_w / 2:.1f}" '
                f'y="{(top - 3) if v >= baseline else (top + hgt + 9):.1f}" '
                f'text-anchor="middle" font-family="ui-monospace,monospace" '
                f'font-size="8" fill="{colour}">{v:+.2f}</text>'
            )
    lx = ml
    for name, colour, _vals in series:
        body.append(
            f'<rect x="{lx:.0f}" y="{mt + ph + 24:.0f}" width="9" height="9" '
            f'fill="{colour}"/>'
        )
        body.append(
            f'<text x="{lx + 13:.0f}" y="{mt + ph + 32:.0f}" '
            f'font-family="ui-monospace,monospace" font-size="9" '
            f'fill="{_INK}">{name}</text>'
        )
        lx += 16 + len(name) * 6.2
    if y_label:
        body.append(
            f'<text x="12" y="{mt + ph / 2:.0f}" text-anchor="middle" '
            f'font-family="ui-monospace,monospace" font-size="9" fill="{_FAINT}" '
            f'transform="rotate(-90 12 {mt + ph / 2:.0f})">{y_label}</text>'
        )
    if title:
        body.insert(
            0,
            f'<text x="{ml}" y="-4" font-family="Barlow Semi Condensed,sans-serif" '
            f'font-weight="600" font-size="13" fill="{_INK}">{title}</text>',
        )
    return _svg(ml + pw + 12, mt + ph + 44, body, "grouped bar chart",
                pad=16 if title else 8)


# ------------------------------------------------------- strategy_bars_svg

_ACT = ("U", "D", "L", "R", "stay")


def strategy_bars_svg(
    row_p: np.ndarray, col_p: np.ndarray, value: float | None = None,
    row_label: str = "player 0", col_label: str = "player 1",
) -> str:
    """Two stacked horizontal bars -- the row and column mixed strategies."""
    row_p = np.asarray(row_p, dtype=float)
    col_p = np.asarray(col_p, dtype=float)
    w, bar_h, gap, left = 200, 20, 26, 66
    th = 2 * (bar_h + gap) + 24
    body = []
    for k, (p, colour, lab) in enumerate(
        [(row_p, P0, row_label), (col_p, P1, col_label)]
    ):
        y = 8 + k * (bar_h + gap)
        body.append(
            f'<text x="{left - 6}" y="{y + bar_h - 5}" text-anchor="end" '
            f'font-family="ui-monospace,monospace" font-size="10" fill="{_FAINT}">{lab}</text>'
        )
        xoff = left
        for a, prob in enumerate(p):
            if prob < 1e-4:
                continue
            bw = prob * w
            body.append(
                f'<rect x="{xoff:.1f}" y="{y}" width="{bw:.1f}" height="{bar_h}" '
                f'fill="{colour}" opacity="{0.35 + 0.6 * prob:.2f}"/>'
            )
            if bw > 16:
                body.append(
                    f'<text x="{xoff + bw / 2:.1f}" y="{y + bar_h - 6}" '
                    f'text-anchor="middle" font-family="ui-monospace,monospace" '
                    f'font-size="9" fill="{PAPER}">{_ACT[a]} {prob * 100:.0f}</text>'
                )
            xoff += bw
    if value is not None:
        body.append(
            f'<text x="{left}" y="{th - 5}" font-family="ui-monospace,monospace" '
            f'font-size="10" fill="{_INK}">value {value:+.3f}</text>'
        )
    return _svg(left + w + 8, th, body, "mixed strategy bars")


# -------------------------------------------------------------- line_chart_svg

def line_chart_svg(
    series: list[tuple[str, str, list[tuple[float, float]]]],
    x_label: str = "",
    y_label: str = "",
    title: str | None = None,
    vline: float | None = None,
) -> str:
    """A small multi-series line chart. ``series`` is ``(name, colour, points)``
    with ``points`` a list of ``(x, y)``. ``vline`` draws a dashed marker at an
    x value (e.g. a threshold)."""
    pw, ph, ml, mt, mr = 300, 170, 44, 18, 78
    xs = [x for _n, _c, pts in series for x, _y in pts]
    ys = [y for _n, _c, pts in series for _x, y in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(0.0, min(ys)), max(ys)
    xr = x1 - x0 or 1.0
    yr = y1 - y0 or 1.0

    def sx(x: float) -> float:
        return ml + (x - x0) / xr * pw

    def sy(y: float) -> float:
        return mt + ph - (y - y0) / yr * ph

    body = [
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" '
        f'stroke="var(--rule-strong, #b7c4b9)" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" '
        f'stroke="var(--rule-strong, #b7c4b9)" stroke-width="1"/>',
    ]
    for frac in (0.0, 0.5, 1.0):
        yv = y0 + frac * yr
        gy = sy(yv)
        body.append(
            f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + pw}" y2="{gy:.1f}" '
            f'stroke="var(--rule, #cdd8ce)" stroke-width="0.6"/>'
        )
        body.append(
            f'<text x="{ml - 6}" y="{gy + 3:.1f}" text-anchor="end" '
            f'font-family="ui-monospace,monospace" font-size="8.5" '
            f'fill="{_FAINT}">{yv:.0f}</text>'
        )
    for xv in (x0, (x0 + x1) / 2, x1):
        body.append(
            f'<text x="{sx(xv):.1f}" y="{mt + ph + 12:.1f}" text-anchor="middle" '
            f'font-family="ui-monospace,monospace" font-size="8.5" '
            f'fill="{_FAINT}">{xv:.1f}</text>'
        )
    if vline is not None:
        vx = sx(vline)
        body.append(
            f'<line x1="{vx:.1f}" y1="{mt}" x2="{vx:.1f}" y2="{mt + ph}" '
            f'stroke="{_EMBER_HEX}" stroke-width="1.2" stroke-dasharray="3 3"/>'
        )
    for name, colour, pts in series:
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(x):.1f} {sy(y):.1f}"
            for i, (x, y) in enumerate(pts)
        )
        body.append(
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round"/>'
        )
        for x, y in pts:
            body.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4" fill="{colour}"/>'
            )
        lx, ly = pts[-1]
        body.append(
            f'<text x="{sx(lx) + 4:.1f}" y="{sy(ly) + 3:.1f}" '
            f'font-family="ui-monospace,monospace" font-size="9" '
            f'fill="{colour}">{name}</text>'
        )
    if x_label:
        body.append(
            f'<text x="{ml + pw / 2:.0f}" y="{mt + ph + 26:.0f}" '
            f'text-anchor="middle" font-family="ui-monospace,monospace" '
            f'font-size="9" fill="{_FAINT}">{x_label}</text>'
        )
    if y_label:
        body.append(
            f'<text x="12" y="{mt + ph / 2:.0f}" text-anchor="middle" '
            f'font-family="ui-monospace,monospace" font-size="9" fill="{_FAINT}" '
            f'transform="rotate(-90 12 {mt + ph / 2:.0f})">{y_label}</text>'
        )
    if title:
        body.insert(
            0,
            f'<text x="{ml}" y="-4" font-family="Barlow Semi Condensed,sans-serif" '
            f'font-weight="600" font-size="13" fill="{_INK}">{title}</text>',
        )
    return _svg(ml + pw + mr, mt + ph + 34, body, "line chart",
                pad=14 if title else 4)


# ------------------------------------------------------------- panel_svg

def panel_svg(items: list[str], cols: int = 2, gap: int = 16) -> str:
    """Tile SVG strings into a grid, each centred in a common cell."""
    boxes = [_viewbox(s) for s in items]
    cw = max(b[2] for b in boxes)
    ch = max(b[3] for b in boxes)
    rows = math.ceil(len(items) / cols)
    body = []
    for i, (svg, (mx, my, vw, vh)) in enumerate(zip(items, boxes)):
        r, c = divmod(i, cols)
        ox = c * (cw + gap) + (cw - vw) / 2 - mx
        oy = r * (ch + gap) + (ch - vh) / 2 - my
        inner = svg.split(">", 1)[1].rsplit("<", 1)[0]
        body.append(f'<g transform="translate({ox:.1f} {oy:.1f})">{inner}</g>')
    return _svg(
        cols * cw + (cols - 1) * gap, rows * ch + (rows - 1) * gap, body, "panel"
    )


def _viewbox(svg: str) -> tuple[float, float, float, float]:
    vb = svg.split('viewBox="', 1)[1].split('"', 1)[0].split()
    return float(vb[0]), float(vb[1]), float(vb[2]), float(vb[3])
