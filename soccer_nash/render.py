"""Draw a soccer game state as a standalone SVG.

Player 0 is blue, player 1 is green, the ball is amber. The ball carrier gets a
dashed ring in their own colour. ``y`` increases upward in the game, so row
``y = height - 1`` is drawn at the top.

Colours are written as ``var(--pN, #hex)`` so the same string renders themed
when dropped into ``docs/report.html`` and plain when saved as a ``.svg`` file.
"""

from __future__ import annotations

from soccer_nash.game import SoccerGame, State

P0 = "var(--p0, #2f6bb0)"   # player 0 -- blue
P1 = "var(--p1, #2f8a52)"   # player 1 -- green
BALL = "var(--ball, #cf7a20)"
BOARD = "var(--tint, #eef2ec)"
LINE = "var(--rule, #cdd8ce)"
EDGE = "var(--rule-strong, #b7c4b9)"
LABEL = "var(--ink-faint, #77817a)"
PAPER = "var(--paper, #ffffff)"

_PLAYER_COLOUR = (P0, P1)
CELL = 44
MARGIN = 30
# kept as private aliases for the existing call sites in this module
_CELL, _MARGIN = CELL, MARGIN


def cell_center(x: int, y: int, height: int) -> tuple[float, float]:
    """Pixel centre of grid cell ``(x, y)``; ``y`` increases upward on screen."""
    cx = MARGIN + x * CELL + CELL / 2
    cy = MARGIN + (height - 1 - y) * CELL + CELL / 2
    return cx, cy


_cell_center = cell_center


def board_frame(game: SoccerGame) -> list[str]:
    """SVG elements for the empty board: the panel, grid lines, and goal mouths
    (green on the left edge for player 1, blue on the right for player 0)."""
    w, h = game.width, game.height
    board_w, board_h = w * CELL, h * CELL
    out = [
        f'<rect x="{MARGIN}" y="{MARGIN}" width="{board_w}" height="{board_h}" '
        f'fill="{BOARD}" stroke="{EDGE}" stroke-width="1.5"/>'
    ]
    for i in range(1, w):
        gx = MARGIN + i * CELL
        out.append(
            f'<line x1="{gx}" y1="{MARGIN}" x2="{gx}" y2="{MARGIN + board_h}" '
            f'stroke="{LINE}" stroke-width="1"/>'
        )
    for j in range(1, h):
        gy = MARGIN + j * CELL
        out.append(
            f'<line x1="{MARGIN}" y1="{gy}" x2="{MARGIN + board_w}" y2="{gy}" '
            f'stroke="{LINE}" stroke-width="1"/>'
        )
    for row in game.goal_rows:
        gy = MARGIN + (h - 1 - row) * CELL
        out.append(f'<rect x="{MARGIN - 6}" y="{gy}" width="6" height="{CELL}" fill="{P1}"/>')
        out.append(f'<rect x="{MARGIN + board_w}" y="{gy}" width="6" height="{CELL}" fill="{P0}"/>')
    return out


def _player(cx: float, cy: float, label: str, colour: str, carrier: bool) -> str:
    parts = [f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="15" fill="{colour}"/>']
    if carrier:
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="21" fill="none" '
            f'stroke="{colour}" stroke-width="1.5" stroke-dasharray="3 4"/>'
        )
    parts.append(
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" '
        f'font-family="sans-serif" font-weight="700" font-size="14" '
        f'fill="{PAPER}">{label}</text>'
    )
    return "".join(parts)


def board_svg(game: SoccerGame, state: State, caption: str | None = None) -> str:
    """Return a standalone ``<svg>`` string for one non-terminal ``state``."""
    if game.is_terminal(state):
        raise ValueError("cannot render a terminal state")

    w, h = game.width, game.height
    x0, y0, x1, y1, b = state
    board_w = w * _CELL
    board_h = h * _CELL
    total_w = board_w + 2 * _MARGIN
    total_h = board_h + 2 * _MARGIN + (18 if caption else 0)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
        f'role="img" aria-label="{w} by {h} soccer grid, '
        f'player 0 (blue) at {x0},{y0}, player 1 (green) at {x1},{y1}, '
        f'ball carried by player {b}.">',
        *board_frame(game),
    ]

    cx0, cy0 = _cell_center(x0, y0, h)
    cx1, cy1 = _cell_center(x1, y1, h)
    out.append(_player(cx0, cy0, "0", P0, carrier=(b == 0)))
    out.append(_player(cx1, cy1, "1", P1, carrier=(b == 1)))

    # the ball sits just above-right of its carrier
    carrier_cx, carrier_cy = (cx0, cy0) if b == 0 else (cx1, cy1)
    out.append(
        f'<circle cx="{carrier_cx + 13:.1f}" cy="{carrier_cy - 13:.1f}" r="5" '
        f'fill="{BALL}" stroke="{PAPER}" stroke-width="1"/>'
    )

    if caption:
        out.append(
            f'<text x="{_MARGIN}" y="{total_h - 6}" font-family="monospace" '
            f'font-size="11" fill="{LABEL}">{caption}</text>'
        )

    out.append("</svg>")
    return "\n".join(out)


def trajectory_svg(game: SoccerGame, states: list[State], per_row: int = 6) -> str:
    """Tile a trajectory (skipping the terminal state) into a filmstrip SVG."""
    frames = [s for s in states if not game.is_terminal(s)]
    if not frames:
        raise ValueError("no non-terminal states to render")

    tile_w = game.width * _CELL + 2 * _MARGIN
    tile_h = game.height * _CELL + 2 * _MARGIN + 18
    cols = min(per_row, len(frames))
    rows = (len(frames) + per_row - 1) // per_row
    gap = 12

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '
        f'{cols * tile_w + (cols - 1) * gap} {rows * tile_h + (rows - 1) * gap}" '
        f'role="img" aria-label="soccer game trajectory, {len(frames)} steps.">'
    ]
    for i, s in enumerate(frames):
        col, row = i % per_row, i // per_row
        dx = col * (tile_w + gap)
        dy = row * (tile_h + gap)
        inner = board_svg(game, s, caption=f"t={i}  {s}")
        inner = inner.split("\n", 1)[1].rsplit("\n", 1)[0]  # strip <svg>/</svg>
        out.append(f'<g transform="translate({dx} {dy})">{inner}</g>')
    out.append("</svg>")
    return "\n".join(out)
