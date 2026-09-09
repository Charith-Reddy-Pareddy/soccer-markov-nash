"""The one-figure summary: the mechanism chain and the algorithmic triangle.

    python scripts/story.py     # -> docs/figures/gallery/story.svg
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

FIG = pathlib.Path("docs/figures/gallery/story.svg")

_INK = "var(--ink, #19211c)"
_FAINT = "var(--ink-faint, #77817a)"
_TINT = "var(--tint, #eef2ec)"
_RULE = "var(--rule-strong, #b7c4b9)"
_P0 = "var(--p0, #2f6bb0)"
_EMBER = "var(--ember, #a94e18)"


def _box(x, y, w, h, text, fill=_TINT, stroke=_RULE, tcol=_INK, fs=11):
    lines = text.split("\n")
    dy = -(len(lines) - 1) * (fs + 2) / 2
    ts = "".join(
        f'<tspan x="{x + w / 2:.0f}" dy="{fs + 2 if i else dy:.0f}">{ln}</tspan>'
        for i, ln in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.2"/>'
        f'<text x="{x + w / 2:.0f}" y="{y + h / 2 + 4:.0f}" text-anchor="middle" '
        f'font-family="Barlow Semi Condensed,sans-serif" font-size="{fs}" '
        f'fill="{tcol}">{ts}</text>'
    )


def _arrow(x1, y1, x2, y2, label="", colour=_RULE):
    mid = f'<text x="{(x1 + x2) / 2 + 6:.0f}" y="{(y1 + y2) / 2 + 3:.0f}" ' \
          f'font-family="ui-monospace,monospace" font-size="9" ' \
          f'fill="{_FAINT}">{label}</text>' if label else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
        f'stroke-width="1.4" marker-end="url(#sa)"/>{mid}'
    )


def build() -> str:
    W, H = 720, 470
    b = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'role="img" aria-label="summary: mechanism chain and algorithmic triangle">',
        '<defs><marker id="sa" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" '
        f'fill="{_RULE}"/></marker></defs>',
        f'<text x="16" y="22" font-family="Barlow Semi Condensed,sans-serif" '
        f'font-weight="700" font-size="15" fill="{_INK}">Where mixing comes from, '
        f'and what the solver does about it</text>',
    ]

    # --- mechanism chain (left column)
    x = 40
    b.append(_box(x, 44, 200, 34, "goal-mouth width"))
    b.append(_arrow(x + 55, 78, x + 30, 104, "= 1"))
    b.append(_arrow(x + 145, 78, x + 170, 104, "&#8805; 2"))
    b.append(_box(x - 8, 104, 96, 34, "PURE\nonly", tcol=_P0))
    b.append(_box(x + 104, 104, 104, 34, "candidate\nmixed states", tcol=_EMBER))
    b.append(_arrow(x + 156, 138, x + 156, 162))
    b.append(_box(x + 40, 162, 168, 34,
                  "+ stochastic move order\n+ defender can't cover both lanes"))
    b.append(_arrow(x + 124, 196, x + 124, 220))
    b.append(_box(x + 40, 220, 168, 32, "crossing best replies"))
    b.append(_arrow(x + 124, 252, x + 124, 276))
    b.append(_box(x + 40, 276, 168, 32, "2x2 matching pennies", tcol=_EMBER))
    b.append(_arrow(x + 124, 308, x + 124, 332))
    b.append(_box(x + 40, 332, 168, 34, "mixed equilibrium\n(LP needed here only)"))

    # --- algorithmic triangle (right)
    tx, ty = 460, 120
    b.append(f'<text x="{tx + 90}" y="{ty - 34}" text-anchor="middle" '
             f'font-family="Barlow Semi Condensed,sans-serif" font-weight="600" '
             f'font-size="12" fill="{_INK}">three backups</text>')
    b.append(f'<path d="M{tx + 90} {ty - 20} L{tx} {ty + 110} L{tx + 180} {ty + 110} Z" '
             f'fill="none" stroke="{_RULE}" stroke-width="1.2"/>')
    b.append(_box(tx + 40, ty - 38, 100, 30, "PURE", fill="none", stroke="none",
                  tcol=_P0, fs=12))
    b.append(f'<text x="{tx + 90}" y="{ty - 2}" text-anchor="middle" '
             f'font-family="ui-monospace,monospace" font-size="8.5" '
             f'fill="{_FAINT}">fast, sometimes short</text>')
    b.append(_box(tx - 46, ty + 112, 92, 26, "HYBRID", fill="none", stroke="none",
                  tcol=_INK, fs=12))
    b.append(f'<text x="{tx}" y="{ty + 150}" text-anchor="middle" '
             f'font-family="ui-monospace,monospace" font-size="8.5" '
             f'fill="{_FAINT}">exact, selective LP</text>')
    b.append(_box(tx + 134, ty + 112, 92, 26, "MIXED", fill="none", stroke="none",
                  tcol=_EMBER, fs=12))
    b.append(f'<text x="{tx + 180}" y="{ty + 150}" text-anchor="middle" '
             f'font-family="ui-monospace,monospace" font-size="8.5" '
             f'fill="{_FAINT}">exact, all LP</text>')

    # --- the experimental result beside it
    b.append(_box(tx - 40, ty + 186, 240, 28,
                  "deterministic:  pure = hybrid = mixed", fs=10))
    b.append(_box(tx - 40, ty + 218, 240, 28,
                  "random order:  pure &lt; hybrid = mixed", fs=10))

    # --- the occupancy callout
    b.append(f'<rect x="{tx - 40}" y="{ty + 256}" width="240" height="46" rx="4" '
             f'fill="{_TINT}" stroke="{_EMBER}" stroke-width="1.2"/>')
    b.append(f'<text x="{tx + 80}" y="{ty + 277}" text-anchor="middle" '
             f'font-family="Barlow Semi Condensed,sans-serif" font-weight="700" '
             f'font-size="12.5" fill="{_EMBER}">3.95% of states &#8594; 41% of the '
             f'equilibrium path</text>')
    b.append(f'<text x="{tx + 80}" y="{ty + 293}" text-anchor="middle" '
             f'font-family="ui-monospace,monospace" font-size="8.5" '
             f'fill="{_FAINT}">rare to store, common to use</text>')

    b.append("</svg>")
    return "\n".join(b)


def main() -> None:
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(build())
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
