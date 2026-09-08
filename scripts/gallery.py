"""Redraw the visualization gallery: policy fans, mixing maps, value
landscapes, and one stage game per matching-pennies template.

    python scripts/gallery.py                 # -> docs/figures/gallery/*.svg + docs/gallery.html
    python scripts/gallery.py --width 5       # smaller, faster board

Every figure is inline SVG themed with the report's CSS variables, so the
same files render standalone and embedded in docs/gallery.html / report.html.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.symmetry import mirror_state
from soccer_nash.templates import mixed_state_templates
from soccer_nash.viz import (
    mixing_map_svg,
    panel_svg,
    policy_svg,
    strategy_bars_svg,
    value_map_svg,
)

OUT = pathlib.Path("docs/figures/gallery")


def _solve(game: SoccerGame, gamma: float = 0.9):
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    return solver, solver.run()


def _is_mixed(p: np.ndarray) -> bool:
    return int((np.asarray(p) > 0.02).sum()) > 1


def _busiest_defender(states, no_saddle, defender_carries=1):
    """Defender cell under which the most carrier positions stay mixed."""
    counts: dict[tuple[int, int], int] = {}
    for s in no_saddle:
        cell = s[2:4] if defender_carries == 1 else s[0:2]
        counts[cell] = counts.get(cell, 0) + 1
    return max(counts, key=counts.get)


def build(width: int, height: int) -> list[tuple[str, str, str]]:
    """Return (slug, caption, svg) triples."""
    goal = tuple(range(1, height - 1)) or (height // 2,)
    rand = SoccerGame(width=width, height=height, goal_rows=goal, move_order="random")
    det = SoccerGame(
        width=width, height=height, goal_rows=goal, move_order="deterministic"
    )
    rsolver, rres = _solve(rand)
    dsolver, dres = _solve(det)
    rmat = lambda s: rsolver._matrix(s, rres.values)  # noqa: E731

    mixed = sorted(rres.no_saddle_states)
    figs: list[tuple[str, str, str]] = []

    # 1. the same state: mixed under random resolution, pure under deterministic
    hero = next(
        s for s in mixed
        if _is_mixed(rres.row_policy[s]) or _is_mixed(rres.col_policy[s])
    )
    figs.append((
        "resolution_contrast",
        "The same state under two move-resolution rules. Random order (Littman) "
        "makes the stage game matching pennies; deterministic carrier-wins "
        "collapses it to a pure best reply.",
        panel_svg([
            policy_svg(rand, hero, rres.row_policy, rres.col_policy,
                       rres.values[hero], title="random order -> mixed"),
            policy_svg(det, hero, dres.row_policy, dres.col_policy,
                       dres.values[hero], title="deterministic -> pure"),
        ], cols=2),
    ))

    # 2. strategy bars for the hero state
    figs.append((
        "hero_strategy",
        f"Equilibrium mixed strategy at state {hero}: each player's weight over "
        "{U, D, L, R}. The support is 2x2 with crossing best replies.",
        strategy_bars_svg(rres.row_policy[hero], rres.col_policy[hero],
                          rres.values[hero]),
    ))

    # 3. mixing maps: three defender anchor cells
    busy = _busiest_defender(rsolver._states, mixed)
    anchors = [busy, (width // 2, height // 2), (max(width - 2, 0), height // 2)]
    seen: set[tuple[int, int]] = set()
    anchors = [a for a in anchors if not (a in seen or seen.add(a))]
    figs.append((
        "mixing_maps",
        "Where the carrier has to guess. Defender pinned at the green disc; each "
        "cell shaded by minimax - maximin of the carrier's stage game.",
        panel_svg([
            mixing_map_svg(rand, rmat, a, title=f"defender at {a}") for a in anchors
        ], cols=len(anchors)),
    ))

    # 4. value landscape
    goal_cell = (width - 1, height // 2)
    figs.append((
        "value_landscape",
        "V* of the random-order game. Left: player 0 carries, swept over the "
        "board, player 1 parked at the mouth. Right: player 1's position swept "
        "while player 0 holds near midfield.",
        panel_svg([
            value_map_svg(rand, rres.values, goal_cell, mover=0, ball=0,
                          title="carrier position (player 0)"),
            value_map_svg(rand, rres.values, (1, height // 2), mover=1, ball=0,
                          title="defender position (player 1)"),
        ], cols=2),
    ))

    # 5. one stage game per matching-pennies template
    templates = mixed_state_templates(rand, mixed, rmat)
    mp = [t for t in templates if t.mp is not None and t.mp.is_matching_pennies]
    if mp:
        figs.append((
            "templates",
            f"{len(mp)} geometric templates cover the 2x2 matching-pennies "
            "states. One representative policy fan per template.",
            panel_svg([
                policy_svg(
                    rand, t.representative, rres.row_policy, rres.col_policy,
                    rres.values[t.representative],
                    title=f"{t.count} states  p*={t.mp.p_row:.2f} q*={t.mp.q_col:.2f}",
                )
                for t in mp
            ], cols=2),
        ))

    # 6. mirror pair -- the symmetry the reduction exploits
    twin = mirror_state(hero, width)
    if twin in rres.row_policy:
        figs.append((
            "mirror_pair",
            "The board's left-right / role-swap symmetry: a mixed state and its "
            "mirror image carry the same equilibrium.",
            panel_svg([
                policy_svg(rand, hero, rres.row_policy, rres.col_policy,
                           rres.values[hero], title=f"{hero}"),
                policy_svg(rand, twin, rres.row_policy, rres.col_policy,
                           rres.values[twin], title=f"mirror {twin}"),
            ], cols=2),
        ))

    # 7. Littman's fifth action: same state, four moves vs five (with stand)
    lit4 = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")
    lit5 = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random",
                      n_actions=5)
    l4s, l4r = _solve(lit4)
    l5s, l5r = _solve(lit5)
    shared = sorted(set(l4r.no_saddle_states) & set(l5r.no_saddle_states))
    lit_state = next(
        (s for s in shared
         if s[4] == 1 and l5r.col_policy[s][4] > 0.2
         and abs(s[0] - s[2]) + abs(s[1] - s[3]) == 1),
        shared[0],
    )
    figs.append((
        "littman_stand",
        "Littman's fifth action. The carrier is pinned near its own goal by an "
        "adjacent defender. With four moves the mix is climb / retreat; adding "
        "STAND, the equilibrium becomes climb / hold -- Littman's Figure 2.",
        panel_svg([
            policy_svg(lit4, lit_state, l4r.row_policy, l4r.col_policy,
                       l4r.values[lit_state], title="four moves"),
            policy_svg(lit5, lit_state, l5r.row_policy, l5r.col_policy,
                       l5r.values[lit_state], title="+ stand (Littman Fig 2)"),
        ], cols=2),
    ))

    return figs


_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Soccer Markov game -- visualization gallery</title>
<style>
  :root {{
    --paper:#f7f8f5; --tint:#eef2ec; --ink:#19211c; --ink-faint:#77817a;
    --rule:#d9e2db; --rule-strong:#c2cec5; --p0:#2f6bb0; --p1:#2f8a52;
    --ball:#cf7a20;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --paper:#11150f; --tint:#191e15; --ink:#e7ece2; --ink-faint:#7c867a;
      --rule:#2b342a; --rule-strong:#3b463a; --p0:#6ea8de; --p1:#7cba95;
    }}
  }}
  body {{ background:var(--paper); color:var(--ink); margin:0 auto; max-width:1040px;
    padding:48px 28px 96px; font-family:"Barlow Semi Condensed",system-ui,sans-serif; }}
  h1 {{ font-size:1.7rem; margin:0 0 .3rem; }}
  p.lede {{ color:var(--ink-faint); margin:0 0 2.5rem; max-width:60ch; }}
  figure {{ margin:0 auto 3rem; border-top:1px solid var(--rule); padding-top:1.4rem; }}
  figure svg {{ display:block; width:100%; height:auto; }}
  figcaption {{ color:var(--ink-faint); font-size:.9rem; margin-top:.7rem; max-width:70ch; }}
</style>
<h1>Visualization gallery</h1>
<p class="lede">Policies and values of the soccer Markov game, drawn rather than
tabulated. Regenerate with <code>python scripts/gallery.py</code>.</p>
{body}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--height", type=int, default=5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    figs = build(args.width, args.height)

    blocks = []
    for slug, caption, svg in figs:
        (OUT / f"{slug}.svg").write_text(svg)
        vb_w = float(svg.split('viewBox="', 1)[1].split()[2])
        cap = min(vb_w, 980)
        blocks.append(
            f'<figure style="max-width:{cap:.0f}px">\n{svg}\n'
            f"<figcaption>{caption}</figcaption>\n</figure>"
        )
        print(f"wrote {OUT / f'{slug}.svg'}")

    page = pathlib.Path("docs/gallery.html")
    page.write_text(_PAGE.format(body="\n".join(blocks)))
    print(f"wrote {page}")


if __name__ == "__main__":
    main()
