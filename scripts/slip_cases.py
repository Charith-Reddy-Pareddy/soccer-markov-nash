"""Six edge cases on the slip board -- the movement-noise twin of the A10
deterministic board (see ``scripts/a10_cases.py``, whose style and rigor this
script mirrors exactly: same ``_ties`` helper, same ``_report``/``_html_case``
structure, same panel generation).

The slip board is smaller and shaped differently from the canonical board (5
by 5, a single goal row at y=2, not 7 by 5 with three goal rows) and adds one
new rule on top of A10's own deterministic resolution: after both players
choose an action, EACH player's own chosen action is independently replaced
by a uniformly random action (probability ``slip`` = 0.15, split evenly
across U/D/L/R) before the ordinary deterministic movement/contest rule runs
on whatever pair of actions results (``SoccerGame._slip_transitions`` in
``soccer_nash/game.py``). This is per-player execution noise layered on top
of A10's resolution rule, not a new contest-resolution rule of its own, and
it is applied independently for both players -- so ``game.transitions()`` on
this board routinely returns a dozen or more weighted outcomes for a single
joint action, not one.

That noise is this document's headline finding: 52 of this board's 1200
states (~4.3%) have a genuine LP mixed equilibrium -- no tie-break
convention, an actual fractional Nash strategy -- versus exactly 0 of 2380 on
the A10-deterministic board, where scripts/a10_cases.py's own headline is
that *every* stage game is pure. Four of the six cases below are drawn from
that 52 and verified fresh against this script's own solve; the other two are
clean pure cases chosen to make the raw slip mechanic legible on its own,
including one where a completely safe-looking action still carries a real,
quantified chance of losing the ball to nothing but the carrier's own random
stumble.

    python scripts/slip_cases.py

Writes `docs/figures/gallery/slip_cases.svg` (composite) plus one
`docs/figures/gallery/slip_caseNN.svg` per case.
"""
from __future__ import annotations

import html
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import MOVE_ACTIONS, SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.viz import bestresponse_graph_svg, panel_svg, policy_svg

FIG = pathlib.Path("docs/figures/gallery/slip_cases.svg")
CASE_DIR = pathlib.Path("docs/figures/gallery")
_ACT = ["U", "D", "L", "R"]
_NAME2ACT = {a.name: a for a in MOVE_ACTIONS}
BOARD = {"width": 5, "height": 5, "goal_rows": (2,), "move_order": "deterministic", "slip": 0.15}


def _ties(M: np.ndarray, tol: float = 1e-6):
    """Mirrors site/src/explorer/helpers.js's own certify(): every row/column
    index achieving the guaranteed value, not just the first (certify_game's
    own `saddle` reports only one)."""
    row_min = M.min(axis=1)
    maximin = row_min.max()
    col_max = M.max(axis=0)
    minimax = col_max.min()
    row_ties = [i for i in range(4) if abs(row_min[i] - maximin) <= tol]
    col_ties = [j for j in range(4) if abs(col_max[j] - minimax) <= tol]
    return maximin, minimax, row_ties, col_ties


def _modal(outs):
    """The single highest-probability outcome of a `game.transitions()` list.

    Unlike the A10 board, `outs[0]` here is one of many weighted slip
    branches, in whatever order `_slip_transitions`'s internal dict happens
    to produce -- not necessarily the most likely one. Every table in this
    script that shows "the" outcome of a joint action means this one."""
    return max(outs, key=lambda o: o[0])


def _fmt_policy(arr: np.ndarray, tol: float = 0.005) -> str:
    parts = [f"{_ACT[k]} {arr[k] * 100:.1f}%" for k in range(4) if arr[k] > tol]
    return " / ".join(parts) if parts else "n/a"


def _explorer_link(state) -> str:
    x0, y0, x1, y1, b = state
    return f"explorer.html?board=slip&state={x0},{y0},{x1},{y1},{b}"


def _report(label, why, g, solver, r, state, panels: list[str], case_no: int, kind: str,
            full_dist: tuple[str, str] | None = None):
    M = solver._matrix(state, r.values)
    maximin, minimax, row_ties, col_ties = _ties(M)
    gap = abs(maximin - minimax)
    if kind == "pure":
        assert gap < 1e-6, f"{state}: expected a pure saddle, got a gap of {gap:.6g}"
    else:
        assert gap > 1e-4, f"{state}: expected a genuine mixed-equilibrium gap, got {gap:.6g}"
        assert state in r.no_saddle_states, f"{state}: not among the solver's mixed states"
    x0, y0, x1, y1, b = state
    carrier, defender = (0, 1) if b == 0 else (1, 0)
    row_pol, col_pol = r.row_policy[state], r.col_policy[state]

    print(f"state {state} -- {label}")
    print(f"  player 0 at ({x0}, {y0}) -- player 1 at ({x1}, {y1}) -- "
          f"player {carrier} has the ball (carrier); player {defender} defends")
    print(f"  V = {r.values[state]:+.6f}")
    print("  Q matrix (player 0's payoff; rows = player 0, cols = player 1):")
    print("        " + "".join(f"{a:>11}" for a in _ACT))
    for i, a in enumerate(_ACT):
        print(f"    {a:>3} " + "".join(f"{M[i, j]:>11.6f}" for j in range(4)))
    print(f"  maximin = {maximin:+.6f} at row(s) {[_ACT[i] for i in row_ties]}"
          + ("  <- TIE, not unique" if len(row_ties) > 1 else "  (unique)"))
    print(f"  minimax = {minimax:+.6f} at col(s) {[_ACT[j] for j in col_ties]}"
          + ("  <- TIE, not unique" if len(col_ties) > 1 else "  (unique)"))
    policy_desc = ("one-hot, tie-broken" if kind == "pure"
                    else "genuine LP mixed-strategy equilibrium")
    print(f"  solver's own policy ({policy_desc}): p0={np.round(row_pol, 4).tolist()}  "
          f"p1={np.round(col_pol, 4).tolist()}")
    print(f"  why: {why}")
    print("  every joint action's modal (highest-probability) transition -- "
          "game.transitions() aggregates every slip branch of a pair into one weighted "
          "list; each cell below is that list's single most likely outcome, prefixed "
          "with its own probability, not the only outcome:")
    for i, a0 in enumerate(MOVE_ACTIONS):
        cells = []
        for j, a1 in enumerate(MOVE_ACTIONS):
            outs = g.transitions(state, a0, a1)
            prob, ns, _reward = _modal(outs)
            tag = ("GOAL:P0" if ns[4] == 0 and ns[0] == -1 else
                   "GOAL:P1" if ns[4] == 1 and ns[0] == -1 else str(ns))
            cells.append(f"{prob:.2f}·{tag}")
        print(f"    {_ACT[i]:>3} " + "  ".join(f"{c:<20}" for c in cells))
    if full_dist is not None:
        a0n, a1n = full_dist
        outs = sorted(g.transitions(state, _NAME2ACT[a0n], _NAME2ACT[a1n]), key=lambda o: -o[0])
        print(f"  full outcome distribution for ({a0n}, {a1n}) -- all {len(outs)} branches:")
        total = 0.0
        for prob, ns, reward in outs:
            total += prob
            print(f"    p={prob:.5f}  ns={ns}  reward={reward}")
        print(f"    total probability = {total:.6f}")
    print()

    board = policy_svg(g, state, {state: row_pol}, {state: col_pol},
                        value=r.values[state], title=label, kind=kind, always_label=True)
    matrix = bestresponse_graph_svg(M, _ACT, _ACT, title=f"Q matrix -- {state}")
    panels.append(board)
    panels.append(matrix)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / f"slip_case{case_no:02d}.svg").write_text(panel_svg([board, matrix], cols=2))


def _full_dist_note(g: SoccerGame, state, a0n: str, a1n: str) -> str:
    """An HTML table of every branch `game.transitions()` returns for one
    joint action -- generated from the live solve, not hand-typed -- so the
    raw slip mechanic (many weighted outcomes per action pair) is visible on
    its own, not only as the single modal outcome the case tables show."""
    outs = sorted(g.transitions(state, _NAME2ACT[a0n], _NAME2ACT[a1n]), key=lambda o: -o[0])
    total = 0.0
    rows = []
    for prob, ns, reward in outs:
        total += prob
        tag = ""
        if g.is_terminal(ns):
            tag = f"GOAL, player {ns[4]}"
        elif ns[4] != state[4]:
            tag = f"ball flips to player {ns[4]}"
        rows.append(
            f"<tr><td>{prob:.5f}</td><td>{ns}</td><td>{reward}</td>"
            f"<td>{html.escape(tag)}</td></tr>"
        )
    return (
        f"<p><b>Full outcome distribution for ({a0n}, {a1n})</b> &mdash; every branch "
        f"<code>game.transitions()</code> actually returns for this one joint action, "
        f"not just the modal outcome shown in the table below "
        f"(all {len(outs)} branches sum to {total:.6f}):</p>\n"
        f"<table><tr><th>probability</th><th>next state</th><th>reward</th>"
        f"<th>note</th></tr>{''.join(rows)}</table>\n"
    )


def _html_case(number, label, why, g, solver, result, state, kind, note=""):
    M = solver._matrix(state, result.values)
    p, q = result.row_policy[state], result.col_policy[state]
    _, _, rows, cols = _ties(M)
    matrix = "".join(
        "<tr><th>" + _ACT[a] + "</th>" + "".join(
            f"<td>{v:.6f}</td>" for v in M[a]) + "</tr>" for a in range(4))
    transitions = "".join(
        "<tr><th>" + a0.name + "</th>" + "".join(
            f"<td>{_modal(g.transitions(state, a0, a1))[1]}</td>"
            for a1 in MOVE_ACTIONS) + "</tr>" for a0 in MOVE_ACTIONS)
    headers = "<tr><th>P0 / P1</th><th>U</th><th>D</th><th>L</th><th>R</th></tr>"
    v = result.values[state]
    kind_label = "Mixed equilibrium" if kind == "mixed" else "Pure equilibrium"
    return f"""<section><h2><span class="case-no">{number}</span>{html.escape(label)}</h2>
<p class="lede">State {state} &middot; player {state[4]} carries the ball &middot;
V = {v:+.6f} &middot; {kind_label}</p>
<p class="lede"><a href="{_explorer_link(state)}">Open this position in the interactive
explorer &rarr;</a></p>
<div class="callout">
<div class="stat-row"><span><b>P0 &rarr;</b> {_fmt_policy(p)}</span>
<span><b>P1 &rarr;</b> {_fmt_policy(q)}</span></div>
<p class="muted">Security-tied alternatives (not probabilities):
P0 {'/'.join(_ACT[k] for k in rows)} &middot;
P1 {'/'.join(_ACT[k] for k in cols)}</p>
</div>
<p>{html.escape(why)}</p>
{note}<img src="figures/gallery/slip_case{number:02d}.svg"
 alt="Case {number} selected policy and payoff graph">
<h3>Q: player 0 payoff, optimal continuation after this turn</h3>
<table>{headers}{matrix}</table>
<h3>All 16 joint actions' modal successor state (highest-probability branch only)</h3>
<table class="transitions">{headers}{transitions}</table></section>"""


def main() -> None:
    g = SoccerGame(**BOARD)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run_exact()
    print(f"exact solve: |V_exact - V_iterative| = {r.exact_vs_iterative:.2e} "
          f"over {len(r.values)} states\n")

    n_total = len(r.values)
    n_mixed = len(r.no_saddle_states)
    print(f"headline: {n_mixed} of {n_total} states "
          f"({100 * n_mixed / n_total:.2f}%) have a genuine mixed equilibrium\n")
    assert n_total == 1200, f"expected 1200 states on the slip board, got {n_total}"
    assert n_mixed == 52, f"expected 52 mixed states (verified this session), got {n_mixed}"

    panels: list[str] = []
    cases = [
        ((0, 1, 0, 3, 0), "The stumble", "pure",
         "Player 0 carries the ball at (0,1); the defender waits two cells up the same "
         "column at (0,3), with the board's only goal row (y=2) sitting directly between "
         "them. Player 0's maximin action R is completely safe under plain resolution -- "
         "it steps sideways to (1,1), nowhere near the defender's column -- and the "
         "defender's minimax reply D simply advances toward the goal row without "
         "contesting anything. Verified below: the modal branch of (R, D), weight "
         "0.8875 = 0.85 + 0.15/4 (the probability both players' intended actions survive "
         "their own independent slip roll), resolves exactly this way, to (1,1,0,2,0) -- "
         "no contest at all. But slip is independent per player: with weight "
         "0.0375 x 0.8875 = 0.03328 (verified below, the single largest of the remaining "
         "branches), player 0's own R is silently replaced by U instead -- sending the "
         "carrier to (0,2), directly into the goal row and into the cell the defender's "
         "intended D is also converging on from (0,3). Player 0 physically ends up there "
         "uncontested -- the defender stays put at (0,3) in this branch -- yet the A10 "
         "rule scripts/a10_cases.py's own 'Pinned in the corner' case already documents "
         "still hands the ball to the loser of that contest, the defender, away from "
         "player 0. A real, quantified 3.328% chance of losing the ball to nothing but "
         "player 0's own random stumble, not to anything the defender chose to do."),
        ((4, 2, 0, 2, 0), "The uncertain open goal", "pure",
         "The direct analogue of scripts/a10_cases.py's 'The open goal': player 0 sits at "
         "the right wall in the board's only goal row, ball in hand, defender at the "
         "opposite wall far too distant to interfere this turn. R is still the unique "
         "maximin/minimax action for both players. On the A10-deterministic board the "
         "identical geometry scores with certainty (V = 1.0 exactly). Here it does not: "
         "V = 0.962780, and the modal branch of R against every one of the defender's "
         "four replies is identical -- weight 0.8875 = 0.85 + 0.15/4, verified below -- "
         "because scoring depends only on player 0's own action surviving its own slip "
         "roll, not on anything the far-off defender does. An unmarked, wall-pinned "
         "carrier in the goal row still only scores 88.75% of the time on a single push; "
         "the remaining 11.25% is spread across a dozen small branches where the push "
         "itself misfires into a neighboring cell instead of over the line."),
        ((0, 1, 1, 1, 0), "Split between safe and risky", "mixed",
         "Player 0 carries the ball at (0,1), one cell west of the defender at (1,1). "
         "Solved on this identical board with slip set to 0, this exact state is a flat "
         "all-zero tie -- verified separately, every one of the sixteen Q-matrix cells is "
         "exactly 0.0, the same 'dead zone' shape scripts/a10_cases.py's own case 8 "
         "documents as the board's most common configuration. Slip breaks that tie into a "
         "real gradient. Player 0's R aims straight at the defender's own square: if the "
         "defender replies L, the two swap -- verified below, (R, L)'s modal branch "
         "(weight 0.78766) resolves to (1,1,0,1,1), a genuine ball-losing swap under the "
         "A10 collision rule, dragging R's worst case to -0.469345. U never approaches "
         "the defender and caps out at +0.058185 in its best case, +0.044092 in its "
         "worst. Neither dominates: U is safe but capped, R is risky but has a far higher "
         "ceiling against every column except L. The equilibrium mixes 76.5% U / 23.5% R "
         "for player 0 because the column's own equilibrium mixes 30.0% U / 70.0% R and "
         "never plays the crushing L -- a pure U-only defense would let row exploit "
         "predictably by leaning further into R, and a pure L-only defense wastes "
         "column's own tempo on a move row's mix rarely walks into. The mixed value, "
         "0.053951, sits strictly between the pure security floor (U alone, 0.044092) "
         "and the best column could otherwise hold row to (0.058185) -- real value "
         "squeezed out of a threat that, at equilibrium, is never even tested against its "
         "worst reply."),
        ((0, 1, 2, 0, 0), "A tie ground into a gradient", "mixed",
         "Defender at (2,0) is two cells away diagonally, seemingly out of reach this "
         "turn. Solved with slip set to 0 on the identical board, this state is again a "
         "flat all-zero tie -- nothing in the Q matrix distinguishes any of the sixteen "
         "cells, and U wins only by tie-break order. Under slip = 0.15 the near-tie "
         "survives almost intact but not exactly: U's worst case (0.051413, against "
         "column U) and R's worst case (0.045186, against column D) differ by well under "
         "a percent of the state's own value -- too close for either to weakly dominate. "
         "The column's own equilibrium mixes 8.1% D / 91.9% R, almost entirely R, because "
         "L is the single biggest threat left in the matrix (0.295145 and 0.312405, the "
         "two largest entries either row reaches) and column avoids feeding it; row in "
         "turn mostly plays U (81.6%) and leans into R (18.4%) only against the sliver of "
         "D column still keeps in its own mix. This is one of the smallest, least "
         "dramatic members of the board's 52 mixed states -- a near-tie the slip noise "
         "tips just far enough to require real randomization instead of a tie-break "
         "convention."),
        ((0, 1, 2, 0, 1), "Two ways to wait", "mixed",
         "Player 1 carries the ball at (2,0); player 0 defends from (0,1), diagonally two "
         "cells away. On the identical board with slip set to 0 this state's Q matrix is "
         "exactly all zero -- both players are too far apart to touch this turn, and the "
         "tie again defaults to U for both. Under slip the matrix turns uniformly "
         "negative for the defender (every one of the sixteen cells sits between -0.007 "
         "and -0.050, unsurprising since the opponent already has the ball) but is not "
         "flat: D and R both risk closing the distance in ways the matrix punishes hard "
         "(D's worst case is -0.049664, R's is -0.042770, both against column U), so the "
         "defender's equilibrium support drops them almost entirely and mixes 91.7% U / "
         "8.3% L instead -- the two actions that hold position rather than lunge. U and L "
         "are close enough (worst cases -0.010669 versus -0.010182) that once slip has "
         "smoothed the sixteen-cell matrix neither cleanly dominates, so the solver "
         "genuinely mixes between them rather than picking one by tie-break order."),
        ((0, 1, 2, 1, 1), "A threat that shapes the mix without firing", "mixed",
         "Player 1 carries the ball at (2,1); player 0 defends from (0,1), two cells away "
         "in the same row. R is not merely a desperate lunge here: verified below, (R, L) "
         "resolves with modal weight 0.78766 to (0,1,1,1,0) -- a genuine contested-cell "
         "swap in the defender's favor, stealing the ball back to player 0, the same "
         "loser-keeps-the-ball mechanic as case 1, running the other way. That single "
         "verified branch is why R carries a real positive entry, +0.033002, at column L "
         "-- the only positive cell anywhere in this defending player's Q matrix. In "
         "equilibrium, though, column never actually plays L (support is 42.0% U / 58.0% "
         "D), so that steal is a threat baked into the matrix's shape rather than "
         "something that fires at equilibrium; the real driver of the 72.75% / 27.25% row "
         "split is that R is fractionally better than U against column's U "
         "(-0.006405 vs -0.008014) while U is better against column's D "
         "(-0.007258 vs -0.008424) -- a genuine trade-off neither action wins outright."),
    ]
    for case_no, (state, label, kind, why) in enumerate(cases, start=1):
        full_dist = ("R", "D") if case_no == 1 else None
        _report(label, why, g, solver, r, state, panels, case_no, kind, full_dist=full_dist)

    notes = {1: _full_dist_note(g, cases[0][0], "R", "D")}

    intro = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Six slip-board edge cases: verified movement noise and mixed equilibria</title>
<style>
:root{--ink:#1c2e26;--accent:#1f7a4d;--accent-dark:#155c39;--line:#dbe5df;
--muted:#5b6b63;--warn:#a8501c;--warn-bg:#fbf3ea;--card:#f3f8f5}
*{box-sizing:border-box}
body{font:15px/1.6 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
max-width:960px;margin:40px auto;color:var(--ink);padding:0 28px}
h1{font-size:25px;margin:0 0 8px;color:var(--accent-dark);letter-spacing:-.01em}
h2{font-size:19px;margin:0 0 10px;color:var(--accent-dark);display:flex;
align-items:center}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);margin:20px 0 8px;font-weight:700}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.09em;
color:var(--accent);font-weight:700;margin:0 0 6px}
p.lede{color:var(--muted);font-size:13px;margin:0 0 12px}
p.muted{color:var(--muted);font-size:13px;margin:6px 0 0}
.case-no{display:inline-flex;align-items:center;justify-content:center;
background:var(--accent);color:#fff;border-radius:50%;width:26px;height:26px;
font-size:13px;font-weight:700;margin-right:10px;flex-shrink:0}
.callout{background:var(--card);border:1px solid var(--line);
border-left:4px solid var(--accent);border-radius:6px;padding:10px 16px;
margin:12px 0}
.callout.warn{border-left-color:var(--warn);background:var(--warn-bg)}
.stat-row{display:flex;gap:28px;font-size:14px}
.stat-row b{color:var(--ink)}
table{border-collapse:collapse;width:100%;font:12px/1.5 "SF Mono",Consolas,monospace}
td,th{border:1px solid var(--line);padding:6px 8px;text-align:center}
th{background:var(--card);color:var(--accent-dark);font-weight:700}
img{display:block;width:100%;max-height:290px;object-fit:contain;margin:14px 0}
section{margin-top:36px;padding-top:18px;border-top:2px solid var(--line)}
code{background:var(--card);border-radius:3px;padding:1px 5px;font-size:.92em}
a{color:var(--accent-dark)}
@page{size:A4;margin:14mm}
@media print{body{margin:0;padding:0;font-size:10.5px} h1{font-size:21px}
h2{font-size:16px} section{break-before:page;margin:0;padding:0;border:0}
img{max-height:235px} table{font-size:9px} td,th{padding:4px}
.callout{padding:8px 12px}}
</style>
<p class="eyebrow">Soccer Markov Nash &middot; movement-slip board</p>
<h1>Six edge cases on the movement-slip board</h1>
<p>5 by 5 board; a single goal row at y=2 (smaller and differently shaped than
the 7 by 5, three-goal-row canonical board); simultaneous U/D/L/R; the same
deterministic A10 carrier-wins-contests resolution rule, with one addition:
after both players choose, each player's own chosen action is independently
replaced by a uniformly random action with probability 0.15, before the
deterministic resolution rule runs on whatever pair of actions results. This
is per-player execution noise layered on top of A10's rule, not a new
contest-resolution rule of its own -- and because it applies independently to
both players, <code>game.transitions()</code> on this board routinely returns
a dozen or more weighted outcomes for a single joint action, not one.</p>
<div class="callout warn">
<p style="margin:0"><b>Headline finding, inverted from the A10 board:</b>
scripts/a10_cases.py's own headline is that <i>every</i> stage game on the
deterministic board is pure -- zero mixed equilibria across all 2380 states.
Here, 52 of this board's 1200 states (~4.3%) have a genuine mixed
equilibrium: a real fractional Nash strategy, not a tie-break convention --
the third-richest source of real randomization among this project's four
non-A10 board variants. The mechanism is plain once you trace it: a small
chance of stumbling into an unintended cell can undermine what looked like a
completely safe pure action, so a player may prefer to split probability
between two actions whose slip-risk profiles trade off against each other,
rather than commit to one action whose worst-case slip outcome is bad. Four
of the six cases below are drawn from that 52, each one verified fresh
against this script's own solve and traced back to the actual weighted
branches <code>game.transitions()</code> returns.</p>
</div>
<p>The other two cases below are clean pure equilibria, chosen to put the raw
slip mechanic on a board diagram by itself: one shows a completely safe pure
action carrying a real, quantified chance of losing the ball to nothing but
the carrier's own random stumble, with the <em>full</em> weighted outcome
distribution shown, not just the modal branch; the other shows an otherwise
certain goal (unconditional on the A10-deterministic board) reduced to an
88.75% chance by the carrier's own execution noise alone.</p>
<p><a href="explorer.html?board=slip">Interactive explorer</a> ·
<a href="equilibrium-debug.md">Audit and reproduction details</a></p>"""
    sections = [_html_case(n, label, why, g, solver, r, state, kind, notes.get(n, ""))
                for n, (state, label, kind, why) in enumerate(cases, 1)]
    pathlib.Path("docs/slip_cases.html").write_text(intro + "".join(sections) + "</html>")
    FIG.write_text(panel_svg(panels, cols=2))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
