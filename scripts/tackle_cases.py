"""Seven edge cases on the ``tackle`` board -- this project's own collision
rule, not Littman's and not the A10 deterministic rule.

``scripts/a10_cases.py`` documents the A10-deterministic board, whose
headline result is that *every* stage game there is pure (0 of 2380 states).
This script's board is the opposite extreme among the project's four
non-A10 variants: 56 of 760 states (about 7.4%) have a genuine mixed
equilibrium -- by far the richest source of real randomization outside
``docs/positions.pdf``'s Littman canonical-random board. The reason is the
rule itself: a *challenge* (the defender's move landing on the carrier's
current cell) is a 50/50 duel, decided by ``tackle_prob``, and neither
"always challenge" nor "never challenge" is a safe pure answer to a fixed
coin -- see the intro callout below and ``docs/tackle.md``.

Every number printed or drawn here is read from a fresh exact solve of
``BOARD`` and from ``game.transitions()`` directly, the same way
``scripts/a10_cases.py`` does it -- nothing is asserted without a matching
transition-engine lookup.

    python scripts/tackle_cases.py

Writes `docs/figures/gallery/tackle_cases.svg` (composite) plus one
`docs/figures/gallery/tackle_caseNN.svg` per case.
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

FIG = pathlib.Path("docs/figures/gallery/tackle_cases.svg")
CASE_DIR = pathlib.Path("docs/figures/gallery")
_ACT = ["U", "D", "L", "R"]
BOARD = {
    "width": 5, "height": 4, "goal_rows": (1, 2),
    "move_order": "tackle", "tackle_prob": 0.5,
}


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


def _outcome_str(g: SoccerGame, state, a0, a1) -> str:
    """Every real outcome of one joint action, exactly as ``game.transitions()``
    returns it -- a single state for a non-challenge turn, or the duel's two
    weighted branches (win-the-ball / lose-the-ball) for a challenge."""
    outs = g.transitions(state, a0, a1)

    def fmt(ns):
        if ns[0] == -1:
            return "GOAL:P0" if ns[4] == 0 else "GOAL:P1"
        return str(ns)

    if len(outs) == 1:
        return fmt(outs[0][1])
    return " / ".join(f"{fmt(ns)} @{p:.0%}" for p, ns, _reward in outs)


def _policy_desc(policy: np.ndarray, tol: float = 0.01) -> str:
    """'U 53.9% / L 46.1%' for a fractional policy, 'R 100%' for a pure one --
    every action carrying real weight, ranked highest first."""
    idxs = sorted((k for k in range(4) if policy[k] > tol), key=lambda k: -policy[k])
    return " / ".join(f"{_ACT[k]} {policy[k] * 100:.1f}%" for k in idxs)


def _report(label: str, why: str, g: SoccerGame, solver, r, state, panels: list[str], case_no: int):
    M = solver._matrix(state, r.values)
    maximin, minimax, row_ties, col_ties = _ties(M)
    mixed = state in r.no_saddle_states
    if mixed:
        assert minimax - maximin > 1e-4, f"{state}: expected a genuine mixed gap, found none"
    else:
        assert abs(maximin - minimax) < 1e-6, f"{state}: expected a pure saddle, got a gap"
    x0, y0, x1, y1, b = state
    carrier, defender = (0, 1) if b == 0 else (1, 0)
    row_pol, col_pol = r.row_policy[state], r.col_policy[state]

    print(f"state {state} -- {label} [{'mixed' if mixed else 'pure'}]")
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
    print(f"  gap (minimax - maximin) = {minimax - maximin:+.6f}"
          + ("  <- no pure saddle: genuinely mixed" if mixed else "  (pure saddle)"))
    print(f"  equilibrium policy: p0={np.round(row_pol, 4).tolist()}  "
          f"p1={np.round(col_pol, 4).tolist()}")
    print(f"  why: {why}")
    print("  every joint action's real outcome(s), from game.transitions() "
          "(rows = player 0, cols = player 1; CHALLENGE cells carry two "
          "weighted branches):")
    for i, a0 in enumerate(MOVE_ACTIONS):
        cells = [_outcome_str(g, state, a0, a1) for a1 in MOVE_ACTIONS]
        print(f"    {_ACT[i]:>3} " + "  ".join(f"{c:<34}" for c in cells))
    print()

    kind = "mixed" if mixed else "pure"
    board = policy_svg(g, state, {state: row_pol}, {state: col_pol},
                        value=r.values[state], title=label, kind=kind, always_label=True)
    matrix = bestresponse_graph_svg(M, _ACT, _ACT, title=f"Q matrix -- {state}")
    panels.append(board)
    panels.append(matrix)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / f"tackle_case{case_no:02d}.svg").write_text(panel_svg([board, matrix], cols=2))


def _html_case(number, label, why, g, solver, result, state, note=""):
    M = solver._matrix(state, result.values)
    p, q = result.row_policy[state], result.col_policy[state]
    _, _, rows, cols = _ties(M)
    mixed = state in result.no_saddle_states
    matrix = "".join(
        "<tr><th>" + _ACT[a] + "</th>" + "".join(
            f"<td>{v:.6f}</td>" for v in M[a]) + "</tr>" for a in range(4))
    transitions = "".join(
        "<tr><th>" + a0.name + "</th>" + "".join(
            f"<td>{html.escape(_outcome_str(g, state, a0, a1))}</td>"
            for a1 in MOVE_ACTIONS) + "</tr>" for a0 in MOVE_ACTIONS)
    headers = "<tr><th>P0 / P1</th><th>U</th><th>D</th><th>L</th><th>R</th></tr>"
    v = result.values[state]
    kind_label = "Mixed equilibrium" if mixed else "Pure equilibrium"
    x0, y0, x1, y1, b = state
    link = f"explorer.html?board=tackle&state={x0},{y0},{x1},{y1},{b}"
    return f"""<section><h2><span class="case-no">{number}</span>{html.escape(label)}</h2>
<p class="lede">State {state} &middot; player {state[4]} carries the ball &middot;
V = {v:+.6f} &middot; {kind_label}</p>
<div class="callout">
<div class="stat-row"><span><b>P0 &rarr;</b> {_policy_desc(p)}</span>
<span><b>P1 &rarr;</b> {_policy_desc(q)}</span></div>
<p class="muted">Security-tied alternatives (not probabilities):
P0 {'/'.join(_ACT[k] for k in rows)} &middot;
P1 {'/'.join(_ACT[k] for k in cols)}</p>
</div>
<p>{html.escape(why)}</p>
{note}<img src="figures/gallery/tackle_case{number:02d}.svg"
 alt="Case {number} selected policy and payoff graph">
<h3>Q: player 0 payoff, optimal continuation after this turn</h3>
<table>{headers}{matrix}</table>
<h3>All 16 successor outcomes, from the transition engine</h3>
<table class="transitions">{headers}{transitions}</table>
<p><a href="{link}">Open this position in the interactive explorer &rarr;</a></p>
</section>"""


NOTES: dict[int, str] = {
    3: (
        "<p><b>No challenge is on the table this turn, and it is still "
        "mixed:</b> every one of the 16 joint actions above resolves to a "
        "single deterministic successor -- the defender at (1, 1) is "
        "diagonally adjacent to the carrier at (0, 0), and none of U/D/L/R "
        "from there lands on (0, 0), so no duel can fire this turn. The gap "
        "between maximin and minimax still does not close, because the "
        "solver's values for several of <em>this turn's</em> successor "
        "states already price in a duel one move later -- e.g. (0,0,1,0,0) "
        "and (0,0,1,0,1), case 1 and case 2 below, are themselves mixed. A "
        "state does not need a challenge reachable in the current stage "
        "game to inherit real randomization from the continuation value; it "
        "only needs to be one move from a state that does.</p>\n"
    ),
    7: (
        "<p><b>Why the goal beats the tackle, exactly:</b> "
        "<code>soccer_nash/game.py</code>'s <code>_resolve_tackle</code> "
        "checks the carrier's own scoring move <em>before</em> resolving "
        "the duel -- <code>if scored: return self._score_result(carrier)</code> "
        "-- so a challenge that would otherwise fire never gets the chance. "
        "Verified above: player 1's R at (3, 1) targets (4, 1), player 0's "
        "current cell, which is a textbook challenge condition, yet every "
        "one of player 1's four replies to player 0's R produces the exact "
        "same single outcome, <code>GOAL:P0</code> at probability 1 -- no "
        "50/50 split anywhere in that row. Compare row R to rows U/D/L, "
        "where the same defender challenge (col R) does split 50/50, "
        "because those rows' own moves do not score outright.</p>\n"
    ),
}


def main() -> None:
    g = SoccerGame(**BOARD)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run_exact()
    print(f"exact solve: |V_exact - V_iterative| = {r.exact_vs_iterative:.2e} "
          f"over {len(r.values)} states")
    print(f"mixed (no pure saddle) states: {len(r.no_saddle_states)} of {len(r.values)} "
          f"({100 * len(r.no_saddle_states) / len(r.values):.1f}%)\n")

    panels: list[str] = []
    cases = [
        ((0, 0, 1, 0, 0), "The corner duel",
         "player 0 carries the ball at (0, 0); player 1 defends from directly "
         "beside it at (1, 0). Player 1's L is the board's own challenge "
         "trigger here -- it targets (0, 0), player 0's current cell -- and it "
         "fires a genuine 50/50 duel against *every* one of player 0's four "
         "replies (verified below: U,L / D,L / L,L / R,L each split into two "
         "weighted branches), because the challenge test only checks where "
         "player 0 is standing at the start of the turn, not what player 0 "
         "chooses to do about it. Player 1 cannot make L safe by committing "
         "to it (a coin the carrier can see coming is no threat) nor make U "
         "(hold back) safe either (giving up 13% of the time on the duel "
         "loses ground it could contest) -- the exact solve settles player 1 "
         "at U 86.9% / L 13.1%. Player 0 answers by mixing U 53.9% / L 46.1%: "
         "U is a real advance to (0, 1); L is player 0's own wall-clamped "
         "hold in place -- neither is safe on its own against a defender who "
         "is themself unpredictable."),
        ((0, 0, 1, 0, 1), "The committed challenge",
         "the mirror of case 1 with the ball on the other foot: player 1 "
         "carries at (1, 0), player 0 defends from (0, 0). Player 0's unique "
         "challenge action is R, targeting (1, 0) -- and here it fires "
         "against every one of player 1's replies (all four R,* cells below "
         "split 50/50), because player 1's own move can never get the ball "
         "out of range before the duel resolves. Player 0 mixes R 79.1% / "
         "U 20.9% -- diving in for the duel most of the time, but not always, "
         "since a challenge that always comes is one player 1 can simply "
         "shepherd around. Player 1 answers by mixing L 71.8% / U 28.2%, "
         "not U/D/R -- L is player 1's own move deepest into open space away "
         "from the defender, so it is the best hedge against a duel that "
         "might or might not land."),
        ((0, 0, 1, 1, 0), "Mixing without a duel on the table",
         "player 0 carries at (0, 0); player 1 defends from the diagonal cell "
         "(1, 1) -- one step too far for any of U/D/L/R to reach (0, 0) this "
         "turn, so (verified below) all 16 joint actions resolve to a single "
         "deterministic successor: no challenge, no randomness, this turn "
         "looks exactly like a plain A10 stage game. It is still genuinely "
         "mixed (minimax - maximin = 0.0595, no pure saddle) -- because "
         "several of its own successors, including case 1 and case 2 above, "
         "are themselves mixed, and that uncertainty is already baked into "
         "V for those states before this turn's Q matrix is even built. "
         "Player 0 mixes U 80.0% / R 20.0%; player 1 mixes L 76.3% / D 23.7%. "
         "See the note below for exactly which cells feed this."),
        ((0, 0, 4, 3, 0), "No challenge, plain A10",
         "the two players sit at opposite corners, the maximum Manhattan "
         "distance (7) the 5x4 board allows. No defender move can land "
         "anywhere close to the carrier's cell, so (verified below) every "
         "one of the 16 joint actions is a single deterministic outcome -- "
         "this state's stage game is byte-for-byte the plain deterministic "
         "A10 rule, exactly as the tackle rule's own docstring promises for "
         "a no-challenge turn. It is also a clean, tie-free pure saddle: R "
         "is player 0's unique best action (0.021644, beating D/L's 0.019479 "
         "and U's 0.007777); D is player 1's unique reply. Nothing here "
         "needs the tackle rule at all -- it is the baseline the rest of "
         "this document's randomness is measured against."),
        ((1, 1, 2, 1, 0), "The 50/50 duel, in full",
         "player 0 carries at (1, 1); player 1 defends from directly beside "
         "it at (2, 1). The equilibrium here is pure -- D for player 0 "
         "(-0.024493, a unique maximin) against L for player 1 (a unique "
         "minimax) -- but (D, L) is itself a live challenge, and both of its "
         "branches are worth reading in full: player 1's L targets (1, 1), "
         "player 0's current cell, so with probability 0.5 the duel is won "
         "by the defender -- the carrier is shoved to (0, 1) along the "
         "defender's own leftward approach, the defender takes the vacated "
         "(1, 1), and the ball changes hands: (0,1,1,1,1). With probability "
         "0.5 the challenge fails -- the defender bounces back to (2, 1) "
         "(unchanged, already there), and the carrier, which was running "
         "clear toward (1, 0), keeps going and keeps the ball: (1,0,2,1,0). "
         "One state, two branches, a coin between them -- exactly the "
         "mechanic that makes 56 states on this board unable to settle on "
         "a pure answer, even though this particular one still can."),
        ((4, 0, 4, 1, 0), "Head-on at the byline",
         "player 0 carries at (4, 0), the right byline; player 1 defends "
         "from directly above at (4, 1). The equilibrium is U for player 0 "
         "against D for player 1, both unique -- and it is a head-on "
         "collision: player 1's D targets (4, 0), the challenge condition, "
         "for every choice player 0 makes, and player 0's own U targets "
         "(4, 1), player 1's current cell, which is not itself a challenge "
         "(only the defender's move onto the carrier's cell counts) but does "
         "mean the carrier is not 'running clear' if the duel fails. Both "
         "branches leave every position exactly where it started -- the win "
         "branch's shove is clamped at y = 0 with nowhere to go, and the "
         "lose branch holds the carrier up rather than letting it complete "
         "U into the defender's own square: (4,0,4,1,1) at 0.5 / "
         "(4,0,4,1,0) at 0.5. Nobody moves either way; only the ball's "
         "owner is ever in doubt."),
        ((4, 1, 3, 1, 0), "The goal that beats the tackle",
         "player 0 carries at (4, 1), already on the right goal line in a "
         "goal row; player 1 defends from directly beside it at (3, 1). "
         "Player 0's R would, off a goal row, just hold in place -- here it "
         "scores outright, and it does so unconditionally: R is the unique "
         "maximin at exactly 1.0, and every one of player 1's four replies "
         "to it ties at the same minimax 1.0, because none of them can stop "
         "it. That includes R, player 1's own challenge action, which "
         "targets (4, 1) -- player 0's cell -- and would ordinarily force a "
         "50/50 duel. See the note below for exactly why the duel never "
         "gets the chance to fire here."),
    ]
    for case_no, (state, label, why) in enumerate(cases, start=1):
        _report(label, why, g, solver, r, state, panels, case_no)

    intro = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Seven tackle-rule edge cases: verified policy selection</title>
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
<p class="eyebrow">Soccer Markov Nash &middot; tackle board</p>
<h1>Seven edge cases on the tackle board</h1>
<p>5 by 4 board; goal rows 1, 2; simultaneous U/D/L/R; this project's own
collision rule, not Littman's and not the A10 rule. A challenge is the
defender's move landing on the carrier's current cell. With no challenge
this turn resolves exactly like the plain deterministic A10 rule. With a
challenge, a duel: probability <code>tackle_prob</code> (0.5 on this board)
the defender wins the ball, shoving the carrier one cell along its own
approach direction (clamped at the wall) and taking the vacated cell;
otherwise the challenge fails, the defender bounces back, and the carrier
keeps the ball, completing its own move only if it was already running
clear of both its own start cell and the defender's.</p>
<div class="callout warn">
<p style="margin:0"><b>Why this board has real, fractional probabilities,
unlike its A10 twin:</b> 56 of this board's 760 states (about 7.4%) have a
genuine mixed equilibrium -- no pure saddle exists. That is the richest
source of real randomization among this project's four non-A10 board
variants, and a sharp contrast with the A10-deterministic board
(<a href="a10_cases.html">a10_cases.html</a>), which has exactly 0 of 2380 --
verified fresh in this same session. The mechanism is the 50/50 duel itself:
whenever a challenge is reachable, neither committing to it nor avoiding it
is a safe pure answer to a fixed coin the opponent also knows is coming, so
both players are pushed into randomizing over it. (The project's other main
source of mixed play is a structurally different mechanism -- Littman's
random move order on the canonical board, <a href="positions.pdf">
positions.pdf</a> -- where the randomness is in whose move applies first,
not in a probabilistic duel.)</p>
</div>
<p>Three of the seven cases below are genuinely mixed (cases 1-3); the other
four are pure but each puts a different facet of the collision rule on a
board diagram: no challenge at all, the duel's two branches in full, a
head-on collision that freezes both positions, and a goal that outraces an
incoming tackle. These are discounted stationary values with gamma 0.9, the
same solver configuration as every other board in this project.</p>
<p><a href="explorer.html?board=tackle">Interactive explorer</a> ·
<a href="equilibrium-debug.md">Audit and reproduction details</a></p>"""
    sections = [_html_case(n, label, why, g, solver, r, state, NOTES.get(n, ""))
                for n, (state, label, why) in enumerate(cases, 1)]
    pathlib.Path("docs/tackle_cases.html").write_text(intro + "".join(sections) + "</html>")
    FIG.write_text(panel_svg(panels, cols=2))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
