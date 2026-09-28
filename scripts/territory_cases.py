"""Seven edge cases on the territory-reward board.

``scripts/a10_cases.py`` covers the deterministic A10 board, whose headline
finding is that *every* stage game there is pure (0 of 2380 states mix).
This board keeps the exact same movement/contest rule -- no randomness in
resolution at all -- but changes ``scoring`` from ``"win"`` to
``"territory"``: the first goal still ends the game outright
(``_score_result`` is untouched), but *every* non-terminal transition now
also pays a small dense per-step reward, ``+/- territory_reward``, based on
which third of the board the ball carrier is standing in after the step
(``SoccerGame._territory_reward``). That per-step reward is a genuine
zero-sum tug-of-war layered on top of the underlying win/lose game -- "a
real objective, not potential-based," in the module docstring's own words --
and it is what produces this board's headline finding: 69 of 2380 states
(~2.9%) have a genuine *mixed* equilibrium, the second-richest source of
real randomization among this project's non-A10 board variants, versus
exactly 0 on A10.

Four cases below are verified mixed equilibria, tracing the actual
maximin/minimax gap and LP support in each one, not merely asserting
"mixed" from a policy_svg auto-classification. One case isolates the
territory-reward mechanic in a single, undiluted Q-matrix cell (a pure
state whose continuation value happens to be exactly 0, so the cell equals
the one-step reward with nothing compounded on top). Two more cases
contrast directly against the A10/win-scoring engine on the identical
board and state, to show precisely what the territory reward adds and does
not add.

    python scripts/territory_cases.py

Writes `docs/figures/gallery/territory_cases.svg` (composite) plus one
`docs/figures/gallery/territory_caseNN.svg` per case.
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

FIG = pathlib.Path("docs/figures/gallery/territory_cases.svg")
CASE_DIR = pathlib.Path("docs/figures/gallery")
_ACT = ["U", "D", "L", "R"]
BOARD = {
    "width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "deterministic",
    "scoring": "territory", "territory_reward": 0.05,
}
WIN_BOARD = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "deterministic"}


def _ties(M: np.ndarray, tol: float = 1e-6):
    """Mirrors site/src/explorer/helpers.js's own certify(): every row/column
    index achieving the guaranteed value, not just the first (certify_game's
    own `saddle` reports only one). On this board the pure maximin and
    minimax frequently *disagree* -- that gap is exactly what a genuine
    mixed equilibrium looks like, so unlike a10_cases.py this helper's
    output is not asserted equal below."""
    row_min = M.min(axis=1)
    maximin = row_min.max()
    col_max = M.max(axis=0)
    minimax = col_max.min()
    row_ties = [i for i in range(4) if abs(row_min[i] - maximin) <= tol]
    col_ties = [j for j in range(4) if abs(col_max[j] - minimax) <= tol]
    return maximin, minimax, row_ties, col_ties


def _policy_str(probs: np.ndarray, tol: float = 0.005) -> str:
    """"U 60% / L 40%"-style summary of every action carrying real weight,
    sorted by weight -- not the single-action, always-100% framing that is
    correct for a10_cases.py but wrong for a genuinely fractional policy."""
    order = np.argsort(-probs)
    parts = [f"{_ACT[i]} {probs[i] * 100:.0f}%" for i in order if probs[i] > tol]
    return " / ".join(parts) if parts else f"{_ACT[int(np.argmax(probs))]} 100%"


def _reward_tag(reward: tuple[float, float]) -> str:
    r0 = reward[0]
    if abs(r0) < 1e-9:
        return ""
    return f" ({r0:+.2f})"


def _report(label: str, why: str, g: SoccerGame, solver, r, state, panels: list[str],
            case_no: int, kind: str):
    M = solver._matrix(state, r.values)
    maximin, minimax, row_ties, col_ties = _ties(M)
    gap = minimax - maximin
    if kind == "pure":
        assert abs(gap) < 1e-6, f"{state}: expected a pure saddle, got a gap of {gap:.6f}"
    else:
        assert gap > 1e-6, f"{state}: expected a genuine mixed gap, got {gap:.6f}"
    x0, y0, x1, y1, b = state
    carrier, defender = (0, 1) if b == 0 else (1, 0)
    row_pol, col_pol = r.row_policy[state], r.col_policy[state]

    print(f"state {state} -- {label} ({kind})")
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
    print(f"  maximin/minimax gap = {gap:+.6f}"
          + ("  (pure saddle)" if kind == "pure" else "  (no pure saddle -> genuine mix)"))
    print(f"  solver's own policy: p0={np.round(row_pol, 4).tolist()}  "
          f"p1={np.round(col_pol, 4).tolist()}")
    print(f"  selected policy (also displayed and played by the site): "
          f"p0={_policy_str(row_pol)}  p1={_policy_str(col_pol)}")
    print(f"  why: {why}")
    print("  every joint action's actual transition (game.transitions(), rows = "
          "player 0, cols = player 1), with this turn's (player 0) reward:")
    for i, a0 in enumerate(MOVE_ACTIONS):
        cells = []
        for j, a1 in enumerate(MOVE_ACTIONS):
            outs = g.transitions(state, a0, a1)
            prob, ns, reward = outs[0]
            tag = ("GOAL:P0" if ns[4] == 0 and ns[0] == -1 else
                   "GOAL:P1" if ns[4] == 1 and ns[0] == -1 else
                   f"{ns}{_reward_tag(reward)}")
            cells.append(tag)
        print(f"    {_ACT[i]:>3} " + "  ".join(f"{c:<22}" for c in cells))
    print()

    board = policy_svg(g, state, {state: row_pol}, {state: col_pol},
                        value=r.values[state], title=label, kind=kind)
    matrix = bestresponse_graph_svg(M, _ACT, _ACT, title=f"Q matrix -- {state}")
    panels.append(board)
    panels.append(matrix)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / f"territory_case{case_no:02d}.svg").write_text(panel_svg([board, matrix], cols=2))


def _html_case(number, label, why, g, solver, result, state, kind, note=""):
    M = solver._matrix(state, result.values)
    p, q = result.row_policy[state], result.col_policy[state]
    _, _, rows, cols = _ties(M)
    matrix = "".join(
        "<tr><th>" + _ACT[a] + "</th>" + "".join(
            f"<td>{v:.6f}</td>" for v in M[a]) + "</tr>" for a in range(4))
    transitions = "".join(
        "<tr><th>" + a0.name + "</th>" + "".join(
            f"<td>{g.transitions(state, a0, a1)[0][1]}"
            f"{_reward_tag(g.transitions(state, a0, a1)[0][2])}</td>"
            for a1 in MOVE_ACTIONS) + "</tr>" for a0 in MOVE_ACTIONS)
    headers = "<tr><th>P0 / P1</th><th>U</th><th>D</th><th>L</th><th>R</th></tr>"
    v = result.values[state]
    kind_label = "mixed equilibrium" if kind == "mixed" else "pure equilibrium"
    callout_cls = "callout warn" if kind == "mixed" else "callout"
    return f"""<section><h2><span class="case-no">{number}</span>{html.escape(label)}</h2>
<p class="lede">State {state} &middot; player {state[4]} carries the ball &middot;
V = {v:+.6f} &middot; {kind_label}</p>
<div class="{callout_cls}">
<div class="stat-row"><span><b>P0 &rarr; {_policy_str(p)}</b></span>
<span><b>P1 &rarr; {_policy_str(q)}</b></span></div>
<p class="muted">Pure security ties (not the mixed support itself):
P0 {'/'.join(_ACT[k] for k in rows)} &middot;
P1 {'/'.join(_ACT[k] for k in cols)}</p>
</div>
<p>{html.escape(why)}</p>
{note}<a href="explorer.html?board=territory&amp;state={','.join(str(v) for v in state)}">
Open this position in the interactive explorer &rarr;</a>
<img src="figures/gallery/territory_case{number:02d}.svg"
 alt="Case {number} selected policy and payoff graph">
<h3>Q: player 0 payoff, optimal continuation after this turn</h3>
<table>{headers}{matrix}</table>
<h3>All 16 successor states and this turn's (player 0) territory reward</h3>
<table class="transitions">{headers}{transitions}</table></section>"""


def main() -> None:
    g = SoccerGame(**BOARD)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run_exact()
    print(f"exact solve: |V_exact - V_iterative| = {r.exact_vs_iterative:.2e} "
          f"over {len(r.values)} states\n")

    def _is_mixed(state, tol=0.02):
        p, q = r.row_policy[state], r.col_policy[state]
        return (p > tol).sum() > 1 or (q > tol).sum() > 1

    mixed_states = [s for s in r.values if not g.is_terminal(s) and _is_mixed(s)]
    print(f"headline check: {len(mixed_states)} of {len(r.values)} states "
          f"({100 * len(mixed_states) / len(r.values):.1f}%) have a genuine mixed "
          f"equilibrium\n")

    # Cross-solve the identical board under plain win-scoring for the two
    # contrast cases below (cases 6 and 7) -- same board, only `scoring` and
    # `territory_reward` removed.
    gw = SoccerGame(**WIN_BOARD)
    solver_w = NashQIteration(gw, gamma=0.9, mode="hybrid", tol=1e-10)
    rw = solver_w.run_exact()

    panels: list[str] = []
    cases = [
        ((0, 1, 6, 1, 0), "The three-way tug of war", "mixed",
         "Player 0 carries the ball at (0,1), deep in its own defending third "
         "(bx=0 < width/3≈2.33); player 1's defender sits at (6,1), all the way at "
         "the opposite wall. Every one of the 16 joint actions still lands the ball "
         "inside that same leftmost third next turn (verified below: every cell "
         "carries the identical -0.05/+0.05 step reward), so none of the Q matrix's "
         "variation comes from this turn's reward -- all of it comes from the 16 "
         "different continuation values. That is exactly what produces a real gap "
         "between the two pure security levels: row-maximin is -0.191873 (unique at "
         "U) but col-minimax is -0.135500 (unique at L), a genuine 0.056373 gap -- "
         "unlike every state in a10_cases.py, where that gap is always exactly zero. "
         "With no pure saddle, the LP instead finds row ≈ U 48% / L 46% / R 6% and "
         "col ≈ U 18% / L 40% / R 42%, with V=-0.160151 sitting strictly between "
         "the two pure bounds, as von Neumann's minimax theorem requires. The "
         "equilibrium's actual support does not even match the naive pure-tie set "
         "(U alone for player 0, L alone for player 1) -- once there is no saddle, "
         "that tie set stops being a guide to the joint mixed solution."),
        ((0, 0, 1, 1, 0), "The corner squeeze", "mixed",
         "Player 0 carries the ball pinned in the exact corner (0,0); player 1's "
         "defender is one diagonal step away at (1,1) -- close combat, and again "
         "deep in player 0's own third the entire turn (every one of the 16 "
         "successors pays the same -0.05/+0.05 step reward, verified below). The "
         "pure security levels disagree -- row-maximin -0.246038, tied between D "
         "and L; col-minimax -0.229760, unique at R -- a 0.016278 gap, again ruling "
         "out a pure saddle. What makes this one worth a second look: the LP's real "
         "mixed support (U 60% / L 40%) does not match the pure-tie set at all -- D "
         "never appears in the actual equilibrium, and U carries the majority "
         "weight despite never being among the maximin ties. Player 1 mirrors this: "
         "minimax is unique at R, yet the LP mostly plays R (97.6%) with a small "
         "2.4% on L, not 100% R."),
        ((0, 1, 1, 1, 0), "A profitable break", "mixed",
         "Player 0 again starts carrying the ball in its own third at (0,1), but "
         "here it has a real breakaway option: R threatens the middle of the "
         "board, and if it gets there (R/U and R/D both reach a continuation "
         "value of +0.609745) the position is very good for player 0. That threat "
         "is why the equilibrium value is positive, V=+0.041147, even though the "
         "ball starts the turn in player 0's own third and this turn's step "
         "reward is the same -0.05/+0.05 on every one of the 16 cells. Row-"
         "maximin (-0.012968, unique at U) and col-minimax (+0.084063, unique at "
         "R) leave a 0.097031 gap, so player 0 mixes U 73.5% / R 26.5% rather "
         "than only playing the safe U, and player 1 mostly plays R (95.8%) to "
         "shut the breakaway down, with a small 4.2% on L to keep player 0 "
         "honest."),
        ((0, 1, 3, 4, 0), "Splitting the wide side", "mixed",
         "A long diagonal, not a close-quarters stand-off: player 0 carries the "
         "ball at (0,1) near its own goal, while player 1's defender is clear "
         "across the board at (3,4), the far top corner. bx again never leaves "
         "the leftmost third across all 16 successors, so every cell again pays "
         "the identical -0.05/+0.05 step reward, and the Q matrix's shape comes "
         "entirely from continuation values, which range from -0.256784 up to "
         "+0.609745 depending on whether player 0's move threatens the open "
         "right side of the board. Row-maximin is -0.151629 (unique at L); col-"
         "minimax is -0.095000 (unique at R); a 0.056629 gap. The equilibrium "
         "mixes row onto L 44.2% / R 55.8% -- not the maximin-unique L alone -- "
         "and column onto D 11.1% / R 88.9% -- not the minimax-unique R alone. "
         "V=-0.112921 lands, as it must, strictly between the two pure bounds."),
        ((1, 0, 4, 2, 0), "The price of your own third", "pure",
         "This is the territory-reward mechanic isolated, with nothing compounded "
         "on top of it. The equilibrium here is pure: R is player 0's unique "
         "maximin (-0.05) and D is player 1's unique minimax (-0.05) -- a genuine "
         "pure saddle, gap exactly 0. Player 0's equilibrium move R takes the "
         "carrier from x=1 to x=2, still inside the leftmost third (2 < 7/3≈"
         "2.333), so it pays the -0.05 own-third penalty (verified: (R, D) -> "
         "(2, 0, 4, 1, 0), reward -0.05). That successor state happens to be worth "
         "exactly 0.0 in the exact solve -- no advantage either way from there -- "
         "so the discounted continuation term 0.9×0.0 vanishes completely, and "
         "the entire state value is the single step's reward, untouched by any "
         "downstream compounding: V=-0.05, exactly. That is the rule's direct "
         "fingerprint on the numbers, not something inferred from a downstream "
         "difference."),
        ((0, 0, 2, 0, 0), "The dead zone wakes up", "pure",
         "This exact state is a10_cases.py's own case 8, \"the dead zone\": the "
         "single most common shape on the whole board under plain scoring='win' "
         "(514 of 2380 states) and worth exactly 0.0 there -- both players "
         "genuinely indifferent among all four actions in every one of the 16 Q-"
         "matrix cells. Cross-solved here with scoring='territory' instead "
         "(nothing else about the board changed), the identical state is worth "
         "-0.229760, not 0. With the ball starting at (0,0), deep in player 0's "
         "own third, and staying there in all 16 of this turn's successors "
         "(bx never reaches 2.333, so every cell pays the identical -0.05/+0.05 "
         "step reward, verified below), player 0 now pays a steady toll just for "
         "standing on its own side of the pitch. It is still a pure equilibrium: "
         "maximin and minimax agree exactly at -0.229760, unique at U for player "
         "0 and tied between U and R for player 1, resolved to R by the same "
         "tie-break convention a10_cases.py documents (preserve the security "
         "value, then maximize mean payoff among exact ties). The territory "
         "reward has quietly broken the tie between \"doing nothing\" and "
         "\"costing you something\" without changing which actions are chosen."),
        ((6, 1, 4, 1, 0), "Goals still trump territory", "pure",
         "This is a10_cases.py's own case 4, \"the open goal,\" resolved here "
         "under scoring='territory' instead. Player 0 sits at (6,1), the right "
         "wall, in a goal row, already carrying the ball; player 1 defends at "
         "(4,1). game.transitions() confirms R still scores unconditionally "
         "against every one of player 1's four replies (R/U, R/D, R/L, R/R all "
         "return the terminal state (-1,-1,-1,-1,0) with reward (1,-1)) -- "
         "exactly as under plain win scoring, because transitions() only applies "
         "the per-step territory reward when the successor is non-terminal. "
         "Cross-solved on the identical board with scoring='win': V=1.0 there "
         "too, bit for bit identical to the territory-scoring value below. The "
         "dense per-step reward this document is otherwise about is never paid "
         "on the scoring transition itself -- carrying the ball into your own "
         "attacking third and finishing forfeits nothing. R is the unique "
         "maximin (1.0); every one of player 1's four columns tie at the same "
         "minimax (1.0), because there is nothing player 1 can do about it."),
    ]
    assert abs(rw.values[(0, 0, 2, 0, 0)]) < 1e-9, "win-scoring dead zone should be exactly 0"
    assert abs(rw.values[(6, 1, 4, 1, 0)] - 1.0) < 1e-9, "win-scoring open goal should be 1.0"

    for case_no, (state, label, kind, why) in enumerate(cases, start=1):
        _report(label, why, g, solver, r, state, panels, case_no, kind)

    intro = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Seven territory-reward edge cases: mixed equilibria and the dense reward</title>
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
<p class="eyebrow">Soccer Markov Nash &middot; Territory-reward board</p>
<h1>Seven edge cases on the territory-reward board</h1>
<p>7 by 5 board; goal rows 1, 2, 3; simultaneous U/D/L/R; the same deterministic
carrier-wins-contests resolution as the A10 board -- no randomness in movement or
contests at all. What changes is <code>scoring="territory"</code>: the first goal
still ends the game outright, exactly as under plain <code>"win"</code> scoring, but
every non-terminal transition additionally pays a small dense per-step reward based
on which third of the board the ball carrier is standing in afterward
(<code>territory_reward=0.05</code>): +0.05 for player 0 / -0.05 for player 1 if the
carrier is in the rightmost third, the reverse in the leftmost third, nothing in the
middle third. Player 0 attacks the right goal, player 1 the left, so this rewards
controlling the ball in your attacking third every single turn, on top of the
underlying win/lose game.</p>
<div class="callout">
<p style="margin:0"><b>Selection rule:</b> where a pure saddle exists (maximin equals
minimax), the same convention as <a href="a10_cases.html">a10_cases.html</a> applies --
preserve the best worst-case payoff, then maximize mean payoff among exact security
ties, with no uniform tie mix substituted for the solver's policy. Where no pure
saddle exists, the solver instead returns the linear program's own mixed Nash
equilibrium probabilities directly, as computed, not rounded toward any single
action.</p>
</div>
<div class="callout warn">
<p style="margin:0"><b>Headline finding: 69 of 2380 states (~2.9%) genuinely mix --
versus exactly 0 of 2380 on the A10 board.</b> A10's headline result is that its
stage game is <i>always</i> pure, because nothing there ever gives the two players
opposing reasons to want the same square at the same time in a way that a single
action can't resolve. The territory reward changes that: it is a second, independent
zero-sum tug-of-war over ball position layered on top of the win/lose game, active
every single turn regardless of who is carrying. In most states the two objectives
point the same direction and a pure equilibrium survives -- but in 69 states the
position-control incentive and the win/lose incentive genuinely conflict, so the row
and column players' pure security levels (maximin and minimax) disagree by a real,
positive gap, and no single action is optimal against a rational opponent. That gap
is exactly what a mixed equilibrium is: real, LP-computed randomization, not a
uniform average glossed over a tie. See <a href="positions.pdf">positions.pdf</a> for
the sibling source of mixed play on this project's <i>random</i>-move-order board --
the two are unrelated mechanisms that happen to produce the same phenomenon.</p>
</div>
<p>Cases 1-4 below are the verified mixed-equilibrium states, each traced back to a
real maximin/minimax gap and an LP support that often does not even match the naive
pure-tie set. Case 5 isolates the territory-reward mechanic in a single Q-matrix
cell whose continuation value happens to be exactly 0, so the cell equals the raw
one-step reward. Cases 6 and 7 cross-solve the identical board and state under plain
<code>scoring="win"</code> to show precisely what the territory reward adds --
and, in case 7, what it deliberately does not touch. These are discounted stationary
values with gamma 0.9, not an undiscounted turn-by-turn assignment.</p>
<p><a href="explorer.html?board=territory">Interactive explorer</a> &middot;
<a href="equilibrium-debug.md">Audit and reproduction details</a></p>"""
    sections = [_html_case(n, label, why, g, solver, r, state, kind)
                for n, (state, label, kind, why) in enumerate(cases, 1)]
    pathlib.Path("docs/territory_cases.html").write_text(intro + "".join(sections) + "</html>")
    FIG.write_text(panel_svg(panels, cols=2))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
