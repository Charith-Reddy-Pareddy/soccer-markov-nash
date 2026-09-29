"""Eight edge cases on the A10-deterministic board, not the Littman cases.

``scripts/positions.py`` / ``docs/positions.pdf`` are Littman's fourteen
cases on the *random*-move-order canonical board, where the point is mixed
equilibria and fractional LP output. This script is the deterministic
(A10-style) twin's own edge cases -- every stage game on that board is
*pure* (this project's own headline result), but "pure" still hides real
structure the site's board explorer surfaces:

* **ties** -- equal security values do not imply equal probabilities. The
  solver preserves maximin/minimax first, then selects the better mean payoff
  among exact ties. This removes weakly dominated choices such as U/L in
  case 1 without hard-coding a direction or changing transition rules.
* **hold vs. score** -- a move past the board edge is a genuine no-op
  everywhere except one place: the ball carrier, in a goal row, pushing past
  *their own* attacking edge, which ends the game instead of holding. Two of
  the eight cases below exist specifically to put that distinction on a board
  diagram (`_wall_mask` in soccer_nash/viz.py draws it correctly now).
* **the collision rule itself** -- winning the race to a contested cell does
  not mean keeping the ball; the loser of a contest or a swap gets it
  regardless of who "won" the square. Every transition printed below is
  read straight from `game.transitions()`, not asserted.

    python scripts/a10_cases.py

Writes `docs/figures/gallery/a10_cases.svg` (composite) plus one
`docs/figures/gallery/a10_caseNN.svg` per case.
"""
from __future__ import annotations

import html
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import MOVE_ACTIONS, SoccerGame
from soccer_nash.nash_dqn import fit_q_to_exact, train_nash_dqn
from soccer_nash.nash_q import NashQIteration
from soccer_nash.viz import bestresponse_graph_svg, panel_svg, policy_svg

DQN_EPOCHS = 400
DQN_SEED = 0

FIG = pathlib.Path("docs/figures/gallery/a10_cases.svg")
CASE_DIR = pathlib.Path("docs/figures/gallery")
_ACT = ["U", "D", "L", "R"]
BOARD = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "deterministic"}


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


def _report(label: str, why: str, g: SoccerGame, solver, r, state, panels: list[str], case_no: int):
    M = solver._matrix(state, r.values)
    maximin, minimax, row_ties, col_ties = _ties(M)
    assert abs(maximin - minimax) < 1e-6, f"{state}: expected a pure saddle, got a gap"
    x0, y0, x1, y1, b = state
    carrier, defender = (0, 1) if b == 0 else (1, 0)
    row_pol, col_pol = r.row_policy[state], r.col_policy[state]
    disp_row, disp_col = row_pol, col_pol

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
    print(f"  solver's own policy (one-hot, tie-broken): p0={np.round(row_pol, 4).tolist()}  "
          f"p1={np.round(col_pol, 4).tolist()}")
    print(f"  selected policy (also displayed and played by the site): "
          f"p0={np.round(disp_row, 4).tolist()}  p1={np.round(disp_col, 4).tolist()}")
    print(f"  why: {why}")
    print("  every joint action's actual transition (game.transitions(), rows = "
          "player 0, cols = player 1):")
    for i, a0 in enumerate(MOVE_ACTIONS):
        cells = []
        for j, a1 in enumerate(MOVE_ACTIONS):
            outs = g.transitions(state, a0, a1)
            prob, ns, _reward = outs[0]
            cells.append("GOAL:P0" if ns[4] == 0 and ns[0] == -1 else
                          "GOAL:P1" if ns[4] == 1 and ns[0] == -1 else str(ns))
        print(f"    {_ACT[i]:>3} " + "  ".join(f"{c:<16}" for c in cells))
    print()

    board = policy_svg(g, state, {state: disp_row}, {state: disp_col},
                        value=r.values[state], title=label, kind="pure", always_label=True)
    matrix = bestresponse_graph_svg(M, _ACT, _ACT, title=f"Q matrix -- {state}")
    panels.append(board)
    panels.append(matrix)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / f"a10_case{case_no:02d}.svg").write_text(panel_svg([board, matrix], cols=2))


def _neural_table(state, M, dqn_zero, dqn_fit):
    """Independent DQN cross-check for this one state: the exact matrix next
    to two networks that never see each other's answer -- one trained from a
    random init via TD bootstrap ("from zero"), one trained by direct
    supervised regression onto the exact matrices ("fit to exact"). Neither
    hits the exact numbers bit-for-bit (networks essentially never do), but
    both should recover the same qualitative shape. Reused verbatim as the
    site's own "Exact vs. DQN, cell by cell" table, with a second DQN column
    added."""
    Qz, Qf = dqn_zero.matrix(state), dqn_fit.matrix(state)
    err_z = float(np.abs(M - Qz).max())
    err_f = float(np.abs(M - Qf).max())
    rows_html = "".join(
        f"<tr><td>{_ACT[a]}/{_ACT[b]}</td><td>{M[a, b]:.4f}</td>"
        f"<td>{Qz[a, b]:.4f}</td><td>{Qf[a, b]:.4f}</td></tr>"
        for a in range(4) for b in range(4)
    )
    return f"""<div class="neural-comparison"><h3>DQN comparison: state {state}</h3>
<p class="muted">max |Q<sub>DQN</sub> &minus; Q<sub>exact</sub>|: from zero =
<b>{err_z:.4f}</b> &middot; fit to exact = <b>{err_f:.4f}</b>. "From zero" is
{DQN_EPOCHS} epochs of TD bootstrap from a random init with no access to the
exact answer -- the honest baseline; "fit to exact" is direct supervised
regression onto the exact matrices, isolating how well a network this size
can even represent Q<sub>exact</sub> before any bootstrapping noise. One
representative run each (seed {DQN_SEED}), not the multi-seed study in
<a href="neural.md">docs/neural.md</a>. Full per-state numbers for every
board, plus a policy-gradient comparison, are in the
<a href="explorer.html?board=canonical_det&amp;state={",".join(map(str, state))}">
interactive explorer</a>'s "Show neural cross-check" panel.</p>
<table><tr><th>P0 / P1</th><th>Exact</th><th>DQN, from zero</th>
<th>DQN, fit to exact</th></tr>{rows_html}</table></div>"""


def _html_case(number, label, why, g, solver, result, state, dqn_zero, dqn_fit, note=""):
    M = solver._matrix(state, result.values)
    p, q = result.row_policy[state], result.col_policy[state]
    _, _, rows, cols = _ties(M)
    i, j = int(np.argmax(p)), int(np.argmax(q))
    matrix = "".join(
        "<tr><th>" + _ACT[a] + "</th>" + "".join(
            f"<td>{v:.6f}</td>" for v in M[a]) + "</tr>" for a in range(4))
    transitions = "".join(
        "<tr><th>" + a0.name + "</th>" + "".join(
            f"<td>{g.transitions(state, a0, a1)[0][1]}</td>"
            for a1 in MOVE_ACTIONS) + "</tr>" for a0 in MOVE_ACTIONS)
    headers = "<tr><th>P0 / P1</th><th>U</th><th>D</th><th>L</th><th>R</th></tr>"
    v = result.values[state]
    neural = _neural_table(state, M, dqn_zero, dqn_fit)
    return f"""<section><h2><span class="case-no">{number}</span>{html.escape(label)}</h2>
<p class="lede">State {state} &middot; player {state[4]} carries the ball &middot;
V = {v:+.6f}</p>
<div class="callout">
<div class="stat-row"><span><b>P0 &rarr; {_ACT[i]}</b> (100%)</span>
<span><b>P1 &rarr; {_ACT[j]}</b> (100%)</span></div>
<p class="muted">Security-tied alternatives (not probabilities):
P0 {'/'.join(_ACT[k] for k in rows)} &middot;
P1 {'/'.join(_ACT[k] for k in cols)}</p>
</div>
<p>{html.escape(why)}</p>
{note}<img src="figures/gallery/a10_case{number:02d}.svg"
 alt="Case {number} selected policy and payoff graph">
<h3>Q: player 0 payoff, optimal continuation after this turn</h3>
<table>{headers}{matrix}</table>
<h3>All 16 successor states, from the transition engine</h3>
<table class="transitions">{headers}{transitions}</table>
{neural}</section>"""


NOTES: dict[int, str] = {
    3: """<p><b>Zero value is not an all-zero Q matrix.</b> This state has 14 zero
entries, but Q(D,L) = 0.656100 and Q(R,L) = -0.590490. V = 0 means the
minimax value is zero; it does not mean every action pair is equally good.</p>
<p>A zero-reward self-loop alone cannot prove V = 0: another action might
force a win. The audit checks every action pair at all 2,380 states and
verifies that both pure security bounds equal the computed V. Exactly 514
states have all 16 raw entries zero, including case 8. No display rounding
is used in this count. See the reward comparison appendix for Jae's matrix
at this same state and the new DQN/PG experiments.</p>"""
}


def main() -> None:
    g = SoccerGame(**BOARD)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run_exact()
    print(f"exact solve: |V_exact - V_iterative| = {r.exact_vs_iterative:.2e} "
          f"over {len(r.values)} states\n")

    print("training DQN cross-check nets (from zero + fit to exact)...")
    dqn_zero = train_nash_dqn(g, gamma=0.9, hidden=64, epochs=DQN_EPOCHS, seed=DQN_SEED).net
    dqn_fit = fit_q_to_exact(
        g, lambda s: solver._matrix(s, r.values), hidden=64, epochs=DQN_EPOCHS, seed=DQN_SEED,
    )
    print("done\n")

    panels: list[str] = []
    cases = [
        ((4, 3, 5, 3, 0), "The swap trap",
         "player 0's U/D/L never touch player 1 at all this turn, so all three "
         "guarantee at least 0. D weakly dominates U and L: it earns 0.729 against "
         "U/L, while U and L earn zero against every reply. The refined policy "
         "is D, not a uniform tie mix. Only R reaches player 1's "
         "square, and against L it triggers a genuine swap (verified below: "
         "(R, L) -> (5,3,4,3,1)) that hands player 1 the ball, dragging R's own "
         "worst case to -0.59. Player 1's unique safe column is R for an unrelated "
         "reason: U/D/L each let a smart player 0 reach a good, independent square "
         "elsewhere in the matrix (0.81/0.81/0.729) that has nothing to do with the "
         "swap cell."),
        ((0, 0, 0, 2, 0), "Pinned in the corner",
         "player 0's U is the one risky action: it targets the empty cell (0,1), "
         "which player 1's D also targets. Neither player is standing there, so "
         "it's an open (non-blocked) contest -- player 0 physically wins the race "
         "((U, D) -> (0,1,0,2,1)) but the A10 rule still gives the ball to the "
         "loser regardless of who reached the cell first, crashing the value to "
         "-0.9. D, L, and R never reach that cell, so they tie safely at 0."),
        ((3, 4, 4, 4, 0), "The standoff",
         "both players are already on the top row, one cell apart. U is wall-"
         "clamped (a genuine no-op) for both -- not a real move upward. R is player 0's one risky "
         "action (it reaches player 1's square: blocked against U, a real swap "
         "against L); L is player 1's mirror risk. D never touches the opponent "
         "for either player, so U/D/L (player 0) and U/D/R (player 1) all tie at "
         "the exact same safe value."),
        ((6, 1, 4, 1, 0), "The open goal",
         "player 0 already sits at the right wall in a goal row, with the ball. "
         "Off a goal row this exact same boundary push would just hold in place -- "
         "here it scores instead: verified, game.transitions() returns a terminal "
         "player-0 win for R against every one of player 1's replies. R is the "
         "unique optimum (maximin 1.0); player 1's columns all tie at minimax 1.0 "
         "too, because there is nothing any of them can do about it."),
        ((0, 0, 0, 1, 1), "The open net",
         "the mirror of the open goal from player 1's side: player 1 is at the "
         "left wall in a goal row, ball in hand, and L scores unconditionally "
         "(verified against every reply). The bleaker half of this case is "
         "player 0's row: every one of U/D/L/R ties at the exact same -1.0 -- "
         "there is nothing player 0 can do differently that changes the outcome, "
         "which is itself worth seeing on a board rather than only in a number."),
        ((0, 1, 0, 3, 0), "The getaway",
         "R is player 0's *unique* safe action (0.531441, no tie) -- an entirely "
         "independent move to (1,1) that never depends on player 1's reply. U "
         "looks just as reasonable but is a trap: against D, both target the same "
         "empty cell (0,2); player 0 wins the race there (verified: (U, D) -> "
         "(0,2,0,3,1)) and immediately loses the ball anyway, crashing to -0.9. "
         "Player 1's columns all tie at 0.531441 -- there is no reply that slows "
         "the escape down."),
        ((1, 0, 0, 1, 0), "The tightest tie",
         "no state on this entire board ever has a fully unique best action for "
         "*both* players simultaneously -- checked exhaustively, (1,1) never "
         "occurs among all 2380 states. This is the smallest tie shape that does "
         "occur (only 16 of 2380 states): player 0's D and R tie exactly; player "
         "1's R is uniquely optimal. U and L each look just as safe as D/R, but "
         "each hides its own contest: (U, R) both target the same cell (1,1), "
         "and (L, D) both target (0,0) -- verified below, the carrier wins the "
         "square but the ball still goes to the defender either way -- while D "
         "and R never target the same cell as player 1 at all."),
        ((0, 0, 2, 0, 0), "The dead zone",
         "the single most common shape on the whole board (514 of 2380 states): "
         "both players are fully indifferent among all four actions. It is not "
         "merely that the guaranteed values tie -- every one of the 16 cells in "
         "the Q matrix is exactly 0.0, including a genuine ball-swapping contest "
         "((R, L) -> a state that is itself also worth exactly 0, verified below). "
         "All successors have the same optimal-continuation value, although their "
         "physical positions differ."),
    ]
    for case_no, (state, label, why) in enumerate(cases, start=1):
        _report(label, why, g, solver, r, state, panels, case_no)

    intro = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Eight A10 edge cases: verified policy selection</title>
<style>
:root{--ink:#1c2e26;--accent:#1DB954;--accent-dark:#178A42;--line:#dbe5df;
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
.callout{padding:8px 12px} .neural-comparison{break-before:page}
.neural-comparison table{break-inside:avoid}}
</style>
<p class="eyebrow">Soccer Markov Nash &middot; A10-deterministic board</p>
<h1>Eight edge cases on the A10 deterministic board</h1>
<p>7 by 5 board; goal rows 1, 2, 3; simultaneous U/D/L/R; deterministic
carrier-wins-contests resolution. Players never occupy the same square.
The loser of a contested square or swap receives the ball.</p>
<div class="callout">
<p style="margin:0"><b>Selection rule:</b> preserve the best worst-case payoff, then maximize
mean payoff over opponent actions among exact security ties. Identical-score
ties use action order. The secondary criterion is a documented selection
convention, not a claim that Nash equilibrium is unique. No uniform tie mix
is substituted for the solver policy.</p>
</div>
<div class="callout warn">
<p style="margin:0"><b>Why every probability below is 0% or 100%, never a fraction:</b>
this project's headline finding is that <i>every</i> stage game on this
deterministic board has a pure equilibrium -- so a single action at 100% is
the correct, expected answer here, not a simplification. Fractional
probabilities only show up in the sibling document, <a href="positions.pdf">
positions.pdf</a>, which covers Littman's cases on the <i>random</i>-move-order
board, where mixed equilibria are the point. If you're looking for the
mixed-strategy numbers, that is the document to read; this one is about what
"pure but not obvious" looks like instead: ties, dominance, and the collision
rule.</p>
</div>
<p>Case 1 selects Down for player 0: it weakly dominates Up and Left.
The four zeros against Right describe optimal continuation from four
new states, not a defender committed to Right forever. These are discounted
stationary values with gamma 0.9, not the undiscounted 100-turn assignment.</p>
<p>This board's own rule keeps every stage game pure. The other four rule
variants each trade that off for genuine mixed equilibria in some states --
see <a href="coinflip_cases.pdf">coinflip_cases.pdf</a> (a fair coin
decides contests instead of the carrier -- still 0 mixed, surprisingly),
<a href="tackle_cases.pdf">tackle_cases.pdf</a> (a 50/50 duel, 56 of 760
mixed -- the richest source), <a href="territory_cases.pdf">territory_cases.pdf</a>
(a dense position reward, 69 of 2380 mixed), and
<a href="slip_cases.pdf">slip_cases.pdf</a> (movement noise, 52 of 1200
mixed).</p>
<p><a href="explorer.html?board=canonical_det">Interactive explorer</a> ·
<a href="equilibrium-debug.md">Audit and reproduction details</a></p>"""
    sections = [_html_case(n, label, why, g, solver, r, state, dqn_zero, dqn_fit, NOTES.get(n, ""))
                for n, (state, label, why) in enumerate(cases, 1)]
    appendix = pathlib.Path("docs/reward_q_appendix.html").read_text()
    pathlib.Path("docs/a10_cases.html").write_text(
        intro + "".join(sections) + appendix + "</html>"
    )
    FIG.write_text(panel_svg(panels, cols=2))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
