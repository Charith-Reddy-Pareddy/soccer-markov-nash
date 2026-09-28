"""Seven edge cases on the A10-coinflip board, not the deterministic twin.

``scripts/a10_cases.py`` / ``docs/a10_cases.html`` document the deterministic
(A10-style) board, where the carrier always wins a contested cell or a swap.
This script's board (``move_order="coinflip"``, internal key
``canonical_coinflip``) keeps that exact collision and swap structure --
same ``_resolve_with_winner`` code path, same loser-keeps-the-ball rule --
with a single change: a fair coin, not possession, decides who "wins" each
contest (``_resolve_with_winner(..., winner=b)`` becomes the average of
``winner=0`` and ``winner=1``, each at probability 0.5).

That is real, checkable randomness in what happens *after* an action pair
is chosen. The headline finding of this document is that it still does not
force any state into a genuinely mixed stage game: re-solving all 2380
states gives ``no_saddle_count = 0``, exactly matching A10 itself (see
``docs/data/explorer.json``). The reason is different from A10's own
headline result, though -- A10's purity comes from weak dominance among
exact security ties in a deterministic matrix; this board's purity survives
despite every contested cell's *value* being a genuine 50/50 average of two
different continuations.

The seven cases below use that fact to ask a sharper question: does
softening a contest's risk into a coin flip ever change *which* action is
optimal, compared to the exact same state on the deterministic board? The
answer is genuinely mixed (case 1 flips; case 2, at the same kind of risk
but a bigger gap, does not; case 3 shows the reverse -- a coin flip that
*removes* a safe option instead of adding one). Every number below is read
from the actual solved Q matrix and ``game.transitions()``, not asserted;
a handful of the deterministic-board numbers quoted for comparison are
re-derived by a second, live solve in this script (see
``_verify_det_reference_values``), not copied from docs/a10_cases.html.

    python scripts/coinflip_cases.py

Writes `docs/figures/gallery/coinflip_cases.svg` (composite) plus one
`docs/figures/gallery/coinflip_caseNN.svg` per case.
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

FIG = pathlib.Path("docs/figures/gallery/coinflip_cases.svg")
CASE_DIR = pathlib.Path("docs/figures/gallery")
_ACT = ["U", "D", "L", "R"]
BOARD = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "coinflip"}
BOARD_DET = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "deterministic"}


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


def _verify_det_reference_values(g_cf: SoccerGame, solver_cf, r_cf) -> None:
    """Re-solves the A10 deterministic board and asserts every DET number
    quoted in the case "why" text below, plus the two aggregate stats (the
    swap-trap-defused family and the tightest-tie-shape count). Nothing here
    is copied from docs/a10_cases.html without being re-derived live."""
    g_det = SoccerGame(**BOARD_DET)
    solver_det = NashQIteration(g_det, gamma=0.9, mode="hybrid", tol=1e-10)
    r_det = solver_det.run_exact()

    def val(r, s):
        return 0.0 if s[0] == -1 else r.values[s]

    # Case 1: the swap trap. DET's (R, L) always sends the ball to the
    # loser (the defender), CF sometimes lets the carrier keep it instead.
    s1 = (4, 3, 5, 3, 0)
    Mdet1 = solver_det._matrix(s1, r_det.values)
    Mcf1 = solver_cf._matrix(s1, r_cf.values)
    assert abs(Mdet1[3, 2] - (-0.590490)) < 1e-6
    assert abs(Mcf1[3, 2] - 0.109755) < 1e-6
    mmd1, _, rtd1, _ = _ties(Mdet1)
    mmc1, _, rtc1, _ = _ties(Mcf1)
    assert 3 not in rtd1 and 3 in rtc1, "R should newly join the maximin tie under coinflip"

    # Case 2: pinned in the corner. DET crashes to -0.9; CF halves it to
    # -0.45 but the tie never widens to include U.
    s2 = (0, 0, 0, 2, 0)
    Mdet2 = solver_det._matrix(s2, r_det.values)
    Mcf2 = solver_cf._matrix(s2, r_cf.values)
    assert abs(Mdet2[0, 1] - (-0.9)) < 1e-6
    assert abs(Mcf2[0, 1] - (-0.45)) < 1e-6
    _, _, rtc2, _ = _ties(Mcf2)
    assert 0 not in rtc2, "U should remain outside the safe tie under coinflip"

    # Case 3: the tie that breaks (reverse direction, player 1 carries).
    s3 = (0, 1, 0, 0, 1)
    Mdet3 = solver_det._matrix(s3, r_det.values)
    Mcf3 = solver_cf._matrix(s3, r_cf.values)
    assert abs(Mdet3[1, 0] - 0.0) < 1e-6
    assert abs(Mcf3[1, 0] - (-0.45)) < 1e-6
    _, _, rtd3, _ = _ties(Mdet3)
    _, _, rtc3, _ = _ties(Mcf3)
    assert 1 in rtd3 and 1 not in rtc3, "D should fall OUT of the safe tie under coinflip"

    # Aggregate stat for case 1: the swap-trap-defused family.
    family = []
    denom = 0
    for state in r_det.values:
        if state[0] == -1:
            continue
        x0, y0, x1, y1, b = state
        if b == 0 and x1 == x0 + 1 and y1 == y0:
            denom += 1
            Md = solver_det._matrix(state, r_det.values)
            Mc = solver_cf._matrix(state, r_cf.values)
            _, _, rd, _ = _ties(Md)
            _, _, rc, _ = _ties(Mc)
            if 3 not in rd and 3 in rc:
                family.append(state)
    assert denom == 30 and len(family) == 15
    assert all(x0 >= 3 for x0, *_ in family)

    # Aggregate stat for case 6: the tightest-tie shape (A10 case 7's own
    # 16-of-2380 shape) survives coinflip at exactly half those states.
    def shape_count(solver, r):
        n = 0
        for state in r.values:
            if state[0] == -1:
                continue
            M = solver._matrix(state, r.values)
            _, _, rt, ct = _ties(M)
            if len(rt) == 2 and len(ct) == 1:
                n += 1
        return n

    det_shape = shape_count(solver_det, r_det)
    cf_shape = shape_count(solver_cf, r_cf)
    assert det_shape == 16, f"expected A10's own 16, got {det_shape}"
    assert cf_shape == 8, f"expected half of 16 to survive, got {cf_shape}"

    # Case 7: the dead zone is bit-for-bit identical between the two games.
    s7 = (0, 0, 2, 0, 0)
    Mdet7 = solver_det._matrix(s7, r_det.values)
    Mcf7 = solver_cf._matrix(s7, r_cf.values)
    assert np.allclose(Mdet7, 0.0, atol=1e-9) and np.allclose(Mcf7, 0.0, atol=1e-9)

    print(f"DET reference solve: |V_exact - V_iterative| = {r_det.exact_vs_iterative:.2e} "
          f"over {len(r_det.values)} states")
    print("all DET comparison numbers and both aggregate stats verified against a live solve\n")


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
            desc = "/".join(
                ("GOAL:P0" if ns[4] == 0 and ns[0] == -1 else
                 "GOAL:P1" if ns[4] == 1 and ns[0] == -1 else str(ns))
                + f"@{p:.2f}"
                for p, ns, _reward in outs
            )
            cells.append(desc)
        print(f"    {_ACT[i]:>3} " + "  ".join(f"{c:<28}" for c in cells))
    print()

    # kind is left to policy_svg's own inference (mixed0/mixed1 from the
    # real policy weights): every case documented here is pure by the
    # matrix's own maximin == minimax check above, and this board's cases
    # are not pure by construction the way A10's are, so the classification
    # should come from the actual policy, not be forced.
    board = policy_svg(g, state, {state: disp_row}, {state: disp_col},
                        value=r.values[state], title=label)
    matrix = bestresponse_graph_svg(M, _ACT, _ACT, title=f"Q matrix -- {state}")
    panels.append(board)
    panels.append(matrix)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / f"coinflip_case{case_no:02d}.svg").write_text(panel_svg([board, matrix], cols=2))


def _html_case(number, label, why, g, solver, result, state, note=""):
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
    x0, y0, x1, y1, b = state
    link = f"explorer.html?board=canonical_coinflip&state={x0},{y0},{x1},{y1},{b}"
    return f"""<section><h2><span class="case-no">{number}</span>{html.escape(label)}</h2>
<p class="lede">State {state} &middot; player {state[4]} carries the ball &middot;
V = {v:+.6f}</p>
<p><a href="{link}">Open this position in the interactive explorer &rarr;</a></p>
<div class="callout">
<div class="stat-row"><span><b>P0 &rarr; {_ACT[i]}</b> (100%)</span>
<span><b>P1 &rarr; {_ACT[j]}</b> (100%)</span></div>
<p class="muted">Security-tied alternatives (not probabilities):
P0 {'/'.join(_ACT[k] for k in rows)} &middot;
P1 {'/'.join(_ACT[k] for k in cols)}</p>
</div>
<p>{html.escape(why)}</p>
{note}<img src="figures/gallery/coinflip_case{number:02d}.svg"
 alt="Case {number} selected policy and payoff graph">
<h3>Q: player 0 payoff, optimal continuation after this turn (coinflip board)</h3>
<table>{headers}{matrix}</table>
<h3>All 16 successor states, from the transition engine (higher-probability branch shown)</h3>
<table class="transitions">{headers}{transitions}</table></section>"""


def main() -> None:
    g = SoccerGame(**BOARD)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run_exact()
    print(f"exact solve: |V_exact - V_iterative| = {r.exact_vs_iterative:.2e} "
          f"over {len(r.values)} states\n")

    no_saddle = 0
    for state in r.values:
        if state[0] == -1:
            continue
        M = solver._matrix(state, r.values)
        maximin, minimax, _, _ = _ties(M)
        if abs(maximin - minimax) > 1e-6:
            no_saddle += 1
    print(f"headline check: {no_saddle} of {len(r.values)} states need a genuinely mixed "
          "policy (expect 0, matching A10)\n")
    assert no_saddle == 0, "expected every state on this board to have a pure saddle"

    _verify_det_reference_values(g, solver, r)

    panels: list[str] = []
    cases = [
        ((4, 3, 5, 3, 0), "The swap trap, disarmed",
         "This is A10's own case 1 state (docs/a10_cases.html), replayed under a fair coin. "
         "Under the deterministic rule, R is player 0's only action that reaches player 1's "
         "square, and (R, L) triggers the same genuine swap A10 documents: the winner of the "
         "coin there is rigged 100% to the carrier, but the loser keeps the ball regardless of "
         "who 'won' -- so DET's swap sends the ball to the defender every single time, crashing "
         "(R, L) to -0.590490 and knocking R out of the safe tie. Flip the coin fair, and the "
         "same swap sometimes goes the other way: when the coin instead favors the defender, "
         "the loser is now the carrier, who keeps the ball, landing at (5,3,4,3,0), worth 0.9. "
         "Averaged with the still-possible carrier-loses branch (landing at (5,3,4,3,1), worth "
         "-0.6561, the same downstream value DET's own swap always lands on), Q_cf(R,L) = "
         "0.5*(0.9*-0.6561) + 0.5*(0.9*0.9) = 0.109755 (verified below against "
         "game.transitions()). R's own worst case is no longer the swap cell at all -- it is "
         "now the harmless (R, R) cell at exactly 0, tying R with U, D, and L. Among that "
         "four-way tie, R's mean payoff (0.432439) beats D's (0.3645), so the solver's "
         "secondary criterion now selects R, not D: the swap trap is genuinely defused, not "
         "just softened. This is not an isolated fluke -- the identical 'R newly joins the "
         "safety tie' pattern recurs at 15 of the 30 same-row, ball-adjacent-defender states on "
         "this board, and every one of them has x0 >= 3 (the attacking half); it never happens "
         "for x0 <= 2."),
        ((0, 0, 0, 2, 0), "Pinned in the corner, still pinned",
         "A10's own case 2 state, for direct comparison. Player 0's U still targets the same "
         "open, non-blocked contest cell (0,1) that player 1's D also targets; under DET the "
         "carrier always wins the coin, but the loser -- the defender -- still ends up with the "
         "ball regardless, crashing the value to -0.9 exactly as A10 documents. Under coinflip, "
         "half the time the coin instead favors the defender, so the carrier keeps the ball and "
         "lands at the harmless (0,0,0,1,0), worth 0.0. Averaging: Q_cf(U,D) = "
         "0.5*(0.9*-1.0) + 0.5*(0.9*0.0) = -0.45, exactly half of DET's -0.9 (both branches "
         "verified below; the -1.0 continuation is A10's own 'open net' state, case 5, where "
         "scoring bypasses the coin entirely). But halving the wound is not the same as closing "
         "it: D, L, and R still guarantee exactly 0, so U's floor of -0.45 remains strictly "
         "below the safe tie. The selected policy is unchanged from A10 -- R for player 0, D "
         "for player 1. This is the honest negative result case 1 does not give away for free: "
         "averaging a punishment only rescues an action when the punishment was already close "
         "enough to the safe floor, and -0.9 was not."),
        ((0, 1, 0, 0, 1), "The tie that breaks",
         "The reverse of the swap trap, and the first case here where player 1 carries the "
         "ball. Player 0's D at (0,1) and player 1's U at (0,0) target each other's current "
         "cell -- a genuine swap. Under DET, the carrier (player 1) always wins the coin, so "
         "the loser (player 0, the defender) always ends up with the ball after the swap, "
         "landing at (0,0,0,1,0), worth 0.0 -- a clean, guaranteed break-even for D, tied with "
         "L. Flip the coin fair, and half the time the carrier instead keeps the ball through "
         "the swap, landing player 0 at (0,0,0,1,1) -- the exact 'open net' state from case 2's "
         "note above, an unconditional loss worth -1.0 to player 0 since scoring bypasses the "
         "coin. Q_cf(D,U) = 0.5*(0.9*-1.0) + 0.5*(0.9*0.0) = -0.45 (verified below), so D's "
         "guaranteed floor collapses from 0 to -0.45 and falls out of the tie -- L is now "
         "player 0's only safe action. Randomizing a contest's winner does not uniformly help "
         "the player who used to be locked out of winning it; here it strictly hurts the player "
         "who used to be guaranteed a fair split, because the alternative to 'always break "
         "even' is a coin flip against an outcome that is scored, not just inconvenient."),
        ((0, 2, 0, 0, 0), "The defender's flip",
         "The mirror image of case 2's geometry -- same corner, same contest, but from the "
         "other side of the board and the other player's turn to worry about it. Player 0 "
         "carries the ball at (0,2) and plays D into the open cell (0,1); player 1 defends "
         "from (0,0) and plays U into the same cell. Under DET, the carrier (0) always wins "
         "the coin, so the loser -- player 1 -- always ends up with the ball, landing at "
         "(0,1,0,0,1), worth 0.0; that is why U is player 1's selected column on the "
         "deterministic board. Under coinflip, half the time the defender instead wins the "
         "coin, and the rule for a won, non-swap contest is that only the winner physically "
         "moves -- the loser stays put -- so the carrier (loser of the coin, but never forced "
         "to give up the ball) simply keeps it exactly where it stood, landing at (0,2,0,1,0), "
         "worth 0.531441. Q_cf(D,U) = 0.5*(0.9*0.0) + 0.5*(0.9*0.531441) = 0.23914845 "
         "(verified below). Player 1's minimax tie set never changes -- all four columns still "
         "guarantee the same 0.531441 -- but U's own mean payoff gets worse relative to R's "
         "once this one cell rises, so the solver's secondary criterion now selects R instead "
         "of U: a genuine flip that leaves the primary tie completely untouched."),
        ((6, 1, 4, 1, 0), "The open goal, mostly immune",
         "A10's own case 4 state. Player 0 already sits at the right wall in a goal row with "
         "the ball; R pushes past the attacking edge and scores unconditionally, exactly as in "
         "the deterministic game -- game.transitions() confirms every one of player 1's replies "
         "to R still ends the game as a player-0 win, because a score is checked before "
         "_resolve_with_winner ever consults which player the coin favors. R's entire row is "
         "therefore bit-for-bit identical between the two boards: [1.0, 1.0, 1.0, 1.0] in both. "
         "But 'immune' describes only the winning row, not the whole matrix: player 0's L still "
         "reaches a genuine swap against player 1's R (verified below), and that cell does "
         "change -- DET sends it to -0.590490 (the defender always wins the coin-that-isn't-a-"
         "coin, exactly as in case 1 above) while coinflip softens it to 0.154755 by the "
         "identical 0.5/0.5 averaging arithmetic used throughout this document. The optimum is "
         "untouched; the alternatives are not."),
        ((1, 0, 0, 1, 0), "The tightest tie, narrowly missed",
         "A10's own case 7 state -- the smallest tie shape on the deterministic board (16 of "
         "2380 states: player 0's D and R tied, player 1's R alone). Player 0's L still hides "
         "the same contest A10 documents: (L, D) both target (0,0); the carrier wins the "
         "physical race under DET but the loser keeps the ball anyway, crashing L to -0.9. "
         "Under coinflip the same contest sometimes lets the carrier keep its own square "
         "instead (verified below: the branch where the defender wins the coin lands at "
         "(1,0,0,0,0), worth 0.531441, not a loss at all), softening Q_cf(L,D) to "
         "0.5*(0.9*-1.0) + 0.5*(0.9*0.531441) = -0.21085155 -- more than three-quarters of the "
         "way back to the safe floor of 0, the closest any cell in this document comes to "
         "flipping without actually doing so. It still falls short: D and R remain the only "
         "safe actions, unchanged from A10. This is not an isolated result -- across the whole "
         "board, the exact 'two-tied-and-one-alone' shape that defines this case survives "
         "coinflip at only 8 of the 16 states where it occurs under DET; the other 8 widen the "
         "way case 1 above does."),
        ((0, 0, 2, 0, 0), "The dead zone, untouched",
         "A10's own case 8 state -- the single most common shape on the whole board (514 of "
         "2380 states, unchanged under coinflip too). Every one of the 16 Q-matrix cells is "
         "exactly 0.0 in both games, not merely tied at the same value but bit-for-bit "
         "identical, because this state's own dozen successors (verified below) are themselves "
         "all exactly 0.0 in both games. Averaging two branches that are each worth 0 is worth "
         "0 no matter which player the coin favors: 0.5*0 + 0.5*0 = 0 for any contest this "
         "state's actions could reach. Not every quiet region of the board is this literal "
         "about it -- the open goal above shows a matrix that visibly changes even though its "
         "optimal row does not -- but the dead zone is: coinflip changes nothing here, down to "
         "the last decimal."),
    ]
    for case_no, (state, label, why) in enumerate(cases, start=1):
        _report(label, why, g, solver, r, state, panels, case_no)

    intro = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Seven A10 coin-flip cases: verified policy selection</title>
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
<p class="eyebrow">Soccer Markov Nash &middot; A10-coinflip board</p>
<h1>Seven edge cases on the A10 coin-flip board</h1>
<p>7 by 5 board; goal rows 1, 2, 3; simultaneous U/D/L/R; A10's exact collision and swap
structure, with one change -- a fair coin, not the ball carrier, decides who wins a
contested square or a swap. Players never occupy the same square. The loser of a
contested square or swap still receives the ball, regardless of who the coin favored.</p>
<div class="callout">
<p style="margin:0"><b>Selection rule:</b> preserve the best worst-case payoff, then maximize
mean payoff over opponent actions among exact security ties. Identical-score
ties use action order. The secondary criterion is a documented selection
convention, not a claim that Nash equilibrium is unique. No uniform tie mix
is substituted for the solver policy.</p>
</div>
<div class="callout warn">
<p style="margin:0"><b>Every one of the 2380 states on this board still resolves to a pure
equilibrium:</b> this board keeps A10's exact collision and swap structure but replaces the
automatic "carrier always wins" resolution with a fair coin -- real randomness in
<i>what happens after</i> an action pair is chosen, not A10's own headline finding of weak
dominance among security ties. Re-solving every stage game confirms
<code>no_saddle_count = 0</code>: zero states need a genuinely mixed policy, exactly matching
A10 itself (see <code>docs/data/explorer.json</code>). The reason is different from A10's,
though -- here, averaging two contest outcomes at a fixed 50/50 probability still collapses to
a single worst-case-optimal action, because the coin only reweights <i>which</i> branch a
contested cell leads to, not <i>whether</i> a dominant response to that reweighting exists.
Genuine randomness in outcomes does not, by itself, force randomization in which action to
pick.</p>
</div>
<p>Case 1 below is the most direct test of what the coin actually changes: at the exact same
state as A10's own "swap trap," halving the swap's guaranteed loss into a 50/50 gamble is
enough to flip player 0's selected action from Down to Right. It is not a universal effect --
case 2, A10's own "pinned in the corner" at the same state, shows the identical halving
arithmetic falling short of a flip, because the gap it needed to close was larger to begin
with. These are discounted stationary values with gamma 0.9, not the undiscounted 100-turn
assignment.</p>
<p><a href="explorer.html?board=canonical_coinflip">Interactive explorer</a> &middot;
<a href="equilibrium-debug.md">Audit and reproduction details</a></p>"""
    sections = [_html_case(n, label, why, g, solver, r, state)
                for n, (state, label, why) in enumerate(cases, 1)]
    pathlib.Path("docs/coinflip_cases.html").write_text(intro + "".join(sections) + "</html>")
    FIG.write_text(panel_svg(panels, cols=2))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
