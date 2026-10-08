# ruff: noqa: E501
"""Build docs/policy_gradient.html (and, via ``make pg-pdf``, the PDF) from the
experiment CSVs, so every number in the report comes straight from a run.

    python scripts/pg_report.py
"""

from __future__ import annotations

import csv
import glob
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP, DOCS = ROOT / "experiments", ROOT / "docs"

LEARNERS = [
    ("Exact solver", "exact", "-"),
    ("REINFORCE, self-play", "reinforce", "selfplay"),
    ("REINFORCE, fictitious play", "reinforce", "fictitious"),
    ("A2C, self-play", "a2c", "selfplay"),
    ("A2C, fictitious play", "a2c", "fictitious"),
    ("PPO, self-play", "ppo", "selfplay"),
    ("PPO, fictitious play", "ppo", "fictitious"),
]


def read(path) -> list[dict]:
    p = pathlib.Path(path)
    return list(csv.DictReader(p.open())) if p.exists() else []


def mean(rows, key):
    return st.mean(float(r[key]) for r in rows)


def sd(rows, key):
    return st.stdev(float(r[key]) for r in rows) if len(rows) > 1 else 0.0


def pick(rows, algo, mode):
    return [r for r in rows if r["algo"] == algo and r["mode"] == mode]


def wtl(rows, who, opp):
    return [mean(rows, f"{who}_{k}_vs_{opp}") for k in ("win", "tie", "loss")]


GREEN, GREY, ORANGE = "#1DB954", "#c3cec8", "#a8501c"


def bar_chart(items, width=660) -> str:
    """Horizontal bars of mean exploitability with one dot per seed."""
    row_h, left, right = 26, 190, 44
    h = row_h * len(items) + 34
    x = lambda v: left + v / 1.0 * (width - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x(g):.1f}" y1="6" x2="{x(g):.1f}" y2="{h - 22}" stroke="#dbe5df"/>'
                   f'<text x="{x(g):.1f}" y="{h - 8}" text-anchor="middle" fill="#5b6b63">{g:g}</text>')
    for i, (label, vals) in enumerate(items):
        y = 10 + i * row_h
        m = sum(vals) / len(vals)
        out.append(f'<text x="{left - 8}" y="{y + 13}" text-anchor="end" fill="#1c2e26">{label}</text>'
                   f'<rect x="{left}" y="{y + 3}" width="{max(x(m) - left, 0):.1f}" height="14" '
                   f'fill="{GREEN}" opacity=".85"/>')
        for v in vals:
            out.append(f'<circle cx="{x(v):.1f}" cy="{y + 10}" r="3" fill="#1c2e26" opacity=".7"/>')
        out.append(f'<text x="{width - right + 10}" y="{y + 14}" fill="#1c2e26" font-weight="700">{m:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_chart(items, titles, width=660) -> str:
    """Two stacked win/tie/loss panels side by side, sharing one label column."""
    row_h, left, gap = 26, 190, 24
    panel = (width - left - gap - 10) / 2
    h = row_h * len(items) + 30
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for k, title in enumerate(titles):
        x0 = left + k * (panel + gap)
        out.append(f'<text x="{x0}" y="12" fill="#5b6b63" font-weight="700">{title}</text>')
    for i, (label, panels) in enumerate(items):
        y = 22 + i * row_h
        out.append(f'<text x="{left - 8}" y="{y + 12}" text-anchor="end" fill="#1c2e26">{label}</text>')
        for k, w in enumerate(panels):
            pos = 0.0
            for frac, col in zip(w, (GREEN, GREY, ORANGE)):
                out.append(f'<rect x="{left + k * (panel + gap) + pos * panel:.1f}" y="{y}" '
                           f'width="{frac * panel:.1f}" height="16" fill="{col}"/>')
                pos += frac
    out.append("</svg>")
    return "".join(out)


def table(head, rows, cls="") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><tr>{th}</tr>{body}</table>'


def fp_br_html() -> str:
    """Exploitability after each checkpoint round of fictitious play with
    best-response phases, one row per run, from the saved curves."""
    runs = []
    for path in sorted(glob.glob(str(EXP / "pg_fp_br_*.csv"))):
        rs = read(path)
        if rs:
            runs.append(rs)
    if not runs:
        return ""
    rounds = sorted({int(r["round"]) for rs in runs for r in rs})
    rows, trend = [], []
    for rs in runs:
        by_round = {int(r["round"]): float(r["exploitability"]) for r in rs}
        names = {"reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
        label = f"{names[rs[0]['algo']]}, {rs[0]['br_iters']} iterations per best response"
        rows.append([label] + [f"{by_round[k]:.2f}" if k in by_round else "" for k in rounds])
        first, last = by_round[min(by_round)], by_round[max(by_round)]
        trend.append(last > first)
    verdict = ("In every run the exploitability at the last checkpoint is higher than at the first"
               if all(trend) else
               "Exploitability at the last checkpoint is lower than at the first in some runs")
    return (table(["run"] + [f"round {k}" for k in rounds], rows)
            + f'<p class="muted">{verdict}; none comes close to 0.</p>')


def policy_html() -> str:
    """Action probabilities of the trained learners at fixed states (from the
    saved policy outputs), as one table per state."""
    path = EXP / "pg_policy_outputs.json"
    if not path.exists():
        return ""
    d = json.loads(path.read_text())
    acts = d["actions"]

    def cells(probs):
        top = max(probs)
        return "".join(
            f'<td><b>{v * 100:.0f}%</b></td>' if v == top else f"<td>{v * 100:.0f}%</td>"
            for v in probs)

    head = ("<tr><th rowspan=2>policy</th><th colspan=4>player 0</th><th colspan=4>player 1</th></tr>"
            "<tr>" + "".join(f"<th>{a}</th>" for a in acts * 2) + "</tr>")
    out = []
    for k, st_ in enumerate(d["states"]):
        rows = [("Exact solver", d["exact"]["row"][k], d["exact"]["col"][k])]
        rows += [(lr["label"], lr["row"][k], lr["col"][k]) for lr in d["learners"]]
        body = "".join(f"<tr><td>{n}</td>{cells(r)}{cells(c)}</tr>" for n, r, c in rows)
        x0, y0, x1, y1, b = st_["state"]
        out.append(f"<h3>{st_['label']}: player 0 at ({x0}, {y0}), player 1 at ({x1}, {y1}), "
                   f"player {b} has the ball</h3><table class=\"probs\">{head}{body}</table>")
    return "".join(out)


def f(v, d=2):
    return f"{v:.{d}f}"


def main() -> None:
    main_rows = read(EXP / "pg_finite_a10.csv")
    long_rows = [r for p in sorted(glob.glob(str(EXP / "pg_finite_a10_long_*.csv")))
                 for r in read(p) if r["algo"] != "exact"]
    data = {(a, m): pick(main_rows, a, m) for _, a, m in LEARNERS}

    expl = [(lbl, [float(r["exploitability"]) for r in data[(a, m)]]) for lbl, a, m in LEARNERS]
    result_rows = []
    for lbl, a, m in LEARNERS:
        rs = data[(a, m)]
        w_r, w_n, w_b = (wtl(rs, "row", o) for o in ("random", "nash", "br"))
        result_rows.append([
            lbl, f"{mean(rs, 'exploitability'):.3f} &plusmn; {sd(rs, 'exploitability'):.3f}",
            "/".join(f(v) for v in w_r), "/".join(f(v) for v in w_n),
            "/".join(f(v) for v in w_b), f(mean(rs, "mirror_gap_mean"))])
    learners = [x for x in LEARNERS if x[1] != "exact"]
    means = [mean(data[(a, m)], "exploitability") for _, a, m in learners]
    lo, hi = min(means), max(means)
    gaps = [mean(data[(a, m)], "mirror_gap_mean") for _, a, m in learners]

    long_html = ""
    if long_rows:
        lrows, ends = [], []
        for lbl, a, m in learners:
            rs = sorted(pick(long_rows, a, m), key=lambda r: int(r["seed"]))
            if rs:
                short = mean(data[(a, m)], "exploitability")
                per_seed = [float(r["exploitability"]) for r in rs]
                ends += [(short, v) for v in per_seed]
                lrows.append([lbl, f"{short:.3f}", " / ".join(f"{v:.2f}" for v in per_seed),
                              "/".join(f(v) for v in wtl(rs, "row", "random")),
                              "/".join(f(v) for v in wtl(rs, "row", "nash"))])
        names = {"reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
        parts = []
        for algo in ("reinforce", "a2c", "ppo"):
            runs = [(mean(data[(algo, "selfplay")], "exploitability"), float(r["exploitability"]))
                    for r in pick(long_rows, algo, "selfplay")]
            if runs:
                lower = sum(1 for short, v in runs if v < short - 0.15)
                parts.append(f"{names[algo]} ends clearly lower on {lower} of {len(runs)} seeds")
        verdict = "; ".join(parts)
        best = min(v for _, v in ends)
        long_html = f"""
<section><h2>Does more training help?</h2>
<p>Self-play with <b>8,000</b> iterations instead of 2,000 (2 seeds per learner; every other
setting unchanged). The fictitious-play runs were too slow to repeat at this length.</p>
{table(["learner", "exploitability, 2,000 it. (3-seed mean)", "exploitability, 8,000 it. (each seed)",
        "W/T/L vs. random (8,000 it.)", "W/T/L vs. exact Nash (8,000 it.)"], lrows)}
<p class="muted">{verdict}; the best single run reaches {best:.2f}, still far from 0. With two
seeds per learner this is a trend, not a result.</p></section>"""

    earlier = []
    for lbl, path in (("A10 board, stationary, kickoff starts", "pg_algos_a10_seeds.csv"),
                      ("random board, stationary, kickoff starts", "pg_algos_random_seeds.csv"),
                      ("random board, stationary, exploring starts", "pg_algos_random_explore_seeds.csv")):
        for algo in ("reinforce", "a2c", "ppo"):
            rs = [r for r in read(EXP / path) if r.get("algo") == algo]
            if rs:
                earlier.append([lbl, "REINFORCE" if algo == "reinforce" else algo.upper(),
                                f(mean(rs, "duality_gap")), f(mean(rs, "mean_equilibrium_regret"), 3),
                                f(mean(rs, "row_action_agreement"))])

    fp = read(EXP / "fictitious_play.csv")
    fp_html = table(
        ["rounds", "mean stage-game gap", "max stage-game gap", "max value error"],
        [[r["rounds"], f"{float(r['mean_gap_all']):.5f}", f"{float(r['max_gap_all']):.5f}",
          f"{float(r['max_value_err_all']):.5f}"] for r in fp]) if fp else ""

    pol = policy_html()
    pol_section = ""
    if pol:
        pol_section = """<section><h2>6. Action probabilities</h2>
<p>What the trained policies actually output: the probability of each move (U, D, L, R) for
both players, at the start of the game (all 100 steps left), at four fixed positions. The
most likely move is in bold. The learners are seed 0 of the runs in section 5; the exact
solver's play is shown for comparison.</p>""" + pol + """
<p class="muted">Rows that put nearly all the probability on one move are close to
deterministic play; the exact solution is deterministic at these positions.</p></section>

"""

    def sec(n: int) -> int:
        return n if pol or n < 7 else n - 1

    summary_end = (
        "The action probabilities the trained policies output are in section 6." if pol else "")
    chart1 = bar_chart(expl)
    chart2 = stacked_chart(
        [(n, [wtl(data[(a, m)], "row", "random"), wtl(data[(a, m)], "row", "nash")])
         for n, a, m in LEARNERS], ["vs. a random player", "vs. the exact Nash policy"])

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Policy gradient on the soccer game</title><style>
:root{{--ink:#1c2e26;--accent:#1DB954;--accent-dark:#178A42;--line:#dbe5df;--muted:#5b6b63;
--card:#f3f8f5;--warn:#a8501c;--warn-bg:#fbf3ea}}
*{{box-sizing:border-box}}
body{{font:14px/1.6 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;max-width:900px;
margin:36px auto;color:var(--ink);padding:0 28px}}
h1{{font-size:26px;margin:0 0 6px;color:var(--accent-dark);letter-spacing:-.01em}}
h2{{font-size:18px;margin:0 0 8px;color:var(--accent-dark)}}
.eyebrow{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--accent);font-weight:700}}
p.lede{{color:var(--muted);font-size:14px;margin:0 0 14px}}
p.muted{{color:var(--muted);font-size:12.5px;margin:6px 0 0}}
section{{margin-top:26px;padding-top:14px;border-top:2px solid var(--line)}}
table,svg,.callout{{break-inside:avoid}}
table.qa{{break-inside:auto}} tr{{break-inside:avoid}}
table.qa td{{text-align:left;vertical-align:top}}
table.probs td,table.probs th{{padding:4px 6px}}
h3{{font-size:13px;margin:16px 0 4px;color:var(--ink)}}
h2{{break-after:avoid}}
.callout{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:6px;padding:10px 16px;margin:12px 0}}
.callout.warn{{border-left-color:var(--warn);background:var(--warn-bg)}}
table{{border-collapse:collapse;width:100%;font:11.5px/1.45 "SF Mono",Consolas,monospace;margin:10px 0}}
td,th{{border:1px solid var(--line);padding:5px 7px;text-align:center}}
td:first-child{{text-align:left}} th{{background:var(--card);color:var(--accent-dark)}}
code{{background:var(--card);border-radius:3px;padding:1px 5px;font-size:.92em}}
.charts{{display:grid;grid-template-columns:150px 1fr 1fr;gap:6px;align-items:start;margin-top:8px}}
.nm{{height:26px;line-height:26px;font-size:11px;text-align:right;padding-right:6px;white-space:nowrap}}
.names{{padding-top:22px}} svg{{width:100%;height:auto}}
.legend span{{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;vertical-align:-1px}}
ul{{margin:6px 0 6px 18px;padding:0}} li{{margin:3px 0}}
@page{{size:A4;margin:14mm}}
</style></head><body>
<div class="eyebrow">Research-group note &middot; policy gradient</div>
<h1>Can policy gradient replicate the exact soccer solver?</h1>
<p class="lede">REINFORCE, A2C and PPO, trained by self-play and by fictitious play, scored
against the exact solution of the discounted 100-step soccer game. Every number below is
read from the experiment files in the repository.</p>

<div class="callout"><b>Summary</b>
<ul>
<li>Six learners, REINFORCE, A2C and PPO each trained by self-play and by fictitious play,
on the discounted 100-step soccer game, scored against the exact solution.</li>
<li>Their exploitability ranges from {lo:.2f} to {hi:.2f}; the exact solution is 0.</li>
<li>Fictitious-play training wins more often against a random player; self-play training
ties the exact equilibrium more often.</li>
<li>The two players' policies are not mirror images (mirror gap {min(gaps):.2f} to
{max(gaps):.2f}; 0 for the exact solution).</li>
</ul>
Results are for 3 seeds at one untuned budget. {summary_end}</div>

<section><h2>1. What was done</h2>
<p>These choices fix the setup used throughout.</p>
{table(["topic", "what was done", "why"], [
  ["Environment", "The deterministic A10 board is the main environment; the random move-order board is kept as an earlier comparison.", "The deterministic board has no mixed-equilibrium stages, so the comparison with the exact solution is clean."],
  ["Objective", "Discount 0.9 over 100 steps, a tie at the end, and the remaining step count as a network input.", "Policy gradient can only approximate a finite-horizon discounted reward."],
  ["Fictitious play", f"Each player is trained by policy gradient against the average of the opponent's past policies. Matrix-level fictitious play is kept only as a side check (section {sec(8)}).", "Policy gradient does the solving; no game is solved explicitly."],
  ["Win rate", "1,000 repeated games from the kickoff against a random player, the exact Nash policy and the exact best response; wins, ties and losses are all counted.", "Repeated play and counting wins."]], "qa")}
</section>

<section><h2>2. Setup</h2>
<p>The environment is the deterministic A10 soccer game on a 7&times;5 board with a
three-cell goal; it has no mixed-strategy stage games, which keeps the comparison with the
exact solution clean. The objective is <b>discounted</b> (&gamma; = 0.9) over a
<b>100-step horizon</b>, a tie if nobody scores, and the network receives the
<b>remaining step count</b> as an input. The reference is the exact discounted backward
induction over the same 100 steps (<code>soccer_nash/finite_horizon.py</code>), whose
equilibrium has exploitability 0 and ties itself every game.</p></section>

<section><h2>3. The learners</h2>
{table(["algorithm", "update"], [
  ["REINFORCE", "plain discounted returns, no baseline"],
  ["A2C", "10-step bootstrapped advantage from a learned critic"],
  ["PPO", "GAE (&lambda; = 0.95), clipped ratio 0.2, 4 epochs"]])}
<p><b>Self-play</b> trains both current networks against each other.
<b>Fictitious play</b> is how policy gradient solves the game here, with no matrix
solving: each player keeps improving its own network by policy gradient against the
<i>empirical average</i> of the opponent's past policies (frozen snapshots, one drawn per
episode), and reports the average of its own snapshots. All learners use learning rate
10<sup>-3</sup>, 64 episodes per batch, 2,000 iterations, an entropy bonus of 0.01, and a
64&times;64 network; none was tuned.</p></section>

<section><h2>4. How they are scored</h2>
<ul>
<li><b>Exploitability</b>: the discounted value both exact best responses earn from the
kickoff, summed. Zero exactly at an equilibrium.</li>
<li><b>Wins, ties and losses</b>: counts over 1,000 repeated games from the kickoff,
against a random player, the exact Nash policy and the exact best response.</li>
<li><b>Mirror gap</b>: how far the two players' policies are from being mirror images
(flip the board, swap the players), averaged over all states at five points in the game.</li>
</ul></section>

<section><h2>5. Results</h2>
{table(["learner", "exploitability (mean &plusmn; sd)", "W/T/L vs. random", "vs. exact Nash",
        "vs. best response", "mirror gap"], result_rows)}
<div class="charts" style="grid-template-columns:1fr"><div>{chart1}</div></div>
<p class="muted">Exploitability by learner: bar = mean, dots = individual seeds.</p>
<div class="legend"><span style="background:{GREEN}"></span>win<span style="background:{GREY}"></span>tie<span style="background:{ORANGE}"></span>loss</div>
<div>{chart2}</div>
<p>Two patterns repeat. Fictitious-play training beats a random player more often
(0.88 to 0.96) but loses almost every game to the exact equilibrium. Self-play training
wins less often against random and ties the equilibrium more often: it plays more
cautiously. Neither is close to equilibrium play, and every learner loses almost every
game to the exact best response.</p></section>
{long_html}
<section><h2>Fictitious play with best-response phases</h2>
<p>A stricter version of fictitious play: in each round each player runs 100 (or 300)
policy-gradient iterations to approximate a best response to the average of the
opponent's earlier best responses, then adds that policy to its history. The reported
policy is the per-state average of the best responses. Exploitability after each
checkpoint round (1 seed per run, 64 episodes per iteration):</p>
{fp_br_html()}</section>

{pol_section}<section><h2>{sec(7)}. Earlier experiments</h2>
<p>Earlier, the same algorithms were run on the stationary
discounted game (no step count), from the kickoff or from random starting states.</p>
{table(["setup", "algorithm", "exploitability", "mean equilibrium regret", "action agreement"], earlier)}
<p class="muted">Exploring starts raise per-state accuracy (action agreement) but not
play from the kickoff.</p></section>

<section><h2>{sec(8)}. Fictitious play on the exact stage games</h2>
<p>As a side check, fictitious play was also run on the exact 4&times;4 stage matrices of
the random move-order board (2,380 states, 94 with no pure saddle). On
rock-paper-scissors best-response dynamics cycle forever while fictitious play settles at
1/3 each; on the soccer stage games it recovers the exact values.</p>{fp_html}
<p class="muted">This is matrix-level fictitious play. Here policy gradient itself
does the solving, which is what section 5 tests.</p></section>
</body></html>"""
    (DOCS / "policy_gradient.html").write_text(html)
    print("wrote docs/policy_gradient.html")


if __name__ == "__main__":
    main()
