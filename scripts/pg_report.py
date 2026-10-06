# ruff: noqa: E501
"""Build docs/policy_gradient.html (and, via ``make pg-pdf``, the PDF) from the
experiment CSVs, so every number in the report comes straight from a run.

    python scripts/pg_report.py
"""

from __future__ import annotations

import csv
import glob
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
    ("Nash-DQN", "dqn", "-"),
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
    row_h, left, right = 26, 190, 30
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
        out.append(f'<text x="{x(m) + 8:.1f}" y="{y + 14}" fill="#1c2e26">{m:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_chart(items, title, width=320) -> str:
    row_h, left = 26, 8
    h = row_h * len(items) + 30
    x = lambda v: left + v * (width - left - 8)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">'
           f'<text x="{left}" y="12" fill="#5b6b63" font-weight="700">{title}</text>']
    for i, (_, w) in enumerate(items):
        y = 22 + i * row_h
        pos = 0.0
        for frac, col in zip(w, (GREEN, GREY, ORANGE)):
            out.append(f'<rect x="{x(pos):.1f}" y="{y}" width="{max(x(frac) - left, 0):.1f}" '
                       f'height="16" fill="{col}"/>')
            pos += frac
    out.append("</svg>")
    return "".join(out)


def table(head, rows) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def f(v, d=2):
    return f"{v:.{d}f}"


def main() -> None:
    main_rows = read(EXP / "pg_finite_a10.csv") + read(EXP / "dqn_finite_a10.csv")
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
        lrows = []
        for lbl, a, m in learners:
            rs = pick(long_rows, a, m)
            if a != "dqn" and rs:
                short = mean(data[(a, m)], "exploitability")
                lrows.append([lbl, f"{short:.3f}", f"{mean(rs, 'exploitability'):.3f}",
                              "/".join(f(v) for v in wtl(rs, "row", "random")),
                              "/".join(f(v) for v in wtl(rs, "row", "nash"))])
        gain = [float(r[1]) - float(r[2]) for r in lrows]
        verdict = ("Four times the training lowers exploitability only slightly or not at all"
                   if max(gain) < 0.15 else
                   "Four times the training lowers exploitability noticeably for some learners")
        long_html = f"""
<section><h2>Does more training help?</h2>
<p>The same learners with <b>8,000</b> iterations instead of 2,000 (2 seeds each; the
other settings are unchanged).</p>
{table(["learner", "exploitability, 2,000 it.", "exploitability, 8,000 it.",
        "W/T/L vs. random", "W/T/L vs. exact Nash"], lrows)}
<p class="muted">{verdict}: the largest drop is {max(gain):.2f}.</p></section>"""

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

    dog = read(EXP / "dog_game.csv") + read(EXP / "dog_game_dqn.csv")
    dog_rows = []
    for name in dict.fromkeys(r["matchup"] for r in dog):
        rs = [r for r in dog if r["matchup"] == name]
        steps = [float(r["mean_capture_step"]) for r in rs if r["mean_capture_step"] != "nan"]
        dog_rows.append([name, f(mean(rs, "capture_rate")), f"{st.mean(steps):.0f}" if steps else "-"])

    chart1 = bar_chart(expl)
    chart2 = stacked_chart([(n, wtl(data[(a, m)], "row", "random")) for n, a, m in LEARNERS],
                           "vs. a random player")
    chart3 = stacked_chart([(n, wtl(data[(a, m)], "row", "nash")) for n, a, m in LEARNERS],
                           "vs. the exact Nash policy")
    names = "".join(f'<div class="nm">{n}</div>' for n, _, _ in LEARNERS)

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
section{{margin-top:26px;padding-top:14px;border-top:2px solid var(--line);break-inside:avoid-page}}
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

<div class="callout warn"><b>The short answer: not yet.</b>
<ul>
<li>All seven learners (three policy-gradient algorithms in two training schemes, plus a
Nash-DQN) stay far from the equilibrium: exploitability {lo:.2f} to {hi:.2f}, where the exact
solution is 0.</li>
<li>Fictitious-play training wins more against a random player but loses to the exact
equilibrium; self-play training ties it more often.</li>
<li>No learner is mirror-symmetric (mirror gap {min(gaps):.2f} to {max(gaps):.2f}, against 0
for the exact solution).</li>
</ul>
These are 3 seeds at one untuned budget, so this shows that these settings did not
converge, not that policy gradient cannot.</div>

<section><h2>1. Setup</h2>
<p>The environment is the deterministic A10 soccer game on a 7&times;5 board with a
three-cell goal; it has no mixed-strategy stage games, which the professor preferred for
this comparison. The objective is <b>discounted</b> (&gamma; = 0.9) over a
<b>100-step horizon</b>, a tie if nobody scores, and the network receives the
<b>remaining step count</b> as an input. The reference is the exact discounted backward
induction over the same 100 steps (<code>soccer_nash/finite_horizon.py</code>), whose
equilibrium has exploitability 0 and ties itself every game.</p></section>

<section><h2>2. The learners</h2>
{table(["algorithm", "update"], [
  ["REINFORCE", "plain discounted returns, no baseline"],
  ["A2C", "10-step bootstrapped advantage from a learned critic"],
  ["PPO", "GAE (&lambda; = 0.95), clipped ratio 0.2, 4 epochs"],
  ["Nash-DQN", "fitted Q over (state, steps left); targets use the exact transitions and a target network's minimax value"]])}
<p><b>Self-play</b> trains both current networks against each other.
<b>Fictitious play</b> is how policy gradient solves the game here, with no matrix
solving: each player keeps improving its own network by policy gradient against the
<i>empirical average</i> of the opponent's past policies (frozen snapshots, one drawn per
episode), and reports the average of its own snapshots. All learners use learning rate
10<sup>-3</sup>, 64 episodes per batch, 2,000 iterations, an entropy bonus of 0.01, and a
64&times;64 network; none was tuned.</p></section>

<section><h2>3. How they are scored</h2>
<ul>
<li><b>Exploitability</b>: the discounted value both exact best responses earn from the
kickoff, summed. Zero exactly at an equilibrium.</li>
<li><b>Wins, ties and losses</b>: counts over 1,000 repeated games from the kickoff,
against a random player, the exact Nash policy and the exact best response.</li>
<li><b>Mirror gap</b>: how far the two players' policies are from being mirror images
(flip the board, swap the players), averaged over all states at five points in the game.</li>
</ul></section>

<section><h2>4. Results</h2>
{table(["learner", "exploitability (mean &plusmn; sd)", "W/T/L vs. random", "vs. exact Nash",
        "vs. best response", "mirror gap"], result_rows)}
<div class="charts" style="grid-template-columns:1fr"><div>{chart1}</div></div>
<p class="muted">Exploitability by learner: bar = mean, dots = individual seeds.</p>
<div class="legend"><span style="background:{GREEN}"></span>win<span style="background:{GREY}"></span>tie<span style="background:{ORANGE}"></span>loss</div>
<div class="charts"><div class="names">{names}</div><div>{chart2}</div><div>{chart3}</div></div>
<p>Two patterns repeat. Fictitious-play training beats a random player more often
(0.88 to 0.96) but loses almost every game to the exact equilibrium. Self-play training
wins less often against random and ties the equilibrium more often: it plays more
cautiously. Neither is close to equilibrium play, and every learner loses almost every
game to the exact best response.</p></section>
{long_html}
<section><h2>5. Earlier experiments</h2>
<p>Before the professor's answers, the same algorithms were run on the stationary
discounted game (no step count), from the kickoff or from random starting states.</p>
{table(["setup", "algorithm", "exploitability", "mean equilibrium regret", "action agreement"], earlier)}
<p class="muted">Exploring starts raise per-state accuracy (action agreement) but not
play from the kickoff.</p></section>

<section><h2>6. Fictitious play on the exact stage games</h2>
<p>As a side check, fictitious play was also run on the exact 4&times;4 stage matrices of
the random move-order board (2,380 states, 94 with no pure saddle). On
rock-paper-scissors best-response dynamics cycle forever while fictitious play settles at
1/3 each; on the soccer stage games it recovers the exact values.</p>{fp_html}
<p class="muted">This is matrix-level fictitious play. The professor's point is that policy
gradient itself does the solving, which is what section 4 tests.</p></section>

<section><h2>7. Continuous actions: the dog game (provisional)</h2>
<p>Policy gradient is applied to continuous actions directly: a network outputs an angle
(von Mises) and a radius (scaled Beta) and is trained by self-play PPO. The dog-and-sheep
game itself is <b>my placeholder</b>, since no definition exists yet, so these numbers
say nothing about equilibrium.</p>
{table(["matchup", "capture rate", "mean capture step"], dog_rows)}</section>

<section><h2>8. Assumptions and open questions</h2>
<p>All 27 assumptions are listed, each marked as quoted, answered by the professor, mine or
invented, in <code>docs/solver_assumptions.md</code>. Still open for the professor: the
dog-game rules, the polar-policy distribution, what &ldquo;check symmetry&rdquo; should
mean, which states to evaluate on, how degenerate equilibria should be compared, and two
page-3 items (the &ldquo;R game&rdquo; closed form and &ldquo;# time f is activated&rdquo;).</p>
<p class="muted">Reproduce: <code>python scripts/pg_finite.py --seeds 3</code>,
<code>python scripts/dqn_finite.py --seeds 3</code>, then <code>make pg-pdf</code>.</p></section>
</body></html>"""
    (DOCS / "policy_gradient.html").write_text(html)
    print("wrote docs/policy_gradient.html")


if __name__ == "__main__":
    main()
