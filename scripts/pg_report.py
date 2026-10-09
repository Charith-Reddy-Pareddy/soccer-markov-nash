# ruff: noqa: E501
"""Build docs/policy_gradient.html (and, via ``make pg-pdf``, the PDF). Every number comes from
``pg_site_data.build()`` and the saved policy outputs, the same sources as the site's tab.

    python scripts/pg_report.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP, DOCS = ROOT / "experiments", ROOT / "docs"
GREEN, GREY, ORANGE = "#1DB954", "#c3cec8", "#a8501c"


def load_data() -> dict:
    spec = importlib.util.spec_from_file_location(
        "pg_site_data", pathlib.Path(__file__).with_name("pg_site_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def pct(v: float) -> str:
    return f"{v:.0%}"


def name(entry: dict) -> str:
    return f"{entry['label']}, {entry['training']}"


def win_cell(t) -> str:
    return f'<b>{t[0]:.0%}</b><span class="sub">tie {t[1]:.0%} &middot; loss {t[2]:.0%}</span>'


def table(head, rows, cls="") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><tr>{th}</tr>{body}</table>'


# ------------------------------------------------------------------------------------ charts
def br_bars(rows, reference, width=660) -> str:
    """Win rate against the best response; the dashed line is the exact solver."""
    row_h, left, right = 26, 210, 54
    h = row_h * len(rows) + 40
    x = lambda v: left + v * (width - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x(g):.1f}" y1="22" x2="{x(g):.1f}" y2="{h - 22}" stroke="#dbe5df"/>'
                   f'<text x="{x(g):.1f}" y="{h - 8}" text-anchor="middle" fill="#5b6b63">{g:.0%}</text>')
    for i, (label, win) in enumerate(rows):
        y = 26 + i * row_h
        out.append(f'<text x="{left - 8}" y="{y + 13}" text-anchor="end" fill="#1c2e26">{label}</text>'
                   f'<rect x="{left}" y="{y + 3}" width="{max(x(win) - left, 0):.1f}" height="14" '
                   f'fill="{GREEN}" opacity=".85"/>'
                   f'<text x="{width - right + 10}" y="{y + 14}" fill="#1c2e26" font-weight="700">{win:.0%}</text>')
    out.append(f'<line x1="{x(reference):.1f}" y1="18" x2="{x(reference):.1f}" y2="{h - 22}" stroke="{ORANGE}" '
               f'stroke-width="2" stroke-dasharray="5 4"/><text x="{x(reference):.1f}" y="12" text-anchor="middle" '
               f'fill="{ORANGE}" font-weight="700">exact solver {reference:.0%}</text></svg>')
    return "".join(out)


def stacked_chart(items, titles, width=660) -> str:
    """Win / tie / loss bars, one panel per opponent, sharing a label column."""
    row_h, left, gap = 26, 210, 24
    n = len(titles)
    panel = (width - left - gap * (n - 1) - 10) / n
    h = row_h * len(items) + 30
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for k, title in enumerate(titles):
        out.append(f'<text x="{left + k * (panel + gap)}" y="12" fill="#5b6b63" font-weight="700">{title}</text>')
    for i, (label, panels) in enumerate(items):
        y = 22 + i * row_h
        out.append(f'<text x="{left - 8}" y="{y + 12}" text-anchor="end" fill="#1c2e26">{label}</text>')
        for k, trip in enumerate(panels):
            pos = 0.0
            for frac, col in zip(trip, (GREEN, GREY, ORANGE)):
                out.append(f'<rect x="{left + k * (panel + gap) + pos * panel:.1f}" y="{y}" '
                           f'width="{frac * panel:.1f}" height="16" fill="{col}"/>')
                pos += frac
    out.append("</svg>")
    return "".join(out)


VARIANT_COLOURS = {"baseline": GREEN, "shared": GREY, "trimmed": ORANGE}


def variant_bars(rows, reference, width=660, field="") -> str:
    """Win rate against the best response per variant: a bar for the mean, a dot per seed."""
    left, right, group, bar = 170, 20, 70, 16
    h = group * len(rows) + 48
    x = lambda v: left + v * (width - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x(g):.1f}" y1="22" x2="{x(g):.1f}" y2="{h - 22}" stroke="#dbe5df"/>'
                   f'<text x="{x(g):.1f}" y="{h - 8}" text-anchor="middle" fill="#5b6b63">{g:.0%}</text>')
    for i, r in enumerate(rows):
        y0 = 28 + i * group
        out.append(f'<text x="{left - 8}" y="{y0 + 24}" text-anchor="end" fill="#1c2e26">'
                   f'{r["label"]}, {r["training"]}</text>')
        for n, key in enumerate(VARIANT_COLOURS):
            v, y = r[key], y0 + n * (bar + 2)
            mean, runs = v[f"{field}mean"], v[f"{field}runs"]
            out.append(f'<rect x="{left}" y="{y}" width="{max(x(mean) - left, 0):.1f}" height="{bar}" '
                       f'fill="{VARIANT_COLOURS[key]}" opacity=".85"/>')
            out += [f'<circle cx="{x(w):.1f}" cy="{y + bar / 2}" r="2.5" fill="#1c2e26"/>' for w in runs]
            out.append(f'<text x="{x(min(max(runs), 0.93)) + 8:.1f}" y="{y + 12}" fill="#1c2e26" '
                       f'font-weight="700">{mean:.0%}</text>')
    out.append(f'<line x1="{x(reference):.1f}" y1="18" x2="{x(reference):.1f}" y2="{h - 22}" stroke="{ORANGE}" '
               f'stroke-width="2" stroke-dasharray="5 4"/><text x="{x(reference):.1f}" y="12" text-anchor="middle" '
               f'fill="{ORANGE}" font-weight="700">exact solver {reference:.0%}</text></svg>')
    return "".join(out)


def variant_section(res: dict) -> str:
    both = res.get("variants")
    if not both or not both.get("random") or not both.get("deterministic"):
        return ""
    rnd, det = both["random"], both["deterministic"]
    pts = lambda v: f"{'+' if v >= 0 else '−'}{abs(round(v * 100))}"  # noqa: E731
    change = lambda rows, key, f: ", ".join(pts(r[key][f] - r["baseline"][f]) for r in rows)  # noqa: E731
    spread = lambda rows, f: round(100 * max(  # noqa: E731
        max(x[f]) - min(x[f]) for r in rows for x in (r["baseline"], r["shared"])))
    pick = {(r["label"], r["training"]): r for r in rnd}
    a2c_s, a2c_f, ppo = (pick[("A2C", "standard")], pick[("A2C", "fictitious play")],
                         pick[("PPO", "standard")])
    dpp = next(r for r in det if (r["label"], r["training"]) == ("PPO", "standard"))
    best = max(x["mean"] for r in det for x in (r["baseline"], r["shared"], r["trimmed"]))
    order = "(A2C standard, A2C fictitious play, PPO standard, PPO fictitious play)"
    return f"""<section><h2>9. Network sharing and trimming</h2>
<p>A2C and PPO, three seeds each, on both boards. <b>Shared network</b>: both players' policies are two output
slices of one network, so each player's update also moves the other. <b>Last 10 steps left out</b>: the final 10 of the 100
steps of every episode are not used in the loss, though they still count in the returns of earlier steps. Bars are the mean over
seeds; dots are single seeds. <span style="color:{GREEN}">■</span> separate networks,
<span style="color:{GREY}">■</span> one shared network, <span style="color:{ORANGE}">■</span> last 10 steps left out.</p>
<h3>Random move-order board: win rate against the best response</h3>
<div>{variant_bars(rnd, res["random"]["exact"]["vs_best_response"][0])}</div>
<h3>Deterministic board: games tied with the exact Nash policy</h3>
<div>{variant_bars(det, res["exact"]["vs_nash"][1], field="tie_nash_")}</div>
<p>Random board: against separate networks, a shared network changes the win rate by {change(rnd, "shared", "mean")} points
{order}; single seeds of the same setup differ by up to {spread(rnd, "runs")} points. Leaving out the last 10 steps
changes A2C by {pts(a2c_s["trimmed"]["mean"] - a2c_s["baseline"]["mean"])} and
{pts(a2c_f["trimmed"]["mean"] - a2c_f["baseline"]["mean"])} points, but PPO with standard training stops scoring: it wins
{pct(ppo["trimmed"]["vs_random"][0])} of games against a random player, against {pct(ppo["baseline"]["vs_random"][0])} before.
No variant gets close to the exact solver.</p>
<p>Deterministic board: no variant wins more than {pct(best)} of games against the best response, as for the baseline (the exact
solver wins 0%, since it ties itself). The share of games tied with the exact Nash policy changes by
{change(det, "shared", "tie_nash_mean")} points with a shared network and by {change(det, "trimmed", "tie_nash_mean")} points
with the last 10 steps left out {order}; single seeds of the same setup differ by up to {spread(det, "tie_nash_runs")} points. A tie does not mean equilibrium
play: PPO with standard training ties all games once the last 10 steps are left out, but wins only
{pct(dpp["trimmed"]["vs_random"][0])} of games against a random player, against {pct(dpp["baseline"]["vs_random"][0])}
before.</p></section>"""


def rps_plot(d: dict, width=660, height=250) -> str:
    lm, rm, tm, bm = 56, 20, 50, 40
    x = lambda k: lm + k / (d["rounds"] - 1) * (width - lm - rm)  # noqa: E731
    y = lambda v: tm + (1 - v) * (height - tm - bm)  # noqa: E731
    path = lambda vals: " ".join(f"{'L' if k else 'M'}{x(k):.1f},{y(v):.1f}" for k, v in enumerate(vals))  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for g, lab in ((0, "0"), (1 / 3, "1/3"), (2 / 3, "2/3"), (1, "1")):
        dash = ' stroke-dasharray="4 3"' if lab == "1/3" else ""
        out.append(f'<line x1="{lm}" y1="{y(g):.1f}" x2="{width - rm}" y2="{y(g):.1f}" stroke="#dbe5df"{dash}/>'
                   f'<text x="{lm - 8}" y="{y(g) + 4:.1f}" text-anchor="end" fill="#5b6b63">{lab}</text>')
    out.append(f'<path d="{path(d["best_response"])}" fill="none" stroke="{ORANGE}" stroke-width="1.6" opacity=".85"/>'
               f'<path d="{path(d["average"])}" fill="none" stroke="{GREEN}" stroke-width="2.6"/>'
               f'<text x="{width - rm}" y="16" text-anchor="end" fill="{ORANGE}">best-response dynamics: the current play flips every round</text>'
               f'<text x="{width - rm}" y="32" text-anchor="end" fill="{GREEN}" font-weight="700">fictitious play: the average settles near 1/3</text>'
               f'<text x="{lm}" y="{height - 8}" fill="#5b6b63">round 1</text>'
               f'<text x="{width - rm}" y="{height - 8}" text-anchor="end" fill="#5b6b63">round {d["rounds"]}</text></svg>')
    return "".join(out)


# --------------------------------------------------------------------------- policy tables
def policy_tables(d: dict) -> str:
    acts = d["actions"]

    def cells(probs):
        top = max(probs)
        return "".join(f"<td><b>{v * 100:.0f}%</b></td>" if v == top else f"<td>{v * 100:.0f}%</td>"
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


def policy_html() -> str:
    path = EXP / "pg_policy_outputs.json"
    return policy_tables(json.loads(path.read_text())) if path.exists() else ""


# --------------------------------------------------------------------------------- sections
def board_section(title: str, note: str, board: dict) -> str:
    exact, learners = board["exact"], board["learners"]
    br = [("Exact solver", exact["vs_best_response"][0])] + [(name(x), x["vs_best_response"][0]) for x in learners]
    stack = [("Exact solver", [exact["vs_random"], exact["vs_nash"], exact["vs_best_response"]])] + [
        (name(x), [x["vs_random"], x["vs_nash"], x["vs_best_response"]]) for x in learners]
    ties = {t: [x["vs_nash"][1] for x in learners if x["training"] == t] for t in ("standard", "fictitious play")}
    tie_sentence = (f" Against the exact Nash policy the standard learners tie {pct(min(ties['standard']))} to "
                    f"{pct(max(ties['standard']))} of games and the fictitious-play learners "
                    f"{pct(min(ties['fictitious play']))} to {pct(max(ties['fictitious play']))}."
                    if max(ties["standard"]) >= 0.01 else "")
    exact_critic = {x["training"]: x["vs_nash"][1] for x in learners if x["label"] == "A2C, exact critic"}
    plain = {x["training"]: x["vs_nash"][1] for x in learners if x["label"] == "A2C"}
    critic_sentence = (f" A2C with the exact critic ties the exact Nash policy in {pct(exact_critic['standard'])} of "
                       f"games, against {pct(plain['standard'])} for A2C with a learned critic."
                       if exact_critic else "")
    wins = [x["vs_best_response"][0] for x in learners]
    return f"""<h3>{title}</h3><p class="muted">{note}</p>
<p><b>Win rate against the best response</b> (mean of {learners[0]['seeds']} seeds; dashed line: the exact
solver). The exact solver wins {pct(exact['vs_best_response'][0])}; the learners win {pct(min(wins))} to {pct(max(wins))}.</p>
<div>{br_bars(br, exact['vs_best_response'][0])}</div>
<p><b>Wins, ties and losses against each opponent</b>.{tie_sentence}{critic_sentence}</p>
<div class="legend"><span style="background:{GREEN}"></span>win<span style="background:{GREY}"></span>tie<span style="background:{ORANGE}"></span>loss</div>
<div>{stacked_chart(stack, ["vs. a random player", "vs. the exact Nash policy", "vs. the best response"])}</div>"""


def build_html(res: dict) -> str:
    det, rnd, cont = res, res["random"], res.get("continuing")
    mixed = res["mixed"]
    boards = [("The deterministic board (A10)", "Players move simultaneously and the carrier wins every contested square, "
               "so the exact solution is pure everywhere and ties itself.", det),
              ("The random move-order board", "Moves are applied in a random order each step; the exact solution "
               "mixes at 94 stage games.", rnd)]
    if cont:
        boards.append(("The continuing game", "Play restarts after every goal and runs for exactly 100 steps; a win is "
                       "more goals than the opponent. Also shows A2C with the exact solver's values as a frozen critic.", cont))
    summary_rows = []
    labels = [name(x) for x in det["learners"]]
    for k, lab in enumerate(labels):
        row = [lab, win_cell(det["learners"][k]["vs_best_response"]), win_cell(rnd["learners"][k]["vs_best_response"])]
        if cont:
            hit = next((x for x in cont["learners"] if name(x) == lab), None)
            row.append(win_cell(hit["vs_best_response"]) if hit else "")
        summary_rows.append(row)
    ref = ["Exact solver", win_cell(det["exact"]["vs_best_response"]), win_cell(rnd["exact"]["vs_best_response"])]
    if cont:
        ref.append(win_cell(cont["exact"]["vs_best_response"]))
    head = ["learner", "deterministic board", "random board"] + (["continuing game"] if cont else [])
    long_rows = [[f"{x['label']}, {x['training']}", win_cell(x["short_vs_nash"]), win_cell(x["vs_nash"]),
                  win_cell(x["vs_random"])] for x in res["longer"]]
    rounds = sorted({c["round"] for r in res["fp_br"] for c in r["checkpoints"]})
    fp_rows = [[f"{r['label']}, {r['best_response_iterations']} iterations per best response"] + [
        next((pct(c["win_vs_best_response"]) for c in r["checkpoints"] if c["round"] == k), "") for k in rounds]
        for r in res["fp_br"]]
    mixed_rows = [["Exact solver", win_cell(mixed["exact"]["vs_nash"]), win_cell(mixed["exact"]["vs_best_response"]), "0", ""]]
    mixed_rows += [[name(x), win_cell(x["vs_nash"]), win_cell(x["vs_best_response"]), f"{x['tv_row']:.2f}",
                    pct(x["share_mixing_row"])] for x in mixed["learners"]]
    pg_br = [x["vs_best_response"][0] for x in rnd["learners"]]
    pg_rand = [x["vs_random"][0] for x in det["learners"]]
    mx = [x["vs_nash"][0] for x in mixed["learners"]]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Policy gradient on the soccer game</title><style>
:root{{--ink:#1c2e26;--accent:#1DB954;--accent-dark:#178A42;--line:#dbe5df;--muted:#5b6b63;--card:#f3f8f5}}
*{{box-sizing:border-box}}
body{{font:14px/1.6 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;max-width:900px;margin:36px auto;color:var(--ink);padding:0 28px}}
h1{{font-size:26px;margin:0 0 6px;color:var(--accent-dark);letter-spacing:-.01em}}
h2{{font-size:18px;margin:0 0 8px;color:var(--accent-dark);break-after:avoid}}
h3{{font-size:13px;margin:16px 0 4px;color:var(--ink)}}
.eyebrow{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--accent);font-weight:700}}
p.lede{{color:var(--muted);font-size:14px;margin:0 0 14px}}
p.muted{{color:var(--muted);font-size:12.5px;margin:6px 0 0}}
section{{margin-top:26px;padding-top:14px;border-top:2px solid var(--line)}}
.callout{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px;padding:10px 16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font:11.5px/1.45 "SF Mono",Consolas,monospace;margin:10px 0}}
td,th{{border:1px solid var(--line);padding:5px 7px;text-align:center}}
td:first-child{{text-align:left}} th{{background:var(--card);color:var(--accent-dark)}}
table,svg,.callout{{break-inside:avoid}} tr{{break-inside:avoid}}
table.text td{{text-align:left;vertical-align:top}}
table.probs td,table.probs th{{padding:4px 6px}}
.sub{{display:block;font-size:10px;color:var(--muted);font-weight:400}}
svg{{width:100%;height:auto}}
.legend span{{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;vertical-align:-1px}}
ul{{margin:6px 0 6px 18px;padding:0}} li{{margin:3px 0}}
@page{{size:A4;margin:14mm}}
</style></head><body>
<div class="eyebrow">Policy gradient &middot; soccer game</div>
<h1>Policy gradient on the soccer game</h1>
<p class="lede">REINFORCE, A2C and PPO, trained the standard way and by fictitious play, scored by one number:
how often the trained player wins when the opponent is the best response to it.</p>

<div class="callout"><b>Summary</b>
<ul>
<li>Against the best response, the exact solver wins {pct(rnd['exact']['vs_best_response'][0])} of games on the random board;
the six learners win {pct(min(pg_br))} to {pct(max(pg_br))}. On the deterministic board the exact solver ties itself, so it wins 0%.</li>
<li>Against a random player the learners win {pct(min(pg_rand))} to {pct(max(pg_rand))} of games.</li>
<li>Starting from the {mixed['mixed_states']} states where the exact answer has to mix, they win {pct(min(mx))} to {pct(max(mx))}
against the exact Nash policy; the exact solver wins {pct(mixed['exact']['vs_nash'][0])}.</li>
<li>Fictitious-play training wins more often against a random player; neither way of training reaches the exact
solver's rate against the best response.</li>
</ul>3 seeds per learner, 1,000 games per opponent.</div>

<section><h2>1. How to read the numbers</h2>
{table(["number", "what it is"], [
 ["Win rate against the best response", "The share of 1,000 games the trained player wins when the opponent plays the exact best response to it, the move that is best against exactly what our player will do. A player who always does the same thing loses every game; the exact solver mixes optimally, so its rate is the ceiling to compare with."],
 ["Win rate against the exact Nash policy", "The same, against the exact equilibrium player."],
 ["Win rate against a random player", "A sanity check: every learner should beat a player who moves at random."],
 ["A win", "The player scores while the opponent does not. On the first two boards the first goal ends the game; in the continuing game play restarts after a goal and a win is more goals than the opponent in 100 steps. Games with no goal, or equal goals, are ties."]], "text")}
<p class="muted">Exploitability, the best-response player's gain, is not reported: it measures the attacker, and it is the player we train that we want to judge.</p></section>

<section><h2>2. How it is implemented</h2>
{table(["setting", "value"], [
 ["Game", "7&times;5 board, goal rows 1&ndash;3. Fixed 100 steps, discount 0.9, and the number of steps left is a network input."],
 ["Returns", "REINFORCE: the discounted return from each step to the end of the episode, no baseline. A2C: 10-step bootstrapped advantage from a learned critic. A2C with the exact critic: the critic is the exact solver's value, frozen, so the advantage is the exact Q(s, a<sub>0</sub>, a<sub>1</sub>) minus V(s). PPO: GAE (&lambda; 0.95), ratio clipped at 0.2, 4 epochs."],
 ["Updates", "Every iteration: 64 episodes of 100 steps, then one gradient step per player on all of that data (four for PPO). 2,000 iterations. Learning rate 10<sup>-3</sup>, entropy bonus 0.01, 64&times;64 network, softmax output, a separate network for each player. No mini-batches, no replay."],
 ["Standard", "Both players' current networks play each other and keep updating: real-time best responses."],
 ["Fictitious play", "Each player trains against a frozen snapshot of the other, drawn at random from all its past snapshots (one every 5 iterations); the policy it reports is the average of its own snapshots."]], "text")}</section>

<section><h2>3. Why fictitious play</h2>
<p>On rock-paper-scissors, answering the opponent's latest move makes the play flip between rock, paper and
scissors for ever. Answering the average of everything it has played converges to one third each.</p>
<div>{rps_plot(res['rps'])}</div></section>

<section><h2>4. Results</h2>
{''.join(board_section(t, n, b) for t, n, b in boards)}
<h3>Win rate against the best response, all boards</h3>
{table(head, [ref, *summary_rows])}</section>

<section><h2>5. Mixed states of the random board</h2>
<p>At {mixed['mixed_states']} states the exact equilibrium has to randomize. Each learner starts
{mixed['games_per_start']} games from every one of them. Distance is the total-variation distance between the
learner's probabilities and the exact mix at the first step (0 = identical, 1 = no overlap); "player 0 mixes"
is the share of states where it puts less than 90% on its top move.</p>
{table(["learner", "wins vs. exact Nash", "wins vs. best response", "distance from the exact mix", "player 0 mixes"], mixed_rows)}
<h3>Three of these states in full</h3>
{policy_tables(mixed['policy_outputs'])}</section>

<section><h2>6. Action probabilities</h2>
<p>The probability each trained network gives to every move, at four fixed positions of the deterministic board
(seed 0), next to the exact solver's move. The most likely move is in bold.</p>
{policy_html()}</section>

<section><h2>7. More training</h2>
<p>Standard training at 8,000 instead of 2,000 iterations (two seeds each), deterministic board.</p>
{table(["learner", "wins vs. exact Nash, 2,000 iterations", "wins vs. exact Nash, 8,000 iterations", "wins vs. random, 8,000 iterations"], long_rows)}</section>

<section><h2>8. Fictitious play with long best responses</h2>
<p>Each round a player trains for 100 (or 300) iterations against the average of its opponent's earlier best
responses, then adds the result to its history. Win rate against the best response at each checkpoint round (one seed per run).</p>
{table(["run"] + [f"round {k}" for k in rounds], fp_rows)}</section>

{variant_section(res)}
</body></html>"""


def main() -> None:
    (DOCS / "policy_gradient.html").write_text(build_html(load_data()))
    print("wrote docs/policy_gradient.html")


if __name__ == "__main__":
    main()
