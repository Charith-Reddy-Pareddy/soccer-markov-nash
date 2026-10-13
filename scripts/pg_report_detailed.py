# ruff: noqa: E501
"""Build the detailed report docs/policy_gradient_detailed.html (the source of its PDF): the full
set of tables, plots and updates. ``pg_report.py`` builds the short report. Every number comes from
``pg_site_data.build()`` and the saved policy outputs, the same sources as the site's tab.

    python scripts/pg_report_detailed.py
"""

from __future__ import annotations

import html
import importlib.util
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("pg_report_short", pathlib.Path(__file__).with_name("pg_report.py"))
_short = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_short)
before_after_chart, pair_groups = _short.before_after_chart, _short.pair_groups
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


def rng(lo: float, hi: float) -> str:
    """A range of shares: "0%" when the ends match, else "2% to 5%"."""
    return pct(lo) if pct(lo) == pct(hi) else f"{pct(lo)} to {pct(hi)}"


def wins_clause(best: float) -> str:
    """What follows "no variant wins": "any games" or "more than 2% of games"."""
    return "any games" if best < 0.005 else f"more than {pct(best)} of games"


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


def variant_bars(rows, reference, width=660, field="", keys=None) -> str:
    """Win rate against the best response per variant: a bar for the mean, a dot per seed."""
    left, right, bar = 170, 20, 16
    group = len(keys or VARIANT_COLOURS) * (bar + 2) + 16
    h = group * len(rows) + 48
    x = lambda v: left + v * (width - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x(g):.1f}" y1="22" x2="{x(g):.1f}" y2="{h - 22}" stroke="#dbe5df"/>'
                   f'<text x="{x(g):.1f}" y="{h - 8}" text-anchor="middle" fill="#5b6b63">{g:.0%}</text>')
    for i, r in enumerate(rows):
        y0 = 28 + i * group
        out.append(f'<text x="{left - 8}" y="{y0 + group / 2 - 6}" text-anchor="end" fill="#1c2e26">'
                   f'{r["label"]}{", " + r["training"] if r["training"] else ""}</text>')
        for n, (key, colour) in enumerate((keys or VARIANT_COLOURS).items()):
            v, y = r[key], y0 + n * (bar + 2)
            mean, runs = v[f"{field}mean"], v[f"{field}runs"]
            out.append(f'<rect x="{left}" y="{y}" width="{max(x(mean) - left, 0):.1f}" height="{bar}" '
                       f'fill="{colour}" opacity=".85"/>')
            out += [f'<circle cx="{x(w):.1f}" cy="{y + bar / 2}" r="2.5" fill="#1c2e26"/>' for w in runs]
            out.append(f'<text x="{x(min(max(runs), 0.93)) + 8:.1f}" y="{y + 12}" fill="#1c2e26" '
                       f'font-weight="700">{mean:.0%}</text>')
    out.append(f'<line x1="{x(reference):.1f}" y1="18" x2="{x(reference):.1f}" y2="{h - 22}" stroke="{ORANGE}" '
               f'stroke-width="2" stroke-dasharray="5 4"/><text x="{x(reference):.1f}" y="12" text-anchor="middle" '
               f'fill="{ORANGE}" font-weight="700">exact solver {reference:.0%}</text></svg>')
    return "".join(out)


MODE_TITLES = {"standard": "Standard: both current networks play each other",
               "fictitious": "Fictitious play: average of softmax snapshots",
               "fictitious_argmax": "Fictitious play: average of argmax snapshots"}


def rps_nn_panels(config: dict, iterations: int, width=660, panel_h=104) -> str:
    """Share of rock over training for each mode: the current network (orange) and the
    aggregate policy (green); the dashed line is the equilibrium, 1/3."""
    lm, rm, tm = 48, 150, 16
    modes = list(config["modes"])
    h = (panel_h + 22) * len(modes)
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for n, mode in enumerate(modes):
        d = config["modes"][mode]
        top = n * (panel_h + 22) + tm
        x = lambda k: lm + k / (iterations - 1) * (width - lm - rm)  # noqa: E731
        y = lambda v, top=top: top + (1 - v) * (panel_h - tm)  # noqa: E731
        path = lambda vals, x=x, y=y: " ".join(  # noqa: E731
            f"{'L' if k else 'M'}{x(k):.1f},{y(v):.1f}" for k, v in enumerate(vals))
        out.append(f'<text x="{lm}" y="{top - 4}" fill="#1c2e26" font-weight="700">{MODE_TITLES[mode]}</text>')
        for g, lab in ((0, "0"), (1 / 3, "1/3"), (1, "1")):
            dash = ' stroke-dasharray="4 3"' if lab == "1/3" else ""
            out.append(f'<line x1="{lm}" y1="{y(g):.1f}" x2="{width - rm}" y2="{y(g):.1f}" stroke="#dbe5df"{dash}/>'
                       f'<text x="{lm - 6}" y="{y(g) + 4:.1f}" text-anchor="end" fill="#5b6b63">{lab}</text>')
        out.append(f'<path d="{path(d["current"])}" fill="none" stroke="{ORANGE}" stroke-width="1" opacity=".8"/>')
        if mode != "standard":
            out.append(f'<path d="{path(d["aggregate"])}" fill="none" stroke="{GREEN}" stroke-width="2.4"/>')
        final = sum(d["final_exploitability"]) / len(d["final_exploitability"])
        out.append(f'<text x="{width - rm + 8}" y="{top + 14}" fill="#5b6b63">exploitability</text>'
                   f'<text x="{width - rm + 8}" y="{top + 30}" fill="#1c2e26" font-weight="700">{final:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def rps_nn_section(res: dict) -> str:
    data = res.get("rps_nn")
    if not data or "configs" not in data:
        return ""
    parts = []
    for name_, cfg in data["configs"].items():
        opt = f"plain gradient steps at {cfg['lr']}" if cfg["sgd"] else f"Adam at {cfg['lr']}"
        parts.append(f"<h3>{name_[0].upper() + name_[1:]}: {opt}, entropy bonus {cfg['entropy']}</h3>"
                     f"<div>{rps_nn_panels(cfg, data['iterations'])}</div>")
    ex = {n: {m: sum(v["final_exploitability"]) / len(v["final_exploitability"]) for m, v in c["modes"].items()}
          for n, c in data["configs"].items()}
    first = next(iter(ex))
    second_text = ""
    if len(ex) > 1:
        sec = list(ex)[1]
        second_text = (f"With the {sec} all three settle close to the equilibrium (standard {ex[sec]['standard']:.2f}, "
                       f"softmax {ex[sec]['fictitious']:.2f}, argmax {ex[sec]['fictitious_argmax']:.2f}): this setting damps the cycling, "
                       "so the standard networks settle too (a stabilization effect, not evidence of convergence to the unregularized equilibrium).")
    return f"""<h3>The same test with neural networks</h3>
<p>Each player is a small network with a softmax output, trained by REINFORCE on sampled games ({data['iterations']:,} iterations,
{data['batch']} games each, 3 seeds; the plots show seed 0). <b>Standard</b> trains the two current networks against each other.
<b>Fictitious play</b> trains each player against a snapshot of the other drawn from all its past snapshots and reports the average of
a player's snapshots; <b>argmax</b> first turns each snapshot into the pure policy that plays its most likely move, so the average is
how often the snapshots play each move. Orange: the current network's share of rock; green: the aggregate. Exploitability (right) is what a perfect attacker could gain against the two networks' policies, summed over both players:
0 at one third each, 2 at worst; it is the mean over seeds and the last 300 iterations.
With the soccer settings the standard networks cycle (exploitability {ex[first]['standard']:.2f}), and neither form of fictitious play reaches
one third (softmax {ex[first]['fictitious']:.2f}, argmax {ex[first]['fictitious_argmax']:.2f}). {second_text}</p>{''.join(parts)}"""


def argmax_section(res: dict) -> str:
    data = res.get("argmax")
    if not data:
        return ""
    keys = {"softmax": GREEN, "argmax": ORANGE}
    rnd, det, cont = data["random"], data["deterministic"], data.get("continuing")
    pts = lambda v: f"{'+' if v >= 0 else '−'}{abs(round(v * 100))}"  # noqa: E731
    change = lambda rows, f: ", ".join(pts(r["argmax"][f] - r["softmax"][f]) for r in rows)  # noqa: E731
    spread = lambda rows, f: round(100 * max(max(x[f]) - min(x[f]) for r in rows for x in (r["softmax"], r["argmax"])))  # noqa: E731
    cont_chart = cont_text = ""
    if cont:
        cont_chart = ("<h3>Continuing game: games tied with the exact Nash policy</h3><div>"
                      + variant_bars(cont, res["continuing"]["exact"]["vs_nash"][1], field="tie_nash_", keys=keys) + "</div>")
        top_value = max(r["argmax"]["mean"] for r in cont)
        cont_text = (f" Continuing game: the share tied with the exact Nash policy changes by {change(cont, 'tie_nash_mean')} points; "
                     f"single seeds differ by up to {spread(cont, 'tie_nash_runs')} points, and no argmax learner wins {wins_clause(top_value)} "
                     "of games against the best response.")
    return f"""<section><h2>Averaging pure policies</h2>
<p>Neural networks cannot output a mixed strategy except through their softmax, so each snapshot is made
a pure policy (the move with the highest probability), those are averaged, and each player best-responds to that average. This is fictitious play in its
original form: each snapshot is a pure best response, and the mix is how often each move was played. Both the opponent that a player trains
against and the policy it reports are averages of pure snapshots. Bars are the mean over 3 seeds and dots single seeds;
<span style="color:{GREEN}">■</span> average of softmax snapshots (the fictitious play used elsewhere),
<span style="color:{ORANGE}">■</span> average of argmax snapshots.</p>
<h3>Random move-order board: win rate against the best response</h3>
<div>{variant_bars(rnd, res["random"]["exact"]["vs_best_response"][0], keys=keys)}</div>
<h3>Deterministic board: games tied with the exact Nash policy</h3>
<div>{variant_bars(det, res["exact"]["vs_nash"][1], field="tie_nash_", keys=keys)}</div>{cont_chart}
<p>Random board: averaging argmax instead of softmax snapshots changes the win rate against the best response by {change(rnd, "mean")} points
(REINFORCE, A2C, PPO); single seeds differ by up to {spread(rnd, "runs")} points. Deterministic board: it changes the share of games tied with the
exact Nash policy by {change(det, "tie_nash_mean")} points; single seeds differ by up to {spread(det, "tie_nash_runs")} points.{cont_text}</p></section>"""


def entropy_section(res: dict) -> str:
    data = res.get("entropy")
    if not data:
        return ""
    keys = {"baseline": GREEN, "larger": ORANGE}
    pts = lambda v: f"{'+' if v >= 0 else '−'}{abs(round(v * 100))}"  # noqa: E731
    boards = [("random", "Random move-order board: win rate against the best response", "", res["random"]["exact"]["vs_best_response"][0], "mean", "Random board", "win rate against the best response"),
              ("deterministic", "Deterministic board: games tied with the exact Nash policy", "tie_nash_", res["exact"]["vs_nash"][1], "tie_nash_mean", "Deterministic board", "share tied with the exact Nash policy"),
              ("continuing", "Continuing game: games tied with the exact Nash policy", "tie_nash_", (res.get("continuing") or {}).get("exact", {}).get("vs_nash", [0, 1])[1], "tie_nash_mean", "Continuing game", "share tied with the exact Nash policy")]
    charts, sentences = [], []
    for key, title, field, ref, f, short, what in boards:
        if key not in data:
            continue
        rows = data[key]
        charts.append(f"<h3>{title}</h3><div>{variant_bars(rows, ref, field=field, keys=keys)}</div>")
        d_main = ", ".join(pts(r["larger"][f] - r["baseline"][f]) for r in rows)
        d_rand = ", ".join(pts(r["larger"]["vs_random"][0] - r["baseline"]["vs_random"][0]) for r in rows)
        sentences.append(f"{short}: the larger bonus changes the {what} by {d_main} points; against a random player the win rate changes by "
                         f"{d_rand} points (REINFORCE, A2C, PPO; standard, fictitious play, argmax).")
    by_algo = ""
    if "random" in data:
        parts = []
        for algo in ("REINFORCE", "A2C", "PPO"):
            rows = [v for v in data["random"] if v["label"] == algo]
            parts.append(f"{algo} {pts(sum(v['larger']['mean'] - v['baseline']['mean'] for v in rows) / len(rows))}")
        by_algo = (f"<p>On the random board, averaged over the three ways of training, the larger bonus moves the win rate against the "
                   f"best response by {', '.join(parts)} points; the exact solver wins {pct(res['random']['exact']['vs_best_response'][0])} "
                   "when it starts with the ball. This is an ablation and does not show convergence to the unregularized equilibrium.</p>")
    return f"""<section><h2>A larger entropy bonus</h2>
<p>On rock-paper-scissors a larger entropy bonus made the networks settle, but entropy regularization changes the optimization problem, so this is a stabilization heuristic reported next to the original setting, not a way to recover the unregularized equilibrium. Here every learner (REINFORCE, A2C, PPO), every way of
training (standard, fictitious play, fictitious play with argmax snapshots) and every board is trained again with the entropy bonus raised
from 0.01 to 0.2; nothing else changes. Bars are the mean over 3 seeds and dots single seeds;
<span style="color:{GREEN}">■</span> entropy bonus 0.01 (used elsewhere), <span style="color:{ORANGE}">■</span> entropy bonus 0.2.</p>
{"".join(charts)}
{by_algo}<p>{" ".join(sentences)}</p></section>"""


def curves_section(res: dict) -> str:
    data = res.get("curves")
    if not data:
        return ""
    parts = []
    for board, c in data.items():
        title = "Random move-order board" if board == "random" else "Deterministic board"
        win = [(f"{s['label']}, {s['training']}", s["win_br"], [ORANGE, GREEN, "#d08a5c", "#178A42"][n])
               for n, s in enumerate(c["series"])]
        moves = []
        for s in c["series"]:
            if s["mode"] == "selfplay":
                moves.append((f"{s['label']}, standard", s["current_move"], ORANGE, 1.4, 1, False))
            else:
                moves.append((f"{s['label']}, current", s["current_move"], "#6f7f77", 1, 0.7, False))
                moves.append((f"{s['label']}, average", s["aggregate_move"], GREEN, 2.6, 1, False))
        parts.append(f"<h3>{title}: win rate against the best response</h3><div>{line_chart(win, c['iterations'], x_label='iteration', label_every=4)}</div>"
                     f"<h3>{title}: probability of the exact solver's kickoff move ({c['move']}; exact solver {pct(c['exact_move_probability'])})</h3>"
                     f"<div>{line_chart(moves, c['iterations'], top=1, x_label='iteration', label_every=4)}</div>")
    ranges = []
    for board, c in data.items():
        half = lambda v: v[len(v) // 2:]  # noqa: E731
        parts_ = "; ".join(f"{s['label']} {s['training']} {rng(min(half(s['win_br'])), max(half(s['win_br'])))}" for s in c["series"])
        exact = (res["random"]["exact"] if board == "random" else res["exact"])["vs_best_response"][0]
        ranges.append(f"<p>{'Random board' if board == 'random' else 'Deterministic board'}, second half of training: the win rate against the best response stays within {parts_}. The exact solver wins {pct(exact)} when it starts with the ball.</p>")
    return f"""<section><h2>Does training converge?</h2>
<p>Best-response dynamics (standard training) can cycle; fictitious play averages the past. Every 100 iterations the policy each learner would
report is scored against the exact best response (mean of 3 seeds), and the probability it gives to the exact solver's kickoff move is recorded
for seed 0: the current network (thin) and, for fictitious play, the average (thick). Orange: standard training's current network. Grey: the
current network under fictitious play. Green: fictitious play's average. A2C and PPO are drawn together in each chart.</p>
{"".join(parts)}{"".join(ranges)}</section>"""


STUDY_PALETTE = [GREEN, "#178A42", GREY, "#6f7f77", ORANGE]


def related_section(res: dict) -> str:
    items = res.get("related")
    if not items:
        return ""

    def entry(r):
        who = f" ({r['authors']}, {r['year']}; {r['venue']})" if r["authors"] else ""
        return f'<li><a href="{r["url"]}">{r["title"]}</a>{who}. {r["says"]}</li>'

    groups = "".join(
        f"<h3>{title}</h3><ul class=\"related\">{''.join(entry(r) for r in items if r['group'] == key)}</ul>"
        for key, title in (("older", "Older work"), ("recent", "Recent work")))
    return f"""<section><h2>Related work</h2>
<p>Has fictitious play been combined with policy gradient before? Yes, in several forms, and the convergence theory is much older than the neural networks. I checked the titles, authors and venues against Google Scholar (its top result for fictitious play, policy gradient and zero-sum Markov games was Liu, Yang and Wang) and compared each one-line summary with the abstract on the paper's own proceedings, journal or arXiv page, or on a bibliographic mirror where the publisher's page would not load (noted in the entry). JSTOR and the publisher pages for the 1951 papers would not load, so Brown and Robinson are cited from the linked secondary source and were not read. I have not checked the proofs. I did not find a result that proves convergence for deep policy gradient with fictitious play in zero-sum Markov games; the closest results are tabular, regularized or for special game classes.</p>{groups}</section>"""


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
    order = "(A2C standard, A2C fictitious play, PPO standard, PPO fictitious play)"
    cont = both.get("continuing")
    cont_html = ""
    if cont:
        cbest = max(x["mean"] for r in cont for x in (r["baseline"], r["shared"], r["trimmed"]))
        cont_html = f"""
<h3>Continuing game: games tied with the exact Nash policy</h3>
<div>{variant_bars(cont, res["continuing"]["exact"]["vs_nash"][1], field="tie_nash_")}</div>"""
        cont_text = f"""<p>Continuing game: no variant wins {wins_clause(cbest)} against the best response, as for the
baseline. The share of games tied with the exact Nash policy changes by
{change(cont, "shared", "tie_nash_mean")} points with a shared network and by
{change(cont, "trimmed", "tie_nash_mean")} points with the last 10 steps left out {order}; single seeds of the same setup
differ by up to {spread(cont, "tie_nash_runs")} points, so three seeds cannot settle changes of this size.</p>"""
    else:
        cont_text = ""
    dpp = next(r for r in det if (r["label"], r["training"]) == ("PPO", "standard"))
    best = max(x["mean"] for r in det for x in (r["baseline"], r["shared"], r["trimmed"]))
    return f"""<section><h2>Network sharing and trimming</h2>
<p>A2C and PPO, three seeds each, on all three boards. <b>Shared network</b>: both players' policies are two output
slices of one network, so each player's update also moves the other. <b>Last 10 steps left out</b>: the final 10 of the 100
steps of every episode are not used in the loss, though they still count in the returns of earlier steps. Bars are the mean over
seeds; dots are single seeds. <span style="color:{GREEN}">■</span> separate networks,
<span style="color:{GREY}">■</span> one shared network, <span style="color:{ORANGE}">■</span> last 10 steps left out.</p>
<h3>Random move-order board: win rate against the best response</h3>
<div>{variant_bars(rnd, res["random"]["exact"]["vs_best_response"][0])}</div>
<h3>Deterministic board: games tied with the exact Nash policy</h3>
<div>{variant_bars(det, res["exact"]["vs_nash"][1], field="tie_nash_")}</div>{cont_html}
<p>Random board: against separate networks, a shared network changes the win rate by {change(rnd, "shared", "mean")} points
{order}; single seeds of the same setup differ by up to {spread(rnd, "runs")} points. Leaving out the last 10 steps
changes A2C by {pts(a2c_s["trimmed"]["mean"] - a2c_s["baseline"]["mean"])} and
{pts(a2c_f["trimmed"]["mean"] - a2c_f["baseline"]["mean"])} points, but PPO with standard training stops scoring: it wins
{pct(ppo["trimmed"]["vs_random"][0])} of games against a random player, against {pct(ppo["baseline"]["vs_random"][0])} before.
No variant gets close to the exact solver.</p>
<p>Deterministic board: no variant wins {wins_clause(best)} against the best response, as for the baseline (the exact
solver wins 0%, since it ties itself). The share of games tied with the exact Nash policy changes by
{change(det, "shared", "tie_nash_mean")} points with a shared network and by {change(det, "trimmed", "tie_nash_mean")} points
with the last 10 steps left out {order}; single seeds of the same setup differ by up to {spread(det, "tie_nash_runs")} points. A tie does not mean equilibrium
play: PPO with standard training ties all games once the last 10 steps are left out, but wins only
{pct(dpp["trimmed"]["vs_random"][0])} of games against a random player, against {pct(dpp["baseline"]["vs_random"][0])}
before.</p>
{cont_text}</section>"""


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


# --------------------------------------------------------------------------- policy charts
MOVE_COLOURS = ["#1DB954", "#8fd3a8", "#c3cec8", "#a8501c"]


def move_chart(rows, actions, width=660) -> str:
    """Probability of each move per policy, one stacked bar for each player; the most likely
    move of each bar is labelled."""
    row_h, left, gap = 24, 190, 30
    panel = (width - left - gap - 10) / 2
    h = row_h * len(rows) + 30
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="10.5">']
    for k, title in enumerate(("player 0", "player 1")):
        out.append(f'<text x="{left + k * (panel + gap)}" y="12" fill="#5b6b63" font-weight="700">{title}</text>')
    for i, (label, pair) in enumerate(rows):
        y = 20 + i * row_h
        out.append(f'<text x="{left - 8}" y="{y + 12}" text-anchor="end" fill="#1c2e26">{label}</text>')
        for k, probs in enumerate(pair):
            pos, top = 0.0, max(range(len(probs)), key=lambda j: probs[j])
            for j, v in enumerate(probs):
                x0 = left + k * (panel + gap) + pos * panel
                out.append(f'<rect x="{x0:.1f}" y="{y}" width="{v * panel:.1f}" height="16" fill="{MOVE_COLOURS[j]}"/>')
                if j == top and v >= 0.12:
                    out.append(f'<text x="{x0 + v * panel / 2:.1f}" y="{y + 12}" text-anchor="middle" '
                               f'fill="{"#fff" if j in (0, 3) else "#1c2e26"}" font-weight="700">{actions[j]} {v:.0%}</text>')
                pos += v
    out.append("</svg>")
    return "".join(out)


def policy_charts(d: dict) -> str:
    acts = d["actions"]
    legend = '<div class="legend">' + "".join(
        f'<span style="background:{MOVE_COLOURS[j]}"></span>{a}' for j, a in enumerate(acts)) + "</div>"
    out = [legend]
    for k, st_ in enumerate(d["states"]):
        rows = [("Exact solver", (d["exact"]["row"][k], d["exact"]["col"][k]))]
        rows += [(lr["label"], (lr["row"][k], lr["col"][k])) for lr in d["learners"]]
        x0, y0, x1, y1, b = st_["state"]
        out.append(f"<h3>{st_['label']}: player 0 at ({x0}, {y0}), player 1 at ({x1}, {y1}), "
                   f"player {b} has the ball</h3><div>{move_chart(rows, acts)}</div>")
    return "".join(out)


def policy_html() -> str:
    path = EXP / "pg_policy_outputs.json"
    return policy_charts(json.loads(path.read_text())) if path.exists() else ""


def line_chart(series, rounds, width=660, height=260, top=None, x_label="round", label_every=1) -> str:
    """Lines over a shared x axis (win rate against the best response by round, or during training);
    a series is ``(label, values)`` or ``(label, values, colour, width, opacity, show_label)``."""
    lm, rm, tm, bm = 48, 170, 20, 34
    if top is None:
        top = max(0.1, max(v for s in series for v in s[1] if v is not None))
        top = min(1.0, round(top * 1.15 + 0.02, 2))
    x = lambda k: lm + (k / max(len(rounds) - 1, 1)) * (width - lm - rm)  # noqa: E731
    y = lambda v: tm + (1 - v / top) * (height - tm - bm)  # noqa: E731
    cols = [GREEN, "#178A42", GREY, "#6f7f77", ORANGE, "#d08a5c"]
    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica,Arial,sans-serif" font-size="11">']
    for g in (0, top / 2, top):
        out.append(f'<line x1="{lm}" y1="{y(g):.1f}" x2="{width - rm}" y2="{y(g):.1f}" stroke="#dbe5df"/>'
                   f'<text x="{lm - 6}" y="{y(g) + 4:.1f}" text-anchor="end" fill="#5b6b63">{g:.0%}</text>')
    for k, lab in enumerate(rounds):
        if k % label_every == 0:
            out.append(f'<text x="{x(k):.1f}" y="{height - 10}" text-anchor="middle" fill="#5b6b63">{lab}</text>')
    out.append(f'<text x="{lm}" y="{height}" fill="#5b6b63">{x_label}</text>')
    for n, s in enumerate(series):
        label, vals = s[0], s[1]
        col = s[2] if len(s) > 2 else cols[n % 6]
        wid = s[3] if len(s) > 3 else 2.2
        op = s[4] if len(s) > 4 else 1
        show = s[5] if len(s) > 5 else True
        pts = [(x(k), y(v)) for k, v in enumerate(vals) if v is not None]
        d = " ".join(f"{'L' if i else 'M'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{wid}" opacity="{op}"/>')
        if show:
            out.append(f'<text x="{width - rm + 8}" y="{pts[-1][1] + 4:.1f}" fill="{col}" font-weight="700">{label}</text>')
    out.append("</svg>")
    return "".join(out)


def seat_rates(entry: dict) -> list[float]:
    """Win rate against the best response when it starts with the ball, when it starts without the ball, and
    their 50/50 average."""
    ball, other = entry["vs_best_response"][0], entry["vs_best_response_column"][0]
    return [ball, other, (ball + other) / 2]


def win_rate_table(boards) -> str:
    """Every win rate against the best response in one table, for both starting positions and their
    average, on every board (``boards``: (title, board) pairs)."""
    seats = ("with ball", "without ball", "average of both")
    names = []
    for _, b in boards:
        names += [name(x) for x in b["learners"] if name(x) not in names]
    head1 = "<th rowspan=2>learner</th>" + "".join(f"<th colspan={len(seats)}>{t}</th>" for t, _ in boards)
    head2 = "".join(f"<th>{s}</th>" for _ in boards for s in seats)
    exact = "<td><b>Exact solver</b></td>" + "".join(
        f"<td><b>{v:.0%}</b></td>" for _, b in boards for v in seat_rates(b["exact"]))
    rows = []
    for n in names:
        cells = []
        for _, b in boards:
            hit = next((x for x in b["learners"] if name(x) == n), None)
            cells += [f"<td>{v:.0%}</td>" for v in seat_rates(hit)] if hit else ["<td>&mdash;</td>"] * len(seats)
        rows.append(f"<tr><td>{n}</td>{''.join(cells)}</tr>")
    return f'<table class="wins"><tr>{head1}</tr><tr>{head2}</tr><tr>{exact}</tr>{"".join(rows)}</table>'


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
solver). The exact solver wins {pct(exact['vs_best_response'][0])}; the learners win {rng(min(wins), max(wins))}.</p>
<div>{br_bars(br, exact['vs_best_response'][0])}</div>
<p><b>Wins, ties and losses against each opponent</b>.{tie_sentence}{critic_sentence}</p>
<div class="legend"><span style="background:{GREEN}"></span>win<span style="background:{GREY}"></span>tie<span style="background:{ORANGE}"></span>loss</div>
<div>{stacked_chart(stack, ["vs. a random player", "vs. the exact Nash policy", "vs. the best response"])}</div>
{table(["learner", "vs. a random player", "vs. the exact Nash policy", "vs. the best response"],
       [[lab, *[win_cell(t) for t in panels]] for lab, panels in stack])}"""


def pts(v: float) -> str:
    points = round(v * 100)
    return f"{'+' if points >= 0 else '−'}{abs(points)}"


def mixed_table(mixed: dict) -> str:
    rows = [["Exact solver", win_cell(mixed["exact"]["vs_nash"]), win_cell(mixed["exact"]["vs_best_response"]), "0", "&mdash;"]]
    rows += [[name(x), win_cell(x["vs_nash"]), win_cell(x["vs_best_response"]), f"{x['tv_row']:.2f}", pct(x["share_mixing_row"])]
             for x in mixed["learners"]]
    return table(["learner", "vs. the exact Nash policy", "vs. the best response", "distance from the exact mix", "player 0 mixes"], rows)


def policy_tables(d: dict) -> str:
    acts = d["actions"]

    def cells(probs):
        top = max(probs)
        return "".join(f"<td><b>{v * 100:.0f}%</b></td>" if v == top else f"<td>{v * 100:.0f}%</td>" for v in probs)

    head = ("<tr><th rowspan=2>policy</th><th colspan=4>player 0</th><th colspan=4>player 1</th></tr>"
            "<tr>" + "".join(f"<th>{a}</th>" for a in acts * 2) + "</tr>")
    out = []
    for k, st_ in enumerate(d["states"]):
        rows = [("Exact solver", d["exact"]["row"][k], d["exact"]["col"][k])]
        rows += [(lr["label"], lr["row"][k], lr["col"][k]) for lr in d["learners"]]
        body = "".join(f"<tr><td>{n}</td>{cells(r)}{cells(c)}</tr>" for n, r, c in rows)
        x0, y0, x1, y1, b = st_["state"]
        out.append(f"<h3>{st_['label']}: player 0 at ({x0}, {y0}), player 1 at ({x1}, {y1}), player {b} has the ball</h3>"
                   f'<table class="probs">{head}{body}</table>')
    return "".join(out)


def policy_table_html() -> str:
    path = EXP / "pg_policy_outputs.json"
    return policy_tables(json.loads(path.read_text())) if path.exists() else ""


def study_table(study: dict) -> str:
    """Mean over seeds (and the range of single seeds) of every setting in a training-choice comparison."""
    field = "tie_nash_" if study["metric"] == "tie" else ""
    head = ["learner"] + [lab for _, lab in study["keys"]]
    rows = []
    for r in study["rows"]:
        cells = []
        for k, _ in study["keys"]:
            runs = r[k][field + "runs"]
            cells.append(f"<b>{r[k][field + 'mean']:.0%}</b><span class=\"sub\">{rng(min(runs), max(runs))}</span>")
        rows.append([name(r) if r["training"] else r["label"], *cells])
    return table(head, rows)


def study_block(s: dict, res: dict) -> str:
    field = "tie_nash_" if s["metric"] == "tie" else ""
    keys = {k: STUDY_PALETTE[i % len(STUDY_PALETTE)] for i, (k, _) in enumerate(s["keys"])}
    ref = s["reference"] if s["reference"] is not None else (
        1.0 if s["metric"] == "tie" else res["random"]["exact"]["vs_best_response"][0])
    first = s["keys"][0][0]
    what = "share tied with the exact Nash policy" if s["metric"] == "tie" else "win rate against the best response"
    lines = "; ".join(f"{lab}: " + ", ".join(pts(r[k][field + "mean"] - r[first][field + "mean"]) for r in s["rows"])
                      for k, lab in s["keys"][1:])
    spread = round(100 * max(max(r[k][field + "runs"]) - min(r[k][field + "runs"]) for r in s["rows"] for k, _ in s["keys"]))
    legend = "".join(f'<span style="color:{keys[k]}">■</span> {lab} &nbsp; ' for k, lab in s["keys"])
    return (f"<h3>{s['title']}</h3><p class=\"muted\">{s['note']}</p><div>{variant_bars(s['rows'], ref, field=field, keys=keys)}</div>"
            f"<p class=\"muted\">{legend}</p>{study_table(s)}"
            f"<p>Change in the {what}, in points, from \"{s['keys'][0][1]}\" (rows in the order shown): {lines}. "
            f"Single seeds of one setting differ by up to {spread} points.</p>")


def choices_section(res: dict) -> str:
    studies = res.get("choices")
    if not studies:
        return ""
    return f"""<section><h2>Training choices</h2>
<p>Each comparison changes one thing from the usual setting (2,000 iterations of 64 episodes, all data used once, a snapshot every
5 iterations, all 100 steps in the loss) and keeps the rest. Bars are the mean over 3 seeds and dots single seeds; in the tables the
mean is bold and the range of single seeds is beneath.</p>
{"".join(study_block(s, res) for s in studies)}</section>"""


def more_seeds_section(res: dict) -> str:
    data = res.get("more_seeds")
    if not data:
        return ""
    exact = res["random"]["exact"]
    ball, other = exact["vs_best_response"][0], exact["vs_best_response_column"][0]
    ci = lambda v: rng(v[1], v[2])  # noqa: E731
    dci = lambda v: f"{pts(v[0])} points ({pts(v[1])} to {pts(v[2])})"  # noqa: E731
    labels = {"a2c_fp_base": "A2C, fictitious play, 2,000 iterations", "a2c_fp_10k": "A2C, fictitious play, 10,000 iterations",
              "ppo_sp_base": "PPO, standard, 2,000 iterations", "ppo_sp_replay4": "PPO, standard, replay of 4 iterations"}
    table_rows = []
    for key, lab in labels.items():
        c = data["conditions"][key]
        table_rows.append([lab] + [f"<b>{pct(c[m + '_ci'][0])}</b><span class=\"sub\">{ci(c[m + '_ci'])}</span>" for m in ("ball", "other", "balanced")]
                          + [rng(min(c['ball']), max(c['ball']))])
    parts = []
    for pair in data["pairs"]:
        b, v = data["conditions"][pair["baseline_key"]], data["conditions"][pair["variant_key"]]
        names = (pair["baseline_name"], pair["variant_name"])
        parts.append(
            f"<h3>{pair['label']}: {pair['baseline_name']} against {pair['variant_name']}</h3>"
            f"<div>{before_after_chart(pair_groups(data, pair, exact), names)}</div>"
            f"<p>Starts with the ball {pct(b['ball_ci'][0])} ({ci(b['ball_ci'])}) → {pct(v['ball_ci'][0])} ({ci(v['ball_ci'])}), a change of {dci(pair['ball_diff'])}; "
            f"averaged over both starts {pct(b['balanced_ci'][0])} → {pct(v['balanced_ci'][0])}, {dci(pair['balanced_diff'])}; when it starts without the ball the change is "
            f"{dci(pair['other_diff'])}. The interval for the change when it starts with the ball {'excludes' if pair['ball_diff'][1] > 0 else 'includes'} zero.</p>")
    return f"""<section><h2>More seeds</h2>
<p>With 3 seeds, A2C with fictitious play looked better at 10,000 iterations than at 2,000, and PPO with standard training looked better with
replayed data. Each of these two settings and its usual-setting baseline was run on seeds 0 to {data['seeds'] - 1} (random move-order board). Each bar is the
mean of the 10 seeds and the thin line is its 95% interval (Student's t, kept within 0% and 100%). The exact solver wins {pct(ball)} when it starts with the ball,
{pct(other)} when it starts without the ball and {pct((ball + other) / 2)} averaged over both starts.</p>
{table(["setting", "starts with the ball (95% interval)", "starts without the ball", "average of both", "single seeds, starts with the ball"], table_rows)}
{"".join(parts)}</section>"""


def methods_table() -> str:
    """The experiment files and the commands that produce them, from docs/methods.md."""
    def clean(cell: str) -> str:
        cell = html.escape(cell.replace("\\|", "|"))
        cell = re.sub(r"`([^`]*)`", r"<code>\1</code>", cell)
        return re.sub(r"\*\*([^*]*)\*\*", r"<b>\1</b>", cell)

    rows = []
    for line in (DOCS / "methods.md").read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) >= 4 and re.search(r"pg_(finite|variant|study|more|curves|entropy|exact|random_kickoff|policy|mixed|fp_br)|rps_nn", cells[0]):
            rows.append([clean(c) for c in cells[:4]])
    if not rows:
        return ""
    head = "".join(f"<th>{h}</th>" for h in ("file", "command", "what", "time"))
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<table class="small methods"><colgroup><col style="width:24%"><col style="width:34%"><col style="width:32%"><col style="width:10%"></colgroup>'
            f"<tr>{head}</tr>{body}</table>")


def reproduce_section(res: dict) -> str:
    return f"""<section><h2>Reproducing the numbers</h2>
<p>Every number in this report is computed by <code>scripts/pg_site_data.py</code> from the result files in <code>experiments/</code>; the same data
feeds the website's Policy gradient tab. The runner is <code>scripts/pg_finite.py</code> (2,000 iterations of 64 episodes and 1,000 evaluation games by
default); each output row records the number of iterations. <code>make pg-report</code> and <code>make pg-pdf</code> rebuild the short report and
<code>make pg-detailed-pdf</code> this one.</p>
{methods_table()}</section>"""


def whats_new(res: dict) -> str:
    items = ["The win-rate table reports both starting positions (with and without the ball) and their 50/50 average; the exact solver is context, not a target.",
             "A third board, the continuing game (fixed 100 steps, play restarts after a goal), with A2C using the exact solver's values as a frozen critic."]
    if res.get("variants"):
        items.append("Comparisons of one shared network and of leaving out the last steps of each episode, on all three boards.")
    if res.get("argmax"):
        items.append("Fictitious play over pure (argmax) snapshots, next to the usual softmax average.")
    if res.get("entropy"):
        items.append("An entropy-bonus ablation (0.01 against 0.2) for every learner, training scheme and board.")
    if res.get("curves"):
        items.append("Training curves: win rate against the best response and the kickoff policy during training.")
    if res.get("choices"):
        items.append("A sweep of training choices: how many last steps to drop, snapshot interval, five times the iterations, mini-batches, replayed data, the exact critic on every board, weight averaging.")
    if res.get("more_seeds"):
        items.append("Ten seeds with 95% intervals for the two best-looking settings.")
    if res.get("rps_nn"):
        items.append("Rock-paper-scissors with neural-network players, with and without a larger entropy bonus.")
    items.append("A related-work list of older and recent papers, checked against Google Scholar and each paper's own page where it would load.")
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def build_html(res: dict) -> str:
    det, rnd, mixed, cont = res, res["random"], res["mixed"], res.get("continuing")
    boards = [("The deterministic board (A10)", "Players move simultaneously and the carrier wins every contested square, "
               "so the exact solution is pure everywhere and ties itself.", det),
              ("The random move-order board", "Moves are applied in a random order each step; the exact solution "
               "mixes at 94 stage games.", rnd)]
    if cont:
        boards.append(("The continuing game", "Play restarts after every goal and runs for exactly 100 steps; a win is "
                       "more goals than the opponent. Also shows A2C with the exact solver's values as a frozen critic.", cont))
    seat_row, seat_col = rnd["exact"]["vs_best_response"][0], rnd["exact"]["vs_best_response_column"][0]
    seat_value = rnd.get("kickoff_value", 0.0)
    seat_chart = stacked_chart(
        [("Exact solver", [rnd["exact"]["vs_best_response"], rnd["exact"]["vs_best_response_column"]])]
        + [(name(x), [x["vs_best_response"], x["vs_best_response_column"]]) for x in rnd["learners"]],
        ["Random board, starting with the ball", "Random board, starting without the ball"])
    long_chart = stacked_chart(
        [(f"{x['label']}, {x['training']}", [x["short_vs_nash"], x["vs_nash"], x["vs_random"]]) for x in res["longer"]],
        ["vs. exact Nash, 2,000 iterations", "vs. exact Nash, 8,000 iterations", "vs. random, 8,000 iterations"])
    long_table = table(["learner", "vs. exact Nash, 2,000 iterations", "vs. exact Nash, 8,000 iterations", "vs. random, 8,000 iterations"],
                       [[f"{x['label']}, {x['training']}", win_cell(x["short_vs_nash"]), win_cell(x["vs_nash"]), win_cell(x["vs_random"])]
                        for x in res["longer"]])
    rounds = sorted({c["round"] for r in res["fp_br"] for c in r["checkpoints"]})
    fp_series = [(f"{r['label']}, {r['best_response_iterations']} iterations", [
        next((c["win_vs_best_response"] for c in r["checkpoints"] if c["round"] == k), None) for k in rounds])
        for r in res["fp_br"]]
    fp_table = table(["run"] + [f"round {k}" for k in rounds],
                     [[f"{r['label']}, {r['best_response_iterations']} iterations per best response"] + [
                         next((pct(c["win_vs_best_response"]) for c in r["checkpoints"] if c["round"] == k), "") for k in rounds]
                      for r in res["fp_br"]])
    mixed_chart = stacked_chart(
        [("Exact solver", [mixed["exact"]["vs_nash"], mixed["exact"]["vs_best_response"]])]
        + [(name(x), [x["vs_nash"], x["vs_best_response"]]) for x in mixed["learners"]],
        ["vs. the exact Nash policy", "vs. the best response"])
    tv = [x["tv_row"] for x in mixed["learners"]]
    mixing = [x["share_mixing_row"] for x in mixed["learners"]]
    pg_br = [x["vs_best_response"][0] for x in rnd["learners"]]
    pg_rand = [x["vs_random"][0] for x in det["learners"]]
    mx = [x["vs_nash"][0] for x in mixed["learners"]]
    legend = f'<div class="legend"><span style="background:{GREEN}"></span>win<span style="background:{GREY}"></span>tie<span style="background:{ORANGE}"></span>loss</div>'

    definitions = table(["number", "what it is"], [
        ["Win rate against the best response", "The share of 1,000 games the trained player wins when the opponent plays the exact best response to it, the move that minimizes its expected discounted score difference."],
        ["Starts with the ball / starts without the ball / average of both", "At kickoff one player has the ball and the other has to defend. Starts with the ball: our learner is the player who has the ball. Starts without the ball: our learner is the other player. Average of both: the mean of the two, one fair number that does not depend on who starts with the ball."],
        ["Win rate against the exact Nash policy", "The same, against the exact equilibrium player."],
        ["Win rate against a random player", "A sanity check: every learner should beat a player who moves at random."],
        ["A win", "The player scores while the opponent does not. On the first two boards the first goal ends the game; in the continuing game play restarts after a goal and a win is more goals than the opponent in 100 steps. Games with no goal, or equal goals, are ties."],
        ["Distance from the exact mix", "Total-variation distance between a learner's probabilities and the exact solver's at the first step: 0 is identical, 1 is no overlap."],
        ["Exploitability", "Exploitability is about attackers, not about the player who has the ball. Take the two players' policies as fixed, replace each player in turn by the exact best response to the other, and add up what the two attackers gain in expected discounted score (goals scored minus goals conceded, discounted). It is 0 exactly at an equilibrium, such as the exact solver, and grows the easier the pair of policies is to beat. It is a property of the pair, not of one player, and the kickoff advantage cancels out of the sum. It is a score, not a win rate, so it cannot be compared with the percentages in the tables. We keep it in the result files and use it in the rock-paper-scissors test, where 0 means one third each and 2 means both players are always beaten. For soccer we report the trained player's win rate against the best response instead, because that judges the player we train: how often it wins against a perfect opponent."]], "text")

    setup = table(["setting", "value"], [
        ["Game", "7&times;5 board, goal rows 1&ndash;3. Fixed 100 steps, discount 0.9, and the number of steps left is a network input."],
        ["Boards", "Deterministic (A10): simultaneous moves, the carrier wins contested squares. Random move order: moves are applied in a random order each step. Continuing game: fixed 100 steps, play restarts after a goal."],
        ["Returns", "REINFORCE: the discounted return from each step to the end of the episode, no baseline. A2C: 10-step bootstrapped advantage from a learned critic. A2C with the exact critic: the critic is the exact solver's value, frozen, so the advantage is the exact Q(s, a<sub>0</sub>, a<sub>1</sub>) minus V(s). PPO: GAE (&lambda; 0.95), ratio clipped at 0.2, 4 epochs."],
        ["Updates", "Every iteration: 64 episodes of 100 steps, then one gradient step per player on all of that data (four for PPO). 2,000 iterations. Learning rate 10<sup>-3</sup>, entropy bonus 0.01, 64&times;64 network, softmax output, a separate network for each player. No mini-batches, no replay (both are tested in the training choices)."],
        ["Standard", "Both players' current networks play each other and keep updating: real-time best responses."],
        ["Fictitious play", "Each player trains against a frozen snapshot of the other, drawn at random from all its past snapshots (one every 5 iterations, a fixed budget of 5 policy-gradient iterations); the policy it reports is the average of its own snapshots. Policies, never network weights, are averaged."],
        ["Evaluation", "1,000 repeated games from the kickoff against a random player, the exact Nash policy and the exact best response; wins, ties and losses counted. 3 seeds per learner unless stated."]], "text")

    sections = [
        f"""<section><h2>Win rates against the best response</h2>
<p>Two players face each other. At kickoff one of them has the ball and the other has to defend, and having the ball is an advantage. Each learner is trained for both players, so it is measured in both positions. <b>Starts with the ball</b>: our learner is the player who has the ball at kickoff. <b>Starts without the ball</b>: our learner is the other player. <b>Average of both</b>: the mean of the two, one fair number that does not depend on who starts with the ball.</p>
{win_rate_table([(t, b) for t, _, b in boards])}
<p>The share of 1,000 games each learner wins when the opponent plays the exact best response to it, when it starts with the ball, when it starts without the ball, and averaged over both
(mean of 3 seeds). The exact solver is the top row, as context rather than a target. A dash means the learner was not run on that board.</p>
<h3>How to read the numbers</h3>{definitions}
<h3>Why the exact solver does not win 0% against the best response</h3>
<p>The best response is the move that minimizes our player's <em>expected discounted score difference</em>, not the one that minimizes its chance of winning.
At the random board's kickoff the player holding the ball is ahead: the exact solution is worth {seat_value:+.2f} to it even against a perfect best response. So
the exact solver wins {pct(seat_row)} of games and loses {pct(1 - seat_row)} when it starts with the ball, and wins {pct(seat_col)} when it starts without the ball. On the deterministic
board both starts tie every game.</p>
{legend}<div>{seat_chart}</div></section>""",
        f"""<section><h2>Setup</h2>{setup}</section>""",
        f"""<section><h2>Results by board</h2>
<p>Against the best response the exact solver wins {pct(rnd['exact']['vs_best_response'][0])} of games when it starts with the ball on the random board; the six learners win
{pct(min(pg_br))} to {pct(max(pg_br))}. Against a random player the learners win {pct(min(pg_rand))} to {pct(max(pg_rand))} of games on the deterministic board.
Fictitious-play training wins more often against a random player; neither way of training reaches the exact solver's rate against the best response.</p>
{''.join(board_section(t, n, b) for t, n, b in boards)}</section>""",
        f"""<section><h2>Mixed states of the random board</h2>
<p>At {mixed['mixed_states']} states the exact equilibrium has to randomize. Each learner starts {mixed['games_per_start']} games from every one of them.</p>
{legend}<div>{mixed_chart}</div>{mixed_table(mixed)}
<p>The distance from the exact mix is {min(tv):.2f} to {max(tv):.2f} for every learner. Player 0 mixes (puts under 90% on its top move) at {pct(min(mixing))} to {pct(max(mixing))}
of these states, though not at the states or in the proportions the exact solution does; against the exact Nash policy the learners win {pct(min(mx))} to {pct(max(mx))}
(exact solver {pct(mixed['exact']['vs_nash'][0])}).</p>
<h3>Three of these states in full</h3>{policy_charts(mixed['policy_outputs'])}{policy_tables(mixed['policy_outputs'])}</section>""",
        f"""<section><h2>Action probabilities</h2>
<p>The probability each trained network gives to every move, at four fixed positions of the deterministic board (seed 0), next to the exact solver's move.</p>
{policy_html()}{policy_table_html()}</section>""",
        f"""<section><h2>Why fictitious play</h2>
<p>On rock-paper-scissors, answering the opponent's latest move makes the play flip between rock, paper and scissors for ever. Answering the average of everything it
has played converges to one third each.</p><div>{rps_plot(res['rps'])}</div>{rps_nn_section(res).replace('<h2>', '<h3>', 1).replace('</h2>', '</h3>', 1).replace('<section>', '').replace('</section>', '')}</section>""",
        curves_section(res),
        f"""<section><h2>More training</h2>
<p>Standard training at 8,000 instead of 2,000 iterations (two seeds each), deterministic board.</p>{legend}<div>{long_chart}</div>{long_table}
<p>At 8,000 iterations the learners win {rng(min(x['vs_nash'][0] for x in res['longer']), max(x['vs_nash'][0] for x in res['longer']))} of games against the exact Nash policy
(at 2,000 iterations {rng(min(x['short_vs_nash'][0] for x in res['longer']), max(x['short_vs_nash'][0] for x in res['longer']))}) and
{rng(min(x['vs_random'][0] for x in res['longer']), max(x['vs_random'][0] for x in res['longer']))} against a random player; more standard training does not
close the gap to the exact solver.</p></section>""",
        f"""<section><h2>Fictitious play with long best responses</h2>
<p>Each round a player trains for 100 (or 300) iterations against the average of its opponent's earlier best responses, then adds the result to its history. Win rate against
the best response at each checkpoint round (one seed per run); every line ends below {pct(max(c['win_vs_best_response'] for r in res['fp_br'] for c in r['checkpoints']))}.</p>
<div>{line_chart(fp_series, rounds)}</div>{fp_table}</section>""",
        variant_section(res), argmax_section(res), entropy_section(res),
        choices_section(res), more_seeds_section(res), related_section(res), reproduce_section(res),
    ]
    numbered, k = [], 0
    for sec in sections:
        if not sec:
            continue
        k += 1
        numbered.append(re.sub(r"<h2>", f"<h2>{k}. ", sec, count=1))
    toc = "".join(f"<li>{m}</li>" for sec in numbered for m in re.findall(r"<h2>(.*?)</h2>", sec)[:1])
    body = "\n\n".join(numbered)
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
ul.toc{{list-style:none;margin-left:0;columns:2}} ul.related{{font-size:12px;line-height:1.45}} ul.related li{{margin:2px 0}}
table.methods{{table-layout:fixed;word-break:break-word}} table.methods code{{font-size:9px}}
table.small td,table.small th{{font-size:10px;padding:3px 5px}} table.small td{{text-align:left}}
table.wins td,table.wins th{{padding:4px 5px;font-size:10.5px}} table.wins td:first-child{{white-space:nowrap}}
.sub{{display:block;font-size:10px;color:var(--muted);font-weight:400}}
svg{{width:100%;height:auto}}
.legend span{{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;vertical-align:-1px}}
ul{{margin:6px 0 6px 18px;padding:0}} li{{margin:3px 0}}
@page{{size:A4;margin:14mm}}
</style></head><body>
<div class="eyebrow">Policy gradient &middot; soccer game &middot; detailed report</div>
<h1>Policy gradient on the soccer game</h1>
<p class="lede">REINFORCE, A2C and PPO, trained the standard way and by fictitious play, scored by how often the trained player wins when the opponent is the best
response to it. This is the detailed report with every table and plot; a short version is <code>policy_gradient.pdf</code>.</p>

<div class="callout"><b>Summary</b>
<ul>
<li>On the random board the exact solver wins {pct(seat_row)} when it starts with the ball, {pct(seat_col)} when it starts without the ball and {pct((seat_row + seat_col) / 2)} averaged over both starts; the six learners win
{pct(min(pg_br))} to {pct(max(pg_br))}, {pct(min(x['vs_best_response_column'][0] for x in rnd['learners']))} to {pct(max(x['vs_best_response_column'][0] for x in rnd['learners']))} and
{pct(min((x['vs_best_response'][0] + x['vs_best_response_column'][0]) / 2 for x in rnd['learners']))} to {pct(max((x['vs_best_response'][0] + x['vs_best_response_column'][0]) / 2 for x in rnd['learners']))}.
On the deterministic board the exact solver ties itself, so it wins 0%.</li>
<li>Against a random player the learners win {pct(min(pg_rand))} to {pct(max(pg_rand))} of games (deterministic board).</li>
<li>Starting from the {mixed['mixed_states']} states where the exact answer has to mix, they win {pct(min(mx))} to {pct(max(mx))} against the exact Nash policy; the exact solver wins {pct(mixed['exact']['vs_nash'][0])}.</li>
<li>No setting tried replicates the exact solver. The best, on 10 seeds, wins {pct(res['more_seeds']['conditions']['a2c_fp_10k']['ball_ci'][0]) if res.get('more_seeds') else '—'} when it starts with the ball (A2C, fictitious play, 10,000 iterations).</li>
</ul>3 seeds per learner unless stated, 1,000 games per opponent.</div>

<h3>Contents</h3><ul class="toc">{toc}</ul>
<h3>What this report adds</h3>{whats_new(res)}

{body}
</body></html>"""


def main() -> None:
    (DOCS / "policy_gradient_detailed.html").write_text(build_html(load_data()))
    print("wrote docs/policy_gradient_detailed.html")


if __name__ == "__main__":
    main()
