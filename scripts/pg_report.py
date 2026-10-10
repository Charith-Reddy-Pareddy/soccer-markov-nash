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


def variant_bars(rows, reference, width=660, field="", keys=None) -> str:
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
                       "so the standard networks settle too.")
    return f"""<h3>The same test with neural networks</h3>
<p>Each player is a small network with a softmax output, trained by REINFORCE on sampled games ({data['iterations']:,} iterations,
{data['batch']} games each, 3 seeds; the plots show seed 0). <b>Standard</b> trains the two current networks against each other.
<b>Fictitious play</b> trains each player against a snapshot of the other drawn from all its past snapshots and reports the average of
a player's snapshots; <b>argmax</b> first turns each snapshot into the pure policy that plays its most likely move, so the average is
how often the snapshots play each move. Orange: the current network's share of rock; green: the aggregate. Exploitability, on the right,
is the mean over seeds and the last 300 iterations, and is 0 at one third each.
With the soccer settings the standard networks cycle (exploitability {ex[first]['standard']:.2f}), and neither form of fictitious play reaches
one third (softmax {ex[first]['fictitious']:.2f}, argmax {ex[first]['fictitious_argmax']:.2f}). {second_text}</p>{''.join(parts)}"""


def argmax_section(res: dict) -> str:
    data = res.get("argmax")
    if not data:
        return ""
    keys = {"softmax": GREEN, "argmax": ORANGE}
    rnd, det = data["random"], data["deterministic"]
    pts = lambda v: f"{'+' if v >= 0 else '−'}{abs(round(v * 100))}"  # noqa: E731
    change = lambda rows, f: ", ".join(pts(r["argmax"][f] - r["softmax"][f]) for r in rows)  # noqa: E731
    spread = lambda rows, f: round(100 * max(max(x[f]) - min(x[f]) for r in rows for x in (r["softmax"], r["argmax"])))  # noqa: E731
    return f"""<section><h2>10. Averaging pure policies</h2>
<p>Neural networks cannot output a mixed strategy except through their softmax, so each snapshot is made
a pure policy (the move with the highest probability), those are averaged, and each player best-responds to that average. This is fictitious play in its
original form: each snapshot is a pure best response, and the mix is how often each move was played. Both the opponent that a player trains
against and the policy it reports are averages of pure snapshots. Bars are the mean over 3 seeds and dots single seeds;
<span style="color:{GREEN}">■</span> average of softmax snapshots (the fictitious play used elsewhere),
<span style="color:{ORANGE}">■</span> average of argmax snapshots.</p>
<h3>Random move-order board: win rate against the best response</h3>
<div>{variant_bars(rnd, res["random"]["exact"]["vs_best_response"][0], keys=keys)}</div>
<h3>Deterministic board: games tied with the exact Nash policy</h3>
<div>{variant_bars(det, res["exact"]["vs_nash"][1], field="tie_nash_", keys=keys)}</div>
<p>Random board: averaging argmax instead of softmax snapshots changes the win rate against the best response by {change(rnd, "mean")} points
(REINFORCE, A2C, PPO); single seeds differ by up to {spread(rnd, "runs")} points. Deterministic board: it changes the share of games tied with the
exact Nash policy by {change(det, "tie_nash_mean")} points; single seeds differ by up to {spread(det, "tie_nash_runs")} points.</p></section>"""


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
        cont_text = f"""<p>Continuing game: no variant wins more than {pct(cbest)} of games against the best response, as for the
baseline. The share of games tied with the exact Nash policy changes by
{change(cont, "shared", "tie_nash_mean")} points with a shared network and by
{change(cont, "trimmed", "tie_nash_mean")} points with the last 10 steps left out {order}; single seeds of the same setup
differ by up to {spread(cont, "tie_nash_runs")} points, so three seeds cannot settle changes of this size.</p>"""
    else:
        cont_text = ""
    dpp = next(r for r in det if (r["label"], r["training"]) == ("PPO", "standard"))
    best = max(x["mean"] for r in det for x in (r["baseline"], r["shared"], r["trimmed"]))
    return f"""<section><h2>9. Network sharing and trimming</h2>
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
<p>Deterministic board: no variant wins more than {pct(best)} of games against the best response, as for the baseline (the exact
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


def line_chart(series, rounds, width=660, height=260) -> str:
    """Win rate against the best response after each round of fictitious play with long best
    responses; one line per run."""
    lm, rm, tm, bm = 48, 170, 20, 34
    top = max(0.1, max(v for _, vals in series for v in vals if v is not None))
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
        out.append(f'<text x="{x(k):.1f}" y="{height - 10}" text-anchor="middle" fill="#5b6b63">{lab}</text>')
    out.append(f'<text x="{lm}" y="{height}" fill="#5b6b63">round</text>')
    for n, (label, vals) in enumerate(series):
        pts = [(x(k), y(v)) for k, v in enumerate(vals) if v is not None]
        d = " ".join(f"{'L' if i else 'M'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{cols[n % 6]}" stroke-width="2.2"/>'
                   f'<text x="{width - rm + 8}" y="{pts[-1][1] + 4:.1f}" fill="{cols[n % 6]}" font-weight="700">{label}</text>')
    out.append("</svg>")
    return "".join(out)


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
    long_chart = stacked_chart(
        [(f"{x['label']}, {x['training']}", [x["short_vs_nash"], x["vs_nash"], x["vs_random"]]) for x in res["longer"]],
        ["vs. exact Nash, 2,000 iterations", "vs. exact Nash, 8,000 iterations", "vs. random, 8,000 iterations"])
    rounds = sorted({c["round"] for r in res["fp_br"] for c in r["checkpoints"]})
    fp_series = [(f"{r['label']}, {r['best_response_iterations']} iterations", [
        next((c["win_vs_best_response"] for c in r["checkpoints"] if c["round"] == k), None) for k in rounds])
        for r in res["fp_br"]]
    fp_chart = line_chart(fp_series, rounds)
    mixed_chart = stacked_chart(
        [("Exact solver", [mixed["exact"]["vs_nash"], mixed["exact"]["vs_best_response"]])]
        + [(name(x), [x["vs_nash"], x["vs_best_response"]]) for x in mixed["learners"]],
        ["vs. the exact Nash policy", "vs. the best response"])
    tv = [x["tv_row"] for x in mixed["learners"]]
    mixing = [x["share_mixing_row"] for x in mixed["learners"]]
    seat_row = rnd["exact"]["vs_best_response"][0]
    seat_col = rnd["exact"]["vs_best_response_column"][0]
    seat_value = rnd.get("kickoff_value", 0.0)
    seat_chart = stacked_chart(
        [("Exact solver", [rnd["exact"]["vs_best_response"], rnd["exact"]["vs_best_response_column"]])]
        + [(name(x), [x["vs_best_response"], x["vs_best_response_column"]]) for x in rnd["learners"]],
        ["Random board, ball-holding seat", "Random board, other seat"])
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
<p class="muted">Exploitability, the best-response player's gain, is not reported: it measures the attacker, and it is the player we train that we want to judge.</p>
<h3>Why the exact solver does not win 0% against the best response</h3>
<p>The best response is the move that minimizes our player's <em>expected discounted score difference</em>, not the one that
minimizes its chance of winning. At the random board's kickoff the player holding the ball is ahead: the exact solution is worth
{seat_value:+.2f} to it even against a perfect best response. So the exact solver wins {pct(seat_row)} of games and loses
{pct(1 - seat_row)} from that seat, and wins {pct(seat_col)} from the other seat. The first number is the one reported above (our learners
always sit in the ball-holding seat); the second is what it gets from the seat without the ball. On the deterministic board
both seats tie every game.</p>
<div class="legend"><span style="background:{GREEN}"></span>win<span style="background:{GREY}"></span>tie<span style="background:{ORANGE}"></span>loss</div>
<div>{seat_chart}</div></section>

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
<div>{rps_plot(res['rps'])}</div>
{rps_nn_section(res)}</section>

<section><h2>4. Results</h2>
{''.join(board_section(t, n, b) for t, n, b in boards)}
</section>

<section><h2>5. Mixed states of the random board</h2>
<p>At {mixed['mixed_states']} states the exact equilibrium has to randomize. Each learner starts
{mixed['games_per_start']} games from every one of them. Distance is the total-variation distance between the
learner's probabilities and the exact mix at the first step (0 = identical, 1 = no overlap); "player 0 mixes"
is the share of states where it puts less than 90% on its top move.</p>
<div class="legend"><span style="background:{GREEN}"></span>win<span style="background:{GREY}"></span>tie<span style="background:{ORANGE}"></span>loss</div>
<div>{mixed_chart}</div>
<p>The distance from the exact mix is {min(tv):.2f} to {max(tv):.2f} for every learner. Player 0 mixes
(puts under 90% on its top move) at {pct(min(mixing))} to {pct(max(mixing))} of these states, though not at the
states or in the proportions the exact solution does.</p>
<h3>Three of these states in full</h3>
{policy_charts(mixed['policy_outputs'])}</section>

<section><h2>6. Action probabilities</h2>
<p>The probability each trained network gives to every move, at four fixed positions of the deterministic board
(seed 0), next to the exact solver's move.</p>
{policy_html()}</section>

<section><h2>7. More training</h2>
<p>Standard training at 8,000 instead of 2,000 iterations (two seeds each), deterministic board.</p>
<div class="legend"><span style="background:{GREEN}"></span>win<span style="background:{GREY}"></span>tie<span style="background:{ORANGE}"></span>loss</div>
<div>{long_chart}</div></section>

<section><h2>8. Fictitious play with long best responses</h2>
<p>Each round a player trains for 100 (or 300) iterations against the average of its opponent's earlier best
responses, then adds the result to its history. Win rate against the best response at each checkpoint round (one seed per run); every line ends below {pct(max(c['win_vs_best_response'] for r in res['fp_br'] for c in r['checkpoints']))}.</p>
<div>{fp_chart}</div></section>

{variant_section(res)}

{argmax_section(res)}
</body></html>"""


def main() -> None:
    (DOCS / "policy_gradient.html").write_text(build_html(load_data()))
    print("wrote docs/policy_gradient.html")


if __name__ == "__main__":
    main()
