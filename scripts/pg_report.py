# ruff: noqa: E501
"""Build the short report docs/policy_gradient.html (the source of the PDF): the win-rate table
first, then the setup and the findings that matter. Every number comes from
``pg_site_data.build()``, the same source as the site's tab, which carries the full detail.

    python scripts/pg_report.py
"""

from __future__ import annotations

import importlib.util
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


def table(head, rows, cls="") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><tr>{th}</tr>{body}</table>'


MODE_TITLES = {"standard": "Standard: both current networks play each other",
               "fictitious": "Fictitious play: average of softmax snapshots",
               "fictitious_argmax": "Fictitious play: average of argmax snapshots"}


def variant_bars(rows, reference, width=660, field="", keys=None) -> str:
    """Win rate against the best response per variant: a bar for the mean, a dot per seed."""
    left, right, bar = 170, 20, 16
    group = len(keys) * (bar + 2) + 16
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
        for n, (key, colour) in enumerate(keys.items()):
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
    """Win rate against the best response from the ball-holding seat, from the other seat, and
    their 50/50 average."""
    ball, other = entry["vs_best_response"][0], entry["vs_best_response_column"][0]
    return [ball, other, (ball + other) / 2]


def win_rate_table(boards) -> str:
    """Every win rate against the best response in one table, for both kickoff seats and their
    average, on every board (``boards``: (title, board) pairs)."""
    seats = ("ball seat", "other seat", "balanced")
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


def ten_seeds_block(res: dict) -> str:
    data = res.get("more_seeds")
    if not data:
        return ""
    keys = {"base": GREEN, "variant": ORANGE}
    exact = res["random"]["exact"]
    ball, other = exact["vs_best_response"][0], exact["vs_best_response_column"][0]
    ci = lambda v: f"{pct(v[1])} to {pct(v[2])}"  # noqa: E731
    dci = lambda v: f"{pts(v[0])} points ({pts(v[1])} to {pts(v[2])})"  # noqa: E731
    parts = []
    for pair in data["pairs"]:
        b, v = data["conditions"][pair["baseline_key"]], data["conditions"][pair["variant_key"]]
        rows = [{"label": lab, "training": "",
                 "base": {"mean": sum(b[m]) / len(b[m]), "runs": b[m]},
                 "variant": {"mean": sum(v[m]) / len(v[m]), "runs": v[m]}}
                for m, lab in (("ball", "Ball seat"), ("other", "Other seat"), ("balanced", "Balanced"))]
        parts.append(
            f"<h3>{pair['label']}: {pair['variant']}</h3><div>{variant_bars(rows, ball, keys=keys)}</div>"
            f"<p>Ball seat {pct(b['ball_ci'][0])} ({ci(b['ball_ci'])}) → {pct(v['ball_ci'][0])} ({ci(v['ball_ci'])}), a change of {dci(pair['ball_diff'])}; "
            f"balanced {pct(b['balanced_ci'][0])} → {pct(v['balanced_ci'][0])}, {dci(pair['balanced_diff'])}. "
            f"The interval for the change {'excludes' if pair['ball_diff'][1] > 0 else 'includes'} zero.</p>")
    return (f"<h3>Ten seeds for the two best-looking settings</h3><p>Seeds 0 to {data['seeds'] - 1}, random board; 95% intervals (Student's t) in brackets; "
            f"green: usual setting, orange: changed setting, dots: single seeds. The exact solver wins {pct(ball)} from the ball seat, {pct(other)} from the other "
            f"seat and {pct((ball + other) / 2)} balanced.</p>{''.join(parts)}")


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
    return f"""<section><h2>7. Related work</h2>
<p>Has fictitious play been combined with policy gradient before? Yes, in several forms, and the convergence theory is much older than the neural networks. I checked the titles, authors and venues against Google Scholar (its top result for fictitious play, policy gradient and zero-sum Markov games was Liu, Yang and Wang) and compared each one-line summary with the abstract on the paper's own proceedings, journal or arXiv page, or on a bibliographic mirror where the publisher's page would not load (noted in the entry). JSTOR and the publisher pages for the 1951 papers would not load, so Brown and Robinson are cited from the linked secondary source and were not read. I have not checked the proofs. I did not find a result that proves convergence for deep policy gradient with fictitious play in zero-sum Markov games; the closest results are tabular, regularized or for special game classes.</p>{groups}</section>"""


def pts(v: float) -> str:
    return f"{'+' if v >= 0 else '−'}{abs(round(v * 100))}"


def rps_section(res: dict) -> str:
    data = res.get("rps_nn")
    if not data or "configs" not in data:
        return ""
    names = list(data["configs"])
    first = data["configs"][names[0]]
    ex = {n: {m: sum(v["final_exploitability"]) / len(v["final_exploitability"]) for m, v in c["modes"].items()}
          for n, c in data["configs"].items()}
    cfg2 = data["configs"][names[1]] if len(names) > 1 else {}
    second = (f" With a much larger entropy bonus ({cfg2.get('entropy')}) and plain gradient steps all three settle: exploitability "
              f"{ex[names[1]]['standard']:.2f} standard, {ex[names[1]]['fictitious']:.2f} softmax average, "
              f"{ex[names[1]]['fictitious_argmax']:.2f} argmax average. That is a stabilization effect of the changed objective, "
              "not evidence of convergence to the unregularized equilibrium." if len(names) > 1 else "")
    return f"""<section><h2>5. Rock-paper-scissors with neural networks</h2>
<p>Each player is a small softmax network trained by REINFORCE ({data['iterations']:,} iterations, {data['batch']} games each, {data['seeds']} seeds; seed 0 shown).
Orange: the current network's share of rock. Green: the average over past snapshots (fictitious play), of softmax policies or of argmax (pure) policies.
Exploitability (right) is 0 at one third each. With the soccer settings the standard networks cycle ({ex[names[0]]['standard']:.2f}) and neither average
reaches one third (softmax {ex[names[0]]['fictitious']:.2f}, argmax {ex[names[0]]['fictitious_argmax']:.2f}).{second}</p>
<div>{rps_nn_panels(first, data['iterations'])}</div></section>"""


def curves_section(res: dict) -> str:
    data = (res.get("curves") or {}).get("random")
    if not data:
        return ""
    win = [(f"{s['label']}, {s['training']}", s["win_br"], [ORANGE, GREEN, "#d08a5c", "#178A42"][n])
           for n, s in enumerate(data["series"])]
    moves = []
    for s in data["series"]:
        if s["mode"] == "selfplay":
            moves.append((f"{s['label']}, standard", s["current_move"], ORANGE, 1.4, 1, False))
        else:
            moves.append((f"{s['label']}, current", s["current_move"], "#6f7f77", 1, 0.7, False))
            moves.append((f"{s['label']}, average", s["aggregate_move"], GREEN, 2.6, 1, False))
    half = lambda v: v[len(v) // 2:]  # noqa: E731
    ranges = "; ".join(f"{s['label']} {s['training']} {pct(min(half(s['win_br'])))} to {pct(max(half(s['win_br'])))}"
                       for s in data["series"])
    exact = res["random"]["exact"]["vs_best_response"][0]
    return f"""<section><h2>4. Does training converge?</h2>
<p>Random board. Every 100 iterations the policy each learner would report is scored against the exact best response (mean of 3 seeds), and the
probability it gives to the exact solver's kickoff move ({data['move']}; the exact solver plays it {pct(data['exact_move_probability'])}) is recorded for
seed 0. Orange: standard training's current network. Grey: the current network under fictitious play. Green: fictitious play's average.</p>
<div>{line_chart(win, data['iterations'], x_label='iteration', label_every=4)}</div>
<div>{line_chart(moves, data['iterations'], top=1, x_label='iteration', label_every=4)}</div>
<p>In the second half of training the win rate against the best response stays within {ranges}; the exact solver wins {pct(exact)} from the ball seat.
Standard training keeps swinging; the fictitious-play average is steadier but flat and low.</p></section>"""


def change_range(rows, key, base, field="mean") -> str:
    values = [r[key][field] - r[base][field] for r in rows]
    return f"{pts(min(values))} to {pts(max(values))}"


def finding_lines(res: dict) -> list[str]:
    """One line per training-recipe comparison: the range of changes, in points."""
    out = []
    v = (res.get("variants") or {}).get("random")
    if v:
        out.append(f"<b>One shared network</b> (two output heads), random board: {change_range(v, 'shared', 'baseline')} points.")
        out.append(f"<b>Last 10 of 100 steps left out</b>, random board: {change_range(v, 'trimmed', 'baseline')} points; PPO with standard training stops scoring.")
    a = (res.get("argmax") or {}).get("random")
    if a:
        out.append(f"<b>Averaging argmax (pure) snapshots</b> instead of softmax ones: {change_range(a, 'argmax', 'softmax')} points on the random board; no gain on the other boards.")
    e = res.get("entropy")
    if e and "random" in e:
        by = []
        for algo in ("REINFORCE", "A2C", "PPO"):
            rows = [r for r in e["random"] if r["label"] == algo]
            by.append(f"{algo} {pts(sum(r['larger']['mean'] - r['baseline']['mean'] for r in rows) / len(rows))}")
        out.append(f"<b>Entropy bonus 0.2 instead of 0.01</b> (an ablation; it changes the optimization problem, so it does not show convergence to the unregularized equilibrium), random board: {', '.join(by)} points.")
    for s in res.get("choices") or []:
        field = "tie_nash_mean" if s["metric"] == "tie" else "mean"
        first = s["keys"][0][0]
        parts = "; ".join(f"{lab} {change_range(s['rows'], k, first, field)}" for k, lab in s["keys"][1:])
        what = "share tied with the exact Nash policy" if s["metric"] == "tie" else "win rate"
        out.append(f"<b>{s['title']}</b>: {what}, change in points from \"{s['keys'][0][1]}\": {parts}.")
    return out


def findings_section(res: dict) -> str:
    det, rnd, mixed = res, res["random"], res["mixed"]
    cont = res.get("continuing")
    learners = rnd["learners"]
    ball = [x["vs_best_response"][0] for x in learners]
    other = [x["vs_best_response_column"][0] for x in learners]
    bal = [(a + b) / 2 for a, b in zip(ball, other)]
    ex = rnd["exact"]
    ex_ball, ex_other = ex["vs_best_response"][0], ex["vs_best_response_column"][0]
    rand = [x["vs_random"][0] for x in det["learners"]]
    mx = [x["vs_nash"][0] for x in mixed["learners"]]
    lines = [
        f"<b>No learner replicates the exact solver.</b> On the random board the exact solver wins {pct(ex_ball)} from the ball seat, {pct(ex_other)} from the other seat and "
        f"{pct((ex_ball + ex_other) / 2)} balanced; the six learners win {pct(min(ball))} to {pct(max(ball))}, {pct(min(other))} to {pct(max(other))} and {pct(min(bal))} to {pct(max(bal))}. "
        "On the deterministic board the exact solver ties itself and the learners win "
        f"{pct(max(x['vs_best_response'][0] for x in det['learners']))} or less"
        f"{'; in the continuing game every learner wins 0%' if cont else ''}.",
        f"Against a random player the learners win {pct(min(rand))} to {pct(max(rand))} of games (deterministic board); fictitious play does better than standard training.",
        f"Starting from the {mixed['mixed_states']} states where the exact answer has to mix, the learners win {pct(min(mx))} to {pct(max(mx))} against the exact Nash policy; the exact solver wins {pct(mixed['exact']['vs_nash'][0])}.",
    ]
    more = res.get("more_seeds")
    if more:
        c = more["conditions"]
        lines.append(
            f"<b>Best settings, on {more['seeds']} seeds:</b> A2C fictitious play at 10,000 iterations wins {pct(c['a2c_fp_10k']['ball_ci'][0])} from the ball seat "
            f"(95% interval {pct(c['a2c_fp_10k']['ball_ci'][1])} to {pct(c['a2c_fp_10k']['ball_ci'][2])}) against {pct(c['a2c_fp_base']['ball_ci'][0])} at 2,000; PPO with replayed data wins "
            f"{pct(c['ppo_sp_replay4']['ball_ci'][0])} ({pct(c['ppo_sp_replay4']['ball_ci'][1])} to {pct(c['ppo_sp_replay4']['ball_ci'][2])}) against {pct(c['ppo_sp_base']['ball_ci'][0])}. Both gains hold up and stay well below the exact solver.")
    return f"""<section><h2>3. Findings</h2><ul>{"".join(f"<li>{x}</li>" for x in lines)}</ul></section>"""


def choices_section(res: dict) -> str:
    lines = finding_lines(res)
    return f"""<section><h2>6. Training choices</h2>
<p>Each comparison changes one thing from the usual setting (2,000 iterations of 64 episodes, a snapshot every 5 iterations, all 100 steps in the loss) on 3 seeds.
Changes are in percentage points; single seeds of one setting differ by 10 to 30 points, so differences of that size are noise.</p>
<ul>{"".join(f"<li>{x}</li>" for x in lines)}</ul>{ten_seeds_block(res)}</section>"""


def build_html(res: dict) -> str:
    boards = [("Deterministic board (A10)", res)]
    boards.append(("Random move-order board", res["random"]))
    if res.get("continuing"):
        boards.append(("Continuing game", res["continuing"]))
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
table.wins td,table.wins th{{padding:4px 5px;font-size:10.5px}} table.wins td:first-child{{white-space:nowrap}}
.sub{{display:block;font-size:10px;color:var(--muted);font-weight:400}}
svg{{width:100%;height:auto}}
.legend span{{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;vertical-align:-1px}}
ul{{margin:6px 0 6px 18px;padding:0}} li{{margin:3px 0}}
ul.related{{font-size:12px;line-height:1.45}} ul.related li{{margin:2px 0}}
@page{{size:A4;margin:14mm}}
</style></head><body>
<div class="eyebrow">Policy gradient &middot; soccer game</div>
<h1>Policy gradient on the soccer game</h1>
<p class="lede">REINFORCE, A2C and PPO, trained the standard way and by fictitious play, scored by how often the trained player wins when the
opponent is the best response to it.</p>

<section><h2>1. Win rates against the best response</h2>
{win_rate_table(boards)}
<ul>
<li><b>Win rate</b>: the share of 1,000 games the trained player wins (it scores while the opponent does not; games with no goal or equal goals are ties), when the
opponent plays the exact best response to it, the move that minimizes its expected discounted score difference. Mean of 3 seeds.</li>
<li><b>Ball seat</b>: the player that holds the ball at kickoff. <b>Other seat</b>: the same learner as the other player. <b>Balanced</b>: their 50/50 average, one number that does not
depend on who starts with the ball. The exact solver is the top row, as context rather than a target.</li>
<li>The random board's kickoff is worth {res["random"].get("kickoff_value", 0.0):+.2f} to the ball holder even against a perfect best response, so the exact solver wins more than half its games from that seat.
On the deterministic board the exact solver ties itself; in the continuing game a win is more goals than the opponent in 100 steps. A dash means the learner was not run on that board.</li>
</ul></section>

<section><h2>2. Setup</h2>
{table(["setting", "value"], [
 ["Game", "7&times;5 board, goal rows 1&ndash;3, fixed 100 steps, discount 0.9; the number of steps left is a network input. Three boards: deterministic (A10), random move order, and the continuing game (play restarts after a goal)."],
 ["Algorithms", "REINFORCE (discounted returns), A2C (10-step bootstrapped advantage; also with the exact solver's values as a frozen critic), PPO (GAE, clip 0.2, 4 epochs)."],
 ["Updates", "Each iteration: 64 episodes of 100 steps, one gradient step per player on all the data. 2,000 iterations, learning rate 10<sup>-3</sup>, entropy bonus 0.01, 64&times;64 networks, a separate network per player."],
 ["Standard", "Both current networks play each other and keep updating."],
 ["Fictitious play", "Each player trains against a frozen snapshot of the other drawn from all its past snapshots (one every 5 iterations); it reports the average of its own snapshots. Policies, never weights, are averaged."]], "text")}
</section>

{findings_section(res)}

{curves_section(res)}

{rps_section(res)}

{choices_section(res)}

{related_section(res)}
</body></html>"""


def main() -> None:
    (DOCS / "policy_gradient.html").write_text(build_html(load_data()))
    print("wrote docs/policy_gradient.html")


if __name__ == "__main__":
    main()
