import { useState } from "react";
import Nav from "./Nav.jsx";
import Footer from "./Footer.jsx";
import PolicyOutputs from "./PolicyOutputs.jsx";
import pgPolicies from "./pgPolicies.json";
import results from "./pgResults.json";
import "./landing.css";

const pct = (v) => `${Math.round(v * 100)}%`;
const name = (l) => `${l.label}, ${l.training}`;

// The headline number: how often the trained player wins when the opponent is the best response
// to it, with the exact solver's own rate as the reference line.
function BestResponseBars({ rows, reference }) {
  const W = 660, left = 210, right = 54, rowH = 26;
  const H = rowH * rows.length + 40;
  const x = (v) => left + v * (W - left - right);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="win rate against the best response" className="pg-bars">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={x(g)} y1="22" x2={x(g)} y2={H - 22} stroke="var(--rule)" />
          <text x={x(g)} y={H - 8} textAnchor="middle" fontSize="11" fill="var(--ink-faint)">{pct(g)}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = 26 + i * rowH;
        return (
          <g key={r.name}>
            <text x={left - 8} y={y + 13} textAnchor="end" fontSize="12" fill="var(--ink)">{r.name}</text>
            <rect x={left} y={y + 3} width={Math.max(x(r.win) - left, 0)} height="14" fill="var(--pitch)" opacity=".85" />
            <text x={W - right + 10} y={y + 14} fontSize="12" fontWeight="700" fill="var(--ink)">{pct(r.win)}</text>
          </g>
        );
      })}
      <line x1={x(reference)} y1="18" x2={x(reference)} y2={H - 22} stroke="var(--ember)" strokeWidth="2" strokeDasharray="5 4" />
      <text x={x(reference)} y="12" textAnchor="middle" fontSize="11" fontWeight="700" fill="var(--ember)">exact solver {pct(reference)}</text>
    </svg>
  );
}

// Every win rate against the best response in one table, for both starting positions and their 50/50
// average, on every board that has results.
const SEATS = [["ball", "With ball"], ["other", "Without ball"], ["balanced", "Average of both"]];
const seatRates = (x) => {
  const ball = x.vs_best_response[0], other = x.vs_best_response_column[0];
  return { ball, other, balanced: (ball + other) / 2 };
};
function WinRateTable({ boards }) {
  const names = [];
  boards.forEach(([, , b]) => b.learners.forEach((l) => { if (!names.includes(name(l))) names.push(name(l)); }));
  return (
    <div className="tbl-wrap">
      <table className="win-table">
        <thead>
          <tr>
            <th rowSpan={2}>Learner</th>
            {boards.map(([k, title]) => <th key={k} colSpan={SEATS.length + 1}>{title}</th>)}
          </tr>
          <tr>{boards.flatMap(([k]) => [...SEATS.map(([f, s]) => <th key={k + "-" + f}>{s}</th>), <th key={k + "-gap"}>Nash gap</th>])}</tr>
        </thead>
        <tbody>
          <tr>
            <td>Exact solver</td>
            {boards.flatMap(([k, , b]) => [...SEATS.map(([f]) => <td key={k + "-" + f}>{pct(seatRates(b.exact)[f])}</td>), <td key={k + "-gap"}>{b.exact.exploitability.mean.toFixed(3)}</td>])}
          </tr>
          {names.map((n) => (
            <tr key={n}>
              <td>{n}</td>
              {boards.flatMap(([k, , b]) => {
                const l = b.learners.find((x) => name(x) === n);
                return [...SEATS.map(([f]) => <td key={k + "-" + f}>{l ? pct(seatRates(l)[f]) : "\u2014"}</td>), <td key={k + "-gap"}>{l ? l.exploitability.mean.toFixed(3) : "\u2014"}</td>];
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Wins (green), ties (grey) and losses (orange): one stacked bar per learner and one panel per
// opponent or setting, all sharing the label column.
function StackBars({ rows, titles }) {
  const W = 660, left = 210, gap = 24, rowH = 26, n = titles.length;
  const panel = (W - left - gap * (n - 1) - 8) / n;
  const H = rowH * rows.length + 34;
  const cols = ["var(--pitch)", "var(--rule-strong)", "var(--ember)"];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="wins, ties and losses of each learner" className="pg-bars">
      {titles.map((title, k) => (
        <text key={title} x={left + k * (panel + gap)} y="12" fontSize="11" fontWeight="700" fill="var(--ink-faint)">{title}</text>
      ))}
      {rows.map((r, i) => {
        const y = 22 + i * rowH;
        return (
          <g key={r.name}>
            <text x={left - 8} y={y + 12} textAnchor="end" fontSize="12" fill="var(--ink)">{r.name}</text>
            {r.panels.map((tr, k) => {
              let x = left + k * (panel + gap);
              return tr.map((v, jx) => {
                const rect = <rect key={`${k}-${jx}`} x={x} y={y} width={v * panel} height="16" fill={cols[jx]} />;
                x += v * panel;
                return rect;
              });
            })}
          </g>
        );
      })}
    </svg>
  );
}

// Lines over a shared x axis: win rate against the best response after each round of long
// best-response phases, win rate or a move probability during training. Series may carry their own
// colour and width; ``top`` fixes the top of the y axis.
const LINE_COLOURS = ["var(--pitch)", "var(--pitch-dark, #178A42)", "var(--rule-strong)", "var(--ink-faint)", "var(--ember)", "var(--ember-soft, #d08a5c)"];
function LineChart({ series, rounds, top: fixedTop, xLabel = "round", height = 260, labelEvery = 1 }) {
  const W = 660, H = height, lm = 48, rm = 170, tm = 20, bm = 34;
  const top = fixedTop || Math.min(1, Math.round((Math.max(0.1, ...series.flatMap((s) => s.values.filter((v) => v != null))) * 1.15 + 0.02) * 100) / 100);
  const x = (k) => lm + (k / Math.max(rounds.length - 1, 1)) * (W - lm - rm);
  const y = (v) => tm + (1 - v / top) * (H - tm - bm);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="lines over training" className="pg-bars">
      {[0, top / 2, top].map((g) => (
        <g key={g}>
          <line x1={lm} y1={y(g)} x2={W - rm} y2={y(g)} stroke="var(--rule)" />
          <text x={lm - 6} y={y(g) + 4} textAnchor="end" fontSize="11" fill="var(--ink-faint)">{pct(g)}</text>
        </g>
      ))}
      {rounds.map((k, i) => (i % labelEvery === 0 ? <text key={k} x={x(i)} y={H - 10} textAnchor="middle" fontSize="11" fill="var(--ink-faint)">{k}</text> : null))}
      <text x={lm} y={H} fontSize="11" fill="var(--ink-faint)">{xLabel}</text>
      {series.map((s, n) => {
        const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter(Boolean);
        const col = s.color || LINE_COLOURS[n % 6];
        return (
          <g key={s.name}>
            <path d={pts.map(([px, py], i) => `${i ? "L" : "M"}${px},${py}`).join(" ")} fill="none" stroke={col} strokeWidth={s.width || 2.2} opacity={s.opacity || 1} />
            {s.label !== false && <text x={W - rm + 8} y={pts[pts.length - 1][1] + 4} fontSize="11" fontWeight="700" fill={col}>{s.name}</text>}
          </g>
        );
      })}
    </svg>
  );
}

// Rock-paper-scissors with neural-network players: the share of rock over training for the current
// network (orange) and the aggregate policy (green); the dashed line is the equilibrium, 1/3.
const RPS_MODES = { standard: "Standard: both current networks play each other",
  fictitious: "Fictitious play: average of softmax snapshots",
  fictitious_argmax: "Fictitious play: average of argmax snapshots" };
function RpsNetPlot({ config, iterations }) {
  const W = 660, ph = 104, lm = 48, rm = 150, tm = 16;
  const modes = Object.keys(config.modes);
  const H = (ph + 22) * modes.length;
  const x = (k) => lm + (k / (iterations - 1)) * (W - lm - rm);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="share of rock over training" className="pg-bars">
      {modes.map((mode, n) => {
        const d = config.modes[mode], top = n * (ph + 22) + tm;
        const y = (v) => top + (1 - v) * (ph - tm);
        const path = (vals) => vals.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
        const final = d.final_exploitability.reduce((a, b) => a + b, 0) / d.final_exploitability.length;
        return (
          <g key={mode}>
            <text x={lm} y={top - 4} fontSize="12" fontWeight="700" fill="var(--ink)">{RPS_MODES[mode]}</text>
            {[[0, "0"], [1 / 3, "1/3"], [1, "1"]].map(([g, lab]) => (
              <g key={lab}>
                <line x1={lm} y1={y(g)} x2={W - rm} y2={y(g)} stroke="var(--rule)" strokeDasharray={lab === "1/3" ? "4 3" : undefined} />
                <text x={lm - 6} y={y(g) + 4} textAnchor="end" fontSize="11" fill="var(--ink-faint)">{lab}</text>
              </g>
            ))}
            <path d={path(d.current)} fill="none" stroke="var(--ember)" strokeWidth="1" opacity=".8" />
            {mode !== "standard" && <path d={path(d.aggregate)} fill="none" stroke="var(--pitch)" strokeWidth="2.4" />}
            <text x={W - rm + 8} y={top + 14} fontSize="11" fill="var(--ink-faint)">exploitability</text>
            <text x={W - rm + 8} y={top + 30} fontSize="12" fontWeight="700" fill="var(--ink)">{final.toFixed(2)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// Win rate before and after a change, next to the exact solver: one group per starting position, two bars
// with their 95% intervals and a grey bar for the exact solver.
function BeforeAfterChart({ groups, names }) {
  const W = 660, left = 170, right = 24, topPad = 40, bar = 15, gap = 3, maxV = 0.8;
  const block = 3 * (bar + gap) + 22, H = topPad + block * groups.length + 30;
  const x = (v) => left + (v / maxV) * (W - left - right);
  const cols = ["var(--pitch)", "var(--ember)", "var(--rule-strong)"];
  const labels = [names[0], names[1], "Exact solver"];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="win rate before and after a change" className="pg-bars">
      {labels.map((l, k) => {
        const lx = left + labels.slice(0, k).reduce((a, s) => a + 16 + 7 * s.length + 18, 0);
        return (
          <g key={l}>
            <rect x={lx} y="8" width="11" height="11" fill={cols[k]} opacity=".9" />
            <text x={lx + 16} y="18" fontSize="11" fill="var(--ink)">{l}</text>
          </g>
        );
      })}
      {[0, 0.2, 0.4, 0.6, 0.8].map((g) => (
        <g key={g}>
          <line x1={x(g)} y1={topPad - 6} x2={x(g)} y2={H - 28} stroke="var(--rule)" />
          <text x={x(g)} y={H - 12} textAnchor="middle" fontSize="11" fill="var(--ink-faint)">{pct(g)}</text>
        </g>
      ))}
      {groups.map((g, i) => {
        const y0 = topPad + i * block;
        return (
          <g key={g.label}>
            <text x={left - 10} y={y0 + 1.5 * (bar + gap) + 2} textAnchor="end" fontSize="12" fontWeight="700" fill="var(--ink)">{g.label}</text>
            {["base", "variant"].map((key, k) => {
              const [mean, lo, hi] = g[key], y = y0 + k * (bar + gap), mid = y + bar / 2;
              return (
                <g key={key}>
                  <rect x={left} y={y} width={Math.max(x(mean) - left, 0)} height={bar} fill={cols[k]} opacity=".9" />
                  <line x1={x(lo)} y1={mid} x2={x(hi)} y2={mid} stroke="var(--ink)" strokeWidth="1.6" />
                  <line x1={x(lo)} y1={mid - 4} x2={x(lo)} y2={mid + 4} stroke="var(--ink)" strokeWidth="1.6" />
                  <line x1={x(hi)} y1={mid - 4} x2={x(hi)} y2={mid + 4} stroke="var(--ink)" strokeWidth="1.6" />
                  <text x={x(hi) + 8} y={y + 11} fontSize="11" fontWeight="700" fill="var(--ink)">
                    {pct(mean)}{pct(lo) !== pct(hi) && <tspan fontWeight="400" fill="var(--ink-faint)"> ({pct(lo)} to {pct(hi)})</tspan>}
                  </text>
                </g>
              );
            })}
            <rect x={left} y={y0 + 2 * (bar + gap)} width={x(g.exact) - left} height={bar} fill={cols[2]} opacity=".9" />
            <text x={x(g.exact) + 8} y={y0 + 2 * (bar + gap) + 11} fontSize="11" fontWeight="700" fill="var(--ink)">{pct(g.exact)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// Win rate against the best response with separate networks, one shared network, and trimmed
// episodes: a bar per variant (mean of the seeds) and a dot per seed.
const VARIANT_KEYS = [["baseline", "Separate networks", "var(--pitch)"],
  ["shared", "One shared network", "var(--ink-faint)"], ["trimmed", "Last 10 steps left out", "var(--ember)"]];
function VariantBars({ rows, reference, field = "", keys = VARIANT_KEYS }) {
  const mean = (v) => v[`${field}mean`];
  const runs = (v) => v[`${field}runs`];
  const W = 660, left = 170, right = 20, bar = 16, group = keys.length * (bar + 2) + 16;
  const H = group * rows.length + 48;
  const x = (v) => left + v * (W - left - right);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="win rate against the best response by variant" className="pg-bars">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={x(g)} y1="22" x2={x(g)} y2={H - 22} stroke="var(--rule)" />
          <text x={x(g)} y={H - 8} textAnchor="middle" fontSize="11" fill="var(--ink-faint)">{pct(g)}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y0 = 28 + i * group;
        return (
          <g key={`${r.label}-${r.training}`}>
            <text x={left - 8} y={y0 + group / 2 - 6} textAnchor="end" fontSize="12" fill="var(--ink)">{r.label}{r.training ? `, ${r.training}` : ""}</text>
            {keys.map(([k, , col], n) => (
              <g key={k}>
                <rect x={left} y={y0 + n * (bar + 2)} width={Math.max(x(mean(r[k])) - left, 0)} height={bar} fill={col} opacity=".85" />
                {runs(r[k]).map((v, m) => <circle key={m} cx={x(v)} cy={y0 + n * (bar + 2) + bar / 2} r="2.5" fill="var(--ink)" />)}
                <text x={x(Math.min(Math.max(...runs(r[k])), 0.93)) + 8} y={y0 + n * (bar + 2) + 12} fontSize="11" fontWeight="700" fill="var(--ink)">{pct(mean(r[k]))}</text>
              </g>
            ))}
          </g>
        );
      })}
      <line x1={x(reference)} y1="18" x2={x(reference)} y2={H - 22} stroke="var(--ember)" strokeWidth="2" strokeDasharray="5 4" />
      <text x={x(reference)} y="12" textAnchor="middle" fontSize="11" fontWeight="700" fill="var(--ember)">exact solver {pct(reference)}</text>
    </svg>
  );
}

// Exact gradient play: per condition, the mean duality gap over states and the gap at the kickoff, median over runs
// as a bar and each run as a dot.
function GapBars({ conditions }) {
  const W = 660, left = 250, right = 40, bar = 13, group = 2 * (bar + 2) + 14;
  const H = group * conditions.length + 48;
  const x = (v) => left + Math.min(v, 1) * (W - left - right);
  const update = { adam: "simultaneous Adam", sgd: "simultaneous gradient steps", eg: "extragradient" };
  const fields = [["gap_mean", "var(--pitch)"], ["gap_kickoff", "var(--ember)"]];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="exact duality gap by condition" className="pg-bars">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={x(g)} y1="22" x2={x(g)} y2={H - 22} stroke="var(--rule)" />
          <text x={x(g)} y={H - 8} textAnchor="middle" fontSize="11" fill="var(--ink-faint)">{g}</text>
        </g>
      ))}
      <rect x={left} y="6" width="10" height="10" fill="var(--pitch)" /><text x={left + 14} y="15" fontSize="11" fill="var(--ink)">mean gap over states</text>
      <rect x={left + 150} y="6" width="10" height="10" fill="var(--ember)" /><text x={left + 164} y="15" fontSize="11" fill="var(--ink)">gap at the kickoff</text>
      {conditions.map((c, i) => {
        const y0 = 28 + i * group;
        return (
          <g key={`${c.board}-${c.policy}-${c.method}-${c.lr}`}>
            <text x={left - 8} y={y0 + group / 2 - 4} textAnchor="end" fontSize="12" fill="var(--ink)">
              {c.board === "a10" ? "A10" : "random"}, {c.policy}, {update[c.method]} {c.lr}
            </text>
            {fields.map(([f, col], n) => {
              const runs = [...c[f]].sort((a, b) => a - b), mid = runs[Math.floor(runs.length / 2)], y = y0 + n * (bar + 2);
              return (
                <g key={f}>
                  <rect x={left} y={y} width={Math.max(x(mid) - left, 0)} height={bar} fill={col} opacity=".85" />
                  {runs.map((v, m) => <circle key={m} cx={x(v)} cy={y + bar / 2} r="2.3" fill="var(--ink)" />)}
                  <text x={x(runs[runs.length - 1]) + 6} y={y + 10} fontSize="11" fontWeight="700" fill="var(--ink)">{mid.toFixed(2)}</text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

// Rock-paper-scissors: the share of rock in the current play of best-response dynamics (it
// cycles for ever) against in the average of all play so far (it settles at one third).
function RpsPlot({ data }) {
  const W = 660, H = 250, l = 56, r = 20, t = 50, b = 40;
  const x = (k) => l + (k / (data.rounds - 1)) * (W - l - r);
  const y = (v) => t + (1 - v) * (H - t - b);
  const line = (vals) => vals.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="best-response dynamics against fictitious play on rock-paper-scissors" className="pg-bars">
      {[0, 1 / 3, 2 / 3, 1].map((g) => (
        <g key={g}>
          <line x1={l} y1={y(g)} x2={W - r} y2={y(g)} stroke="var(--rule)" strokeDasharray={g === 1 / 3 ? "4 3" : undefined} />
          <text x={l - 8} y={y(g) + 4} textAnchor="end" fontSize="11" fill="var(--ink-faint)">{g === 1 / 3 ? "1/3" : g === 2 / 3 ? "2/3" : g}</text>
        </g>
      ))}
      <path d={line(data.best_response)} fill="none" stroke="var(--ember)" strokeWidth="1.6" opacity=".85" />
      <path d={line(data.average)} fill="none" stroke="var(--pitch)" strokeWidth="2.6" />
      <text x={l} y={H - 8} fontSize="11" fill="var(--ink-faint)">round 1</text>
      <text x={W - r} y={H - 8} textAnchor="end" fontSize="11" fill="var(--ink-faint)">round {data.rounds}</text>
      <text x={W - r} y="16" textAnchor="end" fontSize="12" fill="var(--ember)">best-response dynamics: the current play flips every round</text>
      <text x={W - r} y="32" textAnchor="end" fontSize="12" fontWeight="700" fill="var(--pitch)">fictitious play: the average settles near 1/3</text>
    </svg>
  );
}

const BOARDS = [
  ["deterministic", "Deterministic board (A10)"],
  ["random", "Random move-order board"],
  ["continuing", "Goal-reset game (fixed horizon)"],
];

export default function PolicyPage() {
  const [which, setWhich] = useState("deterministic");
  const { longer, fp_br: fpBr, mixed, rps, variants, argmax, rps_nn: rpsNn, entropy, choices, curves, related, more_seeds: moreSeeds, symmetry, small_games: smallGames, rps_phases: rpsPhases, exact_play: exactPlay } = results;
  const vr = variants && variants.random, vd = variants && variants.deterministic, vc = variants && variants.continuing;
  const boards = { deterministic: results, random: results.random, continuing: results.continuing };
  const available = BOARDS.filter(([k]) => boards[k]);
  const board = boards[which];
  const { learners, exact } = board;
  const rnd = results.random;
  const bestRow = (l) => ({ name: name(l), win: l.vs_best_response[0] });
  const winRows = [
    { name: "Exact solver", panels: [exact.vs_random, exact.vs_nash, exact.vs_best_response] },
    ...learners.map((l) => ({ name: name(l), panels: [l.vs_random, l.vs_nash, l.vs_best_response] })),
  ];
  const mixedRows = mixed && [
    { name: "Exact solver", panels: [mixed.exact.vs_nash, mixed.exact.vs_best_response] },
    ...mixed.learners.map((l) => ({ name: name(l), panels: [l.vs_nash, l.vs_best_response] })),
  ];
  const longRows = longer.map((l) => ({ name: name(l), panels: [l.short_vs_nash, l.vs_nash, l.vs_random] }));
  const fpRounds = [...new Set(fpBr.flatMap((r) => r.checkpoints.map((c) => c.round)))].sort((a, b) => a - b);
  const fpSeries = fpBr.map((r) => ({
    name: `${r.label}, ${r.best_response_iterations} iterations`,
    values: fpRounds.map((k) => { const c = r.checkpoints.find((x) => x.round === k); return c ? c.win_vs_best_response : null; }),
  }));
  const seatRows = [
    { name: "Exact solver", panels: [rnd.exact.vs_best_response, rnd.exact.vs_best_response_column] },
    ...rnd.learners.map((l) => ({ name: name(l), panels: [l.vs_best_response, l.vs_best_response_column] })),
  ];
  const brWins = learners.map((l) => l.vs_best_response[0]);
  const tieRange = (training) => {
    const v = learners.filter((l) => l.training === training).map((l) => l.vs_nash[1]);
    return [Math.min(...v), Math.max(...v)];
  };
  const [sLo, sHi] = tieRange("standard");
  const [fLo, fHi] = tieRange("fictitious play");
  const critic = learners.find((l) => l.label === "A2C, exact critic" && l.training === "standard");
  const plain = learners.find((l) => l.label === "A2C" && l.training === "standard");
  const tieText = (sHi >= 0.01
    ? `Against the exact Nash policy the standard learners tie ${pct(sLo)} to ${pct(sHi)} of games and the fictitious-play learners ${pct(fLo)} to ${pct(fHi)}.`
    : "") + (critic && plain
    ? ` A2C with the exact critic ties the exact Nash policy in ${pct(critic.vs_nash[1])} of games, against ${pct(plain.vs_nash[1])} for A2C with a learned critic.`
    : "");
  const pick = (rows, label, training) => rows.find((v) => v.label === label && v.training === training);
  const rng = (lo, hi) => (pct(lo) === pct(hi) ? pct(lo) : `${pct(lo)} to ${pct(hi)}`);
  const winsClause = (best) => (best < 0.005 ? "any games" : `more than ${pct(best)} of games`);
  const pts = (v) => { const n = Math.round(v * 100); return `${n >= 0 ? "+" : "\u2212"}${Math.abs(n)}`; };
  const spread = (rows, f) => Math.round(100 * Math.max(...rows.flatMap((v) => [v.baseline, v.shared].map((x) => {
    const r = x[f]; return Math.max(...r) - Math.min(...r);
  }))));
  const change = (rows, key, f) => rows.map((v) => pts(v[key][f] - v.baseline[f])).join(", ");
  const order = "(A2C standard, A2C fictitious play, PPO standard, PPO fictitious play)";
  const variantText = vr
    ? `Random board: against separate networks, a shared network changes the win rate by ${change(vr, "shared", "mean")} points ${order}; ` +
      `single seeds of the same setup differ by up to ${spread(vr, "runs")} points. ` +
      `Leaving out the last 10 steps changes A2C by ${pts(pick(vr, "A2C", "standard").trimmed.mean - pick(vr, "A2C", "standard").baseline.mean)} and ${pts(pick(vr, "A2C", "fictitious play").trimmed.mean - pick(vr, "A2C", "fictitious play").baseline.mean)} points, ` +
      `but PPO with standard training stops scoring: it wins ${pct(pick(vr, "PPO", "standard").trimmed.vs_random[0])} of games against a random player, against ${pct(pick(vr, "PPO", "standard").baseline.vs_random[0])} before. ` +
      `No variant gets close to the exact solver.`
    : "";
  const meanOf = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const rpsFirst = rpsNn && rpsNn.configs && Object.values(rpsNn.configs)[0];
  const rpsSecond = rpsNn && rpsNn.configs && Object.values(rpsNn.configs)[1];
  const rpsText = rpsFirst
    ? `With the ${Object.keys(rpsNn.configs)[0]} the standard networks cycle (exploitability ${meanOf(rpsFirst.modes.standard.final_exploitability).toFixed(2)}), and neither form of fictitious play reaches one third (softmax ${meanOf(rpsFirst.modes.fictitious.final_exploitability).toFixed(2)}, argmax ${meanOf(rpsFirst.modes.fictitious_argmax.final_exploitability).toFixed(2)}). ` +
      (rpsSecond ? `With the ${Object.keys(rpsNn.configs)[1]} all three settle close to the equilibrium (standard ${meanOf(rpsSecond.modes.standard.final_exploitability).toFixed(2)}, softmax ${meanOf(rpsSecond.modes.fictitious.final_exploitability).toFixed(2)}, argmax ${meanOf(rpsSecond.modes.fictitious_argmax.final_exploitability).toFixed(2)}): this setting damps the cycling, so the standard networks settle too (a stabilization effect, not evidence of convergence to the unregularized equilibrium).` : "")
    : "";
  const AM_KEYS = [["softmax", "Average of softmax snapshots", "var(--pitch)"], ["argmax", "Average of argmax snapshots", "var(--ember)"]];
  const amChange = (rows, f) => rows.map((v) => pts(v.argmax[f] - v.softmax[f])).join(", ");
  const amSpread = (rows, f) => Math.round(100 * Math.max(...rows.flatMap((v) => [v.softmax, v.argmax].map((x) => Math.max(...x[f]) - Math.min(...x[f])))));
  const amText = argmax
    ? `Random board: averaging argmax instead of softmax snapshots changes the win rate against the best response by ${amChange(argmax.random, "mean")} points (REINFORCE, A2C, PPO); single seeds differ by up to ${amSpread(argmax.random, "runs")} points. ` +
      `Deterministic board: it changes the share of games tied with the exact Nash policy by ${amChange(argmax.deterministic, "tie_nash_mean")} points; single seeds differ by up to ${amSpread(argmax.deterministic, "tie_nash_runs")} points.` +
      (argmax.continuing ? ` Continuing game: the share tied with the exact Nash policy changes by ${amChange(argmax.continuing, "tie_nash_mean")} points; single seeds differ by up to ${amSpread(argmax.continuing, "tie_nash_runs")} points, and no argmax learner wins ${winsClause(Math.max(...argmax.continuing.map((v) => v.argmax.mean)))} against the best response.` : "")
    : "";
  const EN_KEYS = [["baseline", "Entropy bonus 0.01 (used elsewhere)", "var(--pitch)"], ["larger", "Entropy bonus 0.2", "var(--ember)"]];
  const enChange = (rows, f) => rows.map((v) => pts(v.larger[f] - v.baseline[f])).join(", ");
  const enRandom = (rows) => rows.map((v) => pts(v.larger.vs_random[0] - v.baseline.vs_random[0])).join(", ");
  const enBoards = entropy ? [["random", "Random move-order board: win rate against the best response", "", rnd.exact.vs_best_response[0], "mean"],
    ["deterministic", "Deterministic board: games tied with the exact Nash policy", "tie_nash_", results.exact.vs_nash[1], "tie_nash_mean"],
    ["continuing", "Continuing game: games tied with the exact Nash policy", "tie_nash_", results.continuing && results.continuing.exact.vs_nash[1], "tie_nash_mean"]].filter(([k]) => entropy[k]) : [];
  const enByAlgo = entropy && entropy.random
    ? ["REINFORCE", "A2C", "PPO"].map((a) => {
      const rows = entropy.random.filter((v) => v.label === a);
      return `${a} ${pts(rows.reduce((s, v) => s + v.larger.mean - v.baseline.mean, 0) / rows.length)}`;
    }).join(", ")
    : "";
  const STUDY_COLOURS = ["var(--pitch)", "var(--pitch-dark, #178A42)", "var(--rule-strong)", "var(--ink-faint)", "var(--ember)"];
  const studyKeys = (s) => s.keys.map(([k, l], i) => [k, l, STUDY_COLOURS[i % STUDY_COLOURS.length]]);
  const studyField = (s) => (s.metric === "tie" ? "tie_nash_" : "");
  const studyMean = (s, row, k) => row[k][`${studyField(s)}mean`];
  const studyText = (s) => {
    const first = s.keys[0][0], what = s.metric === "tie" ? "share tied with the exact Nash policy" : "win rate against the best response";
    const lines = s.keys.slice(1).map(([k, l]) => `${l}: ${s.rows.map((r) => pts(studyMean(s, r, k) - studyMean(s, r, first))).join(", ")}`);
    const spread = Math.round(100 * Math.max(...s.rows.flatMap((r) => s.keys.map(([k]) => {
      const v = r[k][`${studyField(s)}runs`]; return Math.max(...v) - Math.min(...v);
    }))));
    return `Change in the ${what}, in points, from "${s.keys[0][1]}" (rows in the order shown): ${lines.join("; ")}. Single seeds of one setting differ by up to ${spread} points.`;
  };
  const curveBoards = curves ? Object.entries(curves) : [];
  const msGroups = (pair) => {
    const b = moreSeeds.conditions[pair.baseline_key], v = moreSeeds.conditions[pair.variant_key];
    const eb = rnd.exact.vs_best_response[0], eo = rnd.exact.vs_best_response_column[0];
    return [["ball", "Starts with the ball", eb], ["other", "Starts without the ball", eo], ["balanced", "Average of both", (eb + eo) / 2]]
      .map(([m, label, exact]) => ({ label, base: b[`${m}_ci`], variant: v[`${m}_ci`], exact }));
  };
  const ci = (v) => `${pct(v[0])} (95% interval ${pct(v[1])} to ${pct(v[2])})`;
  const dci = (v) => `${pts(v[0])} points (95% interval ${pts(v[1])} to ${pts(v[2])})`;
  const msText = (pair) => {
    const b = moreSeeds.conditions[pair.baseline_key], v = moreSeeds.conditions[pair.variant_key];
    return `${pair.label}, ${pair.variant}: starts with the ball ${ci(b.ball_ci)} before and ${ci(v.ball_ci)} after, a change of ${dci(pair.ball_diff)}; averaged over both starts ${ci(b.balanced_ci)} before and ${ci(v.balanced_ci)} after, a change of ${dci(pair.balanced_diff)}. The interval for the change when it starts with the ball ${pair.ball_diff[1] > 0 ? "excludes" : "includes"} zero; when it starts without the ball both settings win ${pct(Math.max(...b.other, ...v.other))} or less in every run.`;
  };
  const symNames = symmetry && symmetry.length ? [...new Set(symmetry.flatMap((b) => b.rows.map((r) => r.name)))] : [];
  const symGap = (b, n) => { const r = b.rows.find((x) => x.name === n); return r ? r.gap.toFixed(2) : "\u2014"; };
  const symAll = symmetry ? symmetry.flatMap((b) => b.rows.filter((r) => r.name !== "Exact solver").map((r) => r.gap)) : [];
  const symPairs = symmetry ? symmetry.flatMap((b) => ["REINFORCE", "A2C", "PPO"].map((a) => [b.rows.find((r) => r.name === `${a}, standard`).gap, b.rows.find((r) => r.name === `${a}, fictitious play`).gap])) : [];
  const phaseText = rpsPhases
    ? `Training each best response longer, with snapshots after ${rpsPhases.phases.join(", ")} iterations (${rpsPhases.iterations.toLocaleString()} iterations, ${rpsPhases.seeds} seeds), the exploitability of the average of softmax snapshots is ${rpsPhases.phases.map((p) => { const v = rpsPhases.runs[`${p}/fictitious`].final_exploitability; return (v.reduce((a, b) => a + b, 0) / v.length).toFixed(2); }).join(", ")} and of argmax snapshots ${rpsPhases.phases.map((p) => { const v = rpsPhases.runs[`${p}/fictitious_argmax`].final_exploitability; return (v.reduce((a, b) => a + b, 0) / v.length).toFixed(2); }).join(", ")}.`
    : "";
  const detText = vd
    ? `Deterministic board: no variant wins ${winsClause(Math.max(...vd.flatMap((v) => [v.baseline, v.shared, v.trimmed].map((x) => x.mean))))} against the best response, as for the baseline (the exact solver wins 0%, since it ties itself). ` +
      `The share of games tied with the exact Nash policy changes by ${change(vd, "shared", "tie_nash_mean")} points with a shared network and by ${change(vd, "trimmed", "tie_nash_mean")} points with the last 10 steps left out ${order}; ` +
      `single seeds of the same setup differ by up to ${spread(vd, "tie_nash_runs")} points. ` +
      `A tie does not mean equilibrium play: PPO with standard training ties all games once the last 10 steps are left out, but wins only ${pct(pick(vd, "PPO", "standard").trimmed.vs_random[0])} of games against a random player, against ${pct(pick(vd, "PPO", "standard").baseline.vs_random[0])} before.`
    : "";
  const contText = vc
    ? `Continuing game: no variant wins ${winsClause(Math.max(...vc.flatMap((v) => [v.baseline, v.shared, v.trimmed].map((x) => x.mean))))} against the best response, as for the baseline. ` +
      `The share of games tied with the exact Nash policy changes by ${change(vc, "shared", "tie_nash_mean")} points with a shared network and by ${change(vc, "trimmed", "tie_nash_mean")} points with the last 10 steps left out ${order}; ` +
      `single seeds of the same setup differ by up to ${spread(vc, "tie_nash_runs")} points, so three seeds cannot settle changes of this size.`
    : "";

  return (
    <>
      <Nav current="policy" />

      <section className="hero">
        <div className="hero-grid">
          <div>
            <div className="kicker">Policy gradient &middot; soccer game</div>
            <h1>Policy gradient on the soccer game</h1>
            <p className="lede">This study evaluates first-goal and fixed-horizon goal-reset soccer as complementary formats; neither is designated the primary result. First-goal play is tested on deterministic A10 and random move-order boards. Goal-reset play is tested on deterministic A10. Results stay separate because a win means a different thing in each format. REINFORCE, A2C and PPO use standard training and fictitious play.</p>
            <div className="cta-row">
              <a className="btn btn-primary" href="policy_gradient.pdf">Read the PDF &rarr;</a>
              <a className="btn btn-outline" href="policy_gradient_detailed.pdf">Detailed PDF &rarr;</a>
              <a className="btn btn-outline" href="policy_gradient.md">Full write-up &rarr;</a>
            </div>
          </div>
          <div className="hero-card">
            <div className="stat-list">
              <div>Win rate against the best response, random board: the exact solver wins <b>{pct(rnd.exact.vs_best_response[0])}</b>, the six learners <b>{pct(Math.min(...rnd.learners.map((l) => l.vs_best_response[0])))} to {pct(Math.max(...rnd.learners.map((l) => l.vs_best_response[0])))}</b></div>
              <div>Against a random player the learners win <b>{pct(Math.min(...results.learners.map((l) => l.vs_random[0])))} to {pct(Math.max(...results.learners.map((l) => l.vs_random[0])))}</b> of games</div>
              <div>From the {mixed.mixed_states} states where the exact answer has to mix, they win <b>{pct(Math.min(...mixed.learners.map((l) => l.vs_nash[0])))} to {pct(Math.max(...mixed.learners.map((l) => l.vs_nash[0])))}</b> against the exact Nash policy; the exact solver wins {pct(mixed.exact.vs_nash[0])}</div>
              <div><b>6</b> learners (REINFORCE, A2C, PPO; standard and fictitious play), <b>1,000</b> games per opponent</div>
            </div>
          </div>
        </div>
      </section>

      <section id="setup">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">01</span>How to read the numbers</div>
          <h2>Compare results within each game format</h2>
          <div className="tbl-wrap">
            <table className="text-table">
              <thead><tr><th>Number</th><th>What it is</th></tr></thead>
              <tbody>
                <tr><td>Win rate against the best response</td><td>The share of 1,000 games the trained player wins when the opponent plays the exact best response to it under discounted-score difference. This is an empirical outcome, not a guaranteed upper bound, because the best response does not optimize win probability.</td></tr>
                <tr><td>Nash gap (exploitability)</td><td>The sum of the two players&rsquo; gains in expected discounted score when each, in turn, replaces its policy with an exact best response to the other. It is 0 at the exact Nash equilibrium; smaller is closer to equilibrium. This payoff metric is distinct from win rate.</td></tr>
                <tr><td>Win rate against the exact Nash policy</td><td>The same, against the exact equilibrium player. The exact solver cannot be beaten by it; a learner that mixes well scores close to the exact solver&rsquo;s own rate.</td></tr>
                <tr><td>Win rate against a random player</td><td>A sanity check: every learner should beat a player who moves at random.</td></tr>
                <tr><td>A win</td><td>The player scores while the opponent does not. In the first two boards the first goal ends the game; in the continuing game play restarts after a goal and a win is more goals than the opponent in 100 steps. Games with no goal (or equal goals) are ties.</td></tr>
              </tbody>
            </table>
          </div>
          <h3>Why the Nash gap is also reported</h3>
          <p>The Nash gap (called exploitability in the experiment files) is the sum of both players&rsquo; gains in expected discounted score if each, in turn, switches to an exact best response against the other&rsquo;s fixed policy. It is zero at equilibrium, including the exact solver, and smaller is better. It evaluates the policy pair and is not a percentage or a win rate. The win-rate tables answer a separate question: how often the trained player wins against its exact best response under the discounted-score objective.</p>
          <h3>Why the exact solver does not win 0% against the best response</h3>
          <p>The best response is the move that minimizes our player&rsquo;s <em>expected discounted score difference</em>, not the one that
            minimizes its chance of winning. At the random board&rsquo;s kickoff the player holding the ball is ahead: the exact solution is worth
            {" "}{rnd.kickoff_value >= 0 ? "+" : "\u2212"}{Math.abs(rnd.kickoff_value).toFixed(2)} to it even against an exact best response. Its win rate is a reference outcome, not a theoretical ceiling, because the best response minimizes discounted score rather than win probability. The exact solver wins {pct(rnd.exact.vs_best_response[0])} of
            games and loses {pct(1 - rnd.exact.vs_best_response[0])} when it starts with the ball, and wins {pct(rnd.exact.vs_best_response_column[0])} when it starts without the ball. The first number
            is the one reported on this page (our learners always play the player who starts with the ball). On the deterministic board both starts tie every game.</p>
          <StackBars rows={seatRows} titles={["Random board, starting with the ball", "Random board, starting without the ball"]} />
          <p className="fig-cap">Win rate against the best response when starting with and without the ball. Wins (green), ties (grey) and losses (orange).</p>
        </div>
      </section>

      <section className="band-tint" id="learners">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">02</span>How it is implemented</div>
          <h2>Three algorithms, two ways of training</h2>
          <div className="tbl-wrap">
            <table className="text-table">
              <thead><tr><th>Setting</th><th>Value</th></tr></thead>
              <tbody>
                <tr><td>Game formats</td><td>Both are included as complementary conditions, with no primary format: first-goal termination on deterministic A10 and random move-order boards; fixed-horizon goal-reset play on deterministic A10. Each uses a 7&times;5 board, goal rows 1&ndash;3, 100 steps, discount 0.9, and remaining steps as a network input. First-goal outcomes count who scores first; goal-reset outcomes compare goal difference over 100 steps.</td></tr>
                <tr><td>Returns</td><td>REINFORCE: the discounted return from each step to the end of the episode, no baseline. A2C: 10-step bootstrapped advantage from a learned critic. A2C with the exact critic: the critic is the exact solver&rsquo;s value, frozen, so the advantage is the exact Q(s, a<sub>0</sub>, a<sub>1</sub>) minus V(s). PPO: GAE (&lambda; 0.95), ratio clipped at 0.2, 4 epochs.</td></tr>
                <tr><td>Updates</td><td>The saved comparisons use one REINFORCE/A2C update pass per collected batch; the runner now defaults to five shuffled passes (--updates-per-batch 5). PPO uses four epochs. Set the option to 1 to reproduce the saved REINFORCE/A2C comparisons. Every iteration collects 64 episodes of 100 steps. Other settings: 2,000 iterations, learning rate 10<sup>&minus;3</sup>, entropy bonus 0.01, 64&times;64 network, softmax output, separate network per player. No mini-batches or replay in the baseline comparisons.</td></tr>
                <tr><td>Standard</td><td>Both players&rsquo; current networks play each other and keep updating: real-time best responses.</td></tr>
                <tr><td>Fictitious play</td><td>Each player trains against a frozen snapshot of the other, drawn at random from all its past snapshots (one every 5 iterations); the policy it reports is the average of its own snapshots. Every snapshot follows a fixed budget of 5 policy-gradient iterations, and what is averaged is policies (action probabilities), never network weights.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section id="why">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">03</span>Why fictitious play</div>
          <h2>Best responses cycle; their average settles</h2>
          <p className="lede">On rock-paper-scissors, answering the opponent&rsquo;s latest move makes the play flip between
            rock, paper and scissors for ever. Answering the average of everything it has played converges to the equilibrium,
            one third each.</p>
          <RpsPlot data={rps} />
          {rpsNn && rpsNn.configs && (
            <>
              <h3>The same test with neural networks</h3>
              <p>Each player is a small network with a softmax output, trained by REINFORCE on sampled games
                ({rpsNn.iterations.toLocaleString()} iterations, {rpsNn.batch} games each, {rpsNn.seeds} seeds; the plots show seed 0).
                <b> Standard</b> trains the two current networks against each other. <b>Fictitious play</b> trains each player
                against a snapshot of the other drawn from all its past snapshots and reports the average of a player&rsquo;s
                snapshots; <b>argmax</b> first turns each snapshot into the pure policy that plays its most likely move, so the
                average is how often the snapshots play each move. Orange: the current network&rsquo;s share of rock; green: the
                aggregate. Exploitability (right) is what a perfect attacker could gain against the two networks&rsquo; policies, summed over both players: 0 at one third each, 2 at worst. It is the mean over seeds and the last 300 iterations.</p>
              {rpsText && <p>{rpsText}</p>}
              {phaseText && <p>{phaseText}</p>}
              {Object.entries(rpsNn.configs).map(([cname, cfg]) => (
                <div key={cname}>
                  <h3>{cname.charAt(0).toUpperCase() + cname.slice(1)}: {cfg.sgd ? `plain gradient steps at ${cfg.lr}` : `Adam at ${cfg.lr}`}, entropy bonus {cfg.entropy}</h3>
                  <RpsNetPlot config={cfg} iterations={rpsNn.iterations} />
                </div>
              ))}
            </>
          )}
        </div>
      </section>

      <section className="band-tint" id="results">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">04</span>Results</div>
          <h2>How often the trained player wins</h2>
          <h3>Win rates against the best response, with and without the ball</h3>
          <p className="lede">Two players face each other. At kickoff one of them has the ball and the other has to defend, and having the ball is an advantage.
            Each learner is trained for both players, so it is measured in both positions. <b>Starts with the ball</b>: our learner is the player who has the ball at
            kickoff. <b>Starts without the ball</b>: our learner is the other player. <b>Average of both</b>: the mean of the two, one fair number that does not depend
            on who starts with the ball. Each cell is the share of 1,000 games the learner wins when the opponent plays the exact best response to it (the move that
            minimizes its expected discounted score difference), mean of 3 seeds; the exact solver is the top row, as context rather than a target. The win rate is a check, not an objective: no learner is trained to raise it.</p>
          <WinRateTable boards={available.map(([k, label]) => [k, label, boards[k]])} />
          <p className="fig-cap">The deterministic board is the A10 board: the exact solver ties itself there, so it wins 0% either way.
            A dash means the learner was not run on that board. Wins against the exact Nash policy and a random player are in the charts below.</p>
          <h3>One board in detail</h3>
          <div className="state-pick" role="group" aria-label="board">
            {available.map(([k, label]) => (
              <button key={k} className={which === k ? "on" : ""} onClick={() => setWhich(k)}>{label}</button>
            ))}
          </div>
          <p className="state-note">
            {which === "random"
              ? "Players' moves are applied in a random order each step, so the exact solution mixes at 94 stage games and the dynamics are stochastic."
              : which === "continuing"
                ? "Play restarts after every goal and runs for exactly 100 steps; a win is more goals than the opponent. Includes A2C with the exact solver's values as a frozen critic."
                : "Players move simultaneously and the carrier wins every contested square, so the exact solution is pure everywhere and ties itself."}
          </p>
          <h3>Win rate against the best response</h3>
          <BestResponseBars rows={[{ name: "Exact solver", win: exact.vs_best_response[0] }, ...learners.map(bestRow)]} reference={exact.vs_best_response[0]} />
          <p className="fig-cap">Share of 1,000 games won against the best response, mean of {learners[0].seeds} seeds. Dashed line: the exact solver.</p>
          <h3>Wins, ties and losses against each opponent</h3>
          <StackBars rows={winRows} titles={["Against a random player", "Against the exact Nash policy", "Against the best response"]} />
          <p className="fig-cap">Wins (green), ties (grey) and losses (orange).</p>
          {tieText && <p>{tieText}</p>}
          <p>
            {exact.vs_best_response[0] === 0
              ? `Against the best response the exact solver wins ${pct(exact.vs_best_response[0])}, because it ties itself every game, and the learners win ${rng(Math.min(...brWins), Math.max(...brWins))}. `
              : `Against the best response the exact solver wins ${pct(exact.vs_best_response[0])} and the learners ${rng(Math.min(...brWins), Math.max(...brWins))}. `}
            Fictitious-play training wins more often against a random player; no learner reaches the exact solver&rsquo;s rate against the best response or the exact Nash policy.
          </p>
        </div>
      </section>

      {mixed && (
        <section id="mixed">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p0">05</span>Mixed states</div>
            <h2>Where the exact answer has to mix</h2>
            <p className="lede">At {mixed.mixed_states} states of the random move-order board the exact equilibrium
              has to randomize. Each learner starts {mixed.games_per_start} games from every one of them.</p>
            <StackBars rows={mixedRows} titles={["Against the exact Nash policy", "Against the best response"]} />
            <p className="fig-cap">Wins (green), ties (grey) and losses (orange).</p>
            <p>Every learner is {Math.min(...mixed.learners.map((l) => l.tv_row)).toFixed(2)} to {Math.max(...mixed.learners.map((l) => l.tv_row)).toFixed(2)} away
              from the exact mix (total-variation distance at the first step: 0 = identical, 1 = no overlap). Player 0 mixes
              (puts under 90% on its top move) at {pct(Math.min(...mixed.learners.map((l) => l.share_mixing_row)))} to {pct(Math.max(...mixed.learners.map((l) => l.share_mixing_row)))} of
              these states, though not at the states or in the proportions the exact solution does.</p>
            <h3>Three of these states in full</h3>
            <PolicyOutputs data={mixed.policy_outputs} />
          </div>
        </section>
      )}

      <section className="band-tint" id="probabilities">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">06</span>Action probabilities</div>
          <h2>What the trained policies output</h2>
          <p className="lede">The probability each trained network gives to every move, at four fixed
            positions of the deterministic board, next to the exact solver&rsquo;s move.</p>
          <PolicyOutputs data={pgPolicies} />
        </div>
      </section>

      <section id="more">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">07</span>More training</div>
          <h2>Standard training at four times the iterations</h2>
          <p className="lede">8,000 instead of 2,000 iterations (two seeds each), deterministic board.</p>
          <StackBars rows={longRows} titles={["Exact Nash, 2,000 iterations", "Exact Nash, 8,000 iterations", "Random, 8,000 iterations"]} />
          <p className="fig-cap">Wins (green), ties (grey) and losses (orange).</p>
        </div>
      </section>

      <section className="band-tint" id="best-response">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">08</span>Fictitious play with long best responses</div>
          <h2>Train each best response for 100 or 300 iterations</h2>
          <p className="lede">Each round a player trains for many iterations against the average of its opponent&rsquo;s earlier
            best responses, then adds the result to its history. Win rate against the best response after each checkpoint round (one seed per run).</p>
          <LineChart series={fpSeries} rounds={fpRounds} />
        </div>
      </section>

      {vr && vd && (
        <section id="variants">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p0">09</span>Network sharing and trimming</div>
            <h2>One shared network, and leaving out the last steps</h2>
            <p className="lede">A2C and PPO, three seeds each, on all three boards. <b>Shared network</b>: both players&rsquo; policies are
              two output slices of one network, so each player&rsquo;s update also moves the other. <b>Leaving out the last 10 steps</b>:
              the final 10 of the 100 steps of every episode are not used in the loss, though they still count in the returns of earlier steps.
              Bars are the mean over seeds; dots are single seeds.</p>
            <h3>Random move-order board: win rate against the best response</h3>
            <VariantBars rows={vr} reference={rnd.exact.vs_best_response[0]} />
            <h3>Deterministic board: games tied with the exact Nash policy</h3>
            <VariantBars rows={vd} reference={results.exact.vs_nash[1]} field="tie_nash_" />
            {vc && (
              <>
                <h3>Continuing game: games tied with the exact Nash policy</h3>
                <VariantBars rows={vc} reference={results.continuing.exact.vs_nash[1]} field="tie_nash_" />
              </>
            )}
            <p className="fig-cap">{VARIANT_KEYS.map(([, l, c]) => <span key={l} style={{ marginRight: 14 }}><span style={{ color: c }}>&#9632;</span> {l}</span>)}</p>
            <p>{variantText}</p>
            <p>{detText}</p>
            <p>{contText}</p>
          </div>
        </section>
      )}

      {argmax && (
        <section className="band-tint" id="argmax">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p1">10</span>Averaging pure policies</div>
            <h2>Argmax snapshots, then best-respond to their average</h2>
            <p className="lede">A network can only mix through its softmax, so every snapshot is turned into a pure policy (the move with the
              highest probability), those are averaged, and each player best-responds to that average. This is fictitious play in its
              original form: each snapshot is a pure best response and the mix is how often each move was played. Both the opponent a player
              trains against and the policy it reports are averages of pure snapshots. Bars are the mean over 3 seeds; dots are single seeds.</p>
            <h3>Random move-order board: win rate against the best response</h3>
            <VariantBars rows={argmax.random} reference={rnd.exact.vs_best_response[0]} keys={AM_KEYS} />
            <h3>Deterministic board: games tied with the exact Nash policy</h3>
            <VariantBars rows={argmax.deterministic} reference={results.exact.vs_nash[1]} field="tie_nash_" keys={AM_KEYS} />
            {argmax.continuing && (
              <>
                <h3>Continuing game: games tied with the exact Nash policy</h3>
                <VariantBars rows={argmax.continuing} reference={results.continuing.exact.vs_nash[1]} field="tie_nash_" keys={AM_KEYS} />
              </>
            )}
            <p className="fig-cap">{AM_KEYS.map(([, l, c]) => <span key={l} style={{ marginRight: 14 }}><span style={{ color: c }}>&#9632;</span> {l}</span>)}</p>
            <p>{amText}</p>
          </div>
        </section>
      )}

      {entropy && (
        <section className="band-tint" id="entropy">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p1">11</span>A larger entropy bonus</div>
            <h2>An ablation: the entropy bonus raised to 0.2</h2>
            <p className="lede">On rock-paper-scissors a larger entropy bonus made the networks settle, but entropy regularization changes the optimization problem, so this is a stabilization heuristic reported next to the original setting, not a way to recover the unregularized equilibrium. Here every learner (REINFORCE, A2C, PPO),
              every way of training (standard, fictitious play, fictitious play with argmax snapshots) and every board is trained again with
              the entropy bonus raised from 0.01 to 0.2; nothing else changes. Bars are the mean over 3 seeds; dots are single seeds.</p>
            {enBoards.map(([k, title, f, ref]) => (
              <div key={k}>
                <h3>{title}</h3>
                <VariantBars rows={entropy[k]} reference={ref} field={f} keys={EN_KEYS} />
              </div>
            ))}
            <p className="fig-cap">{EN_KEYS.map(([, l, c]) => <span key={l} style={{ marginRight: 14 }}><span style={{ color: c }}>&#9632;</span> {l}</span>)}</p>
            {enByAlgo && <p>{`On the random board, averaged over the three ways of training, the larger bonus moves the win rate against the best response by ${enByAlgo} points; the exact solver wins ${pct(rnd.exact.vs_best_response[0])} when it starts with the ball. This is an ablation and does not show convergence to the unregularized equilibrium.`}</p>}
          <p>{enBoards.map(([k, , , , f]) => `${k === "random" ? "Random board" : k === "deterministic" ? "Deterministic board" : "Continuing game"}: the larger bonus changes the ${k === "random" ? "win rate against the best response" : "share tied with the exact Nash policy"} by ${enChange(entropy[k], f)} points; against a random player the win rate changes by ${enRandom(entropy[k])} points (REINFORCE, A2C, PPO; standard, fictitious play, argmax).`).join(" ")}</p>
          </div>
        </section>
      )}

      {curves && (
        <section className="band-tint" id="converge">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p1">12</span>Does training converge?</div>
            <h2>Win rate and kickoff policy during training</h2>
            <p className="lede">Best-response dynamics (standard training) can cycle; fictitious play averages the past. Every 100 iterations the policy
              each learner would report is scored against the exact best response (mean of 3 seeds), and the probability it gives to the exact
              solver&rsquo;s kickoff move is recorded for seed 0: the current network (thin) and, for fictitious play, the average (thick).</p>
            {curveBoards.map(([board, c]) => (
              <div key={board}>
                <h3>{board === "random" ? "Random move-order board" : "Deterministic board"}: win rate against the best response</h3>
                <LineChart rounds={c.iterations} xLabel="iteration" labelEvery={4}
                  series={c.series.map((s, n) => ({ name: `${s.label}, ${s.training}`, values: s.win_br, color: LINE_COLOURS[[4, 0, 5, 1][n]] }))} />
                <h3>{board === "random" ? "Random board" : "Deterministic board"}: probability of the exact solver&rsquo;s kickoff move ({c.move}; exact solver {pct(c.exact_move_probability)})</h3>
                <LineChart rounds={c.iterations} xLabel="iteration" top={1} labelEvery={4}
                  series={c.series.flatMap((s) => (s.mode === "selfplay"
                    ? [{ name: `${s.label}, standard`, values: s.current_move, color: "var(--ember)", width: 1.4, label: false }]
                    : [{ name: `${s.label}, current`, values: s.current_move, color: "var(--ink-faint)", width: 1, opacity: 0.7, label: false },
                      { name: `${s.label}, average`, values: s.aggregate_move, color: "var(--pitch)", width: 2.6, label: false }]))} />
              </div>
            ))}
            <p className="fig-cap">Orange: standard training&rsquo;s current network. Grey: the current network under fictitious play. Green: fictitious play&rsquo;s average.
              A2C and PPO are drawn together in each chart.</p>
            {curveBoards.map(([board, c]) => (
              <p key={board}>{`${board === "random" ? "Random board" : "Deterministic board"}, second half of training: the win rate against the best response stays within ${c.series.map((s) => {
                const h = s.win_br.slice(Math.floor(s.win_br.length / 2));
                return `${s.label} ${s.training} ${rng(Math.min(...h), Math.max(...h))}`;
              }).join("; ")}. The exact solver wins ${pct(board === "random" ? rnd.exact.vs_best_response[0] : results.exact.vs_best_response[0])} when it starts with the ball.`}</p>
            ))}
          </div>
        </section>
      )}

      {choices && (
        <section id="choices">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p0">13</span>Training choices</div>
            <h2>What changes when the training recipe changes</h2>
            <p className="lede">Each comparison changes one thing from the usual setting (2,000 iterations of 64 episodes, all data used once, a snapshot every
              5 iterations, all 100 steps in the loss) and keeps the rest. Bars are the mean over 3 seeds; dots are single seeds.</p>
            {choices.map((s) => (
              <div key={s.title}>
                <h3>{s.title}</h3>
                <p className="fig-cap">{s.note}</p>
                <VariantBars rows={s.rows} reference={s.reference == null ? (s.metric === "tie" ? 1 : rnd.exact.vs_best_response[0]) : s.reference} field={studyField(s)} keys={studyKeys(s)} />
                <p className="fig-cap">{studyKeys(s).map(([, l, c]) => <span key={l} style={{ marginRight: 14 }}><span style={{ color: c }}>&#9632;</span> {l}</span>)}</p>
                <p>{studyText(s)}</p>
              </div>
            ))}
          </div>
        </section>
      )}




      {moreSeeds && (
        <section className="band-tint" id="more-seeds">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p1">14</span>More seeds</div>
            <h2>The two best-looking settings, on {moreSeeds.seeds} seeds</h2>
            <p className="lede">With 3 seeds, A2C with fictitious play looked better at 10,000 iterations than at 2,000, and PPO with standard training looked
              better with replayed data. Each of these two settings and its usual-setting baseline was run on seeds 0 to {moreSeeds.seeds - 1} (random move-order board).
              Each bar is the mean of the {moreSeeds.seeds} seeds and the thin line is its 95% interval (Student&rsquo;s t). The exact solver wins {pct(rnd.exact.vs_best_response[0])} when it starts with the ball, {pct(rnd.exact.vs_best_response_column[0])} when it starts without the ball and {pct((rnd.exact.vs_best_response[0] + rnd.exact.vs_best_response_column[0]) / 2)} averaged over both starts.</p>
            {moreSeeds.pairs.map((pair) => (
              <div key={pair.label}>
                <h3>{pair.label}: {pair.baseline_name} against {pair.variant_name}</h3>
                <BeforeAfterChart groups={msGroups(pair)} names={[pair.baseline_name, pair.variant_name]} />
                <p>{msText(pair)}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {symmetry && symmetry.length > 0 && (
        <section className="band-tint" id="symmetry">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p0">15</span>Symmetry</div>
            <h2>Are the two players mirror images?</h2>
            <p className="lede">The board is symmetric: swapping the two players and mirroring the pitch gives the same game. The <b>mirror gap</b> measures how far player 1&rsquo;s
              policy is from the mirror image of player 0&rsquo;s: for every position and several steps it takes the largest difference in move probabilities between player 0&rsquo;s
              policy and player 1&rsquo;s mirrored policy, then averages. 0 means perfectly symmetric, 1 means opposite. The exact solver is symmetric (0). The learners are not:
              their two networks are trained separately and settle on different policies, with gaps of {Math.min(...symAll).toFixed(2)} to {Math.max(...symAll).toFixed(2)};
              fictitious play has the smaller gap in {symPairs.filter(([s, f]) => f < s).length} of the {symPairs.length} learner-and-board pairs.</p>
            <div className="tbl-wrap">
              <table className="win-table">
                <thead><tr><th>Policy</th>{symmetry.map((b) => <th key={b.board}>{b.title}</th>)}</tr></thead>
                <tbody>
                  {symNames.map((n) => <tr key={n}><td>{n}</td>{symmetry.map((b) => <td key={b.board}>{symGap(b, n)}</td>)}</tr>)}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {smallGames && (
        <section id="small-games">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p1">16</span>Other kinds of games</div>
            <h2>Coordination, anti-coordination and a zero-sum game</h2>
            <p className="lede">Best-response dynamics are known to converge in zero-sum games and in potential games, such as coordination games, but only to one of the equilibria and not
              necessarily from every start. <b>Simultaneous</b>: both players switch to the best response to the other&rsquo;s last action. <b>Alternating</b>: one player responds, then the other
              responds to that. <b>Fictitious play</b>: each player best-responds to the average of the other&rsquo;s past play; the last column is the share of the first action played by the
              row and by the column player after 2,000 rounds. Battle of the sexes: both prefer to be together and each prefers a different place (B,B pays 2 and 1, S,S pays 1 and 2,
              otherwise 0). Different choices: 1 each if the players choose differently, otherwise 0. Matching pennies: the row player wants to match, the column player to differ.</p>
            <div className="tbl-wrap">
              <table className="win-table">
                <thead>
                  <tr><th>Game</th><th>Pure equilibria</th><th>Mixed equilibrium (first action)</th><th>Start</th><th>Simultaneous ends in</th><th>Alternating ends in</th><th>Fictitious play</th></tr>
                </thead>
                <tbody>
                  {smallGames.flatMap((g) => g.dynamics.map((d, k) => {
                    const fp = g.fictitious_play[d.start];
                    return (
                      <tr key={`${g.game}-${d.start}`}>
                        <td>{k === 0 ? g.game : ""}</td>
                        <td>{k === 0 ? (g.pure.join(", ") || "none") : ""}</td>
                        <td>{k === 0 ? (g.mixed ? `${g.mixed[0].toFixed(2)} / ${g.mixed[1].toFixed(2)}` : "none") : ""}</td>
                        <td>{d.start}</td><td>{d.simultaneous}</td><td>{d.alternating}</td><td>{fp[0].toFixed(2)} / {fp[2].toFixed(2)}</td>
                      </tr>
                    );
                  }))}
                </tbody>
              </table>
            </div>
            <p>In the coordination game the dynamics end in one of the two pure equilibria, which one depending on the start; started with mismatched choices the simultaneous version
              keeps switching. Where the only equilibrium is mixed (matching pennies) best-response dynamics never settle, but fictitious play&rsquo;s averages go to one half each, as in
              rock-paper-scissors. In the game of different choices, started with the same choice, simultaneous best responses keep switching and fictitious play&rsquo;s averages go to one
              half each while the players always match; started with different choices it stays there.</p>
          </div>
        </section>
      )}

      {exactPlay && (
        <section className="band-tint" id="exact-play">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p0">17</span>Without sampling noise</div>
            <h2>Exact gradient play: a table against a network</h2>
            <p className="lede">To separate the learning rule from the noise of sampled gradients, both players follow the <b>exact</b> gradient of their own value, computed from the
              transition table on the stationary discounted game (discount 0.9, no step count) and summed over every state, as if a game could start anywhere. The <b>duality gap</b>
              at a state is the first player&rsquo;s best-response value against the second&rsquo;s policy minus the second&rsquo;s best-response value against the first&rsquo;s; it is
              0 only at a Markov-perfect equilibrium. A <b>table</b> has one row of move logits per state; a <b>network</b> is one 64&times;64 network per player, shared across states.
              Bars are the median over runs (1 table run, 5 network seeds, 3 seeds for the 6,000-step runs) and dots single runs.</p>
            <GapBars conditions={exactPlay.conditions} />
            {(() => {
              const pick = (b, p, m) => exactPlay.conditions.filter((c) => c.board === b && c.policy === p && c.method === m);
              const span = (cs, f) => { const v = cs.flatMap((c) => c[f]); return [Math.min(...v), Math.max(...v)]; };
              const r = (x, d = 2) => x.toFixed(d);
              const [aT, aN, rN, sgd, eg] = [pick("a10", "table", "adam"), pick("a10", "network", "adam"), pick("random", "network", "adam"),
                pick("random", "table", "sgd"), pick("random", "table", "eg")];
              if (!aT.length || !aN.length || !rN.length || !sgd.length || !eg.length) return null;
              return (
                <p>On the deterministic board, where the equilibrium is pure everywhere, the table reaches it (mean gap {r(aT[0].gap_mean[0], 3)}) and every network stalls
                  (mean gap {r(span(aN, "gap_mean")[0])}&ndash;{r(span(aN, "gap_mean")[1])}, kickoff gap {r(span(aN, "gap_kickoff")[0])}&ndash;{r(span(aN, "gap_kickoff")[1])}), the same
                  scale as the sampled learners&rsquo; exploitability: there the network, not the sampling noise, stops policy gradient. On the random board, where the equilibrium mixes,
                  no exact gradient method tried reaches it even with a table: simultaneous gradient steps leave {r(span(sgd, "gap_mean")[0])}&ndash;{r(span(sgd, "gap_mean")[1])} after
                  {" "}{sgd[0].steps.toLocaleString()} steps, and extragradient, a standard fix for the cycling of simultaneous gradient play, roughly halves that to
                  {" "}{r(span(eg, "gap_mean")[0], 3)}&ndash;{r(span(eg, "gap_mean")[1], 3)} and then stops improving; the networks end at {r(span(rN, "gap_mean")[0])}&ndash;{r(span(rN, "gap_mean")[1])}.
                  Two separate obstacles, then: a shared network, and simultaneous gradient play near mixed equilibria. This is the stationary discounted game, not the finite-horizon games above.</p>
              );
            })()}
          </div>
        </section>
      )}

      {related && (
        <section id="related">
          <div className="wrap">
            <div className="eyebrow"><span className="badge p1">18</span>Related work</div>
            <h2>Has fictitious play been combined with policy gradient before?</h2>
            <p className="lede">Yes, in several forms, and the convergence theory is much older than the neural networks. I checked the titles, authors and venues against Google Scholar (its top result for fictitious play, policy gradient and zero-sum Markov games was Liu, Yang and Wang) and compared each one-line summary with the abstract on the paper&rsquo;s own proceedings, journal or arXiv page, or on a bibliographic mirror where the publisher&rsquo;s page would not load (noted in the entry). JSTOR and the publisher pages for the 1951 papers would not load, so Brown and Robinson are cited from the linked secondary source and were not read. I have not checked the proofs. I did not find a result that proves convergence for deep policy gradient with fictitious play in
              zero-sum Markov games; the closest results are tabular, regularized or for special game classes.</p>
            {[["older", "Older work"], ["recent", "Recent work"]].map(([g, title]) => (
              <div key={g}>
                <h3>{title}</h3>
                <ul className="related-list">
                  {related.filter((r) => r.group === g).map((r) => (
                    <li key={r.title}><a href={r.url}>{r.title}</a>{` (${r.authors}, ${r.year}; ${r.venue})`}. {r.says}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="band-tint" id="end">
        <div className="wrap">
          <div className="cta-row">
            <a className="btn btn-primary" href="policy_gradient.pdf">Read the PDF &rarr;</a>
            <a className="btn btn-outline" href="policy_gradient_detailed.pdf">Detailed PDF &rarr;</a>
            <a className="btn btn-outline" href="https://github.com/Charith-Reddy-Pareddy/soccer-markov-nash">View the code on GitHub &rarr;</a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
