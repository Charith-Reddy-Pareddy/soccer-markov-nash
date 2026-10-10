import { useState } from "react";
import Nav from "./Nav.jsx";
import Footer from "./Footer.jsx";
import PolicyOutputs from "./PolicyOutputs.jsx";
import pgPolicies from "./pgPolicies.json";
import results from "./pgResults.json";
import "./landing.css";

const pct = (v) => `${Math.round(v * 100)}%`;
const name = (l) => `${l.label}, ${l.training}`;

// Share of the 1,000 games the player wins (scores while the opponent does not), with the
// ties (no goal, or equal goals) and the losses beside it.
function WinCell({ t }) {
  return (
    <td>
      <b>{pct(t[0])}</b>
      <span className="sub">tie {pct(t[1])} &middot; loss {pct(t[2])}</span>
    </td>
  );
}

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

// Every win rate against the best response in one table, for both kickoff seats and their 50/50
// average, on every board that has results.
const SEATS = [["ball", "Ball seat"], ["other", "Other seat"], ["balanced", "Balanced"]];
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
            {boards.map(([k, title]) => <th key={k} colSpan={SEATS.length}>{title}</th>)}
          </tr>
          <tr>{boards.flatMap(([k]) => SEATS.map(([f, s]) => <th key={`${k}-${f}`}>{s}</th>))}</tr>
        </thead>
        <tbody>
          <tr>
            <td>Exact solver</td>
            {boards.flatMap(([k, , b]) => SEATS.map(([f]) => <td key={`${k}-${f}`}>{pct(seatRates(b.exact)[f])}</td>))}
          </tr>
          {names.map((n) => (
            <tr key={n}>
              <td>{n}</td>
              {boards.flatMap(([k, , b]) => {
                const l = b.learners.find((x) => name(x) === n);
                return SEATS.map(([f]) => <td key={`${k}-${f}`}>{l ? pct(seatRates(l)[f]) : "\u2014"}</td>);
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

// Win rate against the best response after each round of long best-response phases, one line per run.
function LineChart({ series, rounds }) {
  const W = 660, H = 260, lm = 48, rm = 170, tm = 20, bm = 34;
  const top = Math.min(1, Math.round((Math.max(0.1, ...series.flatMap((s) => s.values.filter((v) => v != null))) * 1.15 + 0.02) * 100) / 100);
  const x = (k) => lm + (k / Math.max(rounds.length - 1, 1)) * (W - lm - rm);
  const y = (v) => tm + (1 - v / top) * (H - tm - bm);
  const cols = ["var(--pitch)", "var(--pitch-dark, #178A42)", "var(--rule-strong)", "var(--ink-faint)", "var(--ember)", "var(--ember-soft, #d08a5c)"];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="win rate against the best response by round" className="pg-bars">
      {[0, top / 2, top].map((g) => (
        <g key={g}>
          <line x1={lm} y1={y(g)} x2={W - rm} y2={y(g)} stroke="var(--rule)" />
          <text x={lm - 6} y={y(g) + 4} textAnchor="end" fontSize="11" fill="var(--ink-faint)">{pct(g)}</text>
        </g>
      ))}
      {rounds.map((k, i) => <text key={k} x={x(i)} y={H - 10} textAnchor="middle" fontSize="11" fill="var(--ink-faint)">{k}</text>)}
      <text x={lm} y={H} fontSize="11" fill="var(--ink-faint)">round</text>
      {series.map((s, n) => {
        const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter(Boolean);
        return (
          <g key={s.name}>
            <path d={pts.map(([px, py], i) => `${i ? "L" : "M"}${px},${py}`).join(" ")} fill="none" stroke={cols[n % 6]} strokeWidth="2.2" />
            <text x={W - rm + 8} y={pts[pts.length - 1][1] + 4} fontSize="11" fontWeight="700" fill={cols[n % 6]}>{s.name}</text>
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
            <text x={left - 8} y={y0 + group / 2 - 6} textAnchor="end" fontSize="12" fill="var(--ink)">{r.label}, {r.training}</text>
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
  ["continuing", "Continuing game (restarts after a goal)"],
];

export default function PolicyPage() {
  const [which, setWhich] = useState("deterministic");
  const { longer, fp_br: fpBr, mixed, rps, variants, argmax, rps_nn: rpsNn, entropy } = results;
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
  const pts = (v) => `${v >= 0 ? "+" : "\u2212"}${Math.abs(Math.round(v * 100))}`;
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
      (argmax.continuing ? ` Continuing game: the share tied with the exact Nash policy changes by ${amChange(argmax.continuing, "tie_nash_mean")} points; single seeds differ by up to ${amSpread(argmax.continuing, "tie_nash_runs")} points, and no argmax learner wins more than ${pct(Math.max(...argmax.continuing.map((v) => v.argmax.mean)))} of games against the best response.` : "")
    : "";
  const EN_KEYS = [["baseline", "Entropy bonus 0.01 (used elsewhere)", "var(--pitch)"], ["larger", "Entropy bonus 0.2", "var(--ember)"]];
  const enChange = (rows, f) => rows.map((v) => pts(v.larger[f] - v.baseline[f])).join(", ");
  const enRandom = (rows) => rows.map((v) => pts(v.larger.vs_random[0] - v.baseline.vs_random[0])).join(", ");
  const enBoards = entropy ? [["random", "Random move-order board: win rate against the best response", "", rnd.exact.vs_best_response[0], "mean"],
    ["deterministic", "Deterministic board: games tied with the exact Nash policy", "tie_nash_", results.exact.vs_nash[1], "tie_nash_mean"],
    ["continuing", "Continuing game: games tied with the exact Nash policy", "tie_nash_", results.continuing && results.continuing.exact.vs_nash[1], "tie_nash_mean"]].filter(([k]) => entropy[k]) : [];
  const detText = vd
    ? `Deterministic board: no variant wins more than ${pct(Math.max(...vd.flatMap((v) => [v.baseline, v.shared, v.trimmed].map((x) => x.mean))))} of games against the best response, as for the baseline (the exact solver wins 0%, since it ties itself). ` +
      `The share of games tied with the exact Nash policy changes by ${change(vd, "shared", "tie_nash_mean")} points with a shared network and by ${change(vd, "trimmed", "tie_nash_mean")} points with the last 10 steps left out ${order}; ` +
      `single seeds of the same setup differ by up to ${spread(vd, "tie_nash_runs")} points. ` +
      `A tie does not mean equilibrium play: PPO with standard training ties all games once the last 10 steps are left out, but wins only ${pct(pick(vd, "PPO", "standard").trimmed.vs_random[0])} of games against a random player, against ${pct(pick(vd, "PPO", "standard").baseline.vs_random[0])} before.`
    : "";
  const contText = vc
    ? `Continuing game: no variant wins more than ${pct(Math.max(...vc.flatMap((v) => [v.baseline, v.shared, v.trimmed].map((x) => x.mean))))} of games against the best response, as for the baseline. ` +
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
            <p className="lede">REINFORCE, A2C and PPO are trained on the discounted 100-step game and scored by
              one number: how often the trained player wins when the opponent is the best response to it.
              Each is trained two ways: the standard way, where both players keep updating against each
              other, and fictitious play, where each player responds to the average of its opponent&rsquo;s past
              strategies.</p>
            <div className="cta-row">
              <a className="btn btn-primary" href="policy_gradient.pdf">Read the PDF &rarr;</a>
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
          <h2>One number: the win rate against the best response</h2>
          <div className="tbl-wrap">
            <table className="text-table">
              <thead><tr><th>Number</th><th>What it is</th></tr></thead>
              <tbody>
                <tr><td>Win rate against the best response</td><td>The share of 1,000 games the trained player wins when the opponent plays the exact best response to it, the move that is best against exactly what our player will do. A player who always does the same thing loses every game; the exact solver mixes optimally, so its rate is the ceiling to compare with.</td></tr>
                <tr><td>Win rate against the exact Nash policy</td><td>The same, against the exact equilibrium player. The exact solver cannot be beaten by it; a learner that mixes well scores close to the exact solver&rsquo;s own rate.</td></tr>
                <tr><td>Win rate against a random player</td><td>A sanity check: every learner should beat a player who moves at random.</td></tr>
                <tr><td>A win</td><td>The player scores while the opponent does not. In the first two boards the first goal ends the game; in the continuing game play restarts after a goal and a win is more goals than the opponent in 100 steps. Games with no goal (or equal goals) are ties.</td></tr>
              </tbody>
            </table>
          </div>
          <p className="fig-cap">Exploitability, the best-response player&rsquo;s gain, is not reported: it measures the attacker, and it is the player we train that we want to judge.</p>
          <h3>Why the exact solver does not win 0% against the best response</h3>
          <p>The best response is the move that minimizes our player&rsquo;s <em>expected discounted score difference</em>, not the one that
            minimizes its chance of winning. At the random board&rsquo;s kickoff the player holding the ball is ahead: the exact solution is worth
            {" "}{rnd.kickoff_value >= 0 ? "+" : "\u2212"}{Math.abs(rnd.kickoff_value).toFixed(2)} to it even against a perfect best response. So the exact solver wins {pct(rnd.exact.vs_best_response[0])} of
            games and loses {pct(1 - rnd.exact.vs_best_response[0])} from that seat, and wins {pct(rnd.exact.vs_best_response_column[0])} from the other seat. The first number
            is the one reported on this page (our learners always sit in the ball-holding seat). On the deterministic board both seats tie every game.</p>
          <StackBars rows={seatRows} titles={["Random board, ball-holding seat", "Random board, other seat"]} />
          <p className="fig-cap">Win rate against the best response from each seat. Wins (green), ties (grey) and losses (orange).</p>
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
                <tr><td>Game</td><td>7&times;5 board, goal rows 1&ndash;3. First goal ends the game (A10 and random-order boards) or play restarts after a goal (continuing game). Fixed 100 steps, discount 0.9, the number of steps left is a network input.</td></tr>
                <tr><td>Returns</td><td>REINFORCE: the discounted return from each step to the end of the episode, no baseline. A2C: 10-step bootstrapped advantage from a learned critic. A2C with the exact critic: the critic is the exact solver&rsquo;s value, frozen, so the advantage is the exact Q(s, a<sub>0</sub>, a<sub>1</sub>) minus V(s). PPO: GAE (&lambda; 0.95), ratio clipped at 0.2, 4 epochs.</td></tr>
                <tr><td>Updates</td><td>Every iteration: 64 episodes of 100 steps, then one gradient step per player on all of that data (four for PPO). 2,000 iterations. Learning rate 10<sup>&minus;3</sup>, entropy bonus 0.01, 64&times;64 network, softmax output, a separate network for each player. No mini-batches, no replay.</td></tr>
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
                aggregate. Exploitability, on the right, is the mean over seeds and over the last 300 iterations; it is 0 at one third each.</p>
              {rpsText && <p>{rpsText}</p>}
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
          <h3>Win rates against the best response, by kickoff seat</h3>
          <p className="lede">The share of 1,000 games each learner wins when the opponent plays the exact best response to it (the
            move that minimizes its expected discounted score difference). <b>Ball seat</b>: the player that holds the ball at kickoff.
            <b> Other seat</b>: the same learner as the other player. <b>Balanced</b>: the 50/50 average of the two, one number that
            does not depend on who starts with the ball. Mean of 3 seeds; the exact solver is the top row, as context rather than a target.</p>
          <WinRateTable boards={available.map(([k, label]) => [k, label, boards[k]])} />
          <p className="fig-cap">The deterministic board is the A10 board: the exact solver ties itself there, so it wins 0% from both seats.
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
              ? `Against the best response the exact solver wins ${pct(exact.vs_best_response[0])}, because it ties itself every game, and the learners win ${pct(Math.min(...brWins))} to ${pct(Math.max(...brWins))}. `
              : `Against the best response the exact solver wins ${pct(exact.vs_best_response[0])} and the learners ${pct(Math.min(...brWins))} to ${pct(Math.max(...brWins))}. `}
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
            <p>{enBoards.map(([k, , , , f]) => `${k === "random" ? "Random board" : k === "deterministic" ? "Deterministic board" : "Continuing game"}: the larger bonus changes the ${k === "random" ? "win rate against the best response" : "share tied with the exact Nash policy"} by ${enChange(entropy[k], f)} points; against a random player the win rate changes by ${enRandom(entropy[k])} points (REINFORCE, A2C, PPO; standard, fictitious play, argmax).`).join(" ")}</p>
          </div>
        </section>
      )}

      <section id="end">
        <div className="wrap">
          <div className="cta-row">
            <a className="btn btn-primary" href="policy_gradient.pdf">Read the PDF &rarr;</a>
            <a className="btn btn-outline" href="https://github.com/Charith-Reddy-Pareddy/soccer-markov-nash">View the code on GitHub &rarr;</a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
