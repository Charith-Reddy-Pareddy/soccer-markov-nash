import { useState } from "react";

const pct = (v) => `${Math.round(v * 100)}%`;

const MOVE_COLOURS = ["var(--pitch)", "#8fd3a8", "var(--rule-strong)", "var(--ember)"];

// One stacked bar of move probabilities per policy and player; the likeliest move is labelled.
function MoveBars({ rows, actions }) {
  const W = 660, left = 190, gap = 30, rowH = 24, panel = (W - left - gap - 10) / 2;
  const H = rowH * rows.length + 30;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="probability of each move" className="pg-bars">
      {["Player 0", "Player 1"].map((title, k) => (
        <text key={title} x={left + k * (panel + gap)} y="12" fontSize="11" fontWeight="700" fill="var(--ink-faint)">{title}</text>
      ))}
      {rows.map((r, i) => {
        const y = 20 + i * rowH;
        return (
          <g key={r.name}>
            <text x={left - 8} y={y + 12} textAnchor="end" fontSize="12" fill="var(--ink)">{r.name}</text>
            {[r.row, r.col].map((probs, k) => {
              const top = probs.indexOf(Math.max(...probs));
              let x = left + k * (panel + gap);
              return probs.map((v, jx) => {
                const x0 = x;
                x += v * panel;
                return (
                  <g key={`${k}-${jx}`}>
                    <rect x={x0} y={y} width={v * panel} height="16" fill={MOVE_COLOURS[jx]} />
                    {jx === top && v >= 0.12 && (
                      <text x={x0 + (v * panel) / 2} y={y + 12} textAnchor="middle" fontSize="10.5" fontWeight="700"
                        fill={jx === 0 || jx === 3 ? "#fff" : "var(--ink)"}>{actions[jx]} {pct(v)}</text>
                    )}
                  </g>
                );
              });
            })}
          </g>
        );
      })}
    </svg>
  );
}

// The action probabilities the trained policy-gradient networks output at a few
// fixed positions, next to the exact solver's move. Values come from
// pgPolicies.json (scripts/pg_policy_outputs.py), never typed by hand.
export default function PolicyOutputs({ data }) {
  const [k, setK] = useState(0);
  const [x0, y0, x1, y1, ball] = data.states[k].state;
  const rows = [
    { name: "Exact solver", row: data.exact.row[k], col: data.exact.col[k] },
    ...data.learners.map((l) => ({ name: l.label, row: l.row[k], col: l.col[k] })),
  ];
  return (
    <div className="policy-outputs">
      <div className="state-pick" role="group" aria-label="position">
        {data.states.map((s, i) => (
          <button key={s.id} className={i === k ? "on" : ""} onClick={() => setK(i)}>{s.label}</button>
        ))}
      </div>
      <p className="state-note">
        Player 0 at ({x0}, {y0}), player 1 at ({x1}, {y1}), player {ball} has the ball.
        Probability of each move at the start of the game: {data.actions.map((a, i) => <span key={a} style={{ marginRight: 10 }}><span style={{ color: MOVE_COLOURS[i] }}>&#9632;</span> {a}</span>)}.
      </p>
      <MoveBars rows={rows} actions={data.actions} />
    </div>
  );
}
