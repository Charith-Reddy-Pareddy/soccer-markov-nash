import { useState } from "react";

const pct = (v) => `${Math.round(v * 100)}%`;

function Cells({ probs }) {
  const top = Math.max(...probs);
  return probs.map((v, i) => <td key={i} className={v === top ? "top" : ""}>{pct(v)}</td>);
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
        Probability of each move at the start of the game; the most likely move is in bold.
      </p>
      <div className="tbl-wrap">
        <table className="probs">
          <thead>
            <tr><th rowSpan={2}>Policy</th><th colSpan={4}>Player 0</th><th colSpan={4}>Player 1</th></tr>
            <tr>{[...data.actions, ...data.actions].map((a, i) => <th key={i}>{a}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}><td>{r.name}</td><Cells probs={r.row} /><Cells probs={r.col} /></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
