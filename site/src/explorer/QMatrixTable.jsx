import { ACT } from "./helpers.js";

function heat(v, lo, hi) {
  let t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
  t = Math.min(Math.max(t, 0), 1);
  if (t < 0.5) return `color-mix(in srgb, var(--ember-soft) ${Math.round((0.5 - t) * 2 * 70)}%, var(--raise))`;
  return `color-mix(in srgb, var(--pitch-soft) ${Math.round((t - 0.5) * 2 * 70)}%, var(--raise))`;
}

export default function QMatrixTable({ M }) {
  const vals = M.flat();
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return (
    <table className="qmatrix">
      <thead>
        <tr><th></th>{ACT.map((a) => <th key={a}>{a}</th>)}</tr>
      </thead>
      <tbody>
        {ACT.map((rowLabel, r) => (
          <tr key={rowLabel}>
            <th>{rowLabel}</th>
            {ACT.map((_, c) => (
              <td key={c} className="cell" style={{ background: heat(M[r][c], lo, hi) }}>
                {M[r][c].toFixed(3)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
