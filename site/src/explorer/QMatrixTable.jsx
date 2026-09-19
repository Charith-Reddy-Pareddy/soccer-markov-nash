import { ACT, heatColor } from "./helpers.js";

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
              <td key={c} className="cell" style={{ background: heatColor(M[r][c], lo, hi) }}>
                {M[r][c].toFixed(6)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
