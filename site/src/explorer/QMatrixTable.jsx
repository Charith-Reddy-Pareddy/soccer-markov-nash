import { ACT, fmtPct, heatColor, support } from "./helpers.js";

// rowPol / colPol are optional: the header percentage row/column and the
// actual-support cell highlight only render when a policy is supplied,
// so this still works as a bare M-only table wherever one is needed.
export default function QMatrixTable({ M, rowPol, colPol }) {
  const vals = M.flat();
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const rowSupport = rowPol ? support(rowPol) : [];
  const colSupport = colPol ? support(colPol) : [];
  const inSupport = (r, c) => rowSupport.includes(r) && colSupport.includes(c);
  return (
    <table className="qmatrix">
      <thead>
        <tr>
          <th></th>
          {ACT.map((a, c) => (
            <th key={a}>
              <div>{a}</div>
              {colPol && <div className="qmatrix-pct">{fmtPct(colPol[c])}</div>}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {ACT.map((rowLabel, r) => (
          <tr key={rowLabel}>
            <th>
              <div>{rowLabel}</div>
              {rowPol && <div className="qmatrix-pct">{fmtPct(rowPol[r])}</div>}
            </th>
            {ACT.map((_, c) => (
              <td
                key={c}
                className={"cell" + (inSupport(r, c) ? " support" : "")}
                style={{ background: heatColor(M[r][c], lo, hi) }}
              >
                {M[r][c].toFixed(6)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
