import { ACT, fmtPct, heatColor, support } from "./helpers.js";

// rowPol / colPol are optional: the header percentage row/column and the
// actual-support cell highlight only render when a policy is supplied,
// so this still works as a bare M-only table wherever one is needed.
// wall0 / wall1 are also optional: when supplied, a header action that's
// edge-clamped from that player's current cell gets the same marker the
// move-bars use -- "hold" (legal to pick, but holds in place rather than
// moving) or "score" (the carrier, in a goal row, pushing past their own
// attacking edge -- this ends the game, it is not a hold) -- so "100%" here
// doesn't read as "definitely goes up" when there's no board above it.
function WallMark({ status }) {
  if (status === "hold") {
    return <sup className="wall-mark" title="wall-clamped: no cell there, this holds in place">&#8862;</sup>;
  }
  if (status === "score") {
    return <sup className="wall-mark score" title="scores here: this ends the game, it does not hold in place">&#9873;</sup>;
  }
  return null;
}

export default function QMatrixTable({ M, rowPol, colPol, wall0, wall1 }) {
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
              <div>{a}{wall1 && <WallMark status={wall1[a]} />}</div>
              {colPol && <div className="qmatrix-pct">{fmtPct(colPol[c])}</div>}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {ACT.map((rowLabel, r) => (
          <tr key={rowLabel}>
            <th>
              <div>{rowLabel}{wall0 && <WallMark status={wall0[rowLabel]} />}</div>
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
