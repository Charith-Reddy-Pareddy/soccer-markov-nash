import { ACT } from "./helpers.js";

// Mirrors soccer_nash/viz.py's bestresponse_graph_svg exactly: one node per
// cell, a blue arrow from a cell to the higher-payoff cell in its column
// (the carrier's reason to switch rows), a green arrow to the lower-payoff
// cell in its row (the defender's reason to switch columns).
export default function QMatrixGraph({ M }) {
  const gapX = 92, gapY = 78, lm = 46, tm = 34, r = 15;
  const cx = (j) => lm + j * gapX;
  const cy = (i) => tm + i * gapY;

  const rowBest = [0, 1, 2, 3].map((j) => {
    const col = [0, 1, 2, 3].map((i) => M[i][j]);
    return col.indexOf(Math.max(...col));
  });
  const colBest = [0, 1, 2, 3].map((i) => M[i].indexOf(Math.min(...M[i])));

  let saddle = null;
  for (let j = 0; j < 4; j++) {
    const i = rowBest[j];
    if (colBest[i] === j) { saddle = [i, j]; break; }
  }

  const vArrows = [];
  for (let j = 0; j < 4; j++) {
    const bi = rowBest[j];
    for (let i = 0; i < 4; i++) {
      if (i === bi) continue;
      const x = cx(j) - 7, y0 = cy(i), y1 = cy(bi);
      const sign = y1 > y0 ? 1 : -1;
      vArrows.push(
        <line key={`v${j}-${i}`} x1={x} y1={y0 + sign * r} x2={x} y2={y1 - sign * r}
          stroke="var(--p0)" strokeWidth={2} markerEnd="url(#ah-p0)" opacity={0.85} />
      );
    }
  }
  const hArrows = [];
  for (let i = 0; i < 4; i++) {
    const bj = colBest[i];
    for (let j = 0; j < 4; j++) {
      if (j === bj) continue;
      const y = cy(i) + 7, x0 = cx(j), x1 = cx(bj);
      const sign = x1 > x0 ? 1 : -1;
      hArrows.push(
        <line key={`h${i}-${j}`} x1={x0 + sign * r} y1={y} x2={x1 - sign * r} y2={y}
          stroke="var(--p1)" strokeWidth={2} markerEnd="url(#ah-p1)" opacity={0.85} />
      );
    }
  }

  const nodes = [];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const isSaddle = saddle && saddle[0] === i && saddle[1] === j;
      nodes.push(
        <g key={`n${i}-${j}`}>
          <circle cx={cx(j)} cy={cy(i)} r={r} fill={isSaddle ? "var(--ember)" : "var(--tint)"}
            stroke={isSaddle ? "var(--ember)" : "var(--rule-strong)"} strokeWidth={1.6} />
          <text x={cx(j)} y={cy(i) + 4} textAnchor="middle" fontFamily="ui-monospace,monospace"
            fontWeight="600" fontSize="11" fill={isSaddle ? "var(--raise)" : "var(--ink)"}>
            {M[i][j] >= 0 ? "+" : ""}{M[i][j].toFixed(2)}
          </text>
        </g>
      );
    }
  }

  const sub = saddle
    ? `pure saddle at ${ACT[saddle[0]]}/${ACT[saddle[1]]}`
    : "arrows cycle — no cell is a stable outcome, must mix";
  const th = tm + 3 * gapY + 34;
  const tw = Math.max(lm + 3 * gapX + gapX / 2, lm + sub.length * 6.0);

  return (
    <svg viewBox={`0 0 ${tw} ${th}`} role="img" aria-label="best-response graph">
      <defs>
        <marker id="ah-p0" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="3.6" markerHeight="3.6" orient="auto">
          <path d="M0 0.5 L8 4 L0 7.5 z" fill="var(--p0)" />
        </marker>
        <marker id="ah-p1" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="3.6" markerHeight="3.6" orient="auto">
          <path d="M0 0.5 L8 4 L0 7.5 z" fill="var(--p1)" />
        </marker>
      </defs>
      {ACT.map((lab, j) => (
        <text key={"cl" + j} x={cx(j)} y={tm - 22} textAnchor="middle" fontFamily="ui-monospace,monospace" fontSize="10" fill="var(--p1)">{lab}</text>
      ))}
      {ACT.map((lab, i) => (
        <text key={"rl" + i} x={lm - 34} y={cy(i) + 4} textAnchor="middle" fontFamily="ui-monospace,monospace" fontSize="10" fill="var(--p0)">{lab}</text>
      ))}
      {vArrows}
      {hArrows}
      {nodes}
      <text x={lm} y={th - 4} fontFamily="ui-monospace,monospace" fontSize="9.5" fill={saddle ? "var(--ink-faint)" : "var(--ember)"}>{sub}</text>
    </svg>
  );
}
