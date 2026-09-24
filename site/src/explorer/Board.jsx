import { ACT, ARROW, heatColor, stateKey, wallMask } from "./helpers.js";

const CELL = 62;
const MARGIN = 34;

function cellCenter(x, y, h) {
  return [MARGIN + x * CELL + CELL / 2, MARGIN + (h - 1 - y) * CELL + CELL / 2];
}

// Mirrors soccer_nash/viz.py's _action_fan: a wall-clamped action is drawn as
// a dashed hold ring, never as a directional arrow -- an arrow into a wall
// would show movement that never happens.
//
// showLabels=false keeps the arrows/rings (direction and thickness still
// carry the mix) but drops the percentage text: a label sits near an arrow's
// *tip*, up to ~90% of a cell width from its own centre, so with the V(s)
// heatmap also on (every cell's centre now has its own number) the two
// texts land close enough to overlap illegibly. The exact percentages are
// still shown in the Carrier/Defender lines and the Q matrix's own headers,
// so nothing is lost, just not doubled up on the board itself.
function ActionFan({ cx, cy, pol, colour, wall, showLabels = true }) {
  const hold = ACT.reduce((sum, a, i) => (wall[a] ? sum + pol[i] : sum), 0);
  const top = Math.max(...pol, hold, 1e-9);
  const els = [];

  if (hold >= 0.02) {
    els.push(
      <circle key="hold" cx={cx} cy={cy} r={15} fill="none" stroke={colour}
        strokeWidth={(1.6 + 2.2 * (hold / top)).toFixed(1)} strokeDasharray="2 2"
        opacity={(0.4 + 0.55 * hold).toFixed(2)} />
    );
    if (showLabels && hold < 0.985) {
      els.push(
        <text key="hold-label" x={cx} y={cy + 30} textAnchor="middle"
          fontFamily="ui-monospace,monospace" fontSize="10" fill={colour}>
          hold {Math.round(hold * 100)}%
        </text>
      );
    }
  }

  ACT.forEach((a, idx) => {
    if (wall[a]) return;
    const p = pol[idx];
    if (p < 0.02) return;
    const [dx, dy] = ARROW[a];
    const start = 13, length = start + 8 + 22 * p;
    const sx = cx + dx * start, sy = cy - dy * start;
    const ex = cx + dx * length, ey = cy - dy * length;
    const wgt = 1.9 + 2.6 * (p / top);
    const markerId = colour === "var(--p0)" ? "p0" : "p1";
    els.push(
      <line key={"a-" + a} x1={sx} y1={sy} x2={ex} y2={ey} stroke={colour}
        strokeWidth={wgt.toFixed(1)} strokeLinecap="round"
        opacity={(0.4 + 0.55 * p).toFixed(2)} markerEnd={`url(#ah-${markerId})`} />
    );
    if (showLabels && p < 0.985) {
      const lx = cx + dx * (length + 13), ly = cy - dy * (length + 13);
      els.push(
        <text key={"l-" + a} x={lx} y={ly + 4} textAnchor="middle"
          fontFamily="ui-monospace,monospace" fontSize="10" fill={colour}>
          {Math.round(p * 100)}%
        </text>
      );
    }
  });

  return <>{els}</>;
}

function Player({ cx, cy, label, colour, carrier, active }) {
  return (
    <>
      {active && (
        <circle cx={cx} cy={cy} r={19} fill="none" stroke={colour} strokeWidth={2}
          strokeDasharray="1 3" opacity={0.55} />
      )}
      <circle cx={cx} cy={cy} r={15} fill={colour} />
      {carrier && (
        <circle cx={cx} cy={cy} r={21} fill="none" stroke={colour} strokeWidth={1.5} strokeDasharray="3 4" />
      )}
      <text x={cx} y={cy + 4} textAnchor="middle" fontFamily="sans-serif" fontWeight="700"
        fontSize="14" fill="var(--raise)">{label}</text>
    </>
  );
}

export default function Board({ board, state, activePlayer, onCellClick, heatmap }) {
  const W = board.width, H = board.height, GOALS = board.goal_rows;
  const boardW = W * CELL, boardH = H * CELL;
  const totalW = boardW + 2 * MARGIN, totalH = boardH + 2 * MARGIN;
  const rec = board.states[stateKey(state)];
  const [, rowPol, colPol] = rec;
  const c0 = cellCenter(state.x0, state.y0, H);
  const c1 = cellCenter(state.x1, state.y1, H);
  const bc = state.b === 0 ? c0 : c1;

  const heatVals = heatmap ? [...heatmap.values()] : [];
  const heatLo = Math.min(...heatVals), heatHi = Math.max(...heatVals);

  const cells = [];
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      const [cx, cy] = cellCenter(x, y, H);
      const v = heatmap ? heatmap.get(`${x},${y}`) : undefined;
      cells.push(
        <rect key={`${x},${y}`} className="cell-hit" x={cx - CELL / 2} y={cy - CELL / 2}
          width={CELL} height={CELL}
          fill={v === undefined ? "transparent" : heatColor(v, heatLo, heatHi)}
          onClick={() => onCellClick(x, y)} />
      );
      if (v !== undefined) {
        cells.push(
          <text key={`${x},${y}-v`} x={cx} y={cy + 4} textAnchor="middle" pointerEvents="none"
            fontFamily="ui-monospace,monospace" fontSize="10.5" fill="var(--ink-soft)">
            {v >= 0 ? "+" : ""}{v.toFixed(2)}
          </text>
        );
      }
    }
  }

  return (
    <svg viewBox={`0 0 ${totalW} ${totalH}`} role="img" aria-label="interactive soccer board">
      <defs>
        <marker id="ah-p0" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="3.6" markerHeight="3.6" orient="auto">
          <path d="M0 0.5 L8 4 L0 7.5 z" fill="var(--p0)" />
        </marker>
        <marker id="ah-p1" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="3.6" markerHeight="3.6" orient="auto">
          <path d="M0 0.5 L8 4 L0 7.5 z" fill="var(--p1)" />
        </marker>
      </defs>
      <rect x={MARGIN} y={MARGIN} width={boardW} height={boardH} fill="var(--tint)" stroke="var(--rule-strong)" strokeWidth={1.5} />
      {Array.from({ length: W - 1 }, (_, i) => i + 1).map((i) => {
        const gx = MARGIN + i * CELL;
        return <line key={"v" + i} x1={gx} y1={MARGIN} x2={gx} y2={MARGIN + boardH} stroke="var(--rule)" />;
      })}
      {Array.from({ length: H - 1 }, (_, j) => j + 1).map((j) => {
        const gy = MARGIN + j * CELL;
        return <line key={"h" + j} x1={MARGIN} y1={gy} x2={MARGIN + boardW} y2={gy} stroke="var(--rule)" />;
      })}
      {GOALS.map((row) => {
        const gy = MARGIN + (H - 1 - row) * CELL;
        return (
          <g key={"goal" + row}>
            <rect x={MARGIN - 7} y={gy} width={7} height={CELL} fill="var(--p1)" />
            <rect x={MARGIN + boardW} y={gy} width={7} height={CELL} fill="var(--p0)" />
          </g>
        );
      })}
      {cells}
      <ActionFan cx={c0[0]} cy={c0[1]} pol={rowPol} colour="var(--p0)" wall={wallMask(state.x0, state.y0, W, H)} showLabels={!heatmap} />
      <ActionFan cx={c1[0]} cy={c1[1]} pol={colPol} colour="var(--p1)" wall={wallMask(state.x1, state.y1, W, H)} showLabels={!heatmap} />
      <Player cx={c0[0]} cy={c0[1]} label="0" colour="var(--p0)" carrier={state.b === 0} active={activePlayer === 0} />
      <Player cx={c1[0]} cy={c1[1]} label="1" colour="var(--p1)" carrier={state.b === 1} active={activePlayer === 1} />
      <circle cx={bc[0] + 15} cy={bc[1] - 15} r={5.5} fill="var(--ball)" stroke="var(--raise)" strokeWidth={1.3} />
    </svg>
  );
}
