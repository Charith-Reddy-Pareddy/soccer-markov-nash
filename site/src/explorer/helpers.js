export const ACT = ["U", "D", "L", "R"];
export const ARROW = { U: [0, 1], D: [0, -1], L: [-1, 0], R: [1, 0] };

export function stateKey(s) {
  return `${s.x0},${s.y0},${s.x1},${s.y1},${s.b}`;
}

export function kickoffState(board) {
  return { x0: 0, y0: Math.floor(board.height / 2), x1: board.width - 1, y1: Math.floor(board.height / 2), b: 0 };
}

export function wallMask(x, y, w, h) {
  return { U: y === h - 1, D: y === 0, L: x === 0, R: x === w - 1 };
}

// rows = player0's actions, cols = player1's actions, always
export function orient(Q, b) {
  if (b === 0) return Q;
  const out = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) out[i][j] = -Q[j][i];
  return out;
}

// Mirrors soccer_nash/matrix_games.py's pure_bounds: the row player
// maximises its worst case, the column player minimises its best case.
export function certify(Q, tol = 1e-6) {
  const rowMin = Q.map((row) => Math.min(...row));
  const maximin = Math.max(...rowMin);
  const colMax = [0, 1, 2, 3].map((j) => Math.max(...Q.map((r) => r[j])));
  const minimax = Math.min(...colMax);
  const gap = minimax - maximin;
  if (gap <= tol) {
    return { kind: "pure", i: rowMin.indexOf(maximin), j: colMax.indexOf(minimax), gap };
  }
  return { kind: "mixed", gap };
}

export function support(pol, tol = 0.005) {
  const out = [];
  for (let i = 0; i < 4; i++) if (pol[i] > tol) out.push(i);
  return out;
}

export function fmtPct(p) {
  return (p * 100).toFixed(1) + "%";
}
