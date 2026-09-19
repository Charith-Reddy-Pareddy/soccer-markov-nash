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

// Sample an action index from a length-4 probability vector.
function sampleAction(pol) {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < pol.length; i++) {
    acc += pol[i];
    if (r <= acc) return i;
  }
  return pol.length - 1; // float rounding fallback
}

// One joint action's `trans` entry is either a bare state index / terminal
// sentinel (the single-outcome case) or a list of [index, prob] pairs.
function resolveOutcome(entry) {
  if (typeof entry === "number") return entry;
  const r = Math.random();
  let acc = 0;
  for (const [idx, prob] of entry) {
    acc += prob;
    if (r <= acc) return idx;
  }
  return entry[entry.length - 1][0];
}

// Self-play the exact equilibrium policies against each other from
// `startKey`, sampling real `game.transitions()` outcomes (precomputed by
// scripts/explorer_data.py, not a second transition engine reimplemented
// here) -- the same "simulate N games, tally win/draw" a member's own site
// demoed, run against this project's exact solve instead of an approximate
// one. `maxSteps` mirrors the project's own `SoccerGame.max_steps` (100):
// a game that hasn't ended by then counts as a draw, same as the A10 rule.
export function simulateGames(board, startKey, trials, maxSteps = 100) {
  const { state_list, states } = board;
  let p0 = 0, p1 = 0, draw = 0;
  for (let t = 0; t < trials; t++) {
    let key = startKey;
    let winner = null;
    for (let step = 0; step < maxSteps; step++) {
      const [, rowPol, colPol, , trans] = states[key];
      const a0 = sampleAction(rowPol), a1 = sampleAction(colPol);
      const next = resolveOutcome(trans[a0 * 4 + a1]);
      if (next === -1) { winner = 0; break; }
      if (next === -2) { winner = 1; break; }
      key = state_list[next];
    }
    if (winner === 0) p0++;
    else if (winner === 1) p1++;
    else draw++;
  }
  return { p0, p1, draw, trials };
}

// Shared by QMatrixTable (payoff cells) and Board (value heatmap): ember for
// low, pitch-green for high, centred at the midpoint of the given range.
export function heatColor(v, lo, hi) {
  let t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
  t = Math.min(Math.max(t, 0), 1);
  if (t < 0.5) return `color-mix(in srgb, var(--ember-soft) ${Math.round((0.5 - t) * 2 * 70)}%, var(--raise))`;
  return `color-mix(in srgb, var(--pitch-soft) ${Math.round((t - 0.5) * 2 * 70)}%, var(--raise))`;
}
