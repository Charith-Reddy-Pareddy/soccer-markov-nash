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

export const POLICY_TYPES = ["minimax", "left", "random", "br"];
export const POLICY_LABELS = {
  minimax: "Minimax (exact)",
  left: "Always left",
  random: "Random",
  br: "Best response",
};

const UNIFORM = [0.25, 0.25, 0.25, 0.25];
const ALWAYS_LEFT = [0, 0, 1, 0]; // absolute direction, not carrier-relative -- same convention as scripts/tournament4.py's always_left

// Row player (0) maximises `M`; best pure reply to the column player's mix `q`.
function bestResponseRow(M, q) {
  let best = 0, bestV = -Infinity;
  for (let i = 0; i < 4; i++) {
    let v = 0;
    for (let j = 0; j < 4; j++) v += M[i][j] * q[j];
    if (v > bestV) { bestV = v; best = i; }
  }
  const out = [0, 0, 0, 0]; out[best] = 1; return out;
}

// Column player (1) minimises `M`; best pure reply to the row player's mix `p`.
function bestResponseCol(M, p) {
  let best = 0, bestV = Infinity;
  for (let j = 0; j < 4; j++) {
    let v = 0;
    for (let i = 0; i < 4; i++) v += p[i] * M[i][j];
    if (v < bestV) { bestV = v; best = j; }
  }
  const out = [0, 0, 0, 0]; out[best] = 1; return out;
}

// Resolve both players' actual mix at one state given their chosen policy
// *type* -- "minimax" uses the exact equilibrium strategy already in the
// data, "left"/"random" are fixed, and "br" best-responds to whatever the
// other side turns out to play. If both sides are "br" there's no order to
// resolve first, so both fall back to the exact equilibrium -- which is, not
// coincidentally, the actual fixed point of "best-respond to a best-response".
function resolvePolicies(type0, type1, rowPol, colPol, M) {
  const fixed = (type, pol) => (type === "minimax" ? pol : type === "left" ? ALWAYS_LEFT : type === "random" ? UNIFORM : null);
  let p0 = fixed(type0, rowPol);
  let p1 = fixed(type1, colPol);
  if (p0 === null && p1 === null) return [rowPol, colPol];
  if (p1 === null) p1 = bestResponseCol(M, p0);
  if (p0 === null) p0 = bestResponseRow(M, p1);
  return [p0, p1];
}

// Self-play the chosen policies against each other from `startKey`, sampling
// real `game.transitions()` outcomes (precomputed by scripts/explorer_data.py,
// not a second transition engine reimplemented here) -- the same "simulate N
// games, tally win/draw" a member's own site demoed, run against this
// project's exact solve instead of an approximate one, and against the same
// policy menu (minimax / always-left / random / best-response) as
// docs/tournament.md's reproduction of Littman's Table 3. `maxSteps` mirrors
// the project's own `SoccerGame.max_steps` (100): a game that hasn't ended by
// then counts as a draw, same as the A10 rule.
export function simulateGames(board, startKey, trials, type0, type1, maxSteps = 100) {
  const { state_list, states } = board;
  let p0Wins = 0, p1Wins = 0, draw = 0;
  for (let t = 0; t < trials; t++) {
    let key = startKey;
    let winner = null;
    for (let step = 0; step < maxSteps; step++) {
      const [, rowPol, colPol, M, trans] = states[key];
      const [p0, p1] = resolvePolicies(type0, type1, rowPol, colPol, M);
      const a0 = sampleAction(p0), a1 = sampleAction(p1);
      const next = resolveOutcome(trans[a0 * 4 + a1]);
      if (next === -1) { winner = 0; break; }
      if (next === -2) { winner = 1; break; }
      key = state_list[next];
    }
    if (winner === 0) p0Wins++;
    else if (winner === 1) p1Wins++;
    else draw++;
  }
  return { p0: p0Wins, p1: p1Wins, draw, trials };
}

// Shared by QMatrixTable (payoff cells) and Board (value heatmap): ember for
// low, pitch-green for high, centred at the midpoint of the given range.
export function heatColor(v, lo, hi) {
  let t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
  t = Math.min(Math.max(t, 0), 1);
  if (t < 0.5) return `color-mix(in srgb, var(--ember-soft) ${Math.round((0.5 - t) * 2 * 70)}%, var(--raise))`;
  return `color-mix(in srgb, var(--pitch-soft) ${Math.round((t - 0.5) * 2 * 70)}%, var(--raise))`;
}
