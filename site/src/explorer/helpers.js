export const ACT = ["U", "D", "L", "R"];
export const ARROW = { U: [0, 1], D: [0, -1], L: [-1, 0], R: [1, 0] };

export function stateKey(s) {
  return `${s.x0},${s.y0},${s.x1},${s.y1},${s.b}`;
}

export function kickoffState(board) {
  return { x0: 0, y0: Math.floor(board.height / 2), x1: board.width - 1, y1: Math.floor(board.height / 2), b: 0 };
}

// Per action, whether it's clamped at the board edge -- and if so, whether
// that's a genuine no-op ("hold": there's no cell there, the player just
// stays put) or a goal ("score": this exact player, in a goal row, with the
// ball, pushing further past the boundary -- soccer_nash.game.SoccerGame
// ._target's own scoring condition). Only the forward action (R for player
// 0, L for player 1) can ever score; U/D never move a player past the x
// boundary so they can never trigger it. goalRows/isCarrier/playerIdx are
// optional -- omitting them (or not being the carrier) just means every
// edge clamp reports as a plain "hold", the old geometry-only behaviour.
export function wallMask(x, y, w, h, goalRows, isCarrier, playerIdx) {
  const atGoalRow = !!goalRows && goalRows.includes(y) && isCarrier;
  const scoresR = playerIdx === 0 && atGoalRow && x === w - 1;
  const scoresL = playerIdx === 1 && atGoalRow && x === 0;
  return {
    U: y === h - 1 ? "hold" : null,
    D: y === 0 ? "hold" : null,
    L: x === 0 ? (scoresL ? "score" : "hold") : null,
    R: x === w - 1 ? (scoresR ? "score" : "hold") : null,
  };
}

// The best simple fraction approximating p, denominator <= maxDenominator --
// continued-fraction convergents, the standard algorithm for "closest nice
// ratio," not just rounding to the nearest 1/2, 1/3, 1/4, ... in a fixed
// list. Purely a *display* aid: every computation stays in exact floating
// percentages, this only formats the result afterward, so it never affects
// which action the site says is optimal.
export function nearestNiceFraction(p, maxDenominator = 20) {
  if (!(p > 0)) return "0";
  if (p >= 1) return "1";
  let h0 = 0, h1 = 1, k0 = 1, k1 = 0;
  let b = p;
  let num = 0, den = 1;
  for (let i = 0; i < 30; i++) {
    const a = Math.floor(b);
    const h2 = a * h1 + h0, k2 = a * k1 + k0;
    if (k2 > maxDenominator) break;
    num = h2; den = k2;
    h0 = h1; h1 = h2; k0 = k1; k1 = k2;
    const frac = b - a;
    if (frac < 1e-9) break;
    b = 1 / frac;
  }
  return `${num}/${den}`;
}

// Each action's expected value against the *opponent's actual mix* -- the
// literal indifference condition a mixed equilibrium satisfies: every
// action in a player's support must tie for the best available expected
// value, and every action outside it must do strictly worse. Mirrors
// scripts/positions.py's `_indifference` (e_row = M @ colPol, e_col =
// -(rowPol @ M)), not a separate reimplementation of the game-theory.
export function expectedValues(M, rowPol, colPol) {
  const eRow = M.map((row) => row.reduce((s, v, j) => s + v * colPol[j], 0));
  const eCol = ACT.map((_, j) => -M.reduce((s, row, i) => s + rowPol[i] * row[j], 0));
  return { eRow, eCol };
}

// Where a specific joint action (a0, a1) actually leads, in plain terms --
// resolved from the same precomputed `trans` outcomes simulateGames and
// stepPolicy sample from, not a second transition engine. A pure saddle's
// destination is exactly this, at the saddle's own (i, j).
export function describeOutcome(board, key, a0Idx, a1Idx) {
  const [, , , , trans] = board.states[key];
  const entry = trans[a0Idx * 4 + a1Idx];
  const describe = (idx) => {
    if (idx === -1) return "player 0 scores";
    if (idx === -2) return "player 1 scores";
    const [x0, y0, x1, y1, b] = board.state_list[idx].split(",").map(Number);
    return `(${x0}, ${y0}, ${x1}, ${y1}, ${b})`;
  };
  if (typeof entry === "number") return describe(entry);
  return entry.map(([idx, p]) => `${fmtPct(p)}: ${describe(idx)}`).join(" or ");
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
    // A pure saddle's (i, j) is just the *first* row/column achieving the
    // guaranteed value -- when more than one does, every one of them is
    // equally optimal (the LP/argmax picks one arbitrarily, the same
    // "multiple Nash, which one do you report" question this project's own
    // docs/degeneracy.md documents for mixed equilibria). rowTies/colTies
    // list every tied index, not just the displayed one, so the site never
    // implies a unique answer where there isn't one.
    const rowTies = rowMin.reduce((acc, v, idx) => (Math.abs(v - maximin) <= tol ? [...acc, idx] : acc), []);
    const colTies = colMax.reduce((acc, v, idx) => (Math.abs(v - minimax) <= tol ? [...acc, idx] : acc), []);
    return {
      kind: "pure", i: rowMin.indexOf(maximin), j: colMax.indexOf(minimax),
      gap, maximin, minimax, rowTies, colTies,
    };
  }
  return { kind: "mixed", gap, maximin, minimax };
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

// One joint action, policy-resolved and sampled exactly like simulateGames'
// own inner loop (this is that same loop, factored out) -- for the
// interactive Step/Play controls, which need to watch one game unfold on the
// board itself rather than only tally win/draw stats over many of them.
// Returns the action pair and either the next state's index into
// `board.state_list` or a terminal sentinel (-1 player 0 scored, -2 player 1).
export function stepPolicy(board, key, type0, type1) {
  const [, rowPol, colPol, M, trans] = board.states[key];
  const [p0, p1] = resolvePolicies(type0, type1, rowPol, colPol, M);
  const a0 = sampleAction(p0), a1 = sampleAction(p1);
  const next = resolveOutcome(trans[a0 * 4 + a1]);
  return { a0, a1, next };
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
//
// Also tracks each game's length and player 0's discounted return
// (`gamma^step * (+1 win / -1 loss / 0 draw)`, reward is 0 at every
// non-terminal step under the "win" objective) -- the same quantity
// NashQIteration.run_exact()'s `V` is the exact expectation of
// (soccer_nash/nash_q.py: `V(s) = val(E[R + gamma * V(s')])`), so the
// simulated mean return is directly comparable to the certified V shown
// above the Q matrix, not just a separate win/draw readout.
export function simulateGames(board, startKey, trials, type0, type1, gamma, maxSteps = 100) {
  const { state_list, states } = board;
  let p0Wins = 0, p1Wins = 0, draw = 0, totalSteps = 0, totalReturn = 0, decisiveSteps = 0;
  for (let t = 0; t < trials; t++) {
    let key = startKey;
    let winner = null;
    let step = 0;
    for (; step < maxSteps; step++) {
      const [, rowPol, colPol, M, trans] = states[key];
      const [p0, p1] = resolvePolicies(type0, type1, rowPol, colPol, M);
      const a0 = sampleAction(p0), a1 = sampleAction(p1);
      const next = resolveOutcome(trans[a0 * 4 + a1]);
      if (next === -1) { winner = 0; break; }
      if (next === -2) { winner = 1; break; }
      key = state_list[next];
    }
    if (winner === 0) { p0Wins++; totalReturn += gamma ** step; decisiveSteps += step + 1; }
    else if (winner === 1) { p1Wins++; totalReturn -= gamma ** step; decisiveSteps += step + 1; }
    else draw++;
    totalSteps += winner === null ? maxSteps : step + 1;
  }
  const decisive = p0Wins + p1Wins;
  return {
    p0: p0Wins, p1: p1Wins, draw, trials,
    avgSteps: totalSteps / trials,
    meanReturn: totalReturn / trials,
    // A draw always runs the full maxSteps, which can swamp "average game
    // length" if draws are common -- this is the same average restricted to
    // games that actually ended in a goal, so it reads as "how long does a
    // decisive game typically take" rather than a number dragged toward 100.
    decisiveRate: decisive / trials,
    decisiveAvgSteps: decisive > 0 ? decisiveSteps / decisive : null,
  };
}

// Shared by QMatrixTable (payoff cells) and Board (value heatmap): ember for
// low, pitch-green for high, centred at the midpoint of the given range.
export function heatColor(v, lo, hi) {
  let t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
  t = Math.min(Math.max(t, 0), 1);
  if (t < 0.5) return `color-mix(in srgb, var(--ember-soft) ${Math.round((0.5 - t) * 2 * 70)}%, var(--raise))`;
  return `color-mix(in srgb, var(--pitch-soft) ${Math.round((t - 0.5) * 2 * 70)}%, var(--raise))`;
}
