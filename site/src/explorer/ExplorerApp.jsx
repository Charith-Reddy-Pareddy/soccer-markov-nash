import { useEffect, useMemo, useState } from "react";
import Nav from "../Nav.jsx";
import Footer from "../Footer.jsx";
import Board from "./Board.jsx";
import QMatrixTable from "./QMatrixTable.jsx";
import QMatrixGraph from "./QMatrixGraph.jsx";
import { ACT, certify, describeOutcome, expectedValues, fmtPct, kickoffState, nearestNiceFraction, POLICY_LABELS, POLICY_TYPES, simulateGames, stateKey, stepPolicy, support, wallMask } from "./helpers.js";
import "./explorer.css";

const BOARD_ORDER = ["canonical", "canonical_det", "canonical_coinflip", "tackle", "territory", "slip"];

// case number -> {board, state}, coordinates as printed in docs/positions.md
// Each board with its own dedicated case-study PDF (every case there links
// back into this same explorer via ?board=&state=). A board can carry both
// this AND entries from positions.pdf's original 14 (tackle/territory/slip
// do) -- the two are shown as separate lines, filtered by each preset's own
// `doc` field ("edge" for these, unset/"positions" for positions.pdf's own).
const BOARD_CASE_DOCS = {
  canonical_det: { pdf: "a10_cases.pdf", label: "A10-deterministic" },
  canonical_coinflip: { pdf: "coinflip_cases.pdf", label: "coin-flip" },
  tackle: { pdf: "tackle_cases.pdf", label: "tackle-rule" },
  territory: { pdf: "territory_cases.pdf", label: "territory-reward" },
  slip: { pdf: "slip_cases.pdf", label: "movement-slip" },
};

const PRESETS = {
  1: { board: "canonical", state: [4, 0, 5, 0, 0], label: "Case 1 — pure, for contrast" },
  2: { board: "canonical", state: [0, 1, 1, 1, 0], label: "Case 2 — the typical mix" },
  3: { board: "canonical", state: [1, 1, 1, 0, 1], label: "Case 3 — L/R indifference" },
  4: { board: "canonical", state: [0, 0, 1, 1, 0], label: "Case 4 — the corner duel" },
  5: { board: "canonical", state: [0, 0, 2, 0, 0], label: "Case 5 — 3-action mix" },
  6: { board: "canonical", state: [1, 1, 2, 0, 1], label: "Case 6 — near-pure hedge" },
  7: { board: "canonical", state: [5, 1, 6, 1, 1], label: "Case 7 — mirrored" },
  8: { board: "tackle", state: [2, 3, 3, 3, 1], label: "Case 8 — the tackle rule" },
  9: { board: "canonical", state: [0, 2, 1, 2, 0], label: "Case 9 — asymmetric mix" },
  10: { board: "canonical", state: [0, 2, 2, 2, 1], label: "Case 10 — three-lane mix" },
  11: { board: "territory", state: [4, 4, 5, 4, 0], label: "Case 11 — reward forces the mix" },
  12: { board: "slip", state: [1, 3, 1, 4, 1], label: "Case 12 — movement slip" },
  13: { board: "canonical", state: [1, 1, 3, 1, 1], label: "Case 13 — template 3" },
  14: { board: "canonical", state: [0, 2, 2, 2, 0], label: "Case 14 — rarest template" },
  // A10-deterministic edge cases: every deterministic stage game is pure
  // (this project's own headline result), but "pure" still hides real
  // structure in the matrix -- these show it. Numbered 1-8 within their own
  // set (not 15+) since the presets list below only ever shows one set or
  // the other, keyed off which board is selected.
  15: { board: "canonical_det", doc: "edge", state: [4, 3, 5, 3, 0], label: "Edge case 1 — the swap trap" },
  16: { board: "canonical_det", doc: "edge", state: [0, 0, 0, 2, 0], label: "Edge case 2 — pinned in the corner" },
  17: { board: "canonical_det", doc: "edge", state: [3, 4, 4, 4, 0], label: "Edge case 3 — the standoff" },
  // Player 0 sits at the right wall in a goal row: R isn't wall-clamped here
  // -- a goal column is the one place the boundary action ends the game
  // instead of holding in place, verified against game.transitions() (every
  // reply from player 1 gives the same certain win).
  18: { board: "canonical_det", doc: "edge", state: [6, 1, 4, 1, 0], label: "Edge case 4 — the open goal" },
  // The mirror of 18 from player 1's side: L at the left wall in a goal row,
  // ball with player 1, scores outright regardless of player 0's reply.
  19: { board: "canonical_det", doc: "edge", state: [0, 0, 0, 1, 1], label: "Edge case 5 — the open net" },
  // R is the *unique* safe action here (every column gives the same 0.531441
  // -- player 1 can't stop a clean break into open space); U looks equally
  // natural but is a trap -- if player 1 answers with D the two target the
  // same empty cell, player 0 "wins" the race there and immediately hands
  // over the ball anyway, per the A10 contest rule (loser keeps the ball
  // either way). Verified against game.transitions().
  20: { board: "canonical_det", doc: "edge", state: [0, 1, 0, 3, 0], label: "Edge case 6 — the getaway" },
  // Every state on this board ties on at least one side -- (1,1), a fully
  // unique best action for both players simultaneously, never once occurs
  // among all 2380 states (checked exhaustively). This is the smallest tie
  // shape that does occur: only 16 of 2380 states have exactly a 2-way tie
  // opposite a unique action. U and L both look safe but each hides its own
  // contest -- (U, R) and (L, D) both send player 0 and player 1 to the same
  // cell, verified against game.transitions() -- leaving D/R as the only
  // two moves that never reach player 1 at all.
  21: { board: "canonical_det", doc: "edge", state: [1, 0, 0, 1, 0], label: "Edge case 7 — the tightest tie" },
  // The single most common shape on the whole board (514 of 2380 states,
  // rowties=colties=4): both players are fully indifferent among all four
  // actions. It's not merely that the *guaranteed* values tie -- every one
  // of the 16 joint actions in the Q matrix is exactly 0.0, including a
  // genuine ball-swapping contest (verified: (R, L) -> a state that is
  // itself also worth exactly 0). Nothing that happens this turn, for
  // either player, changes anything.
  22: { board: "canonical_det", doc: "edge", state: [0, 0, 2, 0, 0], label: "Edge case 8 — the dead zone" },
  // tackle_cases.pdf: this project's own probabilistic-duel collision rule
  // (docs/tackle.md) produces 56 of 760 genuinely mixed states -- the
  // richest source of real mixing among the non-A10 board variants.
  23: { board: "tackle", doc: "edge", state: [0, 0, 1, 0, 0], label: "Tackle case 1 — the corner duel" },
  24: { board: "tackle", doc: "edge", state: [0, 0, 1, 0, 1], label: "Tackle case 2 — the committed challenge" },
  25: { board: "tackle", doc: "edge", state: [0, 0, 1, 1, 0], label: "Tackle case 3 — mixing without a duel" },
  26: { board: "tackle", doc: "edge", state: [0, 0, 4, 3, 0], label: "Tackle case 4 — no challenge, plain A10" },
  27: { board: "tackle", doc: "edge", state: [1, 1, 2, 1, 0], label: "Tackle case 5 — the 50/50 duel in full" },
  28: { board: "tackle", doc: "edge", state: [4, 0, 4, 1, 0], label: "Tackle case 6 — head-on at the byline" },
  29: { board: "tackle", doc: "edge", state: [4, 1, 3, 1, 0], label: "Tackle case 7 — the goal that beats the tackle" },
  // coinflip_cases.pdf: A10's exact structure with a fair coin (not the
  // carrier) deciding contests -- still 0 of 2380 states genuinely mix,
  // but the coin does change *which* pure action wins in several cases.
  30: { board: "canonical_coinflip", doc: "edge", state: [4, 3, 5, 3, 0], label: "Coinflip case 1 — the swap trap, disarmed" },
  31: { board: "canonical_coinflip", doc: "edge", state: [0, 0, 0, 2, 0], label: "Coinflip case 2 — pinned in the corner, still pinned" },
  32: { board: "canonical_coinflip", doc: "edge", state: [0, 1, 0, 0, 1], label: "Coinflip case 3 — the tie that breaks" },
  33: { board: "canonical_coinflip", doc: "edge", state: [0, 2, 0, 0, 0], label: "Coinflip case 4 — the defender's flip" },
  34: { board: "canonical_coinflip", doc: "edge", state: [6, 1, 4, 1, 0], label: "Coinflip case 5 — the open goal, mostly immune" },
  35: { board: "canonical_coinflip", doc: "edge", state: [1, 0, 0, 1, 0], label: "Coinflip case 6 — the tightest tie, narrowly missed" },
  36: { board: "canonical_coinflip", doc: "edge", state: [0, 0, 2, 0, 0], label: "Coinflip case 7 — the dead zone, untouched" },
  // territory_cases.pdf: a dense per-step reward for ball position creates
  // 69 of 2380 genuinely mixed states, on top of the underlying win/lose game.
  37: { board: "territory", doc: "edge", state: [0, 1, 6, 1, 0], label: "Territory case 1 — the three-way tug of war" },
  38: { board: "territory", doc: "edge", state: [0, 0, 1, 1, 0], label: "Territory case 2 — the corner squeeze" },
  39: { board: "territory", doc: "edge", state: [0, 1, 1, 1, 0], label: "Territory case 3 — a profitable break" },
  40: { board: "territory", doc: "edge", state: [0, 1, 3, 4, 0], label: "Territory case 4 — splitting the wide side" },
  41: { board: "territory", doc: "edge", state: [1, 0, 4, 2, 0], label: "Territory case 5 — the price of your own third" },
  42: { board: "territory", doc: "edge", state: [0, 0, 2, 0, 0], label: "Territory case 6 — the dead zone wakes up" },
  43: { board: "territory", doc: "edge", state: [6, 1, 4, 1, 0], label: "Territory case 7 — goals still trump territory" },
  // slip_cases.pdf: independent per-player movement noise creates 52 of
  // 1200 genuinely mixed states -- a small chance of stumbling undermines
  // what looks like a safe pure action.
  44: { board: "slip", doc: "edge", state: [0, 1, 0, 3, 0], label: "Slip case 1 — the stumble" },
  45: { board: "slip", doc: "edge", state: [4, 2, 0, 2, 0], label: "Slip case 2 — the uncertain open goal" },
  46: { board: "slip", doc: "edge", state: [0, 1, 1, 1, 0], label: "Slip case 3 — split between safe and risky" },
  47: { board: "slip", doc: "edge", state: [0, 1, 2, 0, 0], label: "Slip case 4 — a tie ground into a gradient" },
  48: { board: "slip", doc: "edge", state: [0, 1, 2, 0, 1], label: "Slip case 5 — two ways to wait" },
  49: { board: "slip", doc: "edge", state: [0, 1, 2, 1, 1], label: "Slip case 6 — a threat that shapes the mix without firing" },
};

function toState(arr) {
  return { x0: arr[0], y0: arr[1], x1: arr[2], y1: arr[3], b: arr[4] };
}

export default function ExplorerApp() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [currentBoard, setCurrentBoard] = useState("canonical");
  const [activePlayer, setActivePlayer] = useState(0);
  const [qview, setQview] = useState("table");
  const [st, setSt] = useState(null);
  const [showV, setShowV] = useState(false);
  const [goToInput, setGoToInput] = useState("");
  const [goToError, setGoToError] = useState(null);
  const [showNeural, setShowNeural] = useState(false);
  const [neuralData, setNeuralData] = useState(null);
  const [neuralError, setNeuralError] = useState(null);

  // Lazy-fetched only when the panel is actually opened: a representative
  // DQN + policy-gradient run's predictions for every state on every board,
  // several MB, not needed on first paint or by anyone who never looks at
  // this panel.
  useEffect(() => {
    if (!showNeural || neuralData || neuralError) return;
    fetch(`${import.meta.env.BASE_URL}data/explorer_neural.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(setNeuralData)
      .catch((e) => setNeuralError(e.message));
  }, [showNeural, neuralData, neuralError]);
  const [trials, setTrials] = useState(2000);
  const [simResult, setSimResult] = useState(null);
  const [simming, setSimming] = useState(false);
  const [p0Type, setP0Type] = useState("minimax");
  const [p1Type, setP1Type] = useState("minimax");
  // Step/Play trajectory: playHistory is every real (non-terminal) state
  // visited since the last reset, so "Back" and "Restart game" both have
  // somewhere to go; a goal doesn't extend it (there's no legal `st` for a
  // terminal state), it just sets playDone/winner instead.
  const [playHistory, setPlayHistory] = useState(null);
  const [playDone, setPlayDone] = useState(false);
  const [winner, setWinner] = useState(null);
  const [lastMove, setLastMove] = useState(null);
  const [playRewards, setPlayRewards] = useState([]);
  const [autoPlaying, setAutoPlaying] = useState(false);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/explorer.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((json) => {
        setData(json);
        // Deep link from a case-study PDF or preset button:
        // explorer.html?board=tackle&state=0,0,1,0,0 jumps straight to that
        // position; explorer.html?board=tackle alone just preselects the
        // board and starts at its own kickoff. Falls back to kickoff on the
        // canonical board if the board/state is missing or illegal.
        const params = new URLSearchParams(window.location.search);
        const board = params.get("board");
        const stateParam = params.get("state");
        if (board && json.boards[board]) {
          if (stateParam) {
            const nums = stateParam.split(",").map(Number);
            const key = nums.join(",");
            if (nums.length === 5 && nums.every(Number.isInteger) && json.boards[board].states[key]) {
              setCurrentBoard(board);
              const st0 = toState(nums);
              setSt(st0);
              setPlayHistory([st0]);
              return;
            }
          } else {
            setCurrentBoard(board);
            const st0 = kickoffState(json.boards[board]);
            setSt(st0);
            setPlayHistory([st0]);
            return;
          }
        }
        const kickoff = kickoffState(json.boards.canonical);
        setSt(kickoff);
        setPlayHistory([kickoff]);
      })
      .catch((e) => setError(e.message));
  }, []);

  const board = data ? data.boards[currentBoard] : null;

  const mixedKeys = useMemo(() => {
    if (!board) return [];
    return Object.keys(board.states).filter((k) => certify(board.states[k][3]).kind === "mixed");
  }, [board]);

  // Value heatmap: sweep the active (movable) player over every legal cell,
  // holding the other player and the ball fixed, reading V straight out of
  // the exact solve -- no recomputation, same lookup the single-state view
  // uses. V is always stored as player 0's value, so player 1's own view of
  // it is the zero-sum negation.
  const heatmap = useMemo(() => {
    if (!board || !st || !showV) return null;
    const m = new Map();
    for (let x = 0; x < board.width; x++) {
      for (let y = 0; y < board.height; y++) {
        if (activePlayer === 0 ? (x === st.x1 && y === st.y1) : (x === st.x0 && y === st.y0)) continue;
        const key = activePlayer === 0
          ? stateKey({ x0: x, y0: y, x1: st.x1, y1: st.y1, b: st.b })
          : stateKey({ x0: st.x0, y0: st.y0, x1: x, y1: y, b: st.b });
        const rec = board.states[key];
        if (!rec) continue;
        m.set(`${x},${y}`, activePlayer === 0 ? rec[0] : -rec[0]);
      }
    }
    return m;
  }, [board, st, activePlayer, showV]);

  // "Play": step automatically on a visible interval, so the trajectory can
  // actually be watched move by move rather than jumping straight to the
  // end. Stops itself on a goal, at the max_steps=100 safety cap (matching
  // the game's own draw rule), or when the user pauses. No dependency array
  // on purpose: it re-arms after every render (i.e. after every step, since
  // stepOnce's setSt/setPlayHistory each trigger one), always closing over
  // the render's current st/p0Type/p1Type -- a narrower dependency list
  // would read a stale state or policy mid-sequence.
  useEffect(() => {
    if (!autoPlaying || playDone || !playHistory || playHistory.length - 1 >= 100) return;
    const t = setTimeout(() => stepOnce(), 450);
    return () => clearTimeout(t);
  });

  if (error) {
    return (
      <>
        <Nav current="explorer" />
        <section><div className="wrap"><p className="loading-note">Couldn't load the explorer data ({error}). Try refreshing.</p></div></section>
        <Footer />
      </>
    );
  }
  if (!data || !st || !playHistory) {
    return (
      <>
        <Nav current="explorer" />
        <section><div className="wrap"><p className="loading-note">Loading the exact solve&hellip;</p></div></section>
        <Footer />
      </>
    );
  }

  // Any time the position itself changes (not just a step along it), the
  // Step/Play trajectory restarts from there too -- a stale "discounted
  // return so far" or "player 0 scored" banner from the old position would
  // be actively misleading, not just out of date.
  function resetPosition(newSt) {
    setSt(newSt);
    setSimResult(null);
    setPlayHistory([newSt]);
    setPlayRewards([]);
    setPlayDone(false);
    setWinner(null);
    setLastMove(null);
    setAutoPlaying(false);
  }

  function switchBoard(id) {
    setCurrentBoard(id);
    setActivePlayer(0);
    resetPosition(kickoffState(data.boards[id]));
  }

  function onCellClick(x, y) {
    const other = activePlayer === 0 ? [st.x1, st.y1] : [st.x0, st.y0];
    if (x === other[0] && y === other[1]) return; // occupied, no swap
    resetPosition(activePlayer === 0 ? { ...st, x0: x, y0: y } : { ...st, x1: x, y1: y });
  }

  function randomState() {
    const keys = Object.keys(board.states);
    return toState(keys[Math.floor(Math.random() * keys.length)].split(",").map(Number));
  }
  function randomMixedState() {
    const k = mixedKeys[Math.floor(Math.random() * mixedKeys.length)];
    return toState(k.split(",").map(Number));
  }

  function runSimulation() {
    setSimming(true);
    // setTimeout so the "Simulating..." label actually paints before the
    // (synchronous, but non-trivial for 10k trials) simulation runs.
    setTimeout(() => {
      setSimResult(simulateGames(board, stateKey(st), trials, p0Type, p1Type, data.gamma));
      setSimming(false);
    }, 10);
  }

  function applyPreset(n) {
    const p = PRESETS[n];
    setCurrentBoard(p.board);
    setActivePlayer(0);
    resetPosition(toState(p.state));
  }

  function moveTo(newSt) {
    resetPosition(newSt);
  }

  // Jump straight to a typed state, e.g. "3,4,4,4,0" -- the ball index is
  // optional and defaults to 0, since most of the states named in
  // conversation (screenshots, docs, a professor's own notes) are already
  // given as "(x0, y0, x1, y1)" or the full 5-tuple interchangeably. Rejects
  // anything that isn't a legal state on the *current* board rather than
  // silently clamping or guessing, since a board switch changes which
  // coordinates are even in range.
  function goToState() {
    const parts = goToInput.split(",").map((s) => s.trim()).filter((s) => s.length > 0);
    if (parts.length !== 4 && parts.length !== 5) {
      setGoToError("Enter x0,y0,x1,y1 or x0,y0,x1,y1,ball");
      return;
    }
    const nums = parts.map(Number);
    if (nums.some((n) => !Number.isInteger(n))) {
      setGoToError("All values must be whole numbers");
      return;
    }
    const [x0, y0, x1, y1, b = 0] = nums;
    if (b !== 0 && b !== 1) {
      setGoToError("Ball must be 0 or 1");
      return;
    }
    const key = stateKey({ x0, y0, x1, y1, b });
    if (!board.states[key]) {
      setGoToError(`(${x0}, ${y0}, ${x1}, ${y1}, ${b}) isn't a legal state on this board`);
      return;
    }
    setGoToError(null);
    moveTo({ x0, y0, x1, y1, b });
  }

  // One Step/Play tick: resolve both players' chosen policy at the current
  // state, sample an action pair, and either move to the real next state
  // (pushed onto playHistory) or, if it's a goal, stop without one (there is
  // no legal `st` for a terminal state) -- the same stepPolicy() resolution
  // simulateGames' own inner loop uses, just watched one step at a time.
  function stepOnce() {
    if (playDone || playRewards.length >= 100) return;
    const { a0, a1, next, reward } = stepPolicy(board, stateKey(st), p0Type, p1Type);
    setLastMove({ a0, a1 });
    setPlayRewards(r => [...r, reward]);
    if (next === -1 || next === -2) {
      setPlayDone(true);
      setWinner(next === -1 ? 0 : 1);
      setAutoPlaying(false);
      return;
    }
    const nextSt = toState(board.state_list[next].split(",").map(Number));
    setSt(nextSt);
    setPlayHistory((h) => [...h, nextSt]);
  }

  function backOnce() {
    setPlayRewards(r => r.slice(0, -1));
    setAutoPlaying(false);
    setLastMove(null);
    if (playDone) {
      setPlayDone(false);
      setWinner(null);
      return;
    }
    setPlayHistory((h) => {
      if (h.length <= 1) return h;
      const nh = h.slice(0, -1);
      setSt(nh[nh.length - 1]);
      return nh;
    });
  }

  function restartPlay() {
    setPlayRewards([]);
    setAutoPlaying(false);
    setPlayDone(false);
    setWinner(null);
    setLastMove(null);
    setPlayHistory((h) => {
      setSt(h[0]);
      return [h[0]];
    });
  }

  const rec = board.states[stateKey(st)];
  const [V, rowPol, colPol, Q, , rounding] = rec;
  // Fixed for every state: rows = player 0's actions, columns = player 1's,
  // never reoriented by who has the ball. An earlier version reoriented so
  // rows were always the carrier's actions -- per the project's own research
  // meetings, that made the matrix silently transpose between states, which
  // is exactly what was confusing a reader tracking player 0/player 1
  // physically. Which player is carrying is shown separately below.
  const M = Q;
  const cert = certify(M);
  // One source of policy probabilities for display, stepping, and simulation.
  const displayRowPol = rowPol;
  const displayColPol = colPol;
  const selectedRow = rowPol.indexOf(Math.max(...rowPol));
  const selectedCol = colPol.indexOf(Math.max(...colPol));
  const carrierPol = st.b === 0 ? displayRowPol : displayColPol;
  const defenderPol = st.b === 0 ? displayColPol : displayRowPol;
  const cs = support(carrierPol), ds = support(defenderPol);
  // Which of each player's own four actions are wall-clamped from where
  // they're actually standing right now -- a "move" into a wall is legal to
  // pick but has no effect (you stay exactly where you are), which the raw
  // percentages alone don't say. The one exception is the carrier, in a goal
  // row, pushing past *their own* attacking edge -- that's not a hold, it
  // scores (soccer_nash.game.SoccerGame._target's own condition), which
  // wallMask flags separately so it isn't shown as a no-op.
  const wall0 = wallMask(st.x0, st.y0, board.width, board.height, board.goal_rows, st.b === 0, 0);
  const wall1 = wallMask(st.x1, st.y1, board.width, board.height, board.goal_rows, st.b === 1, 1);

  return (
    <>
      <Nav current="explorer" />
      <section>
        <div className="wrap">
          <div className="eyebrow"><span className="badge">Live</span>Board explorer</div>
          <h1>Place both players anywhere. Watch the equilibrium update live.</h1>
          <p className="lede">All {Object.values(data.boards).reduce((n, b) => n + b.state_count, 0).toLocaleString()} legal positions across {Object.keys(data.boards).length} boards were solved
            exactly, <a href="positions.md">not approximated by an iterative
            solver</a>, with <code>NashQIteration.run_exact()</code>: the canonical
            board and its deterministic, A10-style twin, side by side for comparison,
            plus the three used in <a href="positions.pdf">positions.pdf</a> &mdash; this
            project's own tackle rule, the territory-reward game, and movement slip. Move
            either player, hand either one the ball, switch boards, self-play the
            equilibrium for a win/draw readout, and the equilibrium policy and the full
            4&times;4 Q matrix &mdash; as a table or
            as the same best-response graph the PDF draws &mdash; update instantly:
            no server, no recomputation, just a lookup into the exact solve.</p>

          <div className="explorer-grid">
            <div className="panel">
              <div className="controls" style={{ margin: "0 0 1.2rem" }}>
                <label className="board-select-label" htmlFor="board-select">Board</label>
                <select id="board-select" className="board-select" value={currentBoard}
                  onChange={(e) => switchBoard(e.target.value)}>
                  {BOARD_ORDER.map((id) => (
                    <option key={id} value={id}>
                      {data.boards[id].label} ({data.boards[id].width}&times;{data.boards[id].height})
                    </option>
                  ))}
                </select>
              </div>
              <div className="board-svg-wrap">
                <Board board={board} state={st} activePlayer={activePlayer} onCellClick={onCellClick} heatmap={heatmap} rowPol={displayRowPol} colPol={displayColPol} />
              </div>
              <div className="controls">
                <div className="seg" role="group" aria-label="which player clicking the board moves">
                  <button className={activePlayer === 0 ? "active p0" : ""} onClick={() => setActivePlayer(0)}>Move: Player 0</button>
                  <button className={activePlayer === 1 ? "active p1" : ""} onClick={() => setActivePlayer(1)}>Move: Player 1</button>
                </div>
                <div className="seg" role="group" aria-label="ball possession">
                  <button className={st.b === 0 ? "active p0" : ""} onClick={() => moveTo({ ...st, b: 0 })}>Ball: 0</button>
                  <button className={st.b === 1 ? "active p1" : ""} onClick={() => moveTo({ ...st, b: 1 })}>Ball: 1</button>
                </div>
              </div>
              <div className="controls">
                <label className="v-toggle">
                  <input type="checkbox" checked={showV} onChange={(e) => setShowV(e.target.checked)} />
                  Show V(s)
                </label>
              </div>
              <div className="controls">
                <button className="iconbtn" onClick={() => moveTo(kickoffState(board))}>Kickoff</button>
                <button className="iconbtn" onClick={() => moveTo(randomState())}>Random position</button>
                <button className="iconbtn" onClick={() => moveTo(randomMixedState())}>Random must-guess position</button>
              </div>
              <div className="controls">
                <input
                  type="text"
                  className="go-to-input"
                  placeholder="x0,y0,x1,y1,ball e.g. 3,4,4,4,0"
                  value={goToInput}
                  onChange={(e) => { setGoToInput(e.target.value); setGoToError(null); }}
                  onKeyDown={(e) => { if (e.key === "Enter") goToState(); }}
                />
                <button className="iconbtn" onClick={goToState}>Go to state</button>
              </div>
              {goToError && <p className="hint" style={{ color: "var(--ember)" }}>{goToError}</p>}
              <p className="hint">Click a cell to move the selected player there.{" "}
                <span style={{ color: "var(--p0)" }}>Blue</span> is player 0, attacking the
                right goal; <span style={{ color: "var(--p1)" }}>green</span> is player 1,
                attacking the left. The small dot marks the ball.{" "}
                {showV && <>Every cell is also shaded and labelled with <b>V</b> for player{" "}
                  {activePlayer} if it stood there instead &mdash;{" "}
                  {activePlayer === 0 ? "player 1" : "player 0"} and the ball held fixed where
                  they are now (switch which player moves to sweep the other one).{" "}
                  <span style={{ color: "var(--pitch)" }}>Green</span> is good for the player
                  being swept; <span style={{ color: "var(--ember)" }}>orange</span> is bad
                  &mdash; the same lookup as the single-state <b>V</b> above, run over every
                  legal cell instead of just one.</>}</p>

              <div className="controls" style={{ marginTop: "1.4rem" }}>
                <div className="seg" role="group" aria-label="player 0's policy">
                  {POLICY_TYPES.map((t) => (
                    <button key={t} className={p0Type === t ? "active p0" : ""} onClick={() => setP0Type(t)}>
                      P0: {POLICY_LABELS[t]}
                    </button>
                  ))}
                </div>
              </div>
              <div className="controls">
                <div className="seg" role="group" aria-label="player 1's policy">
                  {POLICY_TYPES.map((t) => (
                    <button key={t} className={p1Type === t ? "active p1" : ""} onClick={() => setP1Type(t)}>
                      P1: {POLICY_LABELS[t]}
                    </button>
                  ))}
                </div>
              </div>
              <div className="controls">
                <button className="iconbtn" onClick={backOnce} disabled={playHistory.length <= 1 && !playDone}>Back</button>
                <button className="iconbtn" onClick={stepOnce} disabled={playDone || playRewards.length >= 100}>Step</button>
                <button className="iconbtn" onClick={() => setAutoPlaying((p) => !p)} disabled={playDone || playRewards.length >= 100}>
                  {autoPlaying ? "Pause" : "Play"}
                </button>
                <button className="iconbtn" onClick={restartPlay}>Restart game</button>
              </div>
              <div className="sim-stats">
                <div>
                  <div className="s-k">V(s)</div>
                  <div className="s-v">{V >= 0 ? "+" : ""}{V.toFixed(6)}</div>
                </div>
                <div>
                  <div className="s-k">Step</div>
                  <div className="s-v">{playRewards.length}</div>
                </div>
                <div>
                  <div className="s-k">Discounted return so far</div>
                  <div className="s-v">
                    {playRewards.reduce((total, r, i) => total + data.gamma ** i * r, 0).toFixed(6)}
                  </div>
                </div>
                <div>
                  <div className="s-k">Last move</div>
                  <div className="s-v">
                    {lastMove ? `P0 ${ACT[lastMove.a0]}, P1 ${ACT[lastMove.a1]}` : "none yet"}
                  </div>
                </div>
              </div>
              {playDone && (
                <p className="hint" style={{ margin: ".6rem 0 0" }}>
                  <b style={{ color: winner === 0 ? "var(--p0)" : "var(--p1)" }}>
                    Player {winner} scored.
                  </b> Restart game to play again.
                </p>
              )}
              <p className="hint">
                Both players act by the policies selected above (same menu <b>Simulate</b>{" "}
                below scores in bulk) &mdash; <b>Step</b> resolves and samples one joint action
                at the current state, <b>Play</b> repeats that automatically, <b>Back</b>{" "}
                undoes the last one, and <b>Restart game</b> returns to the position this
                trajectory started from.
              </p>
            </div>

            <div className="panel">
              <div className={"readout-kind " + cert.kind}>
                {cert.kind === "pure"
                  ? <>Selected pure equilibrium — player 0: <ActionList ties={[selectedRow]} wall={wall0} />
                      , player 1: <ActionList ties={[selectedCol]} wall={wall1} />
                    </>
                  : `Mixed equilibrium — gap ${cert.gap.toFixed(4)}`}
              </div>
              <div className="state-key mono">
                state ({st.x0}, {st.y0}, {st.x1}, {st.y1}, {st.b}) &mdash; {board.label}
              </div>
              <div className="vbar-row">
                <span className="vbar-val">V = {V >= 0 ? "+" : ""}{V.toFixed(3)}</span>
                <div className="vbar">
                  <div className="mid"></div>
                  <div className="fill" style={{ left: `${50 + 50 * Math.max(Math.min(V, 1), -1)}%` }}></div>
                </div>
              </div>

              <div className="controls" style={{ margin: "0 0 .6rem" }}>
                <div className="seg" role="group" aria-label="Q matrix view">
                  <button className={qview === "table" ? "active view" : ""} onClick={() => setQview("table")}>Table</button>
                  <button className={qview === "graph" ? "active view" : ""} onClick={() => setQview("graph")}>Best-response graph</button>
                </div>
              </div>
              <p className="hint mono" style={{ margin: "0 0 .5rem" }}>
                rows = player 0 &middot; columns = player 1 &middot; cells = player 0's
                payoff &mdash; fixed for every state, not reoriented by who has the ball
              </p>
              <p className="hint" style={{ margin: "0 0 .6rem" }}>
                Best single action for player 0 guarantees{" "}
                <span className="mono">{cert.maximin.toFixed(4)}</span> &middot; best single
                action for player 1 holds player 0 to{" "}
                <span className="mono">{cert.minimax.toFixed(4)}</span> &middot; gap{" "}
                <span className="mono">{cert.gap.toFixed(4)}</span>, so{" "}
                {cert.kind === "pure"
                  ? "a pure saddle point exists and no mixing is needed."
                  : "neither side can guarantee more with a single fixed action, so a mixed strategy is required."}
              </p>
              {qview === "table" ? (
                <QMatrixTable M={M} rowPol={displayRowPol} colPol={displayColPol} wall0={wall0} wall1={wall1} />
              ) : (
                <div className="board-svg-wrap" style={{ margin: ".4rem 0 1.3rem" }}><QMatrixGraph M={M} rowPol={rowPol} colPol={colPol} /></div>
              )}
              {qview === "table" && (
                <p className="hint" style={{ margin: ".5rem 0 0" }}>
                  Each cell is what the game is worth to player 0 if this action pair is played
                  now and both play optimally afterwards:{" "}
                  <span className="mono">r + &gamma;&middot;V(next state)</span>. Row and column
                  headers show each player's own mix; <span className="support-swatch"></span>
                  highlighted cells are the pair both players actually use.
                </p>
              )}

              <div className="support-block">
                <div className="row">
                  <span className="dot" style={{ background: st.b === 0 ? "var(--p0)" : "var(--p1)" }}></span>
                  <span><b>Carrier</b> &mdash; {cs.map((i) => `${ACT[i]} ${fmtPct(carrierPol[i])}`).join(" / ")}</span>
                </div>
                <div className="row">
                  <span className="dot" style={{ background: st.b === 0 ? "var(--p1)" : "var(--p0)" }}></span>
                  <span><b>Defender</b> &mdash; {ds.map((i) => `${ACT[i]} ${fmtPct(defenderPol[i])}`).join(" / ")}</span>
                </div>
              </div>

              <div className="mix-tags">
                {support(rowPol).length > 1 && <span className="mix-tag p0">Player 0 uses a mixed policy</span>}
                {support(colPol).length > 1 && <span className="mix-tag p1">Player 1 uses a mixed policy</span>}
              </div>

              {cert.kind === "pure" && (
                <p className="hint" style={{ margin: ".6rem 0 0" }}>
                  {(cert.rowTies.length > 1 || cert.colTies.length > 1) ? (
                    <>
                      Security-tied alternatives: player 0 {cert.rowTies.map(i => ACT[i]).join("/")},
                      player 1 {cert.colTies.map(i => ACT[i]).join("/")}. These ties do not
                      specify probabilities. The selected policy first preserves the best
                      worst-case value, then prefers the higher average payoff against
                      opponent actions. This avoids weakly dominated choices among exact
                      ties; it does not claim a unique equilibrium. The selected pair{" "}
                      <span className="mono">{ACT[selectedRow]}/{ACT[selectedCol]}</span> leads to{" "}
                      <span className="mono">{describeOutcome(board, stateKey(st), selectedRow, selectedCol)}</span>.
                    </>
                  ) : (
                    <>
                      The optimal joint action (<span className="mono">{ACT[selectedRow]}</span> /{" "}
                      <span className="mono">{ACT[selectedCol]}</span>) leads to{" "}
                      <span className="mono">{describeOutcome(board, stateKey(st), selectedRow, selectedCol)}</span>.
                    </>
                  )}
                </p>
              )}

              <div className="move-bars">
                <div className="move-bars-title"><span className="dot" style={{ background: "var(--p0)" }}></span>Player 0 next move</div>
                {ACT.map((a, i) => {
                  const tied = cert.kind === "pure" && cert.rowTies.length > 1 && cert.rowTies.includes(i);
                  return (
                    <div className="move-bar-row" key={"p0-" + a}>
                      <span className="move-bar-label">{a}<WallMark status={wall0[a]} /></span>
                      <span className="move-bar-track">
                        <span className="move-bar-fill p0" style={{ width: `${(displayRowPol[i] * 100).toFixed(2)}%` }}></span>
                      </span>
                      <span className="move-bar-pct">
                        {(displayRowPol[i] * 100).toFixed(4)}%
                        {displayRowPol[i] > 0 && <span className="move-bar-frac"> (&asymp; {nearestNiceFraction(displayRowPol[i])})</span>}
                        {tied && <span className="move-bar-frac"> &mdash; same security value</span>}
                        {displayRowPol[i] > 0 && wall0[a] === "hold" && <span className="move-bar-frac"> &mdash; wall-clamped, holds at ({st.x0}, {st.y0})</span>}
                        {displayRowPol[i] > 0 && wall0[a] === "score" && <span className="move-bar-frac"> &mdash; scores here, ends the game</span>}
                      </span>
                    </div>
                  );
                })}
                <div className="move-bars-title" style={{ marginTop: ".8rem" }}><span className="dot" style={{ background: "var(--p1)" }}></span>Player 1 next move</div>
                {ACT.map((a, i) => {
                  const tied = cert.kind === "pure" && cert.colTies.length > 1 && cert.colTies.includes(i);
                  return (
                    <div className="move-bar-row" key={"p1-" + a}>
                      <span className="move-bar-label">{a}<WallMark status={wall1[a]} /></span>
                      <span className="move-bar-track">
                        <span className="move-bar-fill p1" style={{ width: `${(displayColPol[i] * 100).toFixed(2)}%` }}></span>
                      </span>
                      <span className="move-bar-pct">
                        {(displayColPol[i] * 100).toFixed(4)}%
                        {displayColPol[i] > 0 && <span className="move-bar-frac"> (&asymp; {nearestNiceFraction(displayColPol[i])})</span>}
                        {tied && <span className="move-bar-frac"> &mdash; same security value</span>}
                        {displayColPol[i] > 0 && wall1[a] === "hold" && <span className="move-bar-frac"> &mdash; wall-clamped, holds at ({st.x1}, {st.y1})</span>}
                        {displayColPol[i] > 0 && wall1[a] === "score" && <span className="move-bar-frac"> &mdash; scores here, ends the game</span>}
                      </span>
                    </div>
                  );
                })}
              </div>

              {cert.kind === "mixed" && (() => {
                const { eRow, eCol } = expectedValues(M, rowPol, colPol);
                const rowSup = support(rowPol), colSup = support(colPol);
                return (
                  <div className="indiff-block">
                    <p className="hint" style={{ margin: ".9rem 0 .4rem" }}>
                      <b>Why indifferent</b> &mdash; each action's expected value against the
                      opponent's actual mix, computed exactly (<span className="mono">M &middot; q</span>{" "}
                      for player 0, <span className="mono">&minus;(p &middot; M)</span> for player 1).
                      A support action (bold) must tie the best available value; everything else
                      must do strictly worse, or it wouldn't be an equilibrium.
                    </p>
                    <div className="indiff-row">
                      <span className="dot" style={{ background: "var(--p0)" }}></span>
                      Player 0:{" "}
                      {ACT.map((a, i) => (
                        <span key={a} className={"mono" + (rowSup.includes(i) ? " tied" : "")}>
                          {a}={eRow[i] >= 0 ? "+" : ""}{eRow[i].toFixed(6)}{" "}
                        </span>
                      ))}
                    </div>
                    <div className="indiff-row">
                      <span className="dot" style={{ background: "var(--p1)" }}></span>
                      Player 1:{" "}
                      {ACT.map((a, i) => (
                        <span key={a} className={"mono" + (colSup.includes(i) ? " tied" : "")}>
                          {a}={eCol[i] >= 0 ? "+" : ""}{eCol[i].toFixed(6)}{" "}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })()}

              <div className="legend-row">
                <span className="k"><span className="swatch" style={{ background: "var(--pitch)" }}></span>high for player 0</span>
                <span className="k"><span className="swatch" style={{ background: "var(--ember)" }}></span>low for player 0</span>
                <span className="k">dashed ring &mdash; a wall clamp, not a move (see <a href="positions.md">positions.md</a>)</span>
              </div>

              <RoundingDiagnostic V={V} rowPol={rowPol} colPol={colPol} cert={cert} rounding={rounding} />

              <div className="controls" style={{ marginTop: "1.2rem" }}>
                <label className="v-toggle">
                  <input type="checkbox" checked={showNeural} onChange={(e) => setShowNeural(e.target.checked)} />
                  Show neural cross-check (DQN / policy gradient)
                </label>
              </div>
              {showNeural && (
                <NeuralCrossCheck
                  neuralData={neuralData} neuralError={neuralError} board={currentBoard}
                  stKey={stateKey(st)} M={M} rowPol={rowPol} colPol={colPol}
                  wall0={wall0} wall1={wall1}
                />
              )}
            </div>
          </div>

          {(() => {
            const positionsKeys = Object.keys(PRESETS).filter(
              (n) => PRESETS[n].board === currentBoard && PRESETS[n].doc !== "edge"
            );
            const edgeDoc = BOARD_CASE_DOCS[currentBoard];
            const edgeKeys = edgeDoc
              ? Object.keys(PRESETS).filter((n) => PRESETS[n].board === currentBoard && PRESETS[n].doc === "edge")
              : [];
            return (
              <>
                {positionsKeys.length > 0 && (
                  <div className="presets">
                    <span className="hint" style={{ margin: "0 .3rem 0 0" }}>Jump to a documented case (<a href="positions.pdf">positions.pdf</a>) &mdash; each switches to that case's board:</span>
                    {positionsKeys.map((n) => (
                      <button key={n} className="preset-btn" onClick={() => applyPreset(n)}>{PRESETS[n].label}</button>
                    ))}
                  </div>
                )}
                {edgeKeys.length > 0 && (
                  <div className="presets">
                    <span className="hint" style={{ margin: "0 .3rem 0 0" }}>Jump to an edge case on the {edgeDoc.label} board (<a href={edgeDoc.pdf}>{edgeDoc.pdf}</a>):</span>
                    {edgeKeys.map((n) => (
                      <button key={n} className="preset-btn" onClick={() => applyPreset(n)}>{PRESETS[n].label}</button>
                    ))}
                  </div>
                )}
              </>
            );
          })()}

          <div className="stat-strip">
            <div><div className="s-k">States solved exactly</div><div className="s-v">{board.state_count.toLocaleString()}</div></div>
            <div><div className="s-k">No pure saddle</div><div className="s-v">{board.no_saddle_count} of {board.state_count}</div></div>
            <div><div className="s-k">Exact vs. iterative</div><div className="s-v">{board.exact_vs_iterative.toExponential(1)}</div></div>
          </div>

          <div className="panel" style={{ marginTop: "1.4rem" }}>
            <div className="eyebrow" style={{ margin: "0 0 .6rem" }}>Simulate</div>
            <p className="hint" style={{ margin: "0 0 .8rem" }}>
              Self-play a policy for each side against the other from the current
              position, sampling real <code>game.transitions()</code> outcomes (precomputed,
              not a second transition engine written for the browser) &mdash; a goal ends the
              game, 100 steps with none counts as a draw, same as the A10 rule.
              <b> Best response</b> re-solves the exact best pure reply to whatever the
              other side is actually playing at every state, straight from the loaded
              4&times;4 Q matrix &mdash; the same menu <a href="tournament.md">docs/tournament.md</a>{" "}
              scores Littman's Table 3 against. Compare <b>canonical</b> (random move
              order) against <b>canonical (deterministic)</b> the way a member's own
              site did. Uses the same <b>P0</b> / <b>P1</b> policies selected above the
              board (currently <b>{POLICY_LABELS[p0Type]}</b> / <b>{POLICY_LABELS[p1Type]}</b>).
            </p>
            <div className="controls">
              <div className="seg" role="group" aria-label="number of simulated games">
                {[500, 2000, 10000].map((n) => (
                  <button key={n} className={trials === n ? "active view" : ""} onClick={() => setTrials(n)}>
                    {n.toLocaleString()}
                  </button>
                ))}
              </div>
              <button className="iconbtn" onClick={runSimulation} disabled={simming}>
                {simming ? "Simulating…" : `Simulate ${trials.toLocaleString()} games`}
              </button>
            </div>
            {simResult && (
              <div style={{ marginTop: "1rem" }}>
                <div className="sim-bar">
                  <div className="sim-seg p0" style={{ width: `${(100 * simResult.p0) / simResult.trials}%` }} />
                  <div className="sim-seg draw" style={{ width: `${(100 * simResult.draw) / simResult.trials}%` }} />
                  <div className="sim-seg p1" style={{ width: `${(100 * simResult.p1) / simResult.trials}%` }} />
                </div>
                <div className="legend-row" style={{ marginTop: ".6rem" }}>
                  <span className="k"><span className="swatch" style={{ background: "var(--p0)" }}></span>
                    player 0 ({POLICY_LABELS[p0Type]}) won {fmtPct(simResult.p0 / simResult.trials)}</span>
                  <span className="k"><span className="swatch" style={{ background: "var(--rule-strong)" }}></span>
                    draw (100 steps) {fmtPct(simResult.draw / simResult.trials)}</span>
                  <span className="k"><span className="swatch" style={{ background: "var(--p1)" }}></span>
                    player 1 ({POLICY_LABELS[p1Type]}) won {fmtPct(simResult.p1 / simResult.trials)}</span>
                </div>
                <div className="sim-stats">
                  <div>
                    <div className="s-k">Average game length</div>
                    <div className="s-v">{simResult.avgSteps.toFixed(1)} steps</div>
                  </div>
                  <div>
                    <div className="s-k">Decisive rate</div>
                    <div className="s-v">{fmtPct(simResult.decisiveRate)}</div>
                  </div>
                  <div>
                    <div className="s-k">Average length (decisive games)</div>
                    <div className="s-v">
                      {simResult.decisiveAvgSteps === null ? "n/a" : `${simResult.decisiveAvgSteps.toFixed(1)} steps`}
                    </div>
                  </div>
                  <div>
                    <div className="s-k">Simulated mean return (player 0)</div>
                    <div className="s-v">{simResult.meanReturn >= 0 ? "+" : ""}{simResult.meanReturn.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="s-k">Certified V at this position</div>
                    <div className="s-v">{V >= 0 ? "+" : ""}{V.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="s-k">Gap (simulated &minus; certified)</div>
                    <div className="s-v">{(simResult.meanReturn - V) >= 0 ? "+" : ""}{(simResult.meanReturn - V).toFixed(3)}</div>
                  </div>
                </div>
                <p className="hint" style={{ margin: ".6rem 0 0" }}>
                  <b>Decisive rate</b> is the share of games that ended in a goal rather than
                  hitting the 100-step draw cap; <b>average length (decisive games)</b> excludes
                  draws from the length average above, since every draw runs the full 100 steps
                  and can otherwise drag the plain average toward 100 even when most games end
                  quickly.
                </p>
                <p className="hint" style={{ margin: ".6rem 0 0" }}>
                  The gap is exactly zero in expectation only when <b>both</b> sides play
                  Minimax (exact) &mdash; that's the same discounted return{" "}
                  <code>V(s) = val(E[R + &gamma;&middot;V(s')])</code> the solver itself certifies
                  (<a href="numerics.md">docs/numerics.md</a>). A non-minimax policy on either
                  side should show a gap favoring whoever deviated from equilibrium against a
                  fixed opponent, or a wide one both ways when both deviate.
                </p>
              </div>
            )}
          </div>

          <LittmanTable />
        </div>
      </section>
      <Footer board={currentBoard} />
    </>
  );
}

// The superscript next to an edge-clamped action: "hold" (no cell there, the
// player just stays put) and "score" (the carrier, in a goal row, pushing
// past *their own* attacking edge -- wallMask's own distinction, mirroring
// soccer_nash.game.SoccerGame._target's scoring condition) read as opposite
// things and must not share one icon or tooltip.
function WallMark({ status }) {
  if (status === "hold") {
    return <sup className="wall-mark" title="wall-clamped: no cell there, this holds in place">&#8862;</sup>;
  }
  if (status === "score") {
    return <sup className="wall-mark score" title="scores here: this ends the game, it does not hold in place">&#9873;</sup>;
  }
  return null;
}

// A pure saddle's full tied set, joined "U/D/L" -- not just the one
// certify() happened to check first. Single-action ties (the common case)
// render exactly as before; a genuine multi-way tie is the whole point of
// not tie-breaking the headline the way the rest of the page no longer does.
function ActionList({ ties, wall }) {
  return ties.map((idx, k) => (
    <span key={idx} className="mono">
      {k > 0 && "/"}{ACT[idx]}<WallMark status={wall[ACT[idx]]} />
    </span>
  ));
}

// Precompiled by scripts/explorer_data.py from soccer_nash.numerics
// .rounding_diagnostic: the *same* stage-game matrix, solved fresh after
// rounding it to 3/2/1 decimal places, not just a re-labelled version of the
// exact solve -- the "we definitely need some kind of approximation
// rounding" question from the project's own research meetings, made
// checkable at any position rather than only the fourteen documented cases
// (docs/positions.md, docs/numerics.md). `rounding` is `null` when none of
// the three precisions actually changes the equilibrium's classification or
// support relative to full precision (soccer_nash.numerics
// .is_rounding_artifact); that is true for the overwhelming majority of
// states, so most positions just report that rounding is safe here.
function policyCell(pol) {
  return support(pol).map((i) => `${ACT[i]} ${fmtPct(pol[i])}`).join(" / ");
}

function RoundingDiagnostic({ V, rowPol, colPol, cert, rounding }) {
  const fullRow = {
    label: "Full",
    pure: cert.kind === "pure",
    value: V,
    row: rowPol,
    col: colPol,
    artifact: false,
  };
  const rows = rounding
    ? [
      fullRow,
      ...rounding.map(([decimals, pure, value, row, col, artifact]) => ({
        label: `${decimals} decimal${decimals === 1 ? "" : "s"}`,
        pure,
        value,
        row,
        col,
        artifact,
      })),
    ]
    : null;

  return (
    <div style={{ marginTop: "1.4rem" }}>
      <div className="eyebrow" style={{ margin: "0 0 .5rem" }}>Rounding diagnostic</div>
      {rows ? (
        <>
          <p className="hint" style={{ margin: "0 0 .6rem" }}>
            This matrix, solved again after rounding it to each precision (not just
            checking whether the classification flips). <b>Bold</b> rows are genuine
            artifacts &mdash; the pure/mixed classification or which actions carry
            weight actually changes; a shifted percentage split within the same
            support does not count.
          </p>
          <table className="rounding">
            <thead>
              <tr><th>Precision</th><th>Classification</th><th>Value</th><th>Player 0</th><th>Player 1</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label} className={r.artifact ? "artifact" : ""}>
                  <td>{r.label}</td>
                  <td>{r.pure ? "pure" : "mixed"}</td>
                  <td>{r.value >= 0 ? "+" : ""}{r.value.toFixed(4)}</td>
                  <td>{policyCell(r.row)}</td>
                  <td>{policyCell(r.col)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <p className="hint" style={{ margin: 0 }}>
          Rounding this matrix to 3, 2, or 1 decimal place changes neither the
          pure/mixed classification nor which actions carry weight for either
          player &mdash; the equilibrium shown above is not a rounding artifact.
        </p>
      )}
    </div>
  );
}

// One representative DQN run (fitted-Q, "from zero" -- pure TD bootstrap
// from a random init, the honest baseline docs/neural.md reports first, not
// "fit to exact" which is trained by regressing directly onto the exact
// answer and so isn't an independent check of anything) and one PG run
// (self-play REINFORCE), both seed 0, against the exact solve at whatever
// state is currently selected -- not just the handful of states already
// written up as case studies. docs/neural.md's own multi-seed study (with
// error bars) is the actual research claim; this is a live, single-run
// sanity check anyone can point at any position on the board.
function NeuralCrossCheck({ neuralData, neuralError, board, stKey, M, rowPol, colPol, wall0, wall1 }) {
  if (neuralError) {
    return <p className="hint" style={{ color: "var(--ember)" }}>Couldn't load the neural cross-check data ({neuralError}).</p>;
  }
  if (!neuralData) {
    return <p className="hint">Loading DQN / policy-gradient predictions&hellip;</p>;
  }
  const rec = neuralData.boards[board]?.[stKey];
  if (!rec) {
    return <p className="hint">No neural data for this state (terminal or off this board).</p>;
  }
  const [Qd, pDqn, qDqn, pPg, qPg] = rec;
  const maxErr = Math.max(...M.flatMap((row, i) => row.map((v, j) => Math.abs(v - Qd[i][j]))));
  const { dqn_epochs, pg_iterations, seed } = neuralData.meta;

  return (
    <div style={{ marginTop: "1.4rem" }}>
      <div className="eyebrow" style={{ margin: "0 0 .5rem" }}>Neural cross-check</div>
      <p className="hint" style={{ margin: "0 0 .6rem" }}>
        One representative training run (seed {seed}), not the multi-seed study in{" "}
        <a href="neural.md">docs/neural.md</a> (see that page and{" "}
        <span className="mono">experiments/nash_dqn_seeds.csv</span> /{" "}
        <span className="mono">experiments/policy_gradient_seeds.csv</span> for error bars).
        DQN is fitted-Q trained from a random init for {dqn_epochs} epochs (the honest
        "from zero" baseline, not one fit to the exact answer). PG is self-play REINFORCE
        for {pg_iterations} iterations, and is genuinely on-policy: a state its own rollouts
        rarely reach from kickoff never gets corrected there, so a mismatch below can mean
        that rather than anything wrong with the exact solve.
      </p>
      <h3 style={{ margin: "0 0 .4rem" }}>Exact vs. DQN, cell by cell</h3>
      <p className="support-line" style={{ marginBottom: ".6rem" }}>
        max |Q<sub>DQN</sub> &minus; Q<sub>exact</sub>| at this state ={" "}
        <b>{maxErr.toFixed(4)}</b>
      </p>
      <table className="rounding">
        <thead>
          <tr><th>P0 / P1</th><th>Exact</th><th>DQN (from zero)</th><th>|diff|</th></tr>
        </thead>
        <tbody>
          {M.flatMap((row, i) => row.map((exact, j) => {
            const dqn = Qd[i][j];
            return (
              <tr key={`${i}-${j}`}>
                <td>{ACT[i]}/{ACT[j]}</td>
                <td>{exact.toFixed(4)}</td>
                <td>{dqn.toFixed(4)}</td>
                <td>{Math.abs(exact - dqn).toFixed(4)}</td>
              </tr>
            );
          }))}
        </tbody>
      </table>
      <p className="hint" style={{ margin: ".6rem 0 0" }}>
        Policy gradient trains a policy net directly, not a Q matrix, so it has
        no per-cell payoff prediction to compare here &mdash; see its action
        probabilities in the policy comparison below instead.
      </p>
      <h3 style={{ margin: "1rem 0 .4rem" }}>DQN's predicted Q matrix</h3>
      <QMatrixTable M={Qd} rowPol={pDqn} colPol={qDqn} wall0={wall0} wall1={wall1} />
      <h3 style={{ margin: "1rem 0 .4rem" }}>Policy comparison</h3>
      <table className="rounding">
        <thead>
          <tr><th>Source</th><th>Player 0</th><th>Player 1</th></tr>
        </thead>
        <tbody>
          <tr><td>Exact</td><td>{policyCell(rowPol)}</td><td>{policyCell(colPol)}</td></tr>
          <tr><td>DQN</td><td>{policyCell(pDqn)}</td><td>{policyCell(qDqn)}</td></tr>
          <tr><td>PG (self-play)</td><td>{policyCell(pPg)}</td><td>{policyCell(qPg)}</td></tr>
        </tbody>
      </table>
    </div>
  );
}

// Littman's own published Table 3 (1994) -- MR/MM by minimax-Q, QR/QQ by
// ordinary Q-learning, each scored (his numbers, not this project's) against
// a random opponent, a hand-built opponent, and a challenger trained
// specifically to beat it. docs/tournament.md reproduces the *structure*
// exactly with this project's solver instead of Littman's learned policies;
// this is the historical table itself, kept verbatim for direct comparison.
const LITTMAN_TABLE3 = [
  { policy: "MR (minimax)", random: "99.3%", handBuilt: "48.1%", challenger: "35.0%" },
  { policy: "MM (minimax)", random: "99.3%", handBuilt: "53.7%", challenger: "37.5%" },
  { policy: "QR (Q-learning)", random: "99.4%", handBuilt: "26.1%", challenger: "0.0%" },
  { policy: "QQ (Q-learning)", random: "99.5%", handBuilt: "76.3%", challenger: "0.0%" },
];

function LittmanTable() {
  return (
    <div className="panel" style={{ marginTop: "1.4rem" }}>
      <div className="eyebrow" style={{ margin: "0 0 .6rem" }}>Littman (1994), Table 3</div>
      <p className="hint" style={{ margin: "0 0 .8rem" }}>
        Littman's own published result, kept verbatim (percentage of games won, his
        learned policies, his board) &mdash; not this project's numbers. <b>MR / MM</b>{" "}
        are minimax-Q; <b>QR / QQ</b> are ordinary Q-learning. Every deterministic
        policy (QR, QQ) collapses to <b>0.0%</b> against a challenger trained to beat
        it; only the minimax policies stay ahead. <a href="tournament.md">docs/tournament.md</a>{" "}
        reproduces this structure exactly with this project's own exact solver instead
        of Littman's learned policies &mdash; the <b>Simulate</b> panel above lets you
        recreate the same shape live: set P0 to <b>Minimax</b> and P1 to{" "}
        <b>Best response</b> to see the same "every deterministic offense has a perfect
        defense" result the Q-learning rows show here.
      </p>
      <table className="qmatrix" style={{ width: "100%" }}>
        <thead>
          <tr><th>policy</th><th>vs. random</th><th>vs. hand-built</th><th>vs. its challenger</th></tr>
        </thead>
        <tbody>
          {LITTMAN_TABLE3.map((row) => (
            <tr key={row.policy}>
              <th style={{ textAlign: "left" }}>{row.policy}</th>
              <td className="cell">{row.random}</td>
              <td className="cell">{row.handBuilt}</td>
              <td className="cell">{row.challenger}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
