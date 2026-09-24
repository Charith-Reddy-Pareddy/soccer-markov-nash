import { useEffect, useMemo, useState } from "react";
import Nav from "../Nav.jsx";
import Footer from "../Footer.jsx";
import Board from "./Board.jsx";
import QMatrixTable from "./QMatrixTable.jsx";
import QMatrixGraph from "./QMatrixGraph.jsx";
import { ACT, certify, fmtPct, kickoffState, POLICY_LABELS, POLICY_TYPES, simulateGames, stateKey, stepPolicy, support } from "./helpers.js";
import "./explorer.css";

const BOARD_ORDER = ["canonical", "canonical_det", "tackle", "territory", "slip"];

// case number -> {board, state}, coordinates as printed in docs/positions.md
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
  const [autoPlaying, setAutoPlaying] = useState(false);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/explorer.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((json) => {
        setData(json);
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

  // One Step/Play tick: resolve both players' chosen policy at the current
  // state, sample an action pair, and either move to the real next state
  // (pushed onto playHistory) or, if it's a goal, stop without one (there is
  // no legal `st` for a terminal state) -- the same stepPolicy() resolution
  // simulateGames' own inner loop uses, just watched one step at a time.
  function stepOnce() {
    if (playDone) return;
    const { a0, a1, next } = stepPolicy(board, stateKey(st), p0Type, p1Type);
    setLastMove({ a0, a1 });
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
  const carrierPol = st.b === 0 ? rowPol : colPol;
  const defenderPol = st.b === 0 ? colPol : rowPol;
  // Fixed for every state: rows = player 0's actions, columns = player 1's,
  // never reoriented by who has the ball. An earlier version reoriented so
  // rows were always the carrier's actions -- per the project's own research
  // meetings, that made the matrix silently transpose between states, which
  // is exactly what was confusing a reader tracking player 0/player 1
  // physically. Which player is carrying is shown separately below.
  const M = Q;
  const cert = certify(M);
  const cs = support(carrierPol), ds = support(defenderPol);

  return (
    <>
      <Nav current="explorer" />
      <section>
        <div className="wrap">
          <div className="eyebrow"><span className="badge">Live</span>Board explorer</div>
          <h1>Place both players anywhere. Watch the equilibrium update live.</h1>
          <p className="lede">All 9,100 legal positions across five boards were solved
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
                <Board board={board} state={st} activePlayer={activePlayer} onCellClick={onCellClick} heatmap={heatmap} />
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
                <button className="iconbtn" onClick={stepOnce} disabled={playDone}>Step</button>
                <button className="iconbtn" onClick={() => setAutoPlaying((p) => !p)} disabled={playDone}>
                  {autoPlaying ? "Pause" : "Play"}
                </button>
                <button className="iconbtn" onClick={restartPlay}>Restart game</button>
              </div>
              <p className="hint mono">
                Step {playHistory.length - 1} &middot; discounted return so far{" "}
                {(playDone ? (winner === 0 ? 1 : -1) * data.gamma ** (playHistory.length - 1) : 0).toFixed(6)}
                {" "}&middot; last move{" "}
                {lastMove ? `P0 ${ACT[lastMove.a0]} / P1 ${ACT[lastMove.a1]}` : "none yet"}
                {playDone && <>
                  {" "}&mdash;{" "}
                  <b style={{ color: winner === 0 ? "var(--p0)" : "var(--p1)" }}>player {winner} scored</b>
                </>}
              </p>
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
                  ? `Pure equilibrium — saddle at player 0: ${ACT[cert.i]}, player 1: ${ACT[cert.j]}`
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
                <QMatrixTable M={M} rowPol={rowPol} colPol={colPol} />
              ) : (
                <div className="board-svg-wrap" style={{ margin: ".4rem 0 1.3rem" }}><QMatrixGraph M={M} /></div>
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
              <div className="legend-row">
                <span className="k"><span className="swatch" style={{ background: "var(--pitch)" }}></span>high for player 0</span>
                <span className="k"><span className="swatch" style={{ background: "var(--ember)" }}></span>low for player 0</span>
                <span className="k">dashed ring &mdash; a wall clamp, not a move (see <a href="positions.md">positions.md</a>)</span>
              </div>

              <RoundingDiagnostic V={V} rowPol={rowPol} colPol={colPol} cert={cert} rounding={rounding} />
            </div>
          </div>

          <div className="presets">
            <span className="hint" style={{ margin: "0 .3rem 0 0" }}>Jump to a documented case (<a href="positions.pdf">positions.pdf</a>) &mdash; each switches to that case's board:</span>
            {Object.keys(PRESETS).map((n) => (
              <button key={n} className="preset-btn" onClick={() => applyPreset(n)}>{PRESETS[n].label}</button>
            ))}
          </div>

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
      <Footer />
    </>
  );
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
