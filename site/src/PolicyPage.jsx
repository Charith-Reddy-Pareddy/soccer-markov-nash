import { useState } from "react";
import Nav from "./Nav.jsx";
import Footer from "./Footer.jsx";
import PolicyOutputs from "./PolicyOutputs.jsx";
import pgPolicies from "./pgPolicies.json";
import results from "./pgResults.json";
import "./landing.css";

const f2 = (v) => v.toFixed(2);
const pct = (v) => `${Math.round(v * 100)}%`;
const name = (l) => `${l.label}, ${l.training}`;

// Exploitability of every learner: bar = mean, dots = individual seeds.
function ExploitBars({ rows }) {
  const W = 660, left = 200, right = 48, rowH = 26;
  const H = rowH * rows.length + 34;
  const x = (v) => left + v * (W - left - right);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="exploitability of each learner" className="pg-bars">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={x(g)} y1="6" x2={x(g)} y2={H - 22} stroke="var(--rule)" />
          <text x={x(g)} y={H - 8} textAnchor="middle" fontSize="11" fill="var(--ink-faint)">{g}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = 10 + i * rowH;
        return (
          <g key={r.name}>
            <text x={left - 8} y={y + 13} textAnchor="end" fontSize="12" fill="var(--ink)">{r.name}</text>
            <rect x={left} y={y + 3} width={Math.max(x(r.mean) - left, 0)} height="14" fill="var(--pitch)" opacity=".85" />
            {r.runs.map((v, k) => <circle key={k} cx={x(v)} cy={y + 10} r="3" fill="var(--ink)" opacity=".7" />)}
            <text x={W - right + 10} y={y + 14} fontSize="12" fontWeight="700" fill="var(--ink)">{f2(r.mean)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// Share of the 1,000 games the player wins (scores while the opponent does not),
// with the ties (no goal in 100 steps) and losses beside it.
function WinCell({ t }) {
  return (
    <td>
      <b>{pct(t[0])}</b>
      <span className="sub">tie {pct(t[1])} &middot; loss {pct(t[2])}</span>
    </td>
  );
}

export default function PolicyPage() {
  const [which, setWhich] = useState("deterministic");
  const { longer, fp_br: fpBr } = results;
  const board = which === "random" ? results.random : results;
  const { learners, exact } = board;
  const range = (ls) => {
    const m = ls.map((l) => l.exploitability.mean);
    return [Math.min(...m), Math.max(...m)];
  };
  const [lo, hi] = range(results.learners);
  const [rlo, rhi] = range(results.random.learners);
  const winsVsNash = learners.map((l) => l.vs_nash[0]);
  const bars = [
    { name: "Exact solver", mean: 0, runs: [0] },
    ...learners.map((l) => ({ name: name(l), mean: l.exploitability.mean, runs: l.exploitability.runs })),
  ];
  const best = Math.min(...longer.flatMap((l) => l.runs));
  const rounds = [...new Set(fpBr.flatMap((r) => r.checkpoints.map((c) => c.round)))].sort((x, y) => x - y);

  return (
    <>
      <Nav current="policy" />

      <section className="hero">
        <div className="hero-grid">
          <div>
            <div className="kicker">Policy gradient &middot; soccer game</div>
            <h1>Policy gradient on the soccer game</h1>
            <p className="lede">REINFORCE, A2C and PPO are trained on the discounted 100-step game,
              with the number of steps left as an input, and scored against the exact solution of
              that same game. Each is trained two ways: plain self-play, and fictitious play, where
              every player keeps improving by policy gradient against the average of its opponent&rsquo;s
              past policies, so no game is ever solved explicitly.</p>
            <div className="cta-row">
              <a className="btn btn-primary" href="policy_gradient.pdf">Read the PDF &rarr;</a>
              <a className="btn btn-outline" href="policy_gradient.md">Full write-up &rarr;</a>
            </div>
          </div>
          <div className="hero-card">
            <div className="stat-list">
              <div><b>6</b> learners: REINFORCE, A2C and PPO, each by self-play and by fictitious play</div>
              <div>Exploitability <b>{f2(lo)} to {f2(hi)}</b> (the exact solution is 0)</div>
              <div>On the random move-order board: <b>{f2(rlo)} to {f2(rhi)}</b></div>
              <div>Best single run at four times the training: <b>{f2(best)}</b></div>
              <div>Scored over <b>1,000</b> repeated games per opponent</div>
            </div>
          </div>
        </div>
      </section>

      <section id="setup">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">01</span>What was done</div>
          <h2>The setup</h2>
          <div className="tbl-wrap">
            <table className="text-table">
              <thead><tr><th>Topic</th><th>What was done</th><th>Why</th></tr></thead>
              <tbody>
                <tr><td>Environment</td><td>The deterministic A10 board is the main environment; the random move-order board is kept as an earlier comparison.</td><td>The deterministic board has no mixed-equilibrium stages, so the comparison with the exact solution is clean.</td></tr>
                <tr><td>Objective</td><td>Discount 0.9 over 100 steps, a tie at the end, and the remaining step count as a network input.</td><td>Policy gradient can only approximate a finite-horizon discounted reward.</td></tr>
                <tr><td>Fictitious play</td><td>Each player is trained by policy gradient against the average of the opponent&rsquo;s past policies.</td><td>Policy gradient does the solving; no game is solved explicitly.</td></tr>
                <tr><td>Win rate</td><td>1,000 repeated games from the kickoff against a random player, the exact Nash policy and the exact best response; wins, ties and losses are all counted.</td><td>Repeated play and counting wins.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="band-tint" id="learners">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">02</span>The learners</div>
          <h2>Three algorithms, two ways of training</h2>
          <div className="tbl-wrap">
            <table className="text-table">
              <thead><tr><th>Algorithm</th><th>Update</th></tr></thead>
              <tbody>
                <tr><td>REINFORCE</td><td>plain discounted returns, no baseline</td></tr>
                <tr><td>A2C</td><td>10-step bootstrapped advantage from a learned critic</td></tr>
                <tr><td>PPO</td><td>GAE (&lambda; = 0.95), clipped ratio 0.2, 4 epochs</td></tr>
              </tbody>
            </table>
          </div>
          <p><b>Self-play</b> trains both current networks against each other. <b>Fictitious play</b> keeps
            frozen snapshots of each player&rsquo;s network; the opponent in each episode is one snapshot drawn
            uniformly, and the policy a player reports is the average of its own snapshots. Every learner
            uses learning rate 10<sup>&minus;3</sup>, 64 episodes per batch, 2,000 iterations, an entropy
            bonus of 0.01 and a 64&times;64 network; none was tuned.</p>
          <p>Scoring: <b>exploitability</b> is the discounted value both exact best responses earn from the
            kickoff, summed, and is zero exactly at an equilibrium. The <b>mirror gap</b> measures how far
            the two players&rsquo; policies are from being mirror images of each other.</p>
        </div>
      </section>

      <section id="results">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">03</span>Results</div>
          <h2>Scored against the exact solution</h2>
          <div className="state-pick" role="group" aria-label="board">
            <button className={which === "deterministic" ? "on" : ""} onClick={() => setWhich("deterministic")}>Deterministic board (A10)</button>
            <button className={which === "random" ? "on" : ""} onClick={() => setWhich("random")}>Random move-order board</button>
          </div>
          <p className="state-note">
            {which === "random"
              ? "Players' moves are applied in a random order each step, so the exact solution mixes at 94 stage games and the dynamics are stochastic."
              : "Players move simultaneously and the carrier wins every contested square, so the exact solution is pure everywhere."}
            {" "}Each cell shows the share of 1,000 games the player wins (scores while the opponent does not); the ties
            (no goal in 100 steps) and losses are beneath it.
          </p>
          <div className="tbl-wrap">
            <table>
              <thead>
                <tr><th>Learner</th><th>Training</th><th>Exploitability</th><th>Wins vs. random</th><th>Wins vs. exact Nash</th><th>Wins vs. best response</th><th>Mirror gap</th></tr>
              </thead>
              <tbody>
                <tr><td>Exact solver</td><td>&mdash;</td><td className="hi">0</td><WinCell t={exact.vs_random} /><WinCell t={exact.vs_nash} /><WinCell t={exact.vs_best_response} /><td>0</td></tr>
                {learners.map((l) => (
                  <tr key={name(l)}>
                    <td>{l.label}</td><td>{l.training}</td>
                    <td>{f2(l.exploitability.mean)} &plusmn; {f2(l.exploitability.sd)}</td>
                    <WinCell t={l.vs_random} /><WinCell t={l.vs_nash} /><WinCell t={l.vs_best_response} />
                    <td>{f2(l.mirror_gap)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ExploitBars rows={bars} />
          <p className="fig-cap">Exploitability by learner: bar = mean over {learners[0].seeds} seeds, dots = individual seeds.</p>
          {which === "random" ? (
            <p>Exploitability is {f2(rlo)} to {f2(rhi)}, lower than on the deterministic board ({f2(lo)} to {f2(hi)}),
              though the two boards are scored on their own scales. The learners win {pct(Math.min(...winsVsNash))} to {pct(Math.max(...winsVsNash))} of
              games against the exact Nash policy; the exact solver wins {pct(exact.vs_nash[0])}, helped by starting with the ball.
              The two players&rsquo; policies are not mirror images (mirror gap {f2(Math.min(...learners.map((l) => l.mirror_gap)))} to {f2(Math.max(...learners.map((l) => l.mirror_gap)))}).</p>
          ) : (
            <p>Fictitious-play training wins more often against a random player; self-play training ties
              the exact equilibrium more often. No learner wins a game against the exact Nash policy, because the
              exact solution ties itself every game. The two players&rsquo; policies are not mirror images
              (mirror gap {f2(Math.min(...learners.map((l) => l.mirror_gap)))} to {f2(Math.max(...learners.map((l) => l.mirror_gap)))},
              against 0 for the exact solution). These are {learners[0].seeds} seeds at one untuned budget.</p>
          )}
        </div>
      </section>

      <section className="band-tint" id="probabilities">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">04</span>Action probabilities</div>
          <h2>What the trained policies output</h2>
          <p className="lede">The probability each trained network gives to every move, at four fixed
            positions, next to the exact solver&rsquo;s move. The networks are seed 0 of the runs above.</p>
          <PolicyOutputs data={pgPolicies} />
        </div>
      </section>

      <section id="more">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">05</span>More training</div>
          <h2>Self-play at four times the training</h2>
          <p className="lede">The same self-play learners with 8,000 iterations instead of 2,000
            (two seeds each; every other setting unchanged).</p>
          <div className="tbl-wrap">
            <table>
              <thead>
                <tr><th>Learner</th><th>Exploitability at 2,000 iterations (mean)</th><th>At 8,000 iterations (each seed)</th><th>Wins vs. random</th><th>Wins vs. exact Nash</th></tr>
              </thead>
              <tbody>
                {longer.map((l) => (
                  <tr key={l.label}>
                    <td>{l.label}, {l.training}</td><td>{f2(l.short_mean)}</td>
                    <td>{l.runs.map(f2).join(" and ")}</td><WinCell t={l.vs_random} /><WinCell t={l.vs_nash} />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>With two seeds per learner this is a trend, not a result; the best single run is {f2(best)}.</p>
        </div>
      </section>

      <section className="band-tint" id="best-response">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">06</span>Fictitious play with best responses</div>
          <h2>A stricter fictitious play</h2>
          <p className="lede">In each round every player runs 100 (or 300) policy-gradient iterations to
            approximate a best response to the average of the opponent&rsquo;s earlier best responses, then adds it
            to its history. Exploitability after each checkpoint round (one seed per run).</p>
          <div className="tbl-wrap">
            <table>
              <thead>
                <tr><th>Run</th>{rounds.map((k) => <th key={k}>Round {k}</th>)}</tr>
              </thead>
              <tbody>
                {fpBr.map((r) => (
                  <tr key={`${r.label}-${r.best_response_iterations}`}>
                    <td>{r.label}, {r.best_response_iterations} iterations per best response</td>
                    {rounds.map((k) => {
                      const c = r.checkpoints.find((x) => x.round === k);
                      return <td key={k}>{c ? f2(c.exploitability) : ""}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>In every run the exploitability at the last checkpoint is higher than at the first.</p>
          <div className="cta-row">
            <a className="btn btn-primary" href="policy_gradient.pdf">Read the PDF &rarr;</a>
            <a className="btn btn-outline" href="https://github.com/Charith-Reddy-Pareddy/soccer-markov-nash">View the code on GitHub &rarr;</a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
