import Nav from "./Nav.jsx";
import Footer from "./Footer.jsx";
import "./landing.css";

export default function Landing() {
  return (
    <>
      <Nav current="switch" />

      <a id="top"></a>
      <section className="hero">
        <svg className="pitch-lines" viewBox="0 0 1080 620" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <rect x="40" y="40" width="1000" height="540" fill="none" stroke="var(--pitch-line)" strokeWidth="2"/>
          <line x1="540" y1="40" x2="540" y2="580" stroke="var(--pitch-line)" strokeWidth="2"/>
          <circle cx="540" cy="310" r="88" fill="none" stroke="var(--pitch-line)" strokeWidth="2"/>
          <circle cx="540" cy="310" r="3.5" fill="var(--pitch-line)"/>
          <rect x="40" y="180" width="70" height="260" fill="none" stroke="var(--pitch-line)" strokeWidth="2"/>
          <rect x="970" y="180" width="70" height="260" fill="none" stroke="var(--pitch-line)" strokeWidth="2"/>
        </svg>
        <div className="hero-grid">
          <div>
            <div className="kicker">Multi-agent reinforcement learning &middot; game theory</div>
            <h1>When does soccer need mixed strategies?</h1>
            <p className="lede">Two players fight over a ball on a grid. Almost always, the
            smart move is obvious &mdash; head for the goal, or stand in the way. But at
            a precise, identifiable set of moments, the only rational move is to
            <em> randomize on purpose</em>: to genuinely not know, until the last
            instant, which way you're going to go. This project pins down exactly
            when that switch flips &mdash; and proves it, exactly, without training a
            single neural network.</p>
            <div className="cta-row">
              <a className="btn btn-primary" href="report.pdf">Read the full report &rarr;</a>
              <a className="btn btn-outline" href="https://github.com/Charith-Reddy-Pareddy/soccer-markov-nash">View the code on GitHub &rarr;</a>
            </div>
          </div>
          <div className="hero-card">
            <img src="figures/png/kickoff.png" alt="A soccer Markov game board at kickoff: player 0 in blue carries the ball toward the right-hand goal, player 1 in green defends the left." />
            <div className="fig-cap">Kickoff. <b>Blue</b> carries toward the right goal;
            <b> green</b> defends the left. Every result on this page comes from
            solving games like this one exactly &mdash; to machine precision, board by
            board, no simulation and no learning curve.</div>
          </div>
        </div>
      </section>

      <section id="setup">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">01</span>The game, briefly</div>
          <h2>A grid, a ball, and one rule that makes it a real contest</h2>
          <p>Two players share a small grid. One carries the ball and is trying to
          reach the opponent's goal; the other is trying to stop them. Each turn,
          both players choose a move <em>simultaneously</em> &mdash; neither sees the
          other's choice first &mdash; and the game plays out from there, discounted so
          that a goal scored sooner is worth more than one scored later. It is
          <strong> zero-sum</strong>: every bit of advantage one player gains is
          exactly what the other loses. That is what makes it a clean laboratory for
          a much bigger question &mdash; not "how do you play soccer," but "when,
          exactly, is confident and predictable play the right call, and when does
          the situation itself force you to bluff?"</p>
          <p>To keep the vocabulary unambiguous throughout this page (per this
          project's own convention &mdash; earlier drafts used names like "advance" or
          "cover the lane" and they only added confusion), every move is just one of
          four directions:</p>
          <div className="compass-row">
            <svg className="compass" width="132" height="132" viewBox="0 0 132 132" aria-hidden="true">
              <circle cx="66" cy="66" r="60" fill="none" stroke="var(--rule-strong)" strokeWidth="1.4" strokeDasharray="2 4"/>
              <circle cx="66" cy="66" r="15" fill="none" stroke="var(--ink-faint)" strokeWidth="2"/>
              <g stroke="var(--pitch)" strokeWidth="3" strokeLinecap="round" markerEnd="url(#cArrow)">
                <line x1="66" y1="45" x2="66" y2="16"/>
                <line x1="66" y1="87" x2="66" y2="116"/>
                <line x1="45" y1="66" x2="16" y2="66"/>
                <line x1="87" y1="66" x2="116" y2="66"/>
              </g>
              <defs>
                <marker id="cArrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="4" markerHeight="4" orient="auto">
                  <path d="M0 0.5 L8 4 L0 7.5 z" fill="var(--pitch)"/>
                </marker>
              </defs>
              <text x="66" y="10" textAnchor="middle" fontFamily="IBM Plex Mono, monospace" fontSize="12" fontWeight="600" fill="var(--ink)">U</text>
              <text x="66" y="128" textAnchor="middle" fontFamily="IBM Plex Mono, monospace" fontSize="12" fontWeight="600" fill="var(--ink)">D</text>
              <text x="8" y="70" textAnchor="middle" fontFamily="IBM Plex Mono, monospace" fontSize="12" fontWeight="600" fill="var(--ink)">L</text>
              <text x="124" y="70" textAnchor="middle" fontFamily="IBM Plex Mono, monospace" fontSize="12" fontWeight="600" fill="var(--ink)">R</text>
            </svg>
            <div className="legend">
              <div><span className="k">U / D / L / R</span> &mdash; up, down, left, right, one cell.</div>
              <div>That's the entire action set. Every diagram on this page draws
              moves as arrows in exactly these four directions &mdash; no hidden
              vocabulary.</div>
            </div>
          </div>
        </div>
      </section>

      <section className="band-tint" id="question">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">02</span>Where this starts</div>
          <h2>A 30-year-old open question, made checkable in one glance</h2>
          <p>Michael Littman's 1994 soccer game already proved that some situations
          in a game like this force a coin flip &mdash; his own example is a player
          pinned against its own goal with a defender right beside it, where any
          fixed move can be blocked forever. What Littman's result doesn't say is
          <em> which</em> situations, in general, need that coin flip and which
          don't. That's the question a research group brought to this project: not
          "can this happen," but "exactly when does it happen, and can you tell
          without running an expensive solver on every single situation?"</p>
          <p>The check turns out to be almost embarrassingly simple. Look at one
          matchup at a time. Does either player have a single move that is at least
          as good as anything else, no matter what the other player does? If yes,
          that move is the obvious, unbeatable answer &mdash; play it. If <em>every</em>
          move can be beaten by some response, the two best replies chase each
          other in a circle. That circle is exactly the shape of rock&ndash;paper&ndash;scissors,
          and the only sane move inside it is to genuinely randomize.</p>
          <figure className="breakout">
            <img src="figures/png/rps_vs_soccer.png" alt="Rock-paper-scissors payoff matrix next to a soccer stage game payoff matrix, both showing a cycle of best replies with no pure winning move." />
            <div className="fig-cap"><b>The same shape, twice.</b> Left: rock&ndash;paper&ndash;scissors
            &mdash; the blue bar marks each row's best reply, the green bar each
            column's; they never land in the same cell. Right: an actual soccer
            matchup from this project, at the moment a carrier is pinned against
            its own goal with a defender beside it &mdash; the identical cycle, no
            training required to see it, just checking every cell once.</div>
          </figure>
        </div>
      </section>

      <section id="switch">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">03</span>The main finding</div>
          <h2>One number on the board flips the entire game from predictable to unpredictable</h2>
          <p>Widen the goal from a single defendable cell to two or more cells, and
          every other setting held fixed, the game's character changes completely.
          With a one-cell goal, a defender can always plant itself in the only spot
          that matters &mdash; there's nothing to guess about, so <strong>every single
          situation on the board has an obvious, deterministic best move for both
          players.</strong> The instant the goal is wide enough that the defender
          cannot cover every scoring cell at once, the carrier has a genuine choice
          of lanes and the defender has to guess which one &mdash; and guessing games
          are exactly where mixed strategies become mandatory, not optional.</p>
          <div className="stat-row">
            <div className="stat-card">
              <div className="stat-k">Single-cell goal</div>
              <div className="stat-v">0<span className="unit">of 2,380 situations</span></div>
              <p>Every reachable matchup on the board has a pure, deterministic
              best move for both players &mdash; verified by machine, not assumed.</p>
            </div>
            <div className="stat-card hit">
              <div className="stat-k">Goal &ge; 2 cells wide</div>
              <div className="stat-v">94<span className="unit">situations force a guess</span></div>
              <p>Mixed strategies become provably necessary &mdash; and <strong>68 of
              those 94</strong> reduce to the same simple pattern: the carrier
              picking a lane, the defender guessing it.</p>
            </div>
          </div>
          <p>This switch holds up across board sizes, board shapes, and where
          exactly the goal sits &mdash; it is not a quirk of one particular layout. And
          it survives a much stronger test: it's exactly the shape needed for a fast
          shortcut. Rather than running an expensive general-purpose solver on
          <em> every</em> situation, a solver can check the simple pure-move test
          first and only fall back to the harder mixed-strategy math on the rare
          situations that actually need it &mdash; skipping that expensive step
          entirely on the overwhelming majority of the board.</p>
          <figure className="breakout">
            <img src="figures/png/mechanism.png" alt="A payoff matrix for one mixed soccer matchup, with best-reply markers cycling around the grid so that no single cell is a stable best response for both players." />
            <div className="fig-cap"><b>No safe cell.</b> A payoff grid from one of the
            94 must-guess situations: the blue marker traces player one's best row
            for every column, the green marker traces player two's best column for
            every row. They never overlap &mdash; there is no cell where both players
            are simultaneously doing their best, which is the exact, checkable
            signature of a forced mix.</div>
          </figure>
        </div>
      </section>

      <section className="band-tint" id="seeing">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">04</span>Seeing it happen</div>
          <h2>Six situations, worked through in detail</h2>
          <p>Numbers and heatmaps only go so far, so here are six actual situations,
          solved exactly and drawn directly. Reading the diagrams: the <strong
          style={{ color: "var(--p0)" }}>blue</strong> disc is player one, the <strong
          style={{ color: "var(--p1)" }}>green</strong> disc is player two, the small
          orange dot is the ball. An arrow's <em>thickness</em> and the percentage
          beside it are the equilibrium's exact odds of playing that move &mdash; one
          fat, lone arrow means "always do this"; several arrows sharing the weight
          means the player must genuinely gamble, in precisely those proportions,
          or hand the opponent a predictable pattern to exploit.</p>
          <figure className="breakout">
            <img src="figures/png/showcase.png" alt="Six soccer board diagrams, each showing both players' equilibrium move probabilities as weighted arrows, alongside the exact 4x4 payoff matrix for that situation." />
            <div className="fig-cap">All six situations, with the unrounded 4&times;4
            payoff matrix printed alongside each one in the underlying report &mdash;
            so anyone can check by hand that the mix is real, not a rounding
            artifact.</div>
          </figure>
          <div className="case-grid">
            <div className="case"><span className="n" style={{ background: "var(--p0)" }}>1</span>
              <div><b>A coin-flip at the goal mouth.</b> Near a 50/50 split for
              both players &mdash; the textbook case.</div></div>
            <div className="case"><span className="n" style={{ background: "var(--p1)" }}>2</span>
              <div><b>Still guessing, far from goal.</b> The carrier is as far from
              scoring as this board allows, yet still splits three ways &mdash; what
              matters is the defender's distance, not the goal's.</div></div>
            <div className="case"><span className="n" style={{ background: "var(--p0)" }}>3</span>
              <div><b>A near-pure hedge.</b> A 97.5% / 2.5% split &mdash; genuinely
              mixed, but shallow enough that rounding it to "always" would be a
              real mistake.</div></div>
            <div className="case"><span className="n" style={{ background: "var(--p1)" }}>4</span>
              <div><b>Its mirror image.</b> Flip the board left-right and swap the
              players: the exact same equilibrium comes back, and its value
              negates to 17 decimal places.</div></div>
            <div className="case"><span className="n" style={{ background: "var(--p0)" }}>5</span>
              <div><b>Zero randomness in the rules &mdash; a mix anyway.</b> With every
              movement fully deterministic, a reward that rewards field position
              alone still forces a near-perfect coin flip.</div></div>
            <div className="case"><span className="n" style={{ background: "var(--p1)" }}>6</span>
              <div><b>A different rule, the same duel.</b> Swap in this project's
              own sliding-tackle rule for who wins a contested ball, and the
              identical "dive in or contain" guessing game reappears.</div></div>
          </div>
        </div>
      </section>

      <section id="generalize">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p0">05</span>Stress-testing it</div>
          <h2>Robust to the board. Specific to the reason.</h2>
          <p>The goal-width switch isn't a one-board coincidence: it survives boards
          up to roughly 7,800 states, odd aspect ratios, and goals with gaps in
          them &mdash; a defender only ever fails when it genuinely cannot cover every
          scoring cell at once, regardless of the exact shape. But the switch is
          specific to <em>why</em> the game is uncertain, not to uncertainty in
          general. Add a completely different kind of randomness &mdash; players
          occasionally slipping and moving the wrong way by accident, on
          <em> every</em> square, independent of either player's choice &mdash; and the
          mixing requirement reappears even with a single-cell goal. A shared coin
          everywhere produces guessing everywhere; a coin that only shows up where
          lanes cross produces guessing only there.</p>
          <figure className="breakout">
            <img src="figures/png/rule_fingerprints.png" alt="Six board heatmaps, one per collision rule, showing where mixed strategies are required; deterministic and coinflip rules are blank, random and blend light a thin band near the goal, slip and tackle light a broad region around the defender." />
            <div className="fig-cap"><b>Six rules, six fingerprints.</b> Same board,
            defender pinned at the goal mouth, every carrier cell shaded by how far
            its matchup is from having an obvious answer. Two rules (deterministic,
            a coin-flip tiebreak) are blank &mdash; always predictable. Two more light
            up a thin band right next to the goal. The last two &mdash; movement slip
            and this project's tackle rule &mdash; light up a broad region wherever
            the two players are close, not just near the goal.</div>
          </figure>
        </div>
      </section>

      <section className="band-tint" id="tournament">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">06</span>Put to the test</div>
          <h2>The mixed policy is the only one that can't be exploited</h2>
          <p>Solving for the correct mix is worthless if it doesn't matter in
          practice, so four policies were pitted against four opponents, including
          one built specifically, for each policy, to find and punish its weakness.
          <strong> Minimax</strong> is the exact policy this project solves for &mdash; it
          mixes exactly where the theory says it must, and stays pure everywhere
          else. The other three never mix: <strong>greedy/self</strong> always
          plays it safe, <strong>greedy/rand</strong> plays the single best reply to
          a random opponent, and <strong>hand-built</strong> is a scripted, common-sense
          "drive at the goal, block the direct path" defense.</p>
          <div className="tbl-wrap">
            <table>
              <thead>
                <tr><th>Policy</th><th>vs. always-left</th><th>vs. random</th><th>vs. hand-built</th><th>vs. its own challenger</th></tr>
              </thead>
              <tbody>
                <tr><td>Minimax (this project's policy)</td><td>+0.64</td><td>+0.63</td><td>+0.19</td><td className="hi">+0.13</td></tr>
                <tr><td>Greedy / random-beater</td><td>+0.95</td><td>+0.92</td><td>+0.79</td><td className="lo">&minus;0.54</td></tr>
                <tr><td>Greedy / play-it-safe</td><td>0.00</td><td>+0.07</td><td>0.00</td><td>&minus;0.00</td></tr>
                <tr><td>Hand-built defense</td><td>0.00</td><td>+0.71</td><td>0.00</td><td className="lo">&minus;0.79</td></tr>
              </tbody>
            </table>
          </div>
          <p><span className="tag" style={{ display: "inline-block", fontSize: ".74rem", letterSpacing: ".04em", padding: ".18rem .55rem", borderRadius: "4px", background: "var(--tint)", color: "var(--ink-soft)", border: "1px solid var(--rule)" }}>expected goal difference, exact solve, no simulation noise</span></p>
          <p>Every deterministic policy looks strong against a weak opponent and
          then collapses the moment a challenger is built to exploit it &mdash; because
          a fixed pattern always has a fixed counter, exactly like always throwing
          rock. <strong>Only the policy that knows how, and how much, to randomize
          survives being specifically hunted.</strong> That's the entire practical
          payoff of doing the harder math on the small slice of situations that
          actually require it, instead of a policy that looks clever until someone
          studies it.</p>
          <figure className="breakout">
            <img src="figures/png/tournament4.png" alt="Bar chart reproducing Littman's Table 3 tournament: minimax exploits weak opponents and survives its challenger, while greedy and hand-built policies collapse against a tailored challenger." />
            <div className="fig-cap">The same result Littman reported in 1994, reproduced
            here with an exact solver instead of learned policies, and with the
            plain 4&times;4 <code>{"{U, D, L, R}"}</code> action set &mdash; no fifth
            "stand" action needed for the result to hold.</div>
          </figure>
        </div>
      </section>

      <section id="neural">
        <div className="wrap">
          <div className="eyebrow"><span className="badge p1">07</span>Does deep learning find the same answer?</div>
          <h2>Action accuracy is not equilibrium accuracy</h2>
          <p className="lede">The exact solve above is the ground truth this project checks
            neural approximation against, not the other way around. A Q-network is trained
            three ways &mdash; from a random init by pure TD bootstrap, by supervised
            regression straight onto the exact matrix (no bootstrap at all), and a warm
            start combining both &mdash; and separately, a policy network learns each
            player's strategy directly. Both are then graded the same way: not by whether
            their output <em>looks like</em> the exact answer, but by whether it actually
            behaves like one &mdash; exploitability and equilibrium regret against the real
            game, not just a percentage match, since more than one policy can be a valid
            Nash equilibrium.</p>
          <div className="tbl-wrap">
            <table>
              <thead>
                <tr><th>Method</th><th>Learns</th><th>Checked against</th><th>Action agreement</th><th>Exploitability</th></tr>
              </thead>
              <tbody>
                <tr><td>DQN, from zero</td><td>Q(s, a0, a1)</td><td>Q* (exact matrix)</td><td>43%</td><td>0.44</td></tr>
                <tr><td>DQN, fit to exact</td><td>Q(s, a0, a1)</td><td>Q* (exact matrix)</td><td>45%</td><td>0.47</td></tr>
                <tr><td>DQN, warm start</td><td>Q(s, a0, a1)</td><td>Q* (exact matrix)</td><td>45%</td><td className="hi">0.38</td></tr>
                <tr><td>Policy network</td><td>&pi;<sub>0</sub>, &pi;<sub>1</sub></td><td>Nash conditions</td><td><b>99.9%</b></td><td className="lo">0.84</td></tr>
              </tbody>
            </table>
          </div>
          <p>The policy network names the exact-optimal action at <b>99.9%</b> of states
            &mdash; yet it is <b>more exploitable</b> than every Q-network tried, including
            the one that only gets 45% of actions right. Naming the right action almost
            everywhere is not the same as being an equilibrium: the rare states it gets
            wrong are exactly the ones a best-responder finds and attacks, and near-100%
            per-state accuracy gives no guarantee about the one number that actually
            measures robustness. Fitting a Q-network directly onto the exact matrix, with
            no training noise at all, still only reaches 45% action agreement &mdash; the
            bottleneck is what this size of network can represent, not how it is trained.
            Full detail, every seed, and the deterministic-vs-genuinely-mixed comparison:
            <a href="neural.md"> docs/neural.md</a> and <a href="policy_gradient.md">docs/policy_gradient.md</a>.</p>
        </div>
      </section>

      <section className="band-dark" id="why">
        <div className="wrap">
          <div className="eyebrow"><span className="badge em">08</span>Why it matters</div>
          <h2>A small game, a question that shows up everywhere agents share a world</h2>
          <p className="lede">Strip away the ball and the grid, and the question
          underneath this project is one that appears anywhere two or more
          decision-makers with opposing or partly opposing goals act in the same
          environment: poker-playing agents, adversarial-robustness testing,
          pricing and auction bots, multi-robot coordination. In every one of those
          settings, being perfectly predictable is a liability the moment someone
          else is watching closely enough to exploit it &mdash; but paying the cost of
          "always consider randomizing" everywhere is wasteful, because most
          situations really do have a clean, deterministic best answer.</p>
          <p>This project's contribution is a cheap, exact way to tell which regime
          a given situation is in &mdash; a check that costs almost nothing and never
          guesses wrong &mdash; instead of defaulting to an expensive general solver
          everywhere or, worse, a policy that is confidently deterministic in
          exactly the spots where that confidence is a losing bet.</p>
          <div className="cta-row">
            <a className="btn btn-primary" href="report.pdf">Read the full report &rarr;</a>
            <a className="btn btn-ghost" href="gallery.html">Browse every diagram &rarr;</a>
            <a className="btn btn-ghost" href="https://github.com/Charith-Reddy-Pareddy/soccer-markov-nash">Source on GitHub &rarr;</a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
