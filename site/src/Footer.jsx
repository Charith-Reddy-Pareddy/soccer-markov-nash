export default function Footer() {
  return (
    <footer>
      <div className="wrap">
        <div className="foot-grid">
          <div>
            <div className="credit-mark" style={{ marginBottom: ".9rem" }}><span></span><span></span></div>
            <p>An exact, no-learning-curve solver for a discrete soccer Markov
            game &mdash; and a map of exactly where, and why, its equilibrium stops
            being deterministic.</p>
          </div>
          <div>
            <h4>Read</h4>
            <ul>
              <li><a href="report.pdf">Full report (PDF)</a></li>
              <li><a href="result.md">The goal-width switch</a></li>
              <li><a href="positions.pdf">Twelve positions, as a PDF</a></li>
              <li><a href="tournament.md">The tournament, in full</a></li>
            </ul>
          </div>
          <div>
            <h4>Explore</h4>
            <ul>
              <li><a href="explorer.html">Board explorer</a></li>
              <li><a href="gallery.html">Diagram gallery</a></li>
              <li><a href="generalize.md">How far it generalizes</a></li>
              <li><a href="templates.md">The 8 geometric templates</a></li>
              <li><a href="https://charith-reddy-pareddy.github.io/the-mixed-game/">The Mixed Game (sister project) &rarr;</a></li>
            </ul>
          </div>
          <div>
            <h4>Project</h4>
            <ul>
              <li><a href="https://github.com/Charith-Reddy-Pareddy/soccer-markov-nash">GitHub repository</a></li>
              <li><a href="README.md">Documentation index</a></li>
              <li><a href="methods.md">Methods &amp; reproducibility</a></li>
            </ul>
          </div>
        </div>
      </div>
      <div className="foot-bottom">
        <span>Charith Reddy Pareddy &mdash; soccer-markov-nash</span>
        <span>Solved exactly. Nothing on this page was learned by a neural network.</span>
      </div>
    </footer>
  );
}
