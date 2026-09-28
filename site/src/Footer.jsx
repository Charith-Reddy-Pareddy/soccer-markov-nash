// Which case-study PDF corresponds to each board, for the footer's "Read"
// list to highlight when Footer is rendered with a `board` prop (only the
// explorer page has a notion of "current board"; every other page renders
// the plain, unordered list below).
const BOARD_READ_DOC = {
  canonical: { href: "positions.pdf", text: "Twelve positions, as a PDF" },
  canonical_det: { href: "a10_cases.pdf", text: "A10 edge cases, as a PDF" },
  canonical_coinflip: { href: "coinflip_cases.pdf", text: "Coin-flip edge cases (PDF)" },
  tackle: { href: "tackle_cases.pdf", text: "Tackle-rule edge cases (PDF)" },
  territory: { href: "territory_cases.pdf", text: "Territory edge cases (PDF)" },
  slip: { href: "slip_cases.pdf", text: "Movement-slip edge cases (PDF)" },
};

const READ_LINKS = [
  { href: "report.pdf", text: "Full report (PDF)" },
  { href: "result.md", text: "The goal-width switch" },
  { href: "positions.pdf", text: "Twelve positions, as a PDF" },
  { href: "tournament.md", text: "The tournament, in full" },
  { href: "a10_cases.pdf", text: "A10 edge cases, as a PDF" },
  { href: "coinflip_cases.pdf", text: "Coin-flip edge cases (PDF)" },
  { href: "tackle_cases.pdf", text: "Tackle-rule edge cases (PDF)" },
  { href: "territory_cases.pdf", text: "Territory edge cases (PDF)" },
  { href: "slip_cases.pdf", text: "Movement-slip edge cases (PDF)" },
];

export default function Footer({ board } = {}) {
  const current = BOARD_READ_DOC[board];
  const links = current
    ? [current, ...READ_LINKS.filter((l) => l.href !== current.href)]
    : READ_LINKS;
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
              {links.map((l) => (
                <li key={l.href}>
                  <a href={l.href}>
                    {current && l.href === current.href ? <b>{l.text} &mdash; this board</b> : l.text}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4>Explore</h4>
            <ul>
              <li><a href="explorer.html">Board explorer</a></li>
              <li><a href="gallery.html">Diagram gallery</a></li>
              <li><a href="generalize.md">How far it generalizes</a></li>
              <li><a href="templates.md">The 8 geometric templates</a></li>
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
      </div>
    </footer>
  );
}
