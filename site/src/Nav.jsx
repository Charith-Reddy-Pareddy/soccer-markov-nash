import ThemeToggle from "./ThemeToggle.jsx";
import { useTheme } from "./useTheme.js";

export default function Nav({ current }) {
  const [theme, toggle] = useTheme();
  return (
    <header className="site">
      <nav>
        <a className="brand" href="index.html">
          <span className="mark"><span></span><span></span></span>
          Soccer &times; Nash
        </a>
        <div className="links">
          <a href="index.html#switch" className={current === "switch" ? "current" : ""}>The switch</a>
          <a href="index.html#seeing" className={current === "seeing" ? "current" : ""}>Gallery</a>
          <a href="explorer.html" className={current === "explorer" ? "current" : ""}>Board explorer</a>
          <a href="index.html#tournament" className={current === "tournament" ? "current" : ""}>Tournament</a>
          <a href="report.pdf" className="cta">Full report</a>
          <ThemeToggle theme={theme} onToggle={toggle} />
        </div>
      </nav>
    </header>
  );
}
