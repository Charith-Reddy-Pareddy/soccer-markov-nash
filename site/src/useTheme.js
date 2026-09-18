import { useCallback, useEffect, useState } from "react";

const KEY = "soccer-nash-theme";

// Light by default, regardless of the visitor's OS setting -- dark mode is
// an explicit opt-in via the toggle, remembered per browser.
export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem(KEY) === "dark" ? "dark" : "light";
    } catch {
      return "light";
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // private mode / storage blocked -- theme still works for this load
    }
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  return [theme, toggle];
}
