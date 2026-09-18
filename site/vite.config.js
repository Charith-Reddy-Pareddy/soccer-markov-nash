import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));

// Builds straight into ../docs (the folder GitHub Pages already serves from
// `main`), alongside the Python-generated pages (positions.html, report.html,
// gallery.html -- untouched, emptyOutDir is off so this never deletes them).
export default defineConfig({
  root,
  base: "./",
  plugins: [react()],
  build: {
    outDir: resolve(root, "../docs"),
    emptyOutDir: false,
    rollupOptions: {
      input: {
        main: resolve(root, "index.html"),
        explorer: resolve(root, "explorer.html"),
      },
    },
  },
});
