import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme.css";
import ExplorerApp from "./explorer/ExplorerApp.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ExplorerApp />
  </StrictMode>,
);
