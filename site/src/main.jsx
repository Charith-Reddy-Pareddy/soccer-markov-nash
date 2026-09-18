import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme.css";
import Landing from "./Landing.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Landing />
  </StrictMode>,
);
