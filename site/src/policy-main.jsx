import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme.css";
import PolicyPage from "./PolicyPage.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <PolicyPage />
  </StrictMode>,
);
