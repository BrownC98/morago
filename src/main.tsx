import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { applyCachedTheme } from "./theme";
import "./tokens.css";
import "./styles.css";

applyCachedTheme();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
