import React from "react";

const F1CarBanner: React.FC = () => (
  <div style={{
    width: "100%",
    background: "linear-gradient(90deg, #0d0d0d 0%, #1a1a1a 100%)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "32px 0",
    borderBottom: "2px solid #e10600",
    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.5)"
  }}>
    <svg width="360" height="72" viewBox="0 0 360 72" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="30" width="360" height="8" rx="4" fill="#444" opacity="0.3" />
      <g>
        <rect x="70" y="38" width="220" height="14" rx="7" fill="#e10600" />
        <rect x="90" y="26" width="180" height="18" rx="9" fill="#fff" />
        <rect x="130" y="18" width="100" height="16" rx="8" fill="#2b2b2b" />
        <ellipse cx="90" cy="58" rx="14" ry="9" fill="#1f1f1f" />
        <ellipse cx="270" cy="58" rx="14" ry="9" fill="#1f1f1f" />
        <rect x="165" y="10" width="24" height="8" rx="4" fill="#e10600" />
        <rect x="176" y="2" width="8" height="8" rx="4" fill="#ffffff" />
      </g>
    </svg>
  </div>
);

export default F1CarBanner;