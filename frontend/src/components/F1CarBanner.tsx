import React from "react";

const F1CarBanner: React.FC = () => (
  <div style={{
    width: "100%",
    background: "linear-gradient(90deg, #e10600 0%, #181818 100%)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "24px 0 12px 0"
  }}>
    <svg width="320" height="64" viewBox="0 0 320 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="28" width="320" height="8" rx="4" fill="#222" opacity="0.2" />
      <g>
        <rect x="60" y="36" width="200" height="12" rx="6" fill="#e10600" />
        <rect x="80" y="24" width="160" height="16" rx="8" fill="#fff" />
        <rect x="120" y="16" width="80" height="16" rx="8" fill="#222" />
        <ellipse cx="80" cy="52" rx="12" ry="8" fill="#181818" />
        <ellipse cx="240" cy="52" rx="12" ry="8" fill="#181818" />
        <rect x="150" y="8" width="20" height="8" rx="4" fill="#e10600" />
        <rect x="160" y="0" width="8" height="8" rx="4" fill="#fff" />
      </g>
    </svg>
  </div>
);

export default F1CarBanner; 