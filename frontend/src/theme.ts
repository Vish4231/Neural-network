import { createTheme } from "@mui/material/styles";

const carbonFiber = `
  repeating-linear-gradient(135deg, #222 0 2px, #333 2px 4px),
  repeating-linear-gradient(45deg, #222 0 2px, #444 2px 4px)
`;

export const getTheme = (mode: "light" | "dark") =>
  createTheme({
    palette: {
      mode,
      primary: {
        main: "#e10600",
      },
      background: {
        default: mode === "light" ? "#f5f5f5" : "#181818",
        paper: mode === "light" ? "#fff" : "#222",
      },
    },
    typography: {
      fontFamily: '"Roboto Condensed", "Arial", sans-serif',
      h1: { fontWeight: 900, letterSpacing: 1 },
      h2: { fontWeight: 800, letterSpacing: 1 },
      h3: { fontWeight: 700, letterSpacing: 0.5 },
      h4: { fontWeight: 700 },
      h5: { fontWeight: 600 },
      h6: { fontWeight: 600 },
      button: { fontWeight: 700, letterSpacing: 1 },
    },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            backgroundImage: carbonFiber,
            backgroundSize: '16px 16px',
            borderRadius: 16,
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            backgroundImage: carbonFiber,
            backgroundSize: '16px 16px',
            borderRadius: 8,
            color: '#fff',
            textTransform: 'uppercase',
            fontWeight: 700,
            letterSpacing: 1,
            boxShadow: '0 2px 8px rgba(225,6,0,0.15)',
            transition: 'box-shadow 0.2s, transform 0.2s',
            '&:hover': {
              boxShadow: '0 4px 16px rgba(225,6,0,0.25)',
              transform: 'translateY(-2px) scale(1.03)',
            },
          },
        },
      },
    },
  });
