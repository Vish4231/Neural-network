import { createTheme } from "@mui/material/styles";

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
      fontFamily: 'Inter, Roboto, Arial, sans-serif',
      h1: { fontWeight: 800, letterSpacing: 0.5 },
      h2: { fontWeight: 700, letterSpacing: 0.5 },
      h3: { fontWeight: 600, letterSpacing: 0.2 },
      h4: { fontWeight: 600 },
      h5: { fontWeight: 500 },
      h6: { fontWeight: 500 },
      button: { fontWeight: 700, letterSpacing: 0.5 },
    },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            background: 'none',
            borderRadius: 12,
            boxShadow: '0 2px 16px rgba(0,0,0,0.10)',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            background: '#e10600',
            color: '#fff',
            borderRadius: 8,
            textTransform: 'uppercase',
            fontWeight: 700,
            letterSpacing: 1,
            boxShadow: '0 2px 8px rgba(225,6,0,0.10)',
            transition: 'box-shadow 0.2s, transform 0.2s',
            '&:hover': {
              boxShadow: '0 4px 16px rgba(225,6,0,0.18)',
              transform: 'translateY(-2px) scale(1.03)',
            },
          },
        },
      },
    },
  });
