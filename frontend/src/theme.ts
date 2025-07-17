import { createTheme } from '@mui/material/styles';

export const getTheme = (mode: 'light' | 'dark') =>
  createTheme({
    palette: {
      mode,
      ...(mode === 'light'
        ? {
            primary: { main: '#e10600' }, // F1 red
            secondary: { main: '#222' },
            background: { default: '#f5f5f5', paper: '#fff' },
          }
        : {
            primary: { main: '#e10600' },
            secondary: { main: '#fff' },
            background: { default: '#181818', paper: '#222' },
          }),
    },
    typography: {
      fontFamily: 'Roboto, Arial, sans-serif',
      h4: { fontWeight: 700 },
    },
    shape: { borderRadius: 12 },
  }); 