import React, { useState } from 'react';
import { ThemeProvider, CssBaseline, Container, IconButton, Box, Typography } from '@mui/material';
import { getTheme } from './theme';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
// import PredictionForm from './components/PredictionForm';
// import ResultCard from './components/ResultCard';

const App: React.FC = () => {
  const [mode, setMode] = useState<'light' | 'dark'>('dark');
  // const [result, setResult] = useState<number | null>(null);
  // const [loading, setLoading] = useState(false);

  return (
    <ThemeProvider theme={getTheme(mode)}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', color: 'text.primary' }}>
        <Container maxWidth="sm" sx={{ py: 4 }}>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
            <Typography variant="h4">F1 Top 5 Predictor</Typography>
            <IconButton onClick={() => setMode(mode === 'light' ? 'dark' : 'light')}>
              {mode === 'light' ? <Brightness4Icon /> : <Brightness7Icon />}
            </IconButton>
          </Box>
          {/* <PredictionForm onResult={setResult} loading={loading} setLoading={setLoading} /> */}
          {/* <ResultCard probability={result} loading={loading} /> */}
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default App; 