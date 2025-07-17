import React, { useState } from "react";
import {
  ThemeProvider,
  CssBaseline,
  Container,
  IconButton,
  Box,
  Typography,
} from "@mui/material";
import { getTheme } from "./theme";
import Brightness4Icon from "@mui/icons-material/Brightness4";
import Brightness7Icon from "@mui/icons-material/Brightness7";
import { getPrediction } from "./api";
import PredictionResultCard from "./components/PredictionResultCard";
import PredictionForm from './components/PredictionForm';
import AboutSection from "./components/AboutSection";
// import ResultCard from './components/ResultCard';

const App: React.FC = () => {
  const [mode, setMode] = useState<"light" | "dark">("dark");
  const [result, setResult] = React.useState<any>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | undefined>(undefined);

  const handlePredict = async (formData: any) => {
    setLoading(true);
    setError(undefined);
    setResult(null);
    try {
      const res = await getPrediction(formData);
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={getTheme(mode)}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: "100vh",
          bgcolor: "background.default",
          color: "text.primary",
        }}
      >
        <Container maxWidth="sm">
          <Box sx={{ py: 4 }}>
            <Typography variant="h3" fontWeight={700} gutterBottom align="center" color="primary">
              F1 Top 5 Predictor
            </Typography>
            <Typography variant="subtitle1" align="center" color="text.secondary" gutterBottom>
              Predict the Top 5 finishers of any Formula 1 race using only pre-race data.
            </Typography>
            <AboutSection />
            <PredictionForm onSubmit={handlePredict} loading={loading} />
            <PredictionResultCard loading={loading} error={error} result={result} />
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default App;
