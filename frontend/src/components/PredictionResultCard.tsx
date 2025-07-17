import React from "react";
import { Card, CardContent, Typography, CircularProgress, Box } from "@mui/material";

interface PredictionResultCardProps {
  loading: boolean;
  error?: string;
  result?: any;
}

const PredictionResultCard: React.FC<PredictionResultCardProps> = ({ loading, error, result }) => {
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
        <CircularProgress />
      </Box>
    );
  }
  if (error) {
    return (
      <Card sx={{ mt: 3, border: "2px solid #e10600" }}>
        <CardContent>
          <Typography color="error" variant="h6">{error}</Typography>
        </CardContent>
      </Card>
    );
  }
  if (!result) return null;
  return (
    <Card sx={{ mt: 3, boxShadow: 6, transition: "0.3s", border: "2px solid #e10600" }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Predicted Top 5 Finishers
        </Typography>
        {result.top5 && Array.isArray(result.top5) ? (
          result.top5.map((driver: string, idx: number) => (
            <Typography key={driver} variant="body1">
              {idx + 1}. {driver}
            </Typography>
          ))
        ) : (
          <Typography>No prediction data available.</Typography>
        )}
        {result.probabilities && (
          <Box mt={2}>
            <Typography variant="subtitle2">Probabilities:</Typography>
            {Object.entries(result.probabilities).map(([driver, prob]: [string, any]) => (
              <Typography key={driver} variant="caption">
                {driver}: {(prob * 100).toFixed(1)}%
              </Typography>
            ))}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default PredictionResultCard; 