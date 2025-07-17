import React from "react";
import { Card, CardContent, Typography, CircularProgress, Box, Avatar, Stack } from "@mui/material";
import { motion } from "framer-motion";

interface PredictionResultCardProps {
  loading: boolean;
  error?: string;
  result?: any;
}

const getInitials = (name: string) => {
  const parts = name.split(" ");
  return parts.map((p) => p[0]).join("").toUpperCase();
};

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
    <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <Card sx={{ mt: 3, boxShadow: 6, transition: "0.3s", border: "2px solid #e10600" }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Predicted Top 5 Finishers
          </Typography>
          {result.top5 && Array.isArray(result.top5) ? (
            <Stack spacing={1}>
              {result.top5.map((driver: string, idx: number) => (
                <Box key={driver} display="flex" alignItems="center" gap={2}>
                  <Avatar sx={{ bgcolor: "primary.main", color: "#fff", width: 36, height: 36, fontWeight: 700 }}>
                    {getInitials(driver)}
                  </Avatar>
                  <Typography variant="body1" fontWeight={600} color={idx < 3 ? "primary.main" : "text.primary"}>
                    {idx + 1}. {driver}
                  </Typography>
                </Box>
              ))}
            </Stack>
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
    </motion.div>
  );
};

export default PredictionResultCard; 