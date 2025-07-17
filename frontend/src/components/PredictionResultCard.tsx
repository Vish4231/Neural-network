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

const F1CarSVG = () => (
  <motion.svg
    width="120"
    height="32"
    viewBox="0 0 320 64"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    initial={{ x: -160, opacity: 0 }}
    animate={{ x: 0, opacity: 1 }}
    transition={{ duration: 0.7, type: "spring", bounce: 0.3 }}
    style={{ margin: '0 auto', display: 'block' }}
  >
    <rect x="0" y="28" width="120" height="8" rx="4" fill="#222" opacity="0.2" />
    <g>
      <rect x="20" y="36" width="80" height="12" rx="6" fill="#e10600" />
      <rect x="30" y="24" width="60" height="16" rx="8" fill="#fff" />
      <rect x="45" y="16" width="30" height="16" rx="8" fill="#222" />
      <ellipse cx="30" cy="52" rx="8" ry="6" fill="#181818" />
      <ellipse cx="90" cy="52" rx="8" ry="6" fill="#181818" />
      <rect x="55" y="8" width="10" height="8" rx="4" fill="#e10600" />
      <rect x="60" y="0" width="4" height="8" rx="2" fill="#fff" />
    </g>
  </motion.svg>
);

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
          <F1CarSVG />
          <Typography variant="h5" gutterBottom align="center">
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