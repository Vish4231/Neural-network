import React from "react";
import { Box, Typography, Link } from "@mui/material";

const Footer: React.FC = () => (
  <Box component="footer" sx={{ mt: 6, py: 2, textAlign: "center", bgcolor: "background.paper", borderTop: 1, borderColor: "divider" }}>
    <Typography variant="body2" color="text.secondary">
      © {new Date().getFullYear()} F1 Top 5 Predictor
      {" | "}
      <Link href="https://github.com/your-github" target="_blank" rel="noopener" underline="hover">
        GitHub
      </Link>
      {" | Powered by FastF1, Ergast, and OpenAI"}
    </Typography>
  </Box>
);

export default Footer; 