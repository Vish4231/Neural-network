import React from "react";
import { Accordion, AccordionSummary, AccordionDetails, Typography } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

const AboutSection: React.FC = () => (
  <Accordion sx={{ mb: 3 }}>
    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
      <Typography variant="h6">About This App</Typography>
    </AccordionSummary>
    <AccordionDetails>
      <Typography variant="body1" gutterBottom>
        <b>F1 Top 5 Predictor</b> uses a machine learning model trained on historical Formula 1 data to predict which drivers are most likely to finish in the Top 5 of a race. Only pre-race information (qualifying, grid, team/driver stats, track, weather, etc.) is used, so predictions are available before lights out!
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Enter the latest pre-race data, submit, and see the model's predicted Top 5 finishers. This tool is for educational and entertainment purposes only.
      </Typography>
    </AccordionDetails>
  </Accordion>
);

export default AboutSection; 