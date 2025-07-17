import React from "react";
import { useForm, Controller } from "react-hook-form";
import { Box, Button, MenuItem, TextField, Typography } from "@mui/material";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { keyframes } from "@emotion/react";

const circuits = [
  "Melbourne",
  "Shanghai",
  "Suzuka",
  "Sakhir",
  "Jeddah",
  "Miami",
  "Imola",
  "Monte Carlo",
  "Barcelona",
  "Montreal",
  "Spielberg",
  "Silverstone",
  "Spa-Francorchamps",
  "Budapest",
  "Zandvoort",
  "Monza",
  "Baku",
  "Singapore",
  "Austin",
  "Mexico City",
  "Sao Paulo",
  "Las Vegas",
  "Lusail",
  "Yas Marina"
];

const chevronSlide = keyframes`
  from { transform: translateX(-12px); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
`;

export type PredictionFormData = {
  circuit: string;
};

interface PredictionFormProps {
  onSubmit: (formData: any) => void;
  loading: boolean;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, loading }) => {
  const { control, handleSubmit } = useForm<PredictionFormData>({
    defaultValues: { circuit: "" },
  });

  const onFormSubmit = (data: any) => {
    onSubmit(data);
  };

  const [hover, setHover] = React.useState(false);

  return (
    <form onSubmit={handleSubmit(onFormSubmit)}>
      <Box mb={2}>
        <Typography variant="subtitle2" gutterBottom>
          Select Track
        </Typography>
        <Controller
          name="circuit"
          control={control}
          render={({ field }) => (
            <TextField select label="Circuit" fullWidth required {...field}>
              {circuits.map((c) => (
                <MenuItem key={c} value={c}>
                  {c}
                </MenuItem>
              ))}
            </TextField>
          )}
        />
      </Box>
      <Button
        type="submit"
        variant="contained"
        color="primary"
        fullWidth
        disabled={loading}
        size="large"
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 700,
          fontSize: 18,
          letterSpacing: 1,
          position: 'relative',
          overflow: 'hidden',
        }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
      >
        {loading ? "Predicting..." : "Predict Top 5"}
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            marginLeft: 8,
            animation: hover && !loading ? `${chevronSlide} 0.3s ease` : undefined,
            opacity: hover && !loading ? 1 : 0,
            transition: 'opacity 0.2s',
          }}
        >
          <ChevronRightIcon fontSize="medium" />
        </span>
      </Button>
    </form>
  );
};

export default PredictionForm;
