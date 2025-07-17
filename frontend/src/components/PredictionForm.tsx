import React from "react";
import { useForm, Controller } from "react-hook-form";
import { Box, Button, MenuItem, TextField, Typography } from "@mui/material";

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
      >
        {loading ? "Predicting..." : "Predict Top 5"}
      </Button>
    </form>
  );
};

export default PredictionForm;
