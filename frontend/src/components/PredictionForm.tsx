import React from "react";
import { useForm, Controller } from "react-hook-form";
import {
  Box,
  Button,
  MenuItem,
  TextField,
  Slider,
  Grid,
  Typography,
} from "@mui/material";

const drivers = [
  "Lando NORRIS",
  "Oscar PIASTRI",
  "Max VERSTAPPEN",
  "Lewis HAMILTON",
  "George RUSSELL",
  // ...add more
];
const teams = [
  "McLaren",
  "Red Bull Racing",
  "Ferrari",
  "Mercedes",
  "Aston Martin",
  // ...add more
];
const circuits = [
  "Silverstone",
  "Monaco",
  "Baku",
  "Suzuka",
  "Yas Marina Circuit",
  // ...add more
];

export type PredictionFormData = {
  driver_name: string;
  team_name: string;
  circuit: string;
  grid_position: number;
  qualifying_lap_time: number;
  air_temperature: number;
  humidity: number;
  rainfall: number;
  track_temperature: number;
  wind_speed: number;
  driver_form_last3: number;
  team_form_last3: number;
  qualifying_gap_to_pole: number;
  teammate_grid_delta: number;
  track_type: string;
  overtaking_difficulty: number;
  driver_championship_position: number;
  team_championship_position: number;
  driver_points_season: number;
  team_points_season: number;
};

interface PredictionFormProps {
  onSubmit: (formData: any) => void;
  loading: boolean;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, loading }) => {
  const { control, handleSubmit } = useForm<PredictionFormData>({
    defaultValues: {
      driver_name: "",
      team_name: "",
      circuit: "",
      grid_position: 1,
      qualifying_lap_time: 90,
      air_temperature: 20,
      humidity: 50,
      rainfall: 0,
      track_temperature: 25,
      wind_speed: 5,
      driver_form_last3: 5,
      team_form_last3: 5,
      qualifying_gap_to_pole: 0.5,
      teammate_grid_delta: 0,
      track_type: "permanent",
      overtaking_difficulty: 3,
      driver_championship_position: 5,
      team_championship_position: 3,
      driver_points_season: 50,
      team_points_season: 100,
    },
  });

  const onFormSubmit = (data: any) => {
    onSubmit(data);
  };

  return (
    <form onSubmit={handleSubmit(onFormSubmit)}>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <Controller
            name="driver_name"
            control={control}
            render={({ field }) => (
              <TextField select label="Driver" fullWidth required {...field}>
                {drivers.map((d) => (
                  <MenuItem key={d} value={d}>
                    {d}
                  </MenuItem>
                ))}
              </TextField>
            )}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <Controller
            name="team_name"
            control={control}
            render={({ field }) => (
              <TextField select label="Team" fullWidth required {...field}>
                {teams.map((t) => (
                  <MenuItem key={t} value={t}>
                    {t}
                  </MenuItem>
                ))}
              </TextField>
            )}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
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
        </Grid>
        <Grid item xs={12} sm={6}>
          <Controller
            name="grid_position"
            control={control}
            render={({ field }) => (
              <Box>
                <Typography gutterBottom>Grid Position</Typography>
                <Slider
                  {...field}
                  min={1}
                  max={20}
                  step={1}
                  valueLabelDisplay="auto"
                />
              </Box>
            )}
          />
        </Grid>
        {/* Add more fields as needed, e.g. qualifying_lap_time, weather, form, etc. */}
        <Grid item xs={12}>
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
        </Grid>
      </Grid>
    </form>
  );
};

export default PredictionForm;
