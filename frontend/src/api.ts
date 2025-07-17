// src/api.ts

const DEFAULT_FEATURES = {
  grid_position: 1,
  qualifying_lap_time: 90,
  air_temperature: 20,
  humidity: 50,
  rainfall: 0,
  track_temperature: 25,
  wind_speed: 5,
  team_name: "Red Bull Racing",
  driver_name: "Max VERSTAPPEN",
  circuit: "Melbourne", // will be overwritten
  country_code: "AUS",
  driver_form_last3: 5,
  team_form_last3: 5,
  qualifying_gap_to_pole: 0.5,
  teammate_grid_delta: 0,
  track_type: "permanent",
  overtaking_difficulty: 3,
  driver_championship_position: 1,
  team_championship_position: 1,
  driver_points_season: 100,
  team_points_season: 200,
};

export async function getPrediction(formData: { circuit: string }) {
  const features = { ...DEFAULT_FEATURES, circuit: formData.circuit };
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!response.ok) {
    throw new Error("Prediction request failed");
  }
  return response.json();
} 