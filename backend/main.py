from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import joblib
import numpy as np
import fastf1
from fastf1 import events

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("model/xgb_top5.model")
encoders = joblib.load("model/encoders_top5.pkl")
scaler = joblib.load("model/scaler_top5.pkl")

FEATURES = [
    "grid_position", "qualifying_lap_time", "air_temperature", "humidity", "rainfall",
    "track_temperature", "wind_speed", "team_name", "driver_name", "circuit", "country_code",
    "driver_form_last3", "team_form_last3", "qualifying_gap_to_pole", "teammate_grid_delta",
    "track_type", "overtaking_difficulty",
    "driver_championship_position", "team_championship_position", "driver_points_season", "team_points_season"
]

class PredictionRequest(BaseModel):
    features: dict

# 2025 F1 grid (example, update as needed)
DRIVERS_TEAMS = [
    ("Max VERSTAPPEN", "Red Bull Racing"),
    ("Liam LAWSON", "Red Bull Racing"),
    ("Charles LECLERC", "Ferrari"),
    ("Lewis HAMILTON", "Ferrari"),
    ("George RUSSELL", "Mercedes"),
    ("Andrea KIMI ANTONELLI", "Mercedes"),
    ("Lando NORRIS", "McLaren"),
    ("Oscar PIASTRI", "McLaren"),
    ("Fernando ALONSO", "Aston Martin"),
    ("Lance STROLL", "Aston Martin"),
    ("Pierre GASLY", "Alpine"),
    ("Jack DOOHAN", "Alpine"),
    ("Esteban OCON", "Haas"),
    ("Oliver BEARMAN", "Haas"),
    ("Yuki TSUNODA", "Racing Bulls"),
    ("Isack HADJAR", "Racing Bulls"),
    ("Carlos SAINZ", "Williams"),
    ("Alex ALBON", "Williams"),
    ("Nico HULKENBERG", "Kick Sauber"),
    ("Gabriel BORTOLETO", "Kick Sauber"),
]

DEFAULT_FEATURES = {
    "grid_position": 1,
    "qualifying_lap_time": 90,
    "air_temperature": 20,
    "humidity": 50,
    "rainfall": 0,
    "track_temperature": 25,
    "wind_speed": 5,
    "country_code": "AUS",
    "driver_form_last3": 5,
    "team_form_last3": 5,
    "qualifying_gap_to_pole": 0.5,
    "teammate_grid_delta": 0,
    "track_type": "permanent",
    "overtaking_difficulty": 3,
    "driver_championship_position": 1,
    "team_championship_position": 1,
    "driver_points_season": 100,
    "team_points_season": 200,
}

# Map user-friendly circuit names to FastF1 event names
CIRCUIT_NAME_MAP = {
    "Melbourne": "Australian Grand Prix",
    "Shanghai": "Chinese Grand Prix",
    "Suzuka": "Japanese Grand Prix",
    "Sakhir": "Bahrain Grand Prix",
    "Jeddah": "Saudi Arabian Grand Prix",
    "Miami": "Miami Grand Prix",
    "Imola": "Emilia Romagna Grand Prix",
    "Monte Carlo": "Monaco Grand Prix",
    "Barcelona": "Spanish Grand Prix",
    "Montreal": "Canadian Grand Prix",
    "Spielberg": "Austrian Grand Prix",
    "Silverstone": "British Grand Prix",
    "Spa-Francorchamps": "Belgian Grand Prix",
    "Budapest": "Hungarian Grand Prix",
    "Zandvoort": "Dutch Grand Prix",
    "Monza": "Italian Grand Prix",
    "Baku": "Azerbaijan Grand Prix",
    "Singapore": "Singapore Grand Prix",
    "Austin": "United States Grand Prix",
    "Mexico City": "Mexico City Grand Prix",
    "Sao Paulo": "São Paulo Grand Prix",
    "Las Vegas": "Las Vegas Grand Prix",
    "Lusail": "Qatar Grand Prix",
    "Yas Marina": "Abu Dhabi Grand Prix",
}

@app.post("/predict")
def predict(req: PredictionRequest):
    import datetime
    season = datetime.datetime.now().year
    circuit = req.features.get("circuit", "Melbourne")
    event_name = CIRCUIT_NAME_MAP.get(circuit, circuit)
    # Find the event round for the selected circuit
    schedule = fastf1.get_event_schedule(season)
    event_row = schedule[schedule['EventName'] == event_name]
    if event_row.empty:
        return {"error": f"Event '{event_name}' not found in {season} schedule."}
    rnd = int(event_row.iloc[0]['RoundNumber'])
    # Get entry list for the event
    event = fastf1.get_event(season, rnd)
    session = event.get_race()
    session.load(telemetry=False, weather=False, laps=False)
    entry_list = session.entry_list
    results = []
    probs = {}
    for drv in entry_list:
        driver = entry_list[drv]['FullName']
        team = entry_list[drv]['TeamName']
        features = DEFAULT_FEATURES.copy()
        features["circuit"] = circuit
        features["driver_name"] = driver
        features["team_name"] = team
        # Prepare input
        X = []
        for f in FEATURES:
            val = features.get(f, -1)
            if f in encoders:
                le = encoders[f]
                val = le.transform([str(val) if val is not None else "unknown"])[0]
            X.append(val)
        X = np.array(X).reshape(1, -1)
        # Scale numerics
        num_features = [f for f in FEATURES if f not in encoders]
        X_num = scaler.transform(X[:, [FEATURES.index(f) for f in num_features]])
        for i, f in enumerate(num_features):
            X[0, FEATURES.index(f)] = X_num[0, i]
        prob = xgb_model.predict_proba(X)[0, 1]
        results.append((driver, prob))
        probs[driver] = float(prob)
    # Sort and get top 5
    results.sort(key=lambda x: x[1], reverse=True)
    top5 = [d for d, _ in results[:5]]
    return {"top5": top5, "probabilities": probs} 