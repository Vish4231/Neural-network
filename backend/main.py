from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import joblib
import numpy as np

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

@app.post("/predict")
def predict(req: PredictionRequest):
    # Prepare input
    X = []
    for f in FEATURES:
        val = req.features.get(f, -1)
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
    # Predict
    prob = xgb_model.predict_proba(X)[0, 1]
    return {"probability": float(prob)} 