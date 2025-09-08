import os
import argparse
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow import keras

from feature_engineering import (
    load_and_engineer_features,
    engineer_features_for_prediction,
    track_features,
    normalize_circuit_name,
)


MODELS_DIR = 'model'


def load_models():
    artifacts = {}
    artifacts['encoders'] = joblib.load(os.path.join(MODELS_DIR, 'encoders_top5.pkl'))
    artifacts['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'scaler_top5.pkl'))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, 'xgb_top5.model'))
    artifacts['xgb_model'] = xgb_model

    # Optional models if present
    lgbm_path = os.path.join(MODELS_DIR, 'lgbm_top5.txt')
    if os.path.exists(lgbm_path):
        artifacts['lgbm_model'] = lgb.Booster(model_file=lgbm_path)

    cat_path = os.path.join(MODELS_DIR, 'catboost_top5.cbm')
    if os.path.exists(cat_path):
        cat_model = cb.CatBoostClassifier()
        cat_model.load_model(cat_path)
        artifacts['cat_model'] = cat_model

    nn_path = os.path.join(MODELS_DIR, 'pre_race_model_top5.keras')
    if os.path.exists(nn_path):
        artifacts['nn_model'] = keras.models.load_model(nn_path)

    meta_path = os.path.join(MODELS_DIR, 'meta_model_logreg.pkl')
    if os.path.exists(meta_path):
        artifacts['meta_model'] = joblib.load(meta_path)

    return artifacts


def load_lineup(year: int, circuit: str) -> pd.DataFrame:
    canonical = normalize_circuit_name(circuit)
    # Try curated 2025 dataset first
    dataset_path = 'F1_2025_Dataset/F1_2025_RaceResults.csv'
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        df_loc = df[df['Track'].str.strip().str.lower() == canonical]
        if not df_loc.empty:
            lineup = df_loc[['Driver', 'Team']].drop_duplicates()
            lineup = lineup.rename(columns={'Driver': 'driver_name', 'Team': 'team_name'})
            return lineup
    # Fallback: infer from engineered historical data last known entries at that circuit
    hist = load_and_engineer_features()
    hist_circ = hist[hist['circuit'].str.strip().str.lower().apply(normalize_circuit_name) == canonical]
    if hist_circ.empty:
        # fallback to most recent overall
        last = hist.sort_values(['year','raceId']).groupby(['driver_name', 'team_name']).tail(1)
        return last[['driver_name','team_name']].drop_duplicates()
    last_at_circuit = hist_circ.sort_values(['year','raceId']).groupby(['driver_name','team_name']).tail(1)
    return last_at_circuit[['driver_name','team_name']].drop_duplicates()


def build_prediction_frame(lineup: pd.DataFrame, circuit: str) -> pd.DataFrame:
    canonical = normalize_circuit_name(circuit)
    tf = track_features.get(canonical, {})
    rows = []
    for _, r in lineup.iterrows():
        base = {'driver_name': r['driver_name'], 'team_name': r['team_name'], 'circuit': canonical}
        base.update(tf)
        rows.append(base)
    return pd.DataFrame(rows)


def ensemble_predict(pred_df_enc: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    # Separate cat and num features based on training artifacts
    encoders = artifacts['encoders']
    scaler = artifacts['scaler']

    cat_cols = ['team_name', 'driver_name', 'circuit', 'track_type']
    num_cols = [c for c in pred_df_enc.columns if c not in cat_cols and c not in ['raceId']]

    df = pred_df_enc.copy()
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].map(lambda v: v if v in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            df[col] = le.transform(df[col].astype(str))

    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])

    preds = []
    if 'xgb_model' in artifacts:
        preds.append(artifacts['xgb_model'].predict_proba(df)[:, 1])
    if 'lgbm_model' in artifacts:
        preds.append(artifacts['lgbm_model'].predict(df))
    if 'cat_model' in artifacts:
        preds.append(artifacts['cat_model'].predict_proba(df)[:, 1])

    # Optional NN: expects array-like; if present, use as additional signal
    if 'nn_model' in artifacts:
        try:
            nn_proba = artifacts['nn_model'].predict(df, verbose=0)
            if nn_proba.ndim > 1:
                nn_proba = nn_proba.max(axis=1)
            preds.append(nn_proba)
        except Exception:
            pass

    if not preds:
        raise RuntimeError('No base models available for prediction')

    stack_X = np.vstack(preds).T
    if 'meta_model' in artifacts:
        top5_prob = artifacts['meta_model'].predict_proba(stack_X)[:, 1]
    else:
        top5_prob = stack_X.mean(axis=1)

    out = pred_df_enc[['driver_name', 'team_name']].copy()
    out['prob_top5'] = top5_prob
    return out


def main():
    parser = argparse.ArgumentParser(description='Predict future F1 race outcomes for a given circuit/year')
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--circuit', type=str, required=True, help='Circuit name, e.g., "Spa-Francorchamps"')
    parser.add_argument('--output', type=str, default='predictions_future_race.csv')
    args = parser.parse_args()

    print('Loading models...')
    artifacts = load_models()
    print('Loading historical features...')
    combined = load_and_engineer_features()
    print('Loading lineup...')
    lineup = load_lineup(args.year, args.circuit)
    if lineup.empty:
        print('No lineup found; aborting.')
        return
    print('Building prediction frame...')
    base_pred = build_prediction_frame(lineup, args.circuit)
    print('Engineering features...')
    pred_features = engineer_features_for_prediction(base_pred, combined)

    print('Running ensemble predictions...')
    results = ensemble_predict(pred_features, artifacts)
    results = results.sort_values('prob_top5', ascending=False).reset_index(drop=True)
    results['rank'] = np.arange(1, len(results) + 1)

    print('\nPredicted Top 10 (by Top-5 probability):')
    for _, row in results.head(10).iterrows():
        print(f"{int(row['rank'])}. {row['driver_name']} ({row['team_name']}) - {row['prob_top5']:.3f}")

    results.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == '__main__':
    main()


