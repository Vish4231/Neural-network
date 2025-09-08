import os
import argparse
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow import keras
from sklearn.isotonic import IsotonicRegression
import pickle
import re

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


def load_race_artifacts():
    paths = {
        'race_model': os.path.join(MODELS_DIR, 'race_rf_model.pkl'),
        'race_encoders': os.path.join(MODELS_DIR, 'race_encoders.pkl'),
        'race_scaler': os.path.join(MODELS_DIR, 'race_scaler.pkl'),
        'race_calibrator': os.path.join(MODELS_DIR, 'race_rf_calibrator.pkl'),
    }
    if not all(os.path.exists(p) for p in [paths['race_model'], paths['race_encoders'], paths['race_scaler']]):
        return None
    calib = None
    if os.path.exists(paths['race_calibrator']):
        try:
            with open(paths['race_calibrator'], 'rb') as f:
                calib = pickle.load(f)
        except Exception:
            calib = None
    return {
        'model': joblib.load(paths['race_model']),
        'encoders': joblib.load(paths['race_encoders']),
        'scaler': joblib.load(paths['race_scaler']),
        'calibrator': calib,
        'calibrator_path': paths['race_calibrator'],
    }


def load_lineup(year: int, circuit: str) -> pd.DataFrame:
    canonical = normalize_circuit_name(circuit)
    # Try curated 2025 dataset first
    dataset_path = 'F1_2025_Dataset/F1_2025_RaceResults.csv'
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        # Normalize track names in dataset for robust match
        def _norm(v):
            try:
                return normalize_circuit_name(str(v))
            except Exception:
                return str(v).strip().lower()
        df['_track_norm'] = df['Track'].apply(_norm)
        df_loc = df[df['_track_norm'] == canonical]
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


def fetch_lineup_fastf1(year: int, circuit: str) -> pd.DataFrame:
    """Try to fetch the latest lineup (driver/team) using FastF1 for the given event.

    Strategy:
    - Load the season schedule
    - Fuzzy match event name/location to provided circuit
    - Load Race (preferred) or Qualifying session results
    - Extract driver full name and team name
    """
    try:
        import fastf1
        from fastf1 import plotting as _ff1_plotting  # noqa: F401 (ensures pandas options set)
        fastf1.Cache.enable_cache('fastf1_cache')
    except Exception:
        return pd.DataFrame()

    def _canon(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return s

    target = _canon(circuit)
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception:
        return pd.DataFrame()

    # Best-effort match: event name or location contains circuit tokens
    schedule['_name_canon'] = schedule['EventName'].apply(_canon)
    schedule['_loc_canon'] = schedule['Location'].apply(_canon)
    # direct contains
    cand = schedule[(schedule['_name_canon'].str.contains(target)) | (schedule['_loc_canon'].str.contains(target))]
    if cand.empty:
        # token match
        toks = [t for t in target.split() if len(t) > 3]
        def _tok_hit(x):
            return any(t in x for t in toks) if toks else False
        cand = schedule[schedule['_name_canon'].apply(_tok_hit) | schedule['_loc_canon'].apply(_tok_hit)]
    if cand.empty:
        return pd.DataFrame()

    # Prefer Race session of the best candidate (last occurrence)
    ev = cand.sort_values('EventFormat').tail(1).iloc[0]
    try_order = ['Race', 'Sprint', 'Qualifying']
    session = None
    for sess_name in try_order:
        try:
            s = fastf1.get_session(int(ev['EventDate'].year), ev['RoundNumber'], sess_name)
            s.load()
            session = s
            break
        except Exception:
            continue
    if session is None or session.results is None or session.results.empty:
        return pd.DataFrame()

    res = session.results
    # Columns usually available: Abbreviation, DriverNumber, TeamName, FirstName, LastName
    def _full(row):
        fn = str(row.get('FirstName', '')).strip()
        ln = str(row.get('LastName', '')).strip()
        return (fn + ' ' + ln).strip()
    lineup = pd.DataFrame({
        'driver_name': res.apply(_full, axis=1),
        'team_name': res.get('TeamName').astype(str) if 'TeamName' in res.columns else res.get('Team', pd.Series(dtype=str)).astype(str)
    })
    lineup = lineup.dropna().drop_duplicates()
    if lineup.empty:
        return lineup

    # Normalize team naming to align with our 2025 dataset aliases
    alias = {
        'oracle red bull racing': 'Red Bull Racing Honda RBPT',
        'red bull racing': 'Red Bull Racing Honda RBPT',
        'racing bulls': 'Racing Bulls Honda RBPT',
        'rb': 'Racing Bulls Honda RBPT',
        'mclaren formula 1 team': 'McLaren Mercedes',
        'mercedesamg petronas formula one team': 'Mercedes',
        'mercedes': 'Mercedes',
        'scuderia ferrari': 'Ferrari',
        'aston martin aramco formula one team': 'Aston Martin Aramco Mercedes',
        'aston martin': 'Aston Martin Aramco Mercedes',
        'williams racing': 'Williams Mercedes',
        'kick sauber': 'Kick Sauber Ferrari',
        'alpine f1 team': 'Alpine Renault',
        'haas f1 team': 'Haas Ferrari',
    }
    def _norm_team(t):
        k = _canon(t)
        return alias.get(k, t)
    lineup['team_name'] = lineup['team_name'].apply(_norm_team)
    return lineup[['driver_name','team_name']]


def build_prediction_frame(lineup: pd.DataFrame, circuit: str) -> pd.DataFrame:
    canonical = normalize_circuit_name(circuit)
    tf = track_features.get(canonical, {})
    rows = []
    for _, r in lineup.iterrows():
        base = {'driver_name': r['driver_name'], 'team_name': r['team_name'], 'circuit': canonical}
        base.update(tf)
        rows.append(base)
    return pd.DataFrame(rows)


def prepare_race_features(lineup: pd.DataFrame, circuit: str) -> pd.DataFrame:
    """Build inputs for race RF model using f1db merged history and track features."""
    merged_path = 'data/f1db_merged_2010_2025.csv'
    if not os.path.exists(merged_path):
        return pd.DataFrame()
    df = pd.read_csv(merged_path)
    # Engineer rolling stats
    from feature_engineering import engineer_f1db_features
    df = engineer_f1db_features(df, track_features)
    # Map names to IDs
    name_map = df[['fullName', 'driverId', 'name_constructor', 'constructorId']].dropna().drop_duplicates()
    # Normalize for join
    def norm(s):
        return s.str.strip().str.lower()
    name_map['full_norm'] = norm(name_map['fullName'])
    name_map['team_norm'] = norm(name_map['name_constructor'])
    lineup = lineup.copy()
    lineup['full_norm'] = norm(lineup['driver_name'])
    lineup['team_norm'] = norm(lineup['team_name'])
    joined = lineup.merge(name_map[['full_norm','team_norm','driverId','constructorId']], on=['full_norm','team_norm'], how='left')

    # Attempt to fill missing IDs using recent-year maps and aliases
    recent = df[df.get('year', 0) >= 2020]
    # Driver map by exact fullname (case-insens.)
    drv_latest = recent.sort_values('date').dropna(subset=['fullName','driverId']).drop_duplicates('fullName', keep='last')
    drv_map = {str(n).strip().lower(): v for n, v in zip(drv_latest['fullName'], drv_latest['driverId'])}
    # Constructor map by exact constructor name
    cons_latest = recent.sort_values('date').dropna(subset=['name_constructor','constructorId']).drop_duplicates('name_constructor', keep='last')
    cons_map = {str(n).strip().lower(): v for n, v in zip(cons_latest['name_constructor'], cons_latest['constructorId'])}
    # Team alias normalization for 2025 dataset naming → f1db constructor naming
    team_alias = {
        'red bull racing honda rbpt': 'red bull',
        'mclaren mercedes': 'mclaren',
        'mercedes': 'mercedes',
        'ferrari': 'ferrari',
        'williams mercedes': 'williams',
        'racing bulls honda rbpt': 'racing bulls',
        'alpine renault': 'alpine',
        'haas ferrari': 'haas',
        'aston martin aramco mercedes': 'aston martin',
        'kick sauber ferrari': 'sauber',
    }
    # Fill driverId where missing
    joined['driverId'] = joined['driverId'].fillna(joined['driver_name'].str.strip().str.lower().map(drv_map))
    # Fill constructorId using alias → cons_map
    joined['team_alias'] = joined['team_name'].str.strip().str.lower().map(team_alias).fillna(joined['team_name'].str.strip().str.lower())
    joined['constructorId'] = joined['constructorId'].fillna(joined['team_alias'].map(cons_map))
    # As a final fallback, add explicit 2025 mappings for rookies/renames
    explicit_driver = {
        'kimi antonelli': 'antonke01',
        'liam lawson': 'lawsoli01',
        'gabriel bortoleto': 'bortoga01',
        'oscar piastri': 'piasros01',
        'lando norris': 'norrila01',
        'george russell': 'russgeo01',
        'charles leclerc': 'leclech01',
        'max verstappen': 'verstma01',
        'lewis hamilton': 'hamille01',
        'pierre gasly': 'gaslypi01',
        'yuki tsunoda': 'tsunoyu01',
        'carlos sainz': 'sainzca01',
        'alexander albon': 'albona01',
    }
    explicit_team = {
        'racing bulls': 'racing_bulls',
        'red bull': 'red_bull',
        'aston martin': 'aston_martin',
        'sauber': 'sauber',
        'williams': 'williams',
        'ferrari': 'ferrari',
        'mercedes': 'mercedes',
        'alpine': 'alpine',
        'mclaren': 'mclaren',
        'haas': 'haas',
    }
    joined['driverId'] = joined['driverId'].fillna(joined['driver_name'].str.strip().str.lower().map(explicit_driver))
    joined['constructorId'] = joined['constructorId'].fillna(joined['team_alias'].map(explicit_team))
    # Latest rolling stats per driver
    latest_stats = df.sort_values('date').groupby('driverId').tail(1)[['driverId','driver_skill','driver_form_last3']]
    team_stats = df.sort_values('date').groupby('constructorId').tail(1)[['constructorId','team_form_last3']]
    joined = joined.merge(latest_stats, on='driverId', how='left').merge(team_stats, on='constructorId', how='left')
    # Add track features for target circuit
    canonical = normalize_circuit_name(circuit)
    tf = track_features.get(canonical, {})
    for k, v in tf.items():
        joined[k] = v
    # Select expected columns
    cols = ['driverId','constructorId','driver_name','team_name','driver_skill','driver_form_last3','team_form_last3',
            'length_km','turns','elevation','drs_zones','grip','rain_prob']
    return joined[cols]


def fit_or_load_race_calibrator(race_artifacts):
    if race_artifacts.get('calibrator') is not None:
        return race_artifacts['calibrator']
    merged_path = 'data/f1db_merged_2010_2025.csv'
    if not os.path.exists(merged_path):
        return None
    try:
        df = pd.read_csv(merged_path)
        from feature_engineering import engineer_f1db_features
        df = engineer_f1db_features(df, track_features)
        features = ['driver_skill', 'driver_form_last3', 'team_form_last3', 'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob']
        cat_features = ['driverId', 'constructorId']
        X = df[features + cat_features].fillna(0).copy()
        y = (df['positionDisplayOrder'] <= 5).astype(int)
        for col in cat_features:
            le = race_artifacts['encoders'][col]
            vals = X[col].astype(str)
            vals = vals.map(lambda v: v if v in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            X[col] = le.transform(vals)
        X[features] = race_artifacts['scaler'].transform(X[features])
        base_probs = race_artifacts['model'].predict_proba(X)[:, 1]
        # Isotonic regression calibrates monotonically without assuming sigmoid shape
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        ir.fit(base_probs, y.values)
        with open(race_artifacts['calibrator_path'], 'wb') as f:
            pickle.dump(ir, f)
        return ir
    except Exception:
        return None


def ensemble_predict(pred_df_enc: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    # Separate cat and num features based on training artifacts
    encoders = artifacts['encoders']
    scaler = artifacts['scaler']

    cat_cols = ['team_name', 'driver_name', 'circuit', 'track_type']
    # Use the scaler's training feature names to avoid mismatch
    if hasattr(scaler, 'feature_names_in_'):
        num_cols = [c for c in scaler.feature_names_in_ if c in pred_df_enc.columns]
    else:
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
        # Ensure all required numeric columns exist; fill missing with 0
        for col in num_cols:
            if col not in df.columns:
                df[col] = 0
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
    # Replace NaNs in stacking inputs with column means (fallback to 0)
    if np.isnan(stack_X).any():
        col_means = np.nanmean(stack_X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(stack_X))
        stack_X[inds] = np.take(col_means, inds[1])
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
    parser.add_argument('--use_fastf1', action='store_true', help='Fetch latest lineup from FastF1 if available')
    args = parser.parse_args()

    print('Loading models...')
    artifacts = load_models()
    print('Loading historical features...')
    combined = load_and_engineer_features()
    print('Loading lineup...')
    lineup = pd.DataFrame()
    if args.use_fastf1:
        print('Trying FastF1 lineup...')
        lineup = fetch_lineup_fastf1(args.year, args.circuit)
    if lineup.empty:
        lineup = load_lineup(args.year, args.circuit)
    if lineup.empty:
        print('No lineup found; aborting.')
        return
    print('Building prediction frame...')
    base_pred = build_prediction_frame(lineup, args.circuit)
    print('Engineering features...')
    pred_features = engineer_features_for_prediction(base_pred, combined)

    print('Running ensemble predictions...')
    base_results = ensemble_predict(pred_features, artifacts)
    base_results = base_results.sort_values('prob_top5', ascending=False).reset_index(drop=True)
    base_results['grid_rank_proxy'] = np.arange(1, len(base_results) + 1)

    # Race finish ranking (if RF model available)
    race_art = load_race_artifacts()
    if race_art is not None:
        print('Computing race-finish ranking using race RF model...')
        race_df = prepare_race_features(lineup, args.circuit)
        if not race_df.empty:
            # Encode cat
            X = race_df[['driver_skill','driver_form_last3','team_form_last3','length_km','turns','elevation','drs_zones','grip','rain_prob', 'driverId','constructorId']].copy()
            # Fill missing IDs with 'Unknown' placeholder
            X['driverId'] = X['driverId'].fillna('Unknown').astype(str)
            X['constructorId'] = X['constructorId'].fillna('Unknown').astype(str)
            for col in ['driverId','constructorId']:
                le = race_art['encoders'][col]
                vals = X[col].astype(str)
                # Replace unseen with 'Unknown'
                def safe_map(v):
                    return v if v in le.classes_ else 'Unknown'
                vals = vals.map(safe_map)
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                X[col] = le.transform(vals)
            num_feats = ['driver_skill','driver_form_last3','team_form_last3','length_km','turns','elevation','drs_zones','grip','rain_prob']
            X[num_feats] = race_art['scaler'].transform(X[num_feats])
            proba_top5 = race_art['model'].predict_proba(X)[:, 1]
            # Calibrate if calibrator available or can be fitted quickly
            calibrator = fit_or_load_race_calibrator(race_art)
            if calibrator is not None:
                proba_top5 = calibrator.predict_proba(proba_top5.reshape(-1, 1))[:, 1]
            race_out = race_df[['driver_name','team_name']].copy()
            race_out['race_prob_top5'] = proba_top5
            race_out = race_out.sort_values('race_prob_top5', ascending=False).reset_index(drop=True)
            race_out['race_rank'] = np.arange(1, len(race_out) + 1)
            # Merge
            results = base_results.merge(race_out, on=['driver_name','team_name'], how='left')
        else:
            print('Race RF features not fully available. Falling back to proxy ranking.')
            results = base_results.copy()
            results['race_prob_top5'] = results['prob_top5']
            results['race_rank'] = results['grid_rank_proxy']
    else:
        print('Race RF model not found. Using proxy ranking from ensemble.')
        results = base_results.copy()
        results['race_prob_top5'] = results['prob_top5']
        results['race_rank'] = results['grid_rank_proxy']

    # Final ordering and output
    results = results[['driver_name','team_name','prob_top5','grid_rank_proxy','race_prob_top5','race_rank']]

    print('\nPredicted Grid (proxy) Top 10:')
    for _, row in results.sort_values('grid_rank_proxy').head(10).iterrows():
        print(f"{int(row['grid_rank_proxy'])}. {row['driver_name']} ({row['team_name']}) - score {row['prob_top5']:.3f}")

    print('\nPredicted Race Finish (Top-5 prob) Top 10:')
    for _, row in results.sort_values('race_rank').head(10).iterrows():
        print(f"{int(row['race_rank'])}. {row['driver_name']} ({row['team_name']}) - prob {row['race_prob_top5']:.3f}")

    results.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == '__main__':
    main()


