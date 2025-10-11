import re
import pandas as pd


def _canon(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def _fullname(row: pd.Series) -> str:
    fn = str(row.get('FirstName', '')).strip()
    ln = str(row.get('LastName', '')).strip()
    return (fn + ' ' + ln).strip()


def extract_event_features(year: int, circuit: str) -> pd.DataFrame:
    """Return per-driver FastF1 summary features for the most recent session of the event.

    Columns returned:
      - driver_name, team_name
      - ff1_avg_lap_ms, ff1_best_lap_ms
      - ff1_speed_trap_kph (if available)
      - ff1_rain_prob_est (rough proxy from weather observations)
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
    except Exception:
        return pd.DataFrame()

    target = _canon(circuit)
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception:
        return pd.DataFrame()

    schedule['_name'] = schedule['EventName'].apply(_canon)
    schedule['_loc'] = schedule['Location'].apply(_canon)
    cand = schedule[(schedule['_name'].str.contains(target)) | (schedule['_loc'].str.contains(target))]
    if cand.empty:
        toks = [t for t in target.split() if len(t) > 3]
        def _tok_hit(x):
            return any(t in x for t in toks) if toks else False
        cand = schedule[schedule['_name'].apply(_tok_hit) | schedule['_loc'].apply(_tok_hit)]
    if cand.empty:
        return pd.DataFrame()

    ev = cand.tail(1).iloc[0]
    session = None
    for sess_name in ['Race', 'Sprint', 'Qualifying', 'Practice 3', 'Practice 2', 'Practice 1']:
        try:
            s = fastf1.get_session(int(ev['EventDate'].year), ev['RoundNumber'], sess_name)
            s.load()
            session = s
            break
        except Exception:
            continue
    if session is None:
        return pd.DataFrame()

    # Build per-driver aggregates
    laps = session.laps
    if laps is None or laps.empty:
        return pd.DataFrame()

    # Convert lap times to milliseconds
    def _to_ms(td):
        try:
            return td.total_seconds() * 1000.0
        except Exception:
            return None

    grp = laps.groupby('Driver')
    avg_lap = grp['LapTime'].apply(lambda s: pd.Series([_to_ms(x) for x in s.dropna()]).mean())
    best_lap = grp['LapTime'].apply(lambda s: pd.Series([_to_ms(x) for x in s.dropna()]).min())

    # Speed trap approximation: use max SpeedI1/I2/I3 if available
    trap_cols = [c for c in laps.columns if str(c).lower().startswith('speed')]
    if trap_cols:
        # Per-driver per-column maxima, then take the max across columns
        per_driver_col_max = grp[trap_cols].max()
        if isinstance(per_driver_col_max, pd.DataFrame):
            speed_trap = per_driver_col_max.max(axis=1)
        else:
            speed_trap = per_driver_col_max
    else:
        speed_trap = pd.Series(dtype=float)

    # Weather proxy: mean rainfall % if available
    rain_prob = None
    try:
        w = session.weather_data
        if w is not None and not w.empty and 'Rainfall' in w.columns:
            # Rainfall is boolean sometimes; map to 0/1 and take mean
            rain_prob = (w['Rainfall'].astype(float)).mean() * 100.0
    except Exception:
        rain_prob = None

    # Map per-driver code to names/teams from session.results
    res = session.results
    if res is None or res.empty:
        return pd.DataFrame()
    code_to_name = {str(r['Abbreviation']): _fullname(r) for _, r in res.iterrows() if 'Abbreviation' in r}
    code_to_team = {str(r['Abbreviation']): str(r.get('TeamName', r.get('Team', ''))) for _, r in res.iterrows() if 'Abbreviation' in r}

    rows = []
    for code, avg in avg_lap.items():
        name = code_to_name.get(str(code), None)
        team = code_to_team.get(str(code), None)
        if not name or not team:
            continue
        row = {
            'driver_name': name,
            'team_name': team,
            'ff1_avg_lap_ms': float(avg) if pd.notna(avg) else None,
            'ff1_best_lap_ms': float(best_lap.get(code)) if pd.notna(best_lap.get(code)) else None,
            'ff1_speed_trap_kph': float(speed_trap.get(code)) if pd.notna(speed_trap.get(code)) else None,
            'ff1_rain_prob_est': float(rain_prob) if rain_prob is not None else None,
        }
        rows.append(row)

    df_feat = pd.DataFrame(rows)
    # Normalize team names lightly for downstream joins (leave as-is; predictor has aliasing)
    return df_feat


