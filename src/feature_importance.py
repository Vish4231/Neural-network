import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import numpy as np

features = [
    'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
    'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code',
    'driver_form_last3', 'team_form_last3', 'qualifying_gap_to_pole', 'teammate_grid_delta',
    'track_type', 'overtaking_difficulty',
    'driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season'
]

model = xgb.XGBClassifier()
model.load_model('model/xgb_top5.model')

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print top 10 features
print("Top 10 XGBoost Feature Importances:")
for i in range(10):
    print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

# Plot
plt.figure(figsize=(10,6))
plt.title("XGBoost Feature Importances (Top 10)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), [features[i] for i in indices[:10]], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model/xgb_feature_importance.png')
print("\nFeature importance plot saved to model/xgb_feature_importance.png")
plt.show() 