# F1 Neural Network - 2025 Data Update Status

## Current Status (as of July 2025)

✅ **System Updated**: The neural network system has been successfully updated to support 2025 data collection and processing.

### Data Coverage:
- **2018-2022**: Not available in OpenF1 API
- **2023**: ✅ 85 race data points collected (4 circuits)
- **2024**: ❌ No data available yet in API
- **2025**: ⏳ 14 race sessions scheduled, but no results yet (season hasn't started)

### What's Been Updated:

1. **Data Collection Scripts**:
   - `src/pre_race_data.py`: Updated to include years 2018-2025
   - `src/expanded_data_collection.py`: Updated year range to 2026
   - `src/update_data_2025.py`: New comprehensive data collection script

2. **Dataset Features**:
   - 27 features including weather, qualifying times, team/driver form
   - Circuit characteristics (track type, overtaking difficulty)
   - Driver and team performance metrics
   - Gap to pole position and teammate comparisons

3. **Current Dataset**:
   - **Total**: 85 race entries
   - **Years**: 2023 only (most complete data available)
   - **Circuits**: Baku, Jeddah, Melbourne, Sakhir
   - **Teams**: All 10 F1 teams represented

## How to Use the Updated System

### Training the Model:
```bash
# Train with current data
python src/model_train_pre_race.py

# Retrain when new data becomes available
python src/update_data_2025.py  # Update dataset
python src/model_train_pre_race.py  # Retrain model
```

### Making Predictions:
```bash
# Predict upcoming races
python src/predict_upcoming_race.py
```

### Updating Data:
```bash
# Check for new 2025 data
python src/update_data_2025.py

# Or use the original script
python src/pre_race_data.py
```

## When Will 2025 Data Be Available?

The 2025 F1 season is expected to start in **March 2025**. Once races begin:

1. **Real-time Updates**: Run `src/update_data_2025.py` after each race
2. **Automatic Integration**: The system will automatically incorporate new 2025 data
3. **Model Retraining**: Retrain the model periodically with new data

## Data Sources

- **Primary**: OpenF1 API (https://api.openf1.org/v1)
- **Coverage**: 2023-2025 (when available)
- **Updates**: Real-time as races are completed

## Features Ready for 2025

1. **Enhanced Weather Data**: Air temperature, humidity, rainfall, track temperature, wind speed
2. **Driver/Team Form**: Rolling 3-race averages
3. **Qualifying Analysis**: Gap to pole, teammate comparisons
4. **Circuit Characteristics**: Track type, overtaking difficulty
5. **Championship Context**: Driver/team standings (when available)

## Next Steps

1. **Monitor 2025 Season**: Check for new race results regularly
2. **Expand Historical Data**: Look for additional data sources for 2018-2022
3. **Model Improvements**: Fine-tune with more 2025 data as it becomes available
4. **Feature Engineering**: Add more sophisticated features based on new data patterns

## System Requirements

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, tensorflow, xgboost, lightgbm
- API access to OpenF1 (no key required)

## Troubleshooting

### No 2025 Data Found:
- **Expected**: 2025 season hasn't started yet
- **Solution**: Wait for race season to begin (March 2025)

### Limited 2024 Data:
- **Issue**: OpenF1 API might not have complete 2024 data
- **Solution**: Check API documentation for data availability

### Model Performance:
- **Current**: Trained on 2023 data only
- **Improvement**: Performance will improve with more diverse year coverage

## Contact

For questions or issues with the 2025 update, check:
1. OpenF1 API status: https://api.openf1.org/v1/status
2. Data collection logs in the terminal output
3. Dataset summary in `data/pre_race_features.csv`

---

*Last updated: July 2025*
*System ready for 2025 F1 season data integration*
