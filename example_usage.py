#!/usr/bin/env python3
"""
Example Usage of Advanced F1 Prediction System
Demonstrates how to use the 3-phase prediction system
"""

import pandas as pd
import numpy as np
from src.master_pipeline import MasterF1PredictionPipeline

def main():
    """
    Example usage of the Advanced F1 Prediction System
    """
    print("🏁 Advanced F1 Prediction System - Example Usage")
    print("=" * 60)
    
    # Initialize the master pipeline
    print("\n1. Initializing Master Pipeline...")
    pipeline = MasterF1PredictionPipeline()
    
    # Check system status
    print("\n2. Checking System Status...")
    status = pipeline.get_system_status()
    
    print("System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Run complete pipeline if not already done
    if not status['system_ready']:
        print("\n3. Running Complete Pipeline...")
        print("This will:")
        print("   - Clean and integrate data from FastF1, OpenF1, and historical sources")
        print("   - Analyze patterns and calculate probabilities")
        print("   - Train ensemble ML models")
        print("   - Validate system accuracy")
        
        results = pipeline.run_complete_pipeline(
            years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            retrain_models=True,
            validate_system=True
        )
        
        if results['success']:
            print("✅ Pipeline completed successfully!")
        else:
            print("❌ Pipeline failed. Check errors above.")
            return
    else:
        print("\n3. System already ready - skipping pipeline setup")
    
    # Example 1: Predict Spa 2025 race
    print("\n4. Example Prediction: Spa-Francorchamps 2025")
    print("-" * 50)
    
    # Create example lineup data (in practice, this would come from FastF1 or other sources)
    example_lineup = pd.DataFrame({
        'driver_name': [
            'Lewis Hamilton', 'George Russell', 'Max Verstappen', 'Sergio Perez',
            'Charles Leclerc', 'Carlos Sainz', 'Lando Norris', 'Oscar Piastri',
            'Fernando Alonso', 'Lance Stroll', 'Pierre Gasly', 'Esteban Ocon',
            'Valtteri Bottas', 'Zhou Guanyu', 'Kevin Magnussen', 'Nico Hulkenberg',
            'Yuki Tsunoda', 'Daniel Ricciardo', 'Alex Albon', 'Logan Sargeant'
        ],
        'team_name': [
            'Mercedes', 'Mercedes', 'Red Bull', 'Red Bull',
            'Ferrari', 'Ferrari', 'McLaren', 'McLaren',
            'Aston Martin', 'Aston Martin', 'Alpine', 'Alpine',
            'Sauber', 'Sauber', 'Haas', 'Haas',
            'AlphaTauri', 'AlphaTauri', 'Williams', 'Williams'
        ],
        'circuit': 'Spa-Francorchamps',
        'year': 2025
    })
    
    # Make prediction
    try:
        predictions = pipeline.predict_future_race(
            circuit="Spa-Francorchamps",
            year=2025,
            lineup_data=example_lineup
        )
        
        if not predictions.empty:
            print("\n🏆 Top 10 Predictions for Spa 2025:")
            print("-" * 40)
            
            for i, (_, row) in enumerate(predictions.head(10).iterrows()):
                driver = row['driver_name']
                team = row['team_name']
                prob = row.get('ensemble_top5_prob', 0)
                print(f"{i+1:2d}. {driver} ({team}): {prob:.3f}")
        else:
            print("⚠️ No predictions generated - check data availability")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("This is expected if the system hasn't been fully trained yet")
    
    # Example 2: Show system capabilities
    print("\n5. System Capabilities")
    print("-" * 30)
    
    print("✅ Data Sources:")
    print("   - FastF1 API (real-time telemetry, weather)")
    print("   - OpenF1 API (race data, car telemetry)")
    print("   - Historical F1 database (comprehensive records)")
    
    print("\n✅ ML Models:")
    print("   - Random Forest (200 estimators)")
    print("   - XGBoost (gradient boosting)")
    print("   - LightGBM (fast gradient boosting)")
    print("   - CatBoost (categorical features)")
    print("   - Neural Networks (deep learning)")
    print("   - Support Vector Machines")
    print("   - Logistic Regression")
    print("   - Ensemble (voting classifier)")
    
    print("\n✅ Features:")
    print("   - Driver performance patterns")
    print("   - Team development trends")
    print("   - Circuit characteristics")
    print("   - Weather impact analysis")
    print("   - Strategic factors")
    print("   - Temporal trends")
    
    print("\n✅ Validation:")
    print("   - Position-specific accuracy")
    print("   - Driver/team/circuit analysis")
    print("   - Temporal validation")
    print("   - Confidence calibration")
    print("   - Feature importance")
    
    # Example 3: Show how to use individual components
    print("\n6. Using Individual Components")
    print("-" * 40)
    
    print("You can also use individual components:")
    print("\n📊 Data Pipeline:")
    print("   from src.comprehensive_data_pipeline import ComprehensiveF1DataPipeline")
    print("   pipeline = ComprehensiveF1DataPipeline()")
    print("   data = pipeline.integrate_all_data_sources()")
    
    print("\n🧠 Pattern Recognition:")
    print("   from src.pattern_recognition_engine import F1PatternRecognitionEngine")
    print("   engine = F1PatternRecognitionEngine()")
    print("   patterns = engine.analyze_driver_patterns()")
    
    print("\n🎯 Prediction System:")
    print("   from src.advanced_prediction_system import AdvancedF1PredictionSystem")
    print("   system = AdvancedF1PredictionSystem()")
    print("   predictions = system.predict_future_race(race_data)")
    
    print("\n🔍 Validation:")
    print("   from src.validation_system import F1PredictionValidationSystem")
    print("   validator = F1PredictionValidationSystem()")
    print("   results = validator.comprehensive_validation(test_data, predictions)")
    
    print("\n🎉 Example Complete!")
    print("=" * 60)
    print("The system is now ready for F1 race predictions!")
    print("Use pipeline.predict_future_race() to make predictions for any circuit.")

if __name__ == "__main__":
    main()
