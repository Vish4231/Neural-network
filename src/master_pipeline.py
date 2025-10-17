#!/usr/bin/env python3
"""
Master F1 Prediction Pipeline
Orchestrates the complete 3-phase F1 prediction system for maximum accuracy
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from .comprehensive_data_pipeline import ComprehensiveF1DataPipeline
from .pattern_recognition_engine import F1PatternRecognitionEngine
from .advanced_prediction_system import AdvancedF1PredictionSystem
from .validation_system import F1PredictionValidationSystem

class MasterF1PredictionPipeline:
    def __init__(self, 
                 cache_dir: str = 'fastf1_cache',
                 data_dir: str = 'data',
                 model_dir: str = 'model/advanced'):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Initialize components
        self.data_pipeline = ComprehensiveF1DataPipeline(cache_dir, data_dir)
        self.pattern_engine = F1PatternRecognitionEngine()
        self.prediction_system = AdvancedF1PredictionSystem()
        self.validation_system = F1PredictionValidationSystem(model_dir)
        
        # Pipeline state
        self.pipeline_state = {
            'data_cleaned': False,
            'patterns_analyzed': False,
            'models_trained': False,
            'system_validated': False
        }
        
        print("🚀 Master F1 Prediction Pipeline Initialized")
        print("=" * 60)
    
    def run_complete_pipeline(self, 
                            years: List[int] = None,
                            retrain_models: bool = False,
                            validate_system: bool = True) -> Dict:
        """
        Run the complete 3-phase F1 prediction pipeline
        """
        print("🏁 Starting Complete F1 Prediction Pipeline")
        print("=" * 60)
        
        pipeline_results = {
            'start_time': datetime.now(),
            'phases_completed': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Phase 1: Data Cleaning & Integration
            print("\n📊 PHASE 1: Data Cleaning & Integration")
            print("-" * 40)
            
            integrated_data = self.data_pipeline.integrate_all_data_sources()
            
            if integrated_data.empty:
                error_msg = "Phase 1 failed: No data available"
                pipeline_results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
                return pipeline_results
            
            # Validate data quality
            validation_report = self.data_pipeline.validate_data_quality(integrated_data)
            
            self.pipeline_state['data_cleaned'] = True
            pipeline_results['phases_completed'].append('data_cleaning')
            pipeline_results['data_validation'] = validation_report
            
            print("✅ Phase 1 Complete: Data cleaned and integrated")
            
            # Phase 2: Pattern Recognition
            print("\n🧠 PHASE 2: Pattern Recognition & Analysis")
            print("-" * 40)
            
            # Update pattern engine with integrated data
            self.pattern_engine.data = integrated_data
            
            # Run pattern analysis
            driver_patterns = self.pattern_engine.analyze_driver_patterns()
            team_patterns = self.pattern_engine.analyze_team_patterns()
            circuit_patterns = self.pattern_engine.analyze_circuit_patterns()
            
            # Calculate probabilities
            probabilities = self.pattern_engine.calculate_position_probabilities()
            
            # Identify trends
            trends = self.pattern_engine.identify_trends()
            
            # Perform clustering
            clusters = self.pattern_engine.cluster_analysis()
            
            # Generate prediction features
            prediction_features = self.pattern_engine.generate_prediction_features()
            
            if prediction_features.empty:
                error_msg = "Phase 2 failed: No prediction features generated"
                pipeline_results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
                return pipeline_results
            
            # Save prediction features
            features_path = os.path.join(self.data_dir, 'pattern_analysis', 'prediction_features.csv')
            os.makedirs(os.path.dirname(features_path), exist_ok=True)
            prediction_features.to_csv(features_path, index=False)
            
            self.pipeline_state['patterns_analyzed'] = True
            pipeline_results['phases_completed'].append('pattern_recognition')
            pipeline_results['pattern_analysis'] = {
                'driver_patterns': len(driver_patterns),
                'team_patterns': len(team_patterns),
                'circuit_patterns': len(circuit_patterns),
                'prediction_features': len(prediction_features)
            }
            
            print("✅ Phase 2 Complete: Patterns analyzed and features generated")
            
            # Phase 3: Prediction System
            print("\n🎯 PHASE 3: Advanced Prediction System")
            print("-" * 40)
            
            # Update prediction system with features
            self.prediction_system.data = prediction_features
            
            # Prepare training data
            X, y, feature_names = self.prediction_system.prepare_training_data()
            
            if X.size == 0:
                error_msg = "Phase 3 failed: No training data available"
                pipeline_results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
                return pipeline_results
            
            # Train models
            models = self.prediction_system.train_ensemble_models(X, y, feature_names)
            
            # Save models
            self.prediction_system.save_models(self.model_dir)
            
            self.pipeline_state['models_trained'] = True
            pipeline_results['phases_completed'].append('model_training')
            pipeline_results['model_performance'] = self.prediction_system.prediction_accuracy
            
            print("✅ Phase 3 Complete: Models trained and saved")
            
            # Validation Phase
            if validate_system:
                print("\n🔍 VALIDATION PHASE: System Accuracy Assessment")
                print("-" * 40)
                
                # Validate prediction accuracy
                validation_results = self.prediction_system.validate_prediction_accuracy(prediction_features)
                
                # Comprehensive validation
                comprehensive_validation = self.validation_system.comprehensive_validation(
                    prediction_features, 
                    self.prediction_system.predict_future_race(prediction_features)
                )
                
                self.pipeline_state['system_validated'] = True
                pipeline_results['phases_completed'].append('system_validation')
                pipeline_results['validation_results'] = comprehensive_validation
                
                print("✅ Validation Complete: System accuracy assessed")
            
            # Generate final report
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration'] = pipeline_results['end_time'] - pipeline_results['start_time']
            pipeline_results['success'] = True
            
            print("\n🎉 PIPELINE COMPLETE!")
            print("=" * 60)
            self._print_pipeline_summary(pipeline_results)
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            pipeline_results['errors'].append(error_msg)
            pipeline_results['success'] = False
            print(f"❌ {error_msg}")
        
        return pipeline_results
    
    def predict_future_race(self, 
                          circuit: str,
                          year: int = 2025,
                          lineup_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict future race outcomes
        """
        print(f"🔮 Predicting {circuit} {year} race outcomes...")
        
        # Check if system is ready
        if not self.pipeline_state['models_trained']:
            print("❌ Models not trained. Please run complete pipeline first.")
            return pd.DataFrame()
        
        # Load models if not already loaded
        if not self.prediction_system.models:
            self.prediction_system.load_models(self.model_dir)
        
        # Prepare race data
        if lineup_data is None:
            lineup_data = self._generate_default_lineup(circuit, year)
        
        if lineup_data.empty:
            print("❌ No lineup data available")
            return pd.DataFrame()
        
        # Generate features for prediction
        race_features = self._prepare_race_features(lineup_data, circuit)
        
        if race_features.empty:
            print("❌ Could not generate race features")
            return pd.DataFrame()
        
        # Make predictions
        predictions = self.prediction_system.predict_future_race(race_features)
        
        # Generate prediction report
        report = self.prediction_system.generate_prediction_report(predictions)
        print("\n" + report)
        
        return predictions
    
    def _generate_default_lineup(self, circuit: str, year: int) -> pd.DataFrame:
        """Generate default lineup for prediction"""
        # This would typically fetch from FastF1 or use known lineup
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _prepare_race_features(self, lineup_data: pd.DataFrame, circuit: str) -> pd.DataFrame:
        """Prepare race features for prediction"""
        # This would use the pattern recognition engine to generate features
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _print_pipeline_summary(self, results: Dict):
        """Print pipeline summary"""
        print(f"⏱️  Duration: {results['duration']}")
        print(f"📊 Phases Completed: {len(results['phases_completed'])}/4")
        
        if results['phases_completed']:
            print("✅ Completed Phases:")
            for phase in results['phases_completed']:
                print(f"   - {phase.replace('_', ' ').title()}")
        
        if results.get('errors'):
            print("❌ Errors:")
            for error in results['errors']:
                print(f"   - {error}")
        
        if results.get('model_performance'):
            print("🎯 Model Performance:")
            for model, accuracy in results['model_performance'].items():
                print(f"   - {model}: {accuracy:.3f}")
        
        if results.get('validation_results'):
            print("🔍 Validation Results:")
            basic_metrics = results['validation_results'].get('basic_metrics', {})
            if basic_metrics:
                print(f"   - Overall Accuracy: {basic_metrics.get('accuracy', 0):.3f}")
                print(f"   - F1 Score: {basic_metrics.get('f1_score', 0):.3f}")
                print(f"   - ROC AUC: {basic_metrics.get('roc_auc', 0):.3f}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'pipeline_state': self.pipeline_state,
            'data_available': os.path.exists(os.path.join(self.data_dir, 'integrated_f1_data.csv')),
            'features_available': os.path.exists(os.path.join(self.data_dir, 'pattern_analysis', 'prediction_features.csv')),
            'models_available': os.path.exists(self.model_dir),
            'system_ready': all(self.pipeline_state.values())
        }
    
    def reset_pipeline(self):
        """Reset pipeline state"""
        self.pipeline_state = {
            'data_cleaned': False,
            'patterns_analyzed': False,
            'models_trained': False,
            'system_validated': False
        }
        print("🔄 Pipeline state reset")

def main():
    """Main function to run the master pipeline"""
    print("🚀 Starting Master F1 Prediction Pipeline")
    print("=" * 60)
    
    # Initialize master pipeline
    pipeline = MasterF1PredictionPipeline()
    
    # Check system status
    status = pipeline.get_system_status()
    print("📊 System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        retrain_models=True,
        validate_system=True
    )
    
    if results['success']:
        print("\n🎉 Master Pipeline Complete!")
        print("System ready for future race predictions!")
        
        # Example prediction (would need actual lineup data)
        print("\n🔮 Example prediction usage:")
        print("predictions = pipeline.predict_future_race('Spa-Francorchamps', 2025)")
        
    else:
        print("\n❌ Pipeline failed. Check errors above.")

if __name__ == "__main__":
    main()
