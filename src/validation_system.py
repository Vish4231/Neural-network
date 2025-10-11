#!/usr/bin/env python3
"""
F1 Prediction Validation System
Comprehensive validation and accuracy assessment for F1 race predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
import os

class F1PredictionValidationSystem:
    def __init__(self, model_dir: str = 'model/advanced'):
        self.model_dir = model_dir
        self.validation_results = {}
        self.accuracy_metrics = {}
        self.prediction_history = []
        
        print("🔍 F1 Prediction Validation System Initialized")
    
    def comprehensive_validation(self, test_data: pd.DataFrame, 
                               predictions: pd.DataFrame) -> Dict:
        """
        Perform comprehensive validation of predictions
        """
        print("🔍 Running comprehensive validation...")
        
        validation_results = {}
        
        # 1. Basic accuracy metrics
        validation_results['basic_metrics'] = self._calculate_basic_metrics(test_data, predictions)
        
        # 2. Position-specific accuracy
        validation_results['position_accuracy'] = self._calculate_position_accuracy(test_data, predictions)
        
        # 3. Driver-specific accuracy
        validation_results['driver_accuracy'] = self._calculate_driver_accuracy(test_data, predictions)
        
        # 4. Team-specific accuracy
        validation_results['team_accuracy'] = self._calculate_team_accuracy(test_data, predictions)
        
        # 5. Circuit-specific accuracy
        validation_results['circuit_accuracy'] = self._calculate_circuit_accuracy(test_data, predictions)
        
        # 6. Temporal validation
        validation_results['temporal_validation'] = self._temporal_validation(test_data, predictions)
        
        # 7. Confidence calibration
        validation_results['confidence_calibration'] = self._confidence_calibration(test_data, predictions)
        
        # 8. Feature importance validation
        validation_results['feature_validation'] = self._validate_feature_importance()
        
        self.validation_results = validation_results
        
        print("✅ Comprehensive validation complete")
        return validation_results
    
    def _calculate_basic_metrics(self, test_data: pd.DataFrame, 
                                predictions: pd.DataFrame) -> Dict:
        """Calculate basic accuracy metrics"""
        print("  Calculating basic metrics...")
        
        # Merge test data with predictions
        merged = test_data.merge(predictions, on=['driver_name', 'team_name'], how='inner')
        
        if merged.empty:
            return {'error': 'No matching data for validation'}
        
        # Actual vs predicted
        y_true = (merged['finishing_position'] <= 5).astype(int)
        y_pred = (merged['ensemble_top5_prob'] > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, merged['ensemble_top5_prob']),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def _calculate_position_accuracy(self, test_data: pd.DataFrame, 
                                   predictions: pd.DataFrame) -> Dict:
        """Calculate accuracy for different finishing positions"""
        print("  Calculating position-specific accuracy...")
        
        merged = test_data.merge(predictions, on=['driver_name', 'team_name'], how='inner')
        
        if merged.empty:
            return {'error': 'No matching data for validation'}
        
        position_accuracy = {}
        
        # Top 5 accuracy
        top5_true = (merged['finishing_position'] <= 5).astype(int)
        top5_pred = (merged['ensemble_top5_prob'] > 0.5).astype(int)
        position_accuracy['top5'] = accuracy_score(top5_true, top5_pred)
        
        # Podium accuracy
        podium_true = (merged['finishing_position'] <= 3).astype(int)
        podium_pred = (merged['ensemble_top5_prob'] > 0.7).astype(int)  # Higher threshold for podium
        position_accuracy['podium'] = accuracy_score(podium_true, podium_pred)
        
        # Points accuracy
        points_true = (merged['finishing_position'] <= 10).astype(int)
        points_pred = (merged['ensemble_top5_prob'] > 0.3).astype(int)  # Lower threshold for points
        position_accuracy['points'] = accuracy_score(points_true, points_pred)
        
        # Position ranking accuracy
        position_accuracy['ranking_correlation'] = self._calculate_ranking_correlation(merged)
        
        return position_accuracy
    
    def _calculate_driver_accuracy(self, test_data: pd.DataFrame, 
                                 predictions: pd.DataFrame) -> Dict:
        """Calculate accuracy for individual drivers"""
        print("  Calculating driver-specific accuracy...")
        
        merged = test_data.merge(predictions, on=['driver_name', 'team_name'], how='inner')
        
        if merged.empty:
            return {'error': 'No matching data for validation'}
        
        driver_accuracy = {}
        
        for driver in merged['driver_name'].unique():
            driver_data = merged[merged['driver_name'] == driver]
            
            if len(driver_data) >= 3:  # Minimum races for driver analysis
                y_true = (driver_data['finishing_position'] <= 5).astype(int)
                y_pred = (driver_data['ensemble_top5_prob'] > 0.5).astype(int)
                
                accuracy = accuracy_score(y_true, y_pred)
                driver_accuracy[driver] = {
                    'accuracy': accuracy,
                    'races': len(driver_data),
                    'avg_probability': driver_data['ensemble_top5_prob'].mean(),
                    'actual_top5_rate': y_true.mean()
                }
        
        return driver_accuracy
    
    def _calculate_team_accuracy(self, test_data: pd.DataFrame, 
                               predictions: pd.DataFrame) -> Dict:
        """Calculate accuracy for individual teams"""
        print("  Calculating team-specific accuracy...")
        
        merged = test_data.merge(predictions, on=['driver_name', 'team_name'], how='inner')
        
        if merged.empty:
            return {'error': 'No matching data for validation'}
        
        team_accuracy = {}
        
        for team in merged['team_name'].unique():
            team_data = merged[merged['team_name'] == team]
            
            if len(team_data) >= 5:  # Minimum races for team analysis
                y_true = (team_data['finishing_position'] <= 5).astype(int)
                y_pred = (team_data['ensemble_top5_prob'] > 0.5).astype(int)
                
                accuracy = accuracy_score(y_true, y_pred)
                team_accuracy[team] = {
                    'accuracy': accuracy,
                    'races': len(team_data),
                    'avg_probability': team_data['ensemble_top5_prob'].mean(),
                    'actual_top5_rate': y_true.mean()
                }
        
        return team_accuracy
    
    def _calculate_circuit_accuracy(self, test_data: pd.DataFrame, 
                                  predictions: pd.DataFrame) -> Dict:
        """Calculate accuracy for different circuits"""
        print("  Calculating circuit-specific accuracy...")
        
        merged = test_data.merge(predictions, on=['driver_name', 'team_name'], how='inner')
        
        if merged.empty:
            return {'error': 'No matching data for validation'}
        
        circuit_accuracy = {}
        
        for circuit in merged['circuit'].unique():
            circuit_data = merged[merged['circuit'] == circuit]
            
            if len(circuit_data) >= 5:  # Minimum races for circuit analysis
                y_true = (circuit_data['finishing_position'] <= 5).astype(int)
                y_pred = (circuit_data['ensemble_top5_prob'] > 0.5).astype(int)
                
                accuracy = accuracy_score(y_true, y_pred)
                circuit_accuracy[circuit] = {
                    'accuracy': accuracy,
                    'races': len(circuit_data),
                    'avg_probability': circuit_data['ensemble_top5_prob'].mean(),
                    'actual_top5_rate': y_true.mean()
                }
        
        return circuit_accuracy
    
    def _temporal_validation(self, test_data: pd.DataFrame, 
                           predictions: pd.DataFrame) -> Dict:
        """Validate predictions over time"""
        print("  Performing temporal validation...")
        
        merged = test_data.merge(predictions, on=['driver_name', 'team_name'], how='inner')
        
        if merged.empty or 'year' not in merged.columns:
            return {'error': 'No temporal data available for validation'}
        
        temporal_results = {}
        
        # Yearly accuracy
        yearly_accuracy = {}
        for year in merged['year'].unique():
            year_data = merged[merged['year'] == year]
            
            if len(year_data) >= 10:  # Minimum races for year analysis
                y_true = (year_data['finishing_position'] <= 5).astype(int)
                y_pred = (year_data['ensemble_top5_prob'] > 0.5).astype(int)
                
                accuracy = accuracy_score(y_true, y_pred)
                yearly_accuracy[year] = {
                    'accuracy': accuracy,
                    'races': len(year_data),
                    'avg_probability': year_data['ensemble_top5_prob'].mean()
                }
        
        temporal_results['yearly_accuracy'] = yearly_accuracy
        
        # Trend analysis
        if len(yearly_accuracy) >= 3:
            years = list(yearly_accuracy.keys())
            accuracies = [yearly_accuracy[year]['accuracy'] for year in years]
            
            # Calculate trend
            trend_slope = np.polyfit(years, accuracies, 1)[0]
            temporal_results['accuracy_trend'] = trend_slope
            temporal_results['trend_interpretation'] = 'improving' if trend_slope > 0 else 'declining'
        
        return temporal_results
    
    def _confidence_calibration(self, test_data: pd.DataFrame, 
                              predictions: pd.DataFrame) -> Dict:
        """Validate confidence calibration"""
        print("  Validating confidence calibration...")
        
        merged = test_data.merge(predictions, on=['driver_name', 'team_name'], how='inner')
        
        if merged.empty:
            return {'error': 'No matching data for validation'}
        
        # Bin predictions by confidence level
        confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        calibration_results = {}
        
        for i in range(len(confidence_bins) - 1):
            lower = confidence_bins[i]
            upper = confidence_bins[i + 1]
            
            bin_data = merged[
                (merged['ensemble_top5_prob'] >= lower) & 
                (merged['ensemble_top5_prob'] < upper)
            ]
            
            if len(bin_data) > 0:
                predicted_prob = bin_data['ensemble_top5_prob'].mean()
                actual_rate = (bin_data['finishing_position'] <= 5).mean()
                
                calibration_results[f'bin_{i+1}'] = {
                    'predicted_probability': predicted_prob,
                    'actual_rate': actual_rate,
                    'calibration_error': abs(predicted_prob - actual_rate),
                    'sample_size': len(bin_data)
                }
        
        # Overall calibration error
        overall_error = np.mean([result['calibration_error'] for result in calibration_results.values()])
        calibration_results['overall_calibration_error'] = overall_error
        
        return calibration_results
    
    def _validate_feature_importance(self) -> Dict:
        """Validate feature importance"""
        print("  Validating feature importance...")
        
        try:
            # Load feature importance
            importance_file = f'{self.model_dir}/feature_importance.pkl'
            if os.path.exists(importance_file):
                feature_importance = joblib.load(importance_file)
                
                validation_results = {
                    'feature_importance_loaded': True,
                    'top_features': self._get_top_features(feature_importance),
                    'feature_stability': self._assess_feature_stability(feature_importance)
                }
            else:
                validation_results = {'feature_importance_loaded': False}
                
        except Exception as e:
            validation_results = {'error': str(e)}
        
        return validation_results
    
    def _calculate_ranking_correlation(self, merged_data: pd.DataFrame) -> float:
        """Calculate correlation between predicted and actual rankings"""
        try:
            # Sort by actual finishing position
            actual_ranking = merged_data.sort_values('finishing_position')['driver_name'].tolist()
            
            # Sort by predicted probability
            predicted_ranking = merged_data.sort_values('ensemble_top5_prob', ascending=False)['driver_name'].tolist()
            
            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            
            # Create ranking positions
            actual_positions = [actual_ranking.index(driver) for driver in predicted_ranking]
            predicted_positions = list(range(len(predicted_ranking)))
            
            correlation, _ = spearmanr(actual_positions, predicted_positions)
            return correlation
            
        except Exception:
            return 0.0
    
    def _get_top_features(self, feature_importance: Dict) -> List[Tuple[str, float]]:
        """Get top features by importance"""
        if 'average' in feature_importance:
            avg_importance = feature_importance['average']
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            return sorted_features[:10]
        return []
    
    def _assess_feature_stability(self, feature_importance: Dict) -> Dict:
        """Assess stability of feature importance across models"""
        if len(feature_importance) < 2:
            return {'stability_score': 0.0}
        
        # Calculate variance in feature importance across models
        feature_variance = {}
        
        for feature in feature_importance.get('average', {}).keys():
            importance_values = []
            for model_name, importance_dict in feature_importance.items():
                if model_name != 'average' and feature in importance_dict:
                    importance_values.append(importance_dict[feature])
            
            if len(importance_values) > 1:
                feature_variance[feature] = np.var(importance_values)
        
        if feature_variance:
            avg_variance = np.mean(list(feature_variance.values()))
            stability_score = 1 / (1 + avg_variance)  # Higher = more stable
        else:
            stability_score = 0.0
        
        return {
            'stability_score': stability_score,
            'feature_variance': feature_variance
        }
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return "❌ No validation results available. Please run validation first."
        
        report = []
        report.append("📊 F1 Prediction Validation Report")
        report.append("=" * 50)
        
        # Basic metrics
        if 'basic_metrics' in self.validation_results:
            metrics = self.validation_results['basic_metrics']
            report.append("\n🎯 Basic Performance Metrics:")
            report.append("-" * 30)
            report.append(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
            report.append(f"Precision: {metrics.get('precision', 0):.3f}")
            report.append(f"Recall: {metrics.get('recall', 0):.3f}")
            report.append(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
            report.append(f"ROC AUC: {metrics.get('roc_auc', 0):.3f}")
        
        # Position accuracy
        if 'position_accuracy' in self.validation_results:
            pos_acc = self.validation_results['position_accuracy']
            report.append("\n🏁 Position-Specific Accuracy:")
            report.append("-" * 30)
            report.append(f"Top 5: {pos_acc.get('top5', 0):.3f}")
            report.append(f"Podium: {pos_acc.get('podium', 0):.3f}")
            report.append(f"Points: {pos_acc.get('points', 0):.3f}")
            report.append(f"Ranking Correlation: {pos_acc.get('ranking_correlation', 0):.3f}")
        
        # Driver accuracy
        if 'driver_accuracy' in self.validation_results:
            driver_acc = self.validation_results['driver_accuracy']
            report.append("\n👤 Driver-Specific Accuracy:")
            report.append("-" * 30)
            
            # Sort drivers by accuracy
            sorted_drivers = sorted(driver_acc.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            
            for driver, stats in sorted_drivers[:5]:
                report.append(f"{driver}: {stats['accuracy']:.3f} ({stats['races']} races)")
        
        # Team accuracy
        if 'team_accuracy' in self.validation_results:
            team_acc = self.validation_results['team_accuracy']
            report.append("\n🏎️ Team-Specific Accuracy:")
            report.append("-" * 30)
            
            # Sort teams by accuracy
            sorted_teams = sorted(team_acc.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            
            for team, stats in sorted_teams:
                report.append(f"{team}: {stats['accuracy']:.3f} ({stats['races']} races)")
        
        # Circuit accuracy
        if 'circuit_accuracy' in self.validation_results:
            circuit_acc = self.validation_results['circuit_accuracy']
            report.append("\n🏁 Circuit-Specific Accuracy:")
            report.append("-" * 30)
            
            # Sort circuits by accuracy
            sorted_circuits = sorted(circuit_acc.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            
            for circuit, stats in sorted_circuits[:5]:
                report.append(f"{circuit}: {stats['accuracy']:.3f} ({stats['races']} races)")
        
        # Temporal validation
        if 'temporal_validation' in self.validation_results:
            temp_val = self.validation_results['temporal_validation']
            report.append("\n📅 Temporal Validation:")
            report.append("-" * 30)
            
            if 'yearly_accuracy' in temp_val:
                yearly_acc = temp_val['yearly_accuracy']
                for year, stats in sorted(yearly_acc.items()):
                    report.append(f"{year}: {stats['accuracy']:.3f} ({stats['races']} races)")
            
            if 'accuracy_trend' in temp_val:
                trend = temp_val['accuracy_trend']
                interpretation = temp_val.get('trend_interpretation', 'stable')
                report.append(f"Accuracy Trend: {trend:.3f} ({interpretation})")
        
        # Confidence calibration
        if 'confidence_calibration' in self.validation_results:
            calib = self.validation_results['confidence_calibration']
            report.append("\n🎯 Confidence Calibration:")
            report.append("-" * 30)
            
            if 'overall_calibration_error' in calib:
                report.append(f"Overall Calibration Error: {calib['overall_calibration_error']:.3f}")
            
            for bin_name, stats in calib.items():
                if bin_name.startswith('bin_'):
                    report.append(f"{bin_name}: Pred={stats['predicted_probability']:.3f}, "
                               f"Actual={stats['actual_rate']:.3f}, "
                               f"Error={stats['calibration_error']:.3f}")
        
        # Feature validation
        if 'feature_validation' in self.validation_results:
            feat_val = self.validation_results['feature_validation']
            report.append("\n🔍 Feature Validation:")
            report.append("-" * 30)
            
            if feat_val.get('feature_importance_loaded', False):
                report.append("Feature importance loaded successfully")
                
                if 'top_features' in feat_val:
                    report.append("Top 5 Features:")
                    for i, (feature, importance) in enumerate(feat_val['top_features'][:5]):
                        report.append(f"  {i+1}. {feature}: {importance:.3f}")
                
                if 'feature_stability' in feat_val:
                    stability = feat_val['feature_stability']['stability_score']
                    report.append(f"Feature Stability Score: {stability:.3f}")
            else:
                report.append("Feature importance not available")
        
        # Overall assessment
        report.append("\n📈 Overall Assessment:")
        report.append("-" * 20)
        
        if 'basic_metrics' in self.validation_results:
            accuracy = self.validation_results['basic_metrics'].get('accuracy', 0)
            if accuracy >= 0.8:
                assessment = "Excellent"
            elif accuracy >= 0.7:
                assessment = "Good"
            elif accuracy >= 0.6:
                assessment = "Fair"
            else:
                assessment = "Needs Improvement"
            
            report.append(f"Prediction Quality: {assessment}")
            report.append(f"Overall Accuracy: {accuracy:.3f}")
        
        return "\n".join(report)
    
    def save_validation_results(self, output_path: str = 'data/validation_results.pkl'):
        """Save validation results"""
        print(f"💾 Saving validation results to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.validation_results, output_path)
        
        print("✅ Validation results saved")
    
    def load_validation_results(self, input_path: str = 'data/validation_results.pkl'):
        """Load validation results"""
        print(f"📂 Loading validation results from {input_path}")
        
        try:
            self.validation_results = joblib.load(input_path)
            print("✅ Validation results loaded")
        except FileNotFoundError:
            print("❌ Validation results file not found")
        except Exception as e:
            print(f"❌ Error loading validation results: {e}")

def main():
    """Main function to run validation system"""
    print("🔍 Starting F1 Prediction Validation System")
    print("=" * 50)
    
    # Initialize validation system
    validator = F1PredictionValidationSystem()
    
    # Example usage (would need actual test data and predictions)
    print("📊 Validation system ready!")
    print("\nTo use the validation system:")
    print("1. Load test data and predictions")
    print("2. Call validator.comprehensive_validation(test_data, predictions)")
    print("3. Generate validation report with validator.generate_validation_report()")
    print("4. Save results with validator.save_validation_results()")

if __name__ == "__main__":
    main()
