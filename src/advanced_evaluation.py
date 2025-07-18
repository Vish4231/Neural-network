#!/usr/bin/env python3
"""
Advanced Evaluation Framework for F1 Predictions
Comprehensive metrics and validation for model accuracy
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class F1PredictionEvaluator:
    
    def __init__(self):
        self.results = {}
        self.metrics = {}
        
    def evaluate_predictions(self, predictions, actuals, race_info):
        """Comprehensive evaluation of F1 predictions"""
        
        print("🔍 Evaluating F1 Predictions...")
        
        # Basic classification metrics
        basic_metrics = self.calculate_basic_metrics(predictions, actuals)
        
        # F1-specific metrics
        f1_metrics = self.calculate_f1_specific_metrics(predictions, actuals, race_info)
        
        # Position-based metrics
        position_metrics = self.calculate_position_metrics(predictions, actuals)
        
        # Confidence calibration
        calibration_metrics = self.calculate_calibration_metrics(predictions, actuals)
        
        # Combine all metrics
        all_metrics = {
            'basic': basic_metrics,
            'f1_specific': f1_metrics,
            'position': position_metrics,
            'calibration': calibration_metrics
        }
        
        return all_metrics
    
    def calculate_basic_metrics(self, predictions, actuals):
        """Calculate basic classification metrics"""
        
        # Convert probabilities to binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(actuals, binary_preds),
            'precision': precision_score(actuals, binary_preds, average='weighted'),
            'recall': recall_score(actuals, binary_preds, average='weighted'),
            'f1_score': f1_score(actuals, binary_preds, average='weighted'),
            'auc_roc': roc_auc_score(actuals, predictions)
        }
        
        return metrics
    
    def calculate_f1_specific_metrics(self, predictions, actuals, race_info):
        """Calculate F1-specific prediction metrics"""
        
        metrics = {}
        
        # Top 5 accuracy (how often we correctly predict top 5)
        top5_accuracy = self.calculate_top5_accuracy(predictions, actuals)
        
        # Podium accuracy (top 3)
        podium_accuracy = self.calculate_podium_accuracy(predictions, actuals)
        
        # Winner accuracy
        winner_accuracy = self.calculate_winner_accuracy(predictions, actuals)
        
        # Position delta (average difference between predicted and actual position)
        position_delta = self.calculate_position_delta(predictions, actuals)
        
        # Grid position vs prediction correlation
        grid_correlation = self.calculate_grid_correlation(predictions, race_info)
        
        metrics = {
            'top5_accuracy': top5_accuracy,
            'podium_accuracy': podium_accuracy,
            'winner_accuracy': winner_accuracy,
            'avg_position_delta': position_delta,
            'grid_correlation': grid_correlation
        }
        
        return metrics
    
    def calculate_position_metrics(self, predictions, actuals):
        """Calculate position-based metrics"""
        
        # Sort predictions and get predicted positions
        sorted_indices = np.argsort(predictions)[::-1]  # Descending order
        predicted_positions = np.argsort(sorted_indices) + 1
        
        # Get actual positions
        actual_positions = np.argsort(actuals)[::-1] + 1
        
        # Mean Absolute Error in positions
        mae_positions = np.mean(np.abs(predicted_positions - actual_positions))
        
        # Root Mean Square Error in positions
        rmse_positions = np.sqrt(np.mean((predicted_positions - actual_positions) ** 2))
        
        # Spearman correlation (rank correlation)
        spearman_corr = np.corrcoef(predicted_positions, actual_positions)[0, 1]
        
        return {
            'mae_positions': mae_positions,
            'rmse_positions': rmse_positions,
            'spearman_correlation': spearman_corr
        }
    
    def calculate_calibration_metrics(self, predictions, actuals):
        """Calculate prediction calibration metrics"""
        
        # Reliability diagram data
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = actuals[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                
                calibration_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'prop_in_bin': prop_in_bin
                })
        
        # Expected Calibration Error (ECE)
        ece = sum([
            item['prop_in_bin'] * abs(item['accuracy'] - item['confidence'])
            for item in calibration_data
        ])
        
        return {
            'calibration_data': calibration_data,
            'expected_calibration_error': ece
        }
    
    def calculate_top5_accuracy(self, predictions, actuals):
        """Calculate top 5 prediction accuracy"""
        # Get top 5 predicted drivers
        top5_predicted = np.argsort(predictions)[-5:]
        
        # Get actual top 5 drivers
        top5_actual = np.argsort(actuals)[-5:]
        
        # Calculate intersection
        intersection = len(set(top5_predicted) & set(top5_actual))
        
        return intersection / 5.0
    
    def calculate_podium_accuracy(self, predictions, actuals):
        """Calculate podium (top 3) prediction accuracy"""
        # Get top 3 predicted drivers
        top3_predicted = np.argsort(predictions)[-3:]
        
        # Get actual top 3 drivers
        top3_actual = np.argsort(actuals)[-3:]
        
        # Calculate intersection
        intersection = len(set(top3_predicted) & set(top3_actual))
        
        return intersection / 3.0
    
    def calculate_winner_accuracy(self, predictions, actuals):
        """Calculate winner prediction accuracy"""
        predicted_winner = np.argmax(predictions)
        actual_winner = np.argmax(actuals)
        
        return int(predicted_winner == actual_winner)
    
    def calculate_position_delta(self, predictions, actuals):
        """Calculate average position difference"""
        # Convert to positions
        pred_positions = np.argsort(np.argsort(predictions)[::-1]) + 1
        actual_positions = np.argsort(np.argsort(actuals)[::-1]) + 1
        
        return np.mean(np.abs(pred_positions - actual_positions))
    
    def calculate_grid_correlation(self, predictions, race_info):
        """Calculate correlation between grid position and predictions"""
        if 'grid_position' in race_info:
            grid_positions = race_info['grid_position']
            return np.corrcoef(grid_positions, predictions)[0, 1]
        return 0.0
    
    def create_evaluation_report(self, metrics, race_name, save_path=None):
        """Create comprehensive evaluation report"""
        
        report = {
            'race_name': race_name,
            'evaluation_date': datetime.now().isoformat(),
            'metrics': metrics,
            'summary': self.create_summary(metrics)
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def create_summary(self, metrics):
        """Create summary of key metrics"""
        
        summary = {
            'overall_accuracy': metrics['basic']['accuracy'],
            'top5_accuracy': metrics['f1_specific']['top5_accuracy'],
            'podium_accuracy': metrics['f1_specific']['podium_accuracy'],
            'prediction_quality': self.assess_prediction_quality(metrics),
            'calibration_quality': self.assess_calibration_quality(metrics),
            'recommendations': self.generate_recommendations(metrics)
        }
        
        return summary
    
    def assess_prediction_quality(self, metrics):
        """Assess overall prediction quality"""
        
        # Weight different metrics
        weights = {
            'accuracy': 0.2,
            'top5_accuracy': 0.3,
            'podium_accuracy': 0.2,
            'auc_roc': 0.2,
            'spearman_correlation': 0.1
        }
        
        score = (
            weights['accuracy'] * metrics['basic']['accuracy'] +
            weights['top5_accuracy'] * metrics['f1_specific']['top5_accuracy'] +
            weights['podium_accuracy'] * metrics['f1_specific']['podium_accuracy'] +
            weights['auc_roc'] * metrics['basic']['auc_roc'] +
            weights['spearman_correlation'] * abs(metrics['position']['spearman_correlation'])
        )
        
        if score > 0.8:
            return "Excellent"
        elif score > 0.6:
            return "Good"
        elif score > 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def assess_calibration_quality(self, metrics):
        """Assess prediction calibration quality"""
        
        ece = metrics['calibration']['expected_calibration_error']
        
        if ece < 0.05:
            return "Well-calibrated"
        elif ece < 0.1:
            return "Reasonably calibrated"
        elif ece < 0.2:
            return "Poorly calibrated"
        else:
            return "Very poorly calibrated"
    
    def generate_recommendations(self, metrics):
        """Generate recommendations for improvement"""
        
        recommendations = []
        
        # Low accuracy
        if metrics['basic']['accuracy'] < 0.6:
            recommendations.append("Consider more feature engineering or data collection")
        
        # Poor calibration
        if metrics['calibration']['expected_calibration_error'] > 0.1:
            recommendations.append("Apply probability calibration techniques")
        
        # Low top 5 accuracy
        if metrics['f1_specific']['top5_accuracy'] < 0.6:
            recommendations.append("Focus on improving top 5 prediction accuracy")
        
        # Poor position correlation
        if abs(metrics['position']['spearman_correlation']) < 0.3:
            recommendations.append("Improve position ranking predictions")
        
        return recommendations
    
    def create_visualization(self, metrics, save_path=None):
        """Create visualization of evaluation results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Basic metrics bar chart
        basic_metrics = metrics['basic']
        axes[0, 0].bar(basic_metrics.keys(), basic_metrics.values())
        axes[0, 0].set_title('Basic Classification Metrics')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. F1-specific metrics
        f1_metrics = metrics['f1_specific']
        f1_keys = ['top5_accuracy', 'podium_accuracy', 'winner_accuracy']
        f1_values = [f1_metrics[k] for k in f1_keys]
        axes[0, 1].bar(f1_keys, f1_values)
        axes[0, 1].set_title('F1-Specific Metrics')
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Position metrics
        pos_metrics = metrics['position']
        axes[1, 0].bar(['MAE', 'RMSE'], [pos_metrics['mae_positions'], pos_metrics['rmse_positions']])
        axes[1, 0].set_title('Position Error Metrics')
        
        # 4. Calibration plot
        calib_data = metrics['calibration']['calibration_data']
        if calib_data:
            confidences = [item['confidence'] for item in calib_data]
            accuracies = [item['accuracy'] for item in calib_data]
            axes[1, 1].plot(confidences, accuracies, 'o-', label='Model')
            axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Calibration Plot')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def create_backtesting_framework(historical_data, model, features):
    """Create backtesting framework for historical validation"""
    
    print("🔄 Creating Backtesting Framework...")
    
    # Split data by season/race
    seasons = historical_data['year'].unique()
    results = []
    
    for season in seasons:
        season_data = historical_data[historical_data['year'] == season]
        races = season_data['race_round'].unique()
        
        for race in races:
            # Use data up to this race for training
            train_data = historical_data[
                (historical_data['year'] < season) | 
                ((historical_data['year'] == season) & (historical_data['race_round'] < race))
            ]
            
            # Test on this race
            test_data = season_data[season_data['race_round'] == race]
            
            if len(train_data) > 100 and len(test_data) > 0:
                # Train model on historical data
                X_train = train_data[features]
                y_train = train_data['target']
                
                X_test = test_data[features]
                y_test = test_data['target']
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Evaluate
                evaluator = F1PredictionEvaluator()
                race_info = {'grid_position': test_data['grid_position'].values}
                metrics = evaluator.evaluate_predictions(predictions, y_test, race_info)
                
                results.append({
                    'season': season,
                    'race': race,
                    'circuit': test_data['circuit'].iloc[0],
                    'metrics': metrics
                })
    
    return results

if __name__ == "__main__":
    # Example usage
    evaluator = F1PredictionEvaluator()
    print("F1 Prediction Evaluator Ready!")
