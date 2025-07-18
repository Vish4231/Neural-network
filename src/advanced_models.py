#!/usr/bin/env python3
"""
Advanced Neural Network Architectures for F1 Predictions
Implements sophisticated models for improved accuracy
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd

class AdvancedF1Models:
    
    def build_attention_model(self, input_shape, num_drivers=20):
        """Build model with attention mechanism for driver interactions"""
        
        # Driver features input
        driver_input = layers.Input(shape=input_shape, name='driver_features')
        
        # Create embeddings for categorical features
        x = layers.Dense(128, activation='relu')(driver_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Attention mechanism - drivers influence each other
        attention_weights = layers.Dense(64, activation='tanh')(x)
        attention_weights = layers.Dense(1, activation='softmax')(attention_weights)
        
        # Apply attention
        attended_features = layers.Multiply()([x, attention_weights])
        
        # Additional processing layers
        x = layers.Dense(64, activation='relu')(attended_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='top5_probability')(x)
        
        model = Model(inputs=driver_input, outputs=output)
        return model
    
    def build_lstm_model(self, sequence_length=5, feature_dim=30):
        """Build LSTM model for sequential race data"""
        
        # Time series input (last N races)
        sequence_input = layers.Input(shape=(sequence_length, feature_dim), name='race_sequence')
        
        # LSTM layers for temporal patterns
        lstm_out = layers.LSTM(64, return_sequences=True)(sequence_input)
        lstm_out = layers.LSTM(32, return_sequences=False)(lstm_out)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(lstm_out)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=sequence_input, outputs=output)
        return model
    
    def build_ensemble_model(self, input_shape):
        """Build ensemble model combining multiple architectures"""
        
        # Shared input
        main_input = layers.Input(shape=input_shape, name='main_input')
        
        # Branch 1: Deep network for main features
        branch1 = layers.Dense(128, activation='relu')(main_input)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Dropout(0.3)(branch1)
        branch1 = layers.Dense(64, activation='relu')(branch1)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Dropout(0.2)(branch1)
        
        # Branch 2: Wide network for feature interactions
        branch2 = layers.Dense(256, activation='relu')(main_input)
        branch2 = layers.Dropout(0.4)(branch2)
        branch2 = layers.Dense(128, activation='relu')(branch2)
        branch2 = layers.Dropout(0.3)(branch2)
        
        # Branch 3: Specialized for qualifying/grid features
        branch3 = layers.Dense(32, activation='relu')(main_input)
        branch3 = layers.Dense(16, activation='relu')(branch3)
        
        # Combine branches
        combined = layers.concatenate([branch1, branch2, branch3])
        
        # Final processing
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=main_input, outputs=output)
        return model
    
    def build_transformer_model(self, input_shape, num_drivers=20):
        """Build Transformer model for driver relationships"""
        
        inputs = layers.Input(shape=input_shape)
        
        # Reshape for transformer (treat features as sequence)
        x = layers.Reshape((input_shape[0], 1))(inputs)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(128, activation='relu')(x)
        ff_output = layers.Dense(input_shape[0])(ff_output)
        
        # Add & Norm
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Final layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    def build_multi_task_model(self, input_shape):
        """Build multi-task model predicting multiple outcomes"""
        
        inputs = layers.Input(shape=input_shape)
        
        # Shared layers
        shared = layers.Dense(128, activation='relu')(inputs)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        shared = layers.Dense(64, activation='relu')(shared)
        
        # Task 1: Top 5 probability
        top5_branch = layers.Dense(32, activation='relu')(shared)
        top5_output = layers.Dense(1, activation='sigmoid', name='top5_prob')(top5_branch)
        
        # Task 2: Podium probability
        podium_branch = layers.Dense(32, activation='relu')(shared)
        podium_output = layers.Dense(1, activation='sigmoid', name='podium_prob')(podium_branch)
        
        # Task 3: Points probability
        points_branch = layers.Dense(32, activation='relu')(shared)
        points_output = layers.Dense(1, activation='sigmoid', name='points_prob')(points_branch)
        
        # Task 4: DNF probability
        dnf_branch = layers.Dense(32, activation='relu')(shared)
        dnf_output = layers.Dense(1, activation='sigmoid', name='dnf_prob')(dnf_branch)
        
        model = Model(
            inputs=inputs,
            outputs=[top5_output, podium_output, points_output, dnf_output]
        )
        
        return model

class AdvancedTraining:
    
    def __init__(self):
        self.callbacks = []
        self.custom_metrics = []
    
    def create_custom_callbacks(self):
        """Create custom callbacks for better training"""
        
        # Learning rate scheduler
        def scheduler(epoch, lr):
            if epoch < 20:
                return lr
            elif epoch < 40:
                return lr * 0.5
            else:
                return lr * 0.1
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        # Early stopping with patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'model/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        self.callbacks = [lr_scheduler, early_stopping, checkpoint, reduce_lr]
        return self.callbacks
    
    def custom_loss_function(self, y_true, y_pred):
        """Custom loss function that penalizes confident wrong predictions"""
        
        # Standard binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Penalty for confident wrong predictions
        confidence_penalty = tf.where(
            tf.not_equal(y_true, tf.round(y_pred)),
            tf.square(y_pred - 0.5) * 2,  # Penalty for being confident and wrong
            0.0
        )
        
        return bce + confidence_penalty
    
    def position_aware_loss(self, y_true, y_pred):
        """Loss function that considers position importance"""
        
        # Higher weight for top positions
        position_weights = tf.constant([3.0, 2.5, 2.0, 1.5, 1.0], dtype=tf.float32)
        
        # Expand weights to match batch size
        weights = tf.tile(position_weights, [tf.shape(y_true)[0] // 5])
        
        # Weighted binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return bce * weights

def create_advanced_training_pipeline():
    """Create complete advanced training pipeline"""
    
    print("🏗️ Creating Advanced F1 Training Pipeline...")
    
    # Initialize components
    models = AdvancedF1Models()
    training = AdvancedTraining()
    
    # Available model architectures
    model_architectures = {
        'attention': models.build_attention_model,
        'lstm': models.build_lstm_model,
        'ensemble': models.build_ensemble_model,
        'transformer': models.build_transformer_model,
        'multi_task': models.build_multi_task_model
    }
    
    # Training configurations
    training_configs = {
        'callbacks': training.create_custom_callbacks(),
        'custom_loss': training.custom_loss_function,
        'position_loss': training.position_aware_loss
    }
    
    print("✅ Advanced training pipeline ready!")
    print(f"Available architectures: {list(model_architectures.keys())}")
    
    return model_architectures, training_configs

if __name__ == "__main__":
    # Example usage
    architectures, configs = create_advanced_training_pipeline()
    print("Advanced F1 Models Module Ready!")
