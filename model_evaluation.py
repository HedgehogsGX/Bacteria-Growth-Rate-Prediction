#!/usr/bin/env python3
"""
Model Evaluation - Validation and Test Set Performance Analysis
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class ModelEvaluator:
    def __init__(self, model_path="bacteria_growth_model.h5", dataset_path="bacteria_24h_cleaned_dataset.csv"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.genus_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.data_split_config = "data_split_config.json"
        
    def biological_continuity_loss(self, y_true, y_pred):
        """Custom loss function for model loading"""
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        continuity_loss = tf.reduce_mean(tf.square(diff_true - diff_pred))
        early_pred = y_pred[:, :8]
        early_diff = early_pred[:, 1:] - early_pred[:, :-1]
        monotonic_penalty = tf.reduce_mean(tf.maximum(-early_diff, 0.0))
        zero_penalty = tf.reduce_mean(tf.maximum(3.0 - y_pred, 0.0))
        total_loss = mse_loss + 0.1 * continuity_loss + 0.2 * monotonic_penalty + 0.5 * zero_penalty
        return total_loss
    
    def load_model_and_data(self):
        """Load model and prepare data"""
        
        print(f"📂 Loading model: {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(
                self.model_path, 
                custom_objects={'biological_continuity_loss': self.biological_continuity_loss}
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
        
        print(f"📂 Loading dataset: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        
        # Parse bacteria names
        bacteria_names = df.iloc[:, 0].str.split(' ', expand=True)
        genera = bacteria_names[0]
        species = bacteria_names[1]
        
        # Encode categorical features
        genus_encoded = self.genus_encoder.fit_transform(genera)
        species_encoded = self.species_encoder.fit_transform(species)
        
        # Scale environmental features
        env_features = df[['ph', 'temperature']].values
        env_features_scaled = self.scaler.fit_transform(env_features)
        
        # Prepare density data (log10 transformed)
        density_data = df.iloc[:, 3:27].values  # 24 time points
        density_data = np.maximum(density_data, 1.0)  # Ensure no zeros
        density_log = np.log10(density_data)
        
        return {
            'genus': genus_encoded,
            'species': species_encoded,
            'env': env_features_scaled,
            'density_log': density_log,
            'bacteria_names': df.iloc[:, 0].values
        }
    
    def load_data_split(self):
        """Load existing data split configuration"""

        if not os.path.exists(self.data_split_config):
            raise FileNotFoundError(
                f"❌ Data split configuration not found: {self.data_split_config}\n"
                f"   Please run model training first to create the data split configuration."
            )

        with open(self.data_split_config, 'r') as f:
            split_config = json.load(f)

        print(f"📂 Loaded data split configuration:")
        print(f"   Training set: {split_config['train_size']} samples")
        print(f"   Validation set: {split_config['val_size']} samples")
        print(f"   Test set: {split_config['test_size']} samples")
        print(f"   Created: {split_config['created_timestamp']}")

        return split_config

    def apply_data_split(self, data, split_config):
        """Apply data split configuration to create train/val/test sets"""

        train_indices = np.array(split_config['train_indices'])
        val_indices = np.array(split_config['val_indices'])
        test_indices = np.array(split_config['test_indices'])

        # Create input data structures
        X_train = {
            'genus_input': data['genus'][train_indices],
            'species_input': data['species'][train_indices],
            'env_input': data['env'][train_indices]
        }
        X_val = {
            'genus_input': data['genus'][val_indices],
            'species_input': data['species'][val_indices],
            'env_input': data['env'][val_indices]
        }
        X_test = {
            'genus_input': data['genus'][test_indices],
            'species_input': data['species'][test_indices],
            'env_input': data['env'][test_indices]
        }

        return {
            'train': {'X': X_train, 'y': data['density_log'][train_indices]},
            'val': {'X': X_val, 'y': data['density_log'][val_indices]},
            'test': {'X': X_test, 'y': data['density_log'][test_indices]}
        }
    
    def evaluate_set(self, X, y, set_name="Set"):
        """Evaluate model performance on a dataset"""
        
        print(f"\n🔍 Evaluating {set_name}...")
        
        # Make predictions
        y_pred = self.model.predict(X, verbose=0)
        
        # Calculate metrics for each time point
        mse_per_time = []
        mae_per_time = []
        r2_per_time = []
        
        for t in range(24):
            mse_t = mean_squared_error(y[:, t], y_pred[:, t])
            mae_t = mean_absolute_error(y[:, t], y_pred[:, t])
            r2_t = r2_score(y[:, t], y_pred[:, t])
            
            mse_per_time.append(mse_t)
            mae_per_time.append(mae_t)
            r2_per_time.append(r2_t)
        
        # Overall metrics
        mse_overall = mean_squared_error(y.flatten(), y_pred.flatten())
        mae_overall = mean_absolute_error(y.flatten(), y_pred.flatten())
        r2_overall = r2_score(y.flatten(), y_pred.flatten())
        
        # Custom biological metrics
        bio_metrics = self.calculate_biological_metrics(y, y_pred)
        
        results = {
            'mse_overall': mse_overall,
            'mae_overall': mae_overall,
            'r2_overall': r2_overall,
            'mse_per_time': mse_per_time,
            'mae_per_time': mae_per_time,
            'r2_per_time': r2_per_time,
            'biological_metrics': bio_metrics,
            'predictions': y_pred,
            'actual': y
        }
        
        print(f"📈 {set_name} Results:")
        print(f"   Overall MSE: {mse_overall:.4f}")
        print(f"   Overall MAE: {mae_overall:.4f}")
        print(f"   Overall R²: {r2_overall:.4f}")
        print(f"   Biological Continuity Score: {bio_metrics['continuity_score']:.4f}")
        print(f"   Growth Pattern Accuracy: {bio_metrics['growth_pattern_accuracy']:.4f}")
        
        return results
    
    def calculate_biological_metrics(self, y_true, y_pred):
        """Calculate biological relevance metrics"""
        
        # Convert back to original scale for biological interpretation
        density_true = 10 ** y_true
        density_pred = 10 ** y_pred
        
        # 1. Continuity score (penalize sudden jumps)
        continuity_scores = []
        for i in range(len(y_true)):
            true_diffs = np.abs(np.diff(density_true[i]))
            pred_diffs = np.abs(np.diff(density_pred[i]))
            continuity_score = 1 - np.mean(np.abs(true_diffs - pred_diffs) / (true_diffs + 1e-6))
            continuity_scores.append(max(0, continuity_score))
        
        # 2. Growth pattern accuracy (lag, log, stationary phases)
        pattern_accuracies = []
        for i in range(len(y_true)):
            # Identify growth phases
            true_phases = self.identify_growth_phases(density_true[i])
            pred_phases = self.identify_growth_phases(density_pred[i])
            
            # Calculate phase agreement
            phase_agreement = np.mean(true_phases == pred_phases)
            pattern_accuracies.append(phase_agreement)
        
        # 3. Final density accuracy
        final_density_errors = np.abs(density_true[:, -1] - density_pred[:, -1]) / density_true[:, -1]
        
        return {
            'continuity_score': np.mean(continuity_scores),
            'growth_pattern_accuracy': np.mean(pattern_accuracies),
            'final_density_mape': np.mean(final_density_errors) * 100
        }
    
    def identify_growth_phases(self, density_curve):
        """Identify growth phases: 0=lag, 1=log, 2=stationary"""
        
        # Simple heuristic for phase identification
        phases = np.zeros(len(density_curve))
        
        # Calculate growth rates
        growth_rates = np.diff(np.log(density_curve + 1e-6))
        
        # Thresholds for phase classification
        lag_threshold = 0.1
        log_threshold = 0.3
        
        for i in range(len(growth_rates)):
            if growth_rates[i] < lag_threshold:
                phases[i+1] = 0  # Lag phase
            elif growth_rates[i] > log_threshold:
                phases[i+1] = 1  # Log phase
            else:
                phases[i+1] = 2  # Stationary phase
        
        return phases
    
    def plot_evaluation_results(self, train_results, val_results, test_results):
        """Plot comprehensive evaluation results"""
        
        plt.figure(figsize=(15, 12))
        
        # 1. MSE per time point
        plt.subplot(2, 3, 1)
        time_points = range(24)
        plt.plot(time_points, train_results['mse_per_time'], 'b-', label='Train', linewidth=2)
        plt.plot(time_points, val_results['mse_per_time'], 'g-', label='Validation', linewidth=2)
        plt.plot(time_points, test_results['mse_per_time'], 'r-', label='Test', linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('MSE')
        plt.title('MSE per Time Point')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. R² per time point
        plt.subplot(2, 3, 2)
        plt.plot(time_points, train_results['r2_per_time'], 'b-', label='Train', linewidth=2)
        plt.plot(time_points, val_results['r2_per_time'], 'g-', label='Validation', linewidth=2)
        plt.plot(time_points, test_results['r2_per_time'], 'r-', label='Test', linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('R²')
        plt.title('R² per Time Point')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Overall metrics comparison
        plt.subplot(2, 3, 3)
        metrics = ['MSE', 'MAE', 'R²']
        train_vals = [train_results['mse_overall'], train_results['mae_overall'], train_results['r2_overall']]
        val_vals = [val_results['mse_overall'], val_results['mae_overall'], val_results['r2_overall']]
        test_vals = [test_results['mse_overall'], test_results['mae_overall'], test_results['r2_overall']]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.bar(x - width, train_vals, width, label='Train', alpha=0.8)
        plt.bar(x, val_vals, width, label='Validation', alpha=0.8)
        plt.bar(x + width, test_vals, width, label='Test', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Overall Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        
        # 4. Biological metrics
        plt.subplot(2, 3, 4)
        bio_metrics = ['Continuity', 'Growth Pattern', 'Final Density MAPE']
        train_bio = [train_results['biological_metrics']['continuity_score'],
                    train_results['biological_metrics']['growth_pattern_accuracy'],
                    train_results['biological_metrics']['final_density_mape']/100]
        val_bio = [val_results['biological_metrics']['continuity_score'],
                  val_results['biological_metrics']['growth_pattern_accuracy'],
                  val_results['biological_metrics']['final_density_mape']/100]
        test_bio = [test_results['biological_metrics']['continuity_score'],
                   test_results['biological_metrics']['growth_pattern_accuracy'],
                   test_results['biological_metrics']['final_density_mape']/100]
        
        x = np.arange(len(bio_metrics))
        plt.bar(x - width, train_bio, width, label='Train', alpha=0.8)
        plt.bar(x, val_bio, width, label='Validation', alpha=0.8)
        plt.bar(x + width, test_bio, width, label='Test', alpha=0.8)
        
        plt.xlabel('Biological Metrics')
        plt.ylabel('Score')
        plt.title('Biological Relevance Metrics')
        plt.xticks(x, bio_metrics, rotation=45)
        plt.legend()
        
        # 5. Prediction vs Actual scatter plot (Test set)
        plt.subplot(2, 3, 5)
        test_actual = test_results['actual'].flatten()
        test_pred = test_results['predictions'].flatten()
        
        plt.scatter(test_actual, test_pred, alpha=0.5, s=1)
        plt.plot([test_actual.min(), test_actual.max()], [test_actual.min(), test_actual.max()], 'r--', lw=2)
        plt.xlabel('Actual (log10)')
        plt.ylabel('Predicted (log10)')
        plt.title('Test Set: Predicted vs Actual')
        plt.grid(True, alpha=0.3)
        
        # 6. Residuals plot
        plt.subplot(2, 3, 6)
        residuals = test_pred - test_actual
        plt.scatter(test_pred, residuals, alpha=0.5, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted (log10)')
        plt.ylabel('Residuals')
        plt.title('Test Set: Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_full_evaluation(self):
        """Run complete model evaluation"""
        
        print("🧬 Advanced Bacterial Growth Model Evaluation")
        print("=" * 50)
        
        # Load model and data
        data = self.load_model_and_data()
        if data is None:
            return

        # Load fixed data split configuration
        try:
            split_config = self.load_data_split()
        except FileNotFoundError as e:
            print(e)
            return

        # Apply data split
        splits = self.apply_data_split(data, split_config)
        
        # Evaluate on all sets
        train_results = self.evaluate_set(splits['train']['X'], splits['train']['y'], "Training Set")
        val_results = self.evaluate_set(splits['val']['X'], splits['val']['y'], "Validation Set")
        test_results = self.evaluate_set(splits['test']['X'], splits['test']['y'], "Test Set")
        
        # Plot results
        self.plot_evaluation_results(train_results, val_results, test_results)
        
        # Summary
        print(f"\n📋 Evaluation Summary:")
        print(f"{'Metric':<25} {'Train':<10} {'Validation':<12} {'Test':<10}")
        print("-" * 60)
        print(f"{'MSE':<25} {train_results['mse_overall']:<10.4f} {val_results['mse_overall']:<12.4f} {test_results['mse_overall']:<10.4f}")
        print(f"{'MAE':<25} {train_results['mae_overall']:<10.4f} {val_results['mae_overall']:<12.4f} {test_results['mae_overall']:<10.4f}")
        print(f"{'R²':<25} {train_results['r2_overall']:<10.4f} {val_results['r2_overall']:<12.4f} {test_results['r2_overall']:<10.4f}")
        print(f"{'Continuity Score':<25} {train_results['biological_metrics']['continuity_score']:<10.4f} {val_results['biological_metrics']['continuity_score']:<12.4f} {test_results['biological_metrics']['continuity_score']:<10.4f}")
        print(f"{'Growth Pattern Acc':<25} {train_results['biological_metrics']['growth_pattern_accuracy']:<10.4f} {val_results['biological_metrics']['growth_pattern_accuracy']:<12.4f} {test_results['biological_metrics']['growth_pattern_accuracy']:<10.4f}")
        
        return {
            'train': train_results,
            'validation': val_results,
            'test': test_results
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.run_full_evaluation()
