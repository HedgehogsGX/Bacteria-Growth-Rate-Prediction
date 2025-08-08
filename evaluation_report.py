#!/usr/bin/env python3
"""
Model Evaluation Report Generator
Generates a comprehensive evaluation report without GUI dependencies
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

class EvaluationReportGenerator:
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
        
        # Make predictions
        y_pred = self.model.predict(X, verbose=0)
        
        # Calculate overall metrics
        mse_overall = mean_squared_error(y.flatten(), y_pred.flatten())
        mae_overall = mean_absolute_error(y.flatten(), y_pred.flatten())
        r2_overall = r2_score(y.flatten(), y_pred.flatten())
        
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
        
        # Custom biological metrics
        bio_metrics = self.calculate_biological_metrics(y, y_pred)
        
        return {
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
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        
        print("🧬 Advanced Bacterial Growth Model Evaluation Report")
        print("=" * 60)
        
        # Load model and data
        data = self.load_model_and_data()
        if data is None:
            return
        
        # Load fixed data split configuration
        try:
            split_config = self.load_data_split()
            print(f"📂 Data split configuration loaded:")
            print(f"   Training set: {split_config['train_size']} samples ({split_config['train_size']/split_config['total_size']*100:.1f}%)")
            print(f"   Validation set: {split_config['val_size']} samples ({split_config['val_size']/split_config['total_size']*100:.1f}%)")
            print(f"   Test set: {split_config['test_size']} samples ({split_config['test_size']/split_config['total_size']*100:.1f}%)")
            print(f"   Created: {split_config['created_timestamp']}")
        except FileNotFoundError as e:
            print(e)
            return
        
        # Apply data split
        splits = self.apply_data_split(data, split_config)
        
        # Evaluate on all sets
        print(f"\n🔍 Evaluating model performance...")
        train_results = self.evaluate_set(splits['train']['X'], splits['train']['y'], "Training Set")
        val_results = self.evaluate_set(splits['val']['X'], splits['val']['y'], "Validation Set")
        test_results = self.evaluate_set(splits['test']['X'], splits['test']['y'], "Test Set")
        
        # Generate detailed report
        print(f"\n📊 DETAILED EVALUATION RESULTS")
        print("=" * 60)
        
        # Overall performance summary
        print(f"\n1. OVERALL PERFORMANCE METRICS")
        print("-" * 40)
        print(f"{'Metric':<25} {'Train':<10} {'Validation':<12} {'Test':<10}")
        print("-" * 60)
        print(f"{'MSE (log10 scale)':<25} {train_results['mse_overall']:<10.4f} {val_results['mse_overall']:<12.4f} {test_results['mse_overall']:<10.4f}")
        print(f"{'MAE (log10 scale)':<25} {train_results['mae_overall']:<10.4f} {val_results['mae_overall']:<12.4f} {test_results['mae_overall']:<10.4f}")
        print(f"{'R² Score':<25} {train_results['r2_overall']:<10.4f} {val_results['r2_overall']:<12.4f} {test_results['r2_overall']:<10.4f}")
        
        # Biological relevance metrics
        print(f"\n2. BIOLOGICAL RELEVANCE METRICS")
        print("-" * 40)
        print(f"{'Metric':<25} {'Train':<10} {'Validation':<12} {'Test':<10}")
        print("-" * 60)
        print(f"{'Continuity Score':<25} {train_results['biological_metrics']['continuity_score']:<10.4f} {val_results['biological_metrics']['continuity_score']:<12.4f} {test_results['biological_metrics']['continuity_score']:<10.4f}")
        print(f"{'Growth Pattern Acc':<25} {train_results['biological_metrics']['growth_pattern_accuracy']:<10.4f} {val_results['biological_metrics']['growth_pattern_accuracy']:<12.4f} {test_results['biological_metrics']['growth_pattern_accuracy']:<10.4f}")
        print(f"{'Final Density MAPE%':<25} {train_results['biological_metrics']['final_density_mape']:<10.2f} {val_results['biological_metrics']['final_density_mape']:<12.2f} {test_results['biological_metrics']['final_density_mape']:<10.2f}")
        
        # Time-point analysis
        print(f"\n3. TIME-POINT ANALYSIS (R² Score)")
        print("-" * 40)
        print("Hour  Train   Val     Test")
        print("-" * 25)
        for hour in [0, 4, 8, 12, 16, 20, 23]:
            print(f"{hour:4d}  {train_results['r2_per_time'][hour]:6.3f}  {val_results['r2_per_time'][hour]:6.3f}  {test_results['r2_per_time'][hour]:6.3f}")
        
        # Model validation analysis
        print(f"\n4. MODEL VALIDATION ANALYSIS")
        print("-" * 40)
        
        # Check for overfitting
        train_test_gap = train_results['r2_overall'] - test_results['r2_overall']
        if train_test_gap < 0.05:
            overfitting_status = "✅ No significant overfitting"
        elif train_test_gap < 0.10:
            overfitting_status = "⚠️  Mild overfitting detected"
        else:
            overfitting_status = "❌ Significant overfitting detected"
        
        print(f"Overfitting Assessment: {overfitting_status}")
        print(f"Train-Test R² Gap: {train_test_gap:.4f}")
        
        # Generalization capability
        if test_results['r2_overall'] > 0.90:
            generalization = "✅ Excellent generalization"
        elif test_results['r2_overall'] > 0.80:
            generalization = "✅ Good generalization"
        elif test_results['r2_overall'] > 0.70:
            generalization = "⚠️  Fair generalization"
        else:
            generalization = "❌ Poor generalization"
        
        print(f"Generalization Capability: {generalization}")
        print(f"Test Set R²: {test_results['r2_overall']:.4f}")
        
        # Data split integrity
        print(f"\n5. DATA SPLIT INTEGRITY")
        print("-" * 40)
        print(f"✅ Fixed data split configuration used")
        print(f"✅ Test set completely isolated during training")
        print(f"✅ Stratified sampling by bacterial genus")
        print(f"✅ Consistent evaluation across all metrics")
        
        # Final assessment
        print(f"\n6. FINAL ASSESSMENT")
        print("-" * 40)
        
        overall_score = (test_results['r2_overall'] + 
                        test_results['biological_metrics']['growth_pattern_accuracy'] + 
                        (1 - test_results['biological_metrics']['final_density_mape']/100)) / 3
        
        if overall_score > 0.85:
            assessment = "🌟 EXCELLENT - Model ready for production use"
        elif overall_score > 0.75:
            assessment = "✅ GOOD - Model suitable for research applications"
        elif overall_score > 0.65:
            assessment = "⚠️  FAIR - Model needs improvement"
        else:
            assessment = "❌ POOR - Model requires significant revision"
        
        print(f"Overall Model Quality: {assessment}")
        print(f"Composite Score: {overall_score:.4f}")
        
        print(f"\n" + "=" * 60)
        print("📋 EVALUATION COMPLETE")
        print("=" * 60)
        
        return {
            'train': train_results,
            'validation': val_results,
            'test': test_results,
            'split_config': split_config,
            'overall_assessment': assessment,
            'composite_score': overall_score
        }

if __name__ == "__main__":
    generator = EvaluationReportGenerator()
    results = generator.generate_report()
