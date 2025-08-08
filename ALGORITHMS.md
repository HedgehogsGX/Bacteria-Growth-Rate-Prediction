# 🧬 MicroCurve ML - Algorithm Summary

This document provides a comprehensive overview of all algorithms used in the MicroCurve ML bacterial growth prediction system.

## 🤖 Deep Learning Algorithms

### Neural Network Architecture
- **Type**: Multi-input Deep Neural Network (MLP)
- **Inputs**: 3 channels (genus, species, environment)
- **Architecture**: 256→512→256→128 fully connected layers
- **Output**: 24 time points (24-hour growth curve)

### Activation Functions
```python
- ReLU: Hidden layer activation
- Sigmoid: Output layer activation (Logistic Function)
- Lambda: Custom mapping (x * 6.0 + 3.0)
```

### Regularization
```python
- Dropout: [0.3, 0.3, 0.2, 0.2] (overfitting prevention)
- BatchNormalization: After each layer
- EarlyStopping: patience=15
```

### Optimization
```python
- Adam Optimizer: learning_rate=0.001
- ReduceLROnPlateau: Adaptive learning rate
- Backpropagation: Gradient descent
```

## 📊 Data Processing Algorithms

### Encoding Algorithms
```python
- LabelEncoder: Categorical feature encoding (genus, species)
- StandardScaler: Numerical feature normalization (pH, temperature)
- Log10 Transform: Density data transformation
```

### Data Splitting
```python
- Stratified Split: Stratified sampling by genus
- Train/Val/Test: 60%/20%/20% fixed split
- Random State: 42 (reproducibility)
```

### Data Quality Control
```python
- Anomaly Detection: Zero values, extreme values, sudden fluctuations
- "Death-Revival" Pattern Detection
- Biologically Unreasonable Pattern Filtering
```

## 🧬 Biological Model Algorithms

### Growth Curve Generation
```python
def enhanced_growth_model(t, K, doubling_time, lag, N0):
    # Exponential growth model
    k = ln(2) / doubling_time  # Growth rate constant
    # Piecewise function: Lag → Exponential → Stationary
```

### Environmental Response
```python
# Gaussian response functions
ph_factor = exp(-0.5 * ((ph - optimal_ph) / 0.8)²)
temp_factor = exp(-0.5 * ((temp - optimal_temp) / 8.0)²)
growth_factor = ph_factor * temp_factor
```

### Temperature-Doubling Time Model
```python
# Q10 model (doubling time halves every 10°C increase)
temp_factor = 2^((optimal_temp - temperature) / 10.0)
doubling_time = optimal_doubling_time * temp_factor
```

## 🎯 Loss Function Algorithms

### Biological Continuity Loss
```python
def biological_continuity_loss(y_true, y_pred):
    # 1. Base MSE loss
    mse_loss = mean_squared_error(y_true, y_pred)
    
    # 2. Continuity loss (time series smoothness)
    continuity_loss = mse(diff(y_true), diff(y_pred))
    
    # 3. Monotonicity loss (early growth constraint)
    monotonic_penalty = mean(max(-early_diff, 0))
    
    # 4. Zero penalty (minimum density constraint)
    zero_penalty = mean(max(3.0 - y_pred, 0))
    
    # Weighted combination
    total_loss = mse + 0.1*continuity + 0.2*monotonic + 0.5*zero
```

## 🔍 Matching Algorithms

### Bacterial Name Matching
```python
- String parsing: split() to separate genus and species
- Ecological mapping: genus_eco_map intelligent substitution
- Encoder matching: LabelEncoder.transform()
- Fault tolerance: Default values for unknown bacteria
```

### Parameter Validation
```python
- Range checking: Temperature, pH optimal range validation
- Auto-correction: Use median when out of range
- Ecological constraints: Parameter adjustment based on bacterial ecology
```

## 📈 Evaluation Algorithms

### Traditional ML Metrics
```python
- MSE: Mean Squared Error
- MAE: Mean Absolute Error
- R²: Coefficient of Determination
- Time-point level metrics: Individual evaluation for each time point
```

### Biological Relevance Metrics
```python
- Continuity Score: Penalize sudden jumps
- Growth Pattern Accuracy: Lag/Log/Stationary phase identification
- Final Density MAPE: Final density prediction error
- Growth Phase Recognition Algorithm
```

### Model Quality Assessment
```python
# Comprehensive scoring algorithm
overall_score = (R² + growth_pattern_accuracy + (1 - final_density_mape/100)) / 3

# Graded evaluation
if score > 0.85: "EXCELLENT"
elif score > 0.75: "GOOD" 
elif score > 0.65: "FAIR"
else: "POOR"
```

## 🛠️ Post-processing Algorithms

### Biological Constraints
```python
def _light_biological_constraints(growth_curve):
    # 1. Minimum density constraint
    growth_curve = maximum(growth_curve, 1e3)
    
    # 2. Outlier smoothing
    if current_val > prev_val * 50 or current_val < prev_val * 0.02:
        growth_curve[i] = (prev_val + next_val) / 2
```

### Special Bacterial Corrections
```python
def _apply_thermophile_correction(growth_curve, temperature):
    if temperature < 50:  # Thermophile low-temperature correction
        # Extend lag phase
        corrected_curve[:6] *= 0.3
        # Reduce overall growth rate
        corrected_curve[6:] *= 0.6
```

## 📊 Algorithm Statistics

| **Category** | **Count** | **Key Algorithms** |
|--------------|-----------|-------------------|
| **Deep Learning** | 8 | MLP, Adam, Dropout, BatchNorm |
| **Data Processing** | 6 | LabelEncoder, StandardScaler, Stratified Split |
| **Biological Models** | 4 | Exponential Growth, Gaussian Response, Q10 Model |
| **Loss Functions** | 1 | Biological Continuity Loss (4 components) |
| **Matching** | 4 | String Matching, Ecological Mapping, Encoder Matching |
| **Evaluation** | 8 | MSE/MAE/R², Biological Metrics, Composite Scoring |
| **Post-processing** | 2 | Biological Constraints, Special Bacterial Corrections |
| **Visualization** | 4 | Linear/Log Plots, Scatter Plots, Residual Plots |

**Total: 37 Core Algorithms** forming a complete biological intelligence prediction system.
