# 🧬 MicroCurve ML

**Advanced Bacterial Growth Prediction using Deep Learning with Ecological Intelligence**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MicroCurve ML is an intelligent bacterial growth prediction system that combines deep learning with ecological knowledge to predict 24-hour bacterial growth curves based on species, pH, and temperature conditions.

## ✨ Key Features

- 🧠 **Deep Learning Model**: Multi-input neural network with biological constraints
- 🌱 **Ecological Intelligence**: Automatic parameter validation based on bacterial ecology
- 📈 **24-Hour Prediction**: Complete growth curve prediction with 24 time points
- 🔬 **5 Bacterial Species**: E.coli, S.aureus, P.aeruginosa, B.subtilis, S.pyogenes
- 📊 **Advanced Visualization**: Linear and logarithmic scale growth curves
- ⚡ **Real-time Prediction**: Interactive command-line interface

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MicroCurve-ML.git
cd MicroCurve-ML

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run interactive prediction tool
python bacteria_predictor.py
```

### Example Prediction

```
🧬 Advanced Bacterial Growth Prediction Tool
==================================================

📝 Enter prediction parameters:
Bacteria name (e.g., 'Escherichia coli'): Escherichia coli
pH value (e.g., 7.0): 7.0
Temperature (°C, e.g., 37.0): 37.0

🔄 Predicting growth of Escherichia coli...

📊 Prediction Results:
   - Bacteria: Escherichia coli
   - Conditions: pH=7.0, T=37.0°C
   - Maximum density: 9.22e+08 CFU/mL
   - Final density: 8.58e+08 CFU/mL

📈 24-hour Growth Curve:
Hour | Density (CFU/mL)
-----|------------------
   0 | 1.89e+05
   1 | 1.93e+05
   2 | 3.33e+05
   ...
  23 | 8.58e+08
```

## 🏗️ Architecture

### Model Components

- **Multi-Input Neural Network**: Genus + Species + Environmental conditions
- **Embedding Layers**: 16-dimensional embeddings for categorical features
- **Deep Architecture**: 256→512→256→128 fully connected layers
- **Biological Loss Function**: Custom loss with continuity and monotonicity constraints
- **Ecological Validator**: Automatic parameter correction based on bacterial ecology

### Supported Bacteria

| Bacteria | Optimal pH | Optimal Temp (°C) | Doubling Time |
|----------|------------|-------------------|---------------|
| *Escherichia coli* | 7.0 | 37.0 | 20 min |
| *Staphylococcus aureus* | 7.4 | 25.0 | 2.5 hours |
| *Pseudomonas aeruginosa* | 7.0 | 37.0 | 21 min |
| *Bacillus subtilis* | 7.0 | 30.0 | 24 min |
| *Streptococcus pyogenes* | 7.4 | 37.0 | 40 min |

## 📊 Performance

- **R² Score**: 0.92+ on test set
- **Biological Accuracy**: 95%+ growth pattern recognition
- **Prediction Speed**: <1 second per prediction
- **Data Coverage**: 4,300+ training samples

## 🔬 Scientific Background

The model incorporates several biological principles:

- **Q10 Temperature Effect**: Doubling time changes with temperature
- **Gaussian pH Response**: Optimal growth at species-specific pH ranges
- **Growth Phases**: Lag → Exponential → Stationary phase modeling
- **Ecological Constraints**: Species-specific environmental tolerances

---

*For research and educational purposes only.*
