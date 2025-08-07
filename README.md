# 🧬 Bacterial Growth Rate Prediction

A deep learning-based bacterial growth prediction system that predicts 24-hour growth curves based on bacterial species, pH, and temperature.

## ✨ Features

- 🎯 **Biologically Accurate**: Uses biological constraint loss functions to ensure predictions follow microbiological principles
- 🧠 **Ecological Intelligence**: Automatically validates and corrects parameters based on bacterial ecological characteristics
- 📊 **High-Quality Data**: Trained on cleaned dataset with anomalous patterns removed
- 📈 **Continuous Growth Curves**: Predicts smooth 24-hour growth trajectories
- 🔬 **Multi-Species Support**: Supports 5 common bacterial species

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Prediction Tool

```bash
python bacteria_predictor.py
```

### 3. Enter Parameters

```
Bacteria name (e.g., 'Escherichia coli'): Escherichia coli
pH value (e.g., 7.0): 7.0
Temperature (°C, e.g., 37.0): 30.0
```

### 4. View Results

The program displays a complete 24-hour growth curve including:
- Maximum and final density
- Hourly bacterial density values
- Optional growth curve visualization

## 🧪 Supported Bacteria

- **Escherichia coli** - Mesophile, intestinal bacteria
- **Staphylococcus aureus** - Skin and mucous membrane bacteria
- **Pseudomonas aeruginosa** - Environmental bacteria
- **Bacillus subtilis** - Soil bacteria, spore-forming
- **Streptococcus pyogenes** - Respiratory tract bacteria

## 📊 Example Output

**E. coli growth at pH=7.0, 30°C:**

```
📊 Prediction Results:
   - Bacteria: Escherichia coli
   - Conditions: pH=7.0, T=30.0°C
   - Maximum density: 8.38e+08 CFU/mL
   - Final density: 8.38e+08 CFU/mL

📈 24-hour Growth Curve:
Hour | Density (CFU/mL)
-----|------------------
   0 | 2.08e+05
   8 | 1.53e+07
  16 | 6.80e+08
  23 | 8.38e+08
```

## 🧬 Ecological Intelligence

The system automatically validates input parameters against bacterial ecological characteristics:

```
🧬 Ecological Analysis:
   Naming source: Greek 'thermós' (heat)
   Ecological features: hot springs, thermal environments
   🌡️  Thermus is thermophile, optimal temp 60-80°C
      Input temperature 33.0°C unsuitable, suggest correcting to 70.0°C
```

## 📁 Project Structure

```
├── bacteria_predictor.py              # Main prediction program
├── model_trainer.py                   # Model trainer
├── bacteria_growth_model.h5           # Trained model file
├── bacteria_24h_cleaned_dataset.csv   # Cleaned training data
├── data_quality_checker.py            # Data quality checker
├── generate_24h_dataset.py            # Dataset generator
└── requirements.txt                   # Dependencies
```

## 📝 License

MIT License

---

*This tool is designed for microbiological research and educational purposes. Predictions are for reference only.*
