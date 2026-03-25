# MicroCurve ML

MicroCurve ML is a bacterial growth prediction tool that estimates growth curves and key metrics based on species and environmental conditions.

## Example prediction

```text
Advanced Bacterial Growth Prediction Tool
=========================================
Enter prediction parameters:
Bacteria name (e.g., 'Escherichia coli'): Escherichia coli
pH value (e.g., 7.0): 7.0
Temperature (°C, e.g., 37.0): 37.0

Predicting growth of Escherichia coli...

Prediction Results:
 - Bacteria: Escherichia coli
 - Conditions: pH=7.0, T=37.0°C
 - Maximum density: 9.22e+08 CFU/mL
 - Final density: 8.58e+08 CFU/mL

24-hour Growth Curve:
Hour | Density (CFU/mL)
-----|------------------
 0   | 1.89e+05
 1   | 1.93e+05
 2   | 3.33e+05
 ... | ...
 23  | 8.58e+08


Architecture
Main model components:

Multi-input neural network: genus, species, and environmental conditions.

Embedding layers: 16-dimensional embeddings for categorical features.

Deep architecture: 256 → 512 → 256 → 128 fully connected layers.

Biological loss function: continuity and monotonicity constraints.

Ecological validator: automatic parameter checks based on bacterial ecology.

Supported bacteria
Bacteria	Optimal pH	Optimal Temp (°C)	Doubling time
Escherichia coli	7.0	37.0	20 min
Staphylococcus aureus	7.4	25.0	2.5 hours
Pseudomonas aeruginosa	7.0	37.0	21 min
Bacillus subtilis	7.0	30.0	24 min
Streptococcus pyogenes	7.4	37.0	40 min
Performance
R² score: above 0.92 on the test set.

Biological accuracy: above 95% growth pattern recognition.

Prediction speed: under 1 second per prediction.

Data coverage: more than 4,300 training samples.

Scientific background
The model incorporates several biological principles:

Q10 temperature effect: change of doubling time with temperature.

Gaussian pH response: species-specific optimal pH ranges.

Growth phases: lag, exponential, and stationary phase modeling.

Ecological constraints: species-specific environmental tolerances.

For research and educational purposes only.
