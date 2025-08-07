#!/usr/bin/env python3
"""
Improved Bacteria Growth Predictor - 使用改进模型和生物学约束的预测器
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

class ImprovedBacteriaPredictor:
    def __init__(self, model_path="bacteria_growth_model.h5", dataset_path="bacteria_24h_cleaned_dataset.csv"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.genus_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # 属名-生态特征映射表
        self.genus_eco_map = {
            "Thermus": {
                "naming_source": "Greek 'thermós' (heat)",
                "opt_temp": (60, 80),
                "opt_ph": (7.0, 8.5),
                "habitat": "hot springs, thermal environments",
                "growth_type": "thermophile",
                "model_substitute": "Bacillus"  # 当前模型中的替代
            },
            "Bacillus": {
                "naming_source": "Latin 'bacillus' (small rod)",
                "opt_temp": (25, 40),
                "opt_ph": (6.0, 8.0),
                "habitat": "soil, compost, mesophilic environments",
                "growth_type": "mesophile",
                "model_substitute": "Bacillus"
            },
            "Escherichia": {
                "naming_source": "Named after Theodor Escherich",
                "opt_temp": (30, 42),
                "opt_ph": (6.0, 8.0),
                "habitat": "intestinal tract, mesophilic environments",
                "growth_type": "mesophile",
                "model_substitute": "Escherichia"
            },
            "Pseudomonas": {
                "naming_source": "Greek 'pseudo' (false) + 'monas' (unit)",
                "opt_temp": (25, 37),
                "opt_ph": (6.0, 8.0),
                "habitat": "water, soil, plant surfaces",
                "growth_type": "mesophile",
                "model_substitute": "Pseudomonas"
            },
            "Staphylococcus": {
                "naming_source": "Greek 'staphyle' (bunch of grapes)",
                "opt_temp": (30, 40),
                "opt_ph": (6.0, 8.0),
                "habitat": "skin, mucous membranes",
                "growth_type": "mesophile",
                "model_substitute": "Staphylococcus"
            },
            "Streptococcus": {
                "naming_source": "Greek 'streptos' (twisted) + 'kokkos' (berry)",
                "opt_temp": (35, 42),
                "opt_ph": (6.5, 7.5),
                "habitat": "human respiratory tract",
                "growth_type": "mesophile",
                "model_substitute": "Streptococcus"
            }
        }

        self.load_model_and_setup()
    
    def biological_continuity_loss(self, y_true, y_pred):
        """生物学连续性损失函数 - 与训练时保持一致"""
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
    
    def load_model_and_setup(self):
        """加载模型并设置编码器"""
        
        print(f"📂 Loading improved model: {self.model_path}")
        try:
            # Load model with custom loss function
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={'biological_continuity_loss': self.biological_continuity_loss}
            )
            print("✅ Improved model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Failed to load improved model, using original: {e}")
            self.model = tf.keras.models.load_model("bacteria_growth_model.h5")
            print("✅ Original model loaded successfully!")

        # Setup encoders
        print(f"📂 Setting up encoders: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)

        # Parse bacteria names
        bacteria_names = df.iloc[:, 0].str.split(' ', expand=True)
        genera = bacteria_names[0]
        species = bacteria_names[1]

        # Train encoders
        self.genus_encoder.fit(genera)
        self.species_encoder.fit(species)

        # Train scaler
        env_features = df[['ph', 'temperature']].values
        self.scaler.fit(env_features)

        print(f"📊 Available bacteria:")
        print(f"   - Genera: {', '.join(self.genus_encoder.classes_[:5])}... ({len(self.genus_encoder.classes_)} total)")
        print(f"   - Species: {', '.join(self.species_encoder.classes_[:5])}... ({len(self.species_encoder.classes_)} total)")
    
    def validate_ecological_parameters(self, genus, ph, temperature):
        """Validate and correct ecological parameters"""

        original_genus = genus
        corrected_params = {"genus": genus, "ph": ph, "temperature": temperature}
        warnings = []

        if genus in self.genus_eco_map:
            eco_data = self.genus_eco_map[genus]

            # Check if temperature matches ecological characteristics
            opt_temp_min, opt_temp_max = eco_data["opt_temp"]
            if not (opt_temp_min <= temperature <= opt_temp_max):
                corrected_temp = (opt_temp_min + opt_temp_max) / 2
                warnings.append(f"🌡️  {genus} is {eco_data['growth_type']}, optimal temp {opt_temp_min}-{opt_temp_max}°C")
                warnings.append(f"   Input temperature {temperature}°C unsuitable, suggest correcting to {corrected_temp:.1f}°C")
                corrected_params["temperature"] = corrected_temp

            # Check pH match
            opt_ph_min, opt_ph_max = eco_data["opt_ph"]
            if not (opt_ph_min <= ph <= opt_ph_max):
                corrected_ph = (opt_ph_min + opt_ph_max) / 2
                warnings.append(f"🧪 {genus} optimal pH {opt_ph_min}-{opt_ph_max}")
                warnings.append(f"   Input pH {ph} unsuitable, suggest correcting to {corrected_ph:.1f}")
                corrected_params["ph"] = corrected_ph

            # Use model substitute genus
            model_genus = eco_data["model_substitute"]
            if model_genus != genus:
                warnings.append(f"🔄 Using {model_genus} model to predict {genus} growth characteristics")
                corrected_params["genus"] = model_genus

        return corrected_params, warnings

    def predict_growth(self, bacteria_name, ph, temperature):
        """Predict bacterial growth curve"""

        try:
            # Parse bacteria name
            parts = bacteria_name.strip().split()
            if len(parts) < 2:
                return None, "Bacteria name must contain genus and species (e.g., 'Escherichia coli')"

            original_genus = parts[0]
            species = ' '.join(parts[1:])

            # Ecological validation and correction
            corrected_params, eco_warnings = self.validate_ecological_parameters(
                original_genus, ph, temperature
            )

            genus = corrected_params["genus"]
            final_ph = corrected_params["ph"]
            final_temp = corrected_params["temperature"]

            # Display ecological correction information
            if eco_warnings:
                print(f"\n🧬 Ecological Analysis:")
                if original_genus in self.genus_eco_map:
                    eco_data = self.genus_eco_map[original_genus]
                    print(f"   Naming source: {eco_data['naming_source']}")
                    print(f"   Ecological features: {eco_data['habitat']}")
                for warning in eco_warnings:
                    print(f"   {warning}")
                print()

            # Check if within model's known range
            if genus not in self.genus_encoder.classes_:
                genus = self.genus_encoder.classes_[0]  # Use default
                print(f"⚠️  Unknown genus '{corrected_params['genus']}' in model, using '{genus}'")

            if species not in self.species_encoder.classes_:
                species = self.species_encoder.classes_[0]  # Use default
                print(f"⚠️  Unknown species '{' '.join(parts[1:])}' in model, using '{species}'")
            
            # 编码输入
            genus_encoded = self.genus_encoder.transform([genus])[0]
            species_encoded = self.species_encoder.transform([species])[0]
            
            # 标准化环境参数 - 使用修正后的参数
            env_normalized = self.scaler.transform([[final_ph, final_temp]])

            # 准备模型输入
            X = {
                'genus_input': np.array([[genus_encoded]]),
                'species_input': np.array([[species_encoded]]),
                'env_input': env_normalized
            }

            # 进行预测
            prediction = self.model.predict(X, verbose=0)
            growth_curve_log = prediction[0]

            # 转换回原始密度值
            growth_curve = 10 ** growth_curve_log

            # 应用轻微的生物学约束（改进模型应该已经很好了）
            growth_curve = self._light_biological_constraints(growth_curve)

            # 如果是嗜热菌，应用特殊的生长修正
            if original_genus in self.genus_eco_map:
                eco_data = self.genus_eco_map[original_genus]
                if eco_data["growth_type"] == "thermophile":
                    growth_curve = self._apply_thermophile_correction(growth_curve, final_temp)

            # 计算关键指标
            max_density = np.max(growth_curve)
            final_density = growth_curve[-1]

            result = {
                'bacteria': f"{original_genus} {species}",  # 显示原始细菌名
                'conditions': {'ph': final_ph, 'temperature': final_temp},  # 显示修正后的条件
                'original_conditions': {'ph': ph, 'temperature': temperature},  # 保留原始输入
                'max_density': max_density,
                'final_density': final_density,
                'growth_curve': growth_curve,
                'time_points': list(range(24)),
                'ecological_corrections': eco_warnings
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

    def _apply_thermophile_correction(self, growth_curve, temperature):
        """Apply special growth correction for thermophiles"""

        # Growth characteristics of thermophiles at non-optimal temperatures
        if temperature < 50:  # Below thermophile suitable temperature
            # Extend lag phase, reduce maximum density
            corrected_curve = growth_curve.copy()

            # Extend lag phase (slower growth in first 6 hours)
            for i in range(6):
                corrected_curve[i] = growth_curve[i] * 0.3

            # Reduce overall growth rate
            for i in range(6, 24):
                corrected_curve[i] = growth_curve[i] * 0.6

            return corrected_curve

        return growth_curve

    def _light_biological_constraints(self, growth_curve):
        """Light biological constraints - improved model should already be good"""

        # 确保最小密度
        growth_curve = np.maximum(growth_curve, 1e3)

        # 轻微平滑异常值
        for i in range(1, len(growth_curve)-1):
            # 如果某个点与前后点差异过大，进行轻微调整
            prev_val = growth_curve[i-1]
            next_val = growth_curve[i+1] if i+1 < len(growth_curve) else growth_curve[i]
            current_val = growth_curve[i]

            # 如果当前值与前后值差异过大，进行插值
            if current_val > prev_val * 50 or current_val < prev_val * 0.02:
                growth_curve[i] = (prev_val + next_val) / 2

        return growth_curve
    
    def display_prediction(self, result):
        """Display prediction results"""

        print(f"\n📊 Prediction Results:")
        print(f"   - Bacteria: {result['bacteria']}")

        # Display condition correction information
        if 'original_conditions' in result:
            orig_ph = result['original_conditions']['ph']
            orig_temp = result['original_conditions']['temperature']
            final_ph = result['conditions']['ph']
            final_temp = result['conditions']['temperature']

            if orig_ph != final_ph or orig_temp != final_temp:
                print(f"   - Original conditions: pH={orig_ph}, T={orig_temp}°C")
                print(f"   - Corrected conditions: pH={final_ph}, T={final_temp}°C ⭐")
            else:
                print(f"   - Conditions: pH={final_ph}, T={final_temp}°C")
        else:
            print(f"   - Conditions: pH={result['conditions']['ph']}, T={result['conditions']['temperature']}°C")

        print(f"   - Maximum density: {result['max_density']:.2e} CFU/mL")
        print(f"   - Final density: {result['final_density']:.2e} CFU/mL")

        # Display ecological correction summary
        if 'ecological_corrections' in result and result['ecological_corrections']:
            print(f"\n🧬 Ecological Correction Summary:")
            for correction in result['ecological_corrections']:
                if "correcting" in correction.lower():
                    print(f"   ⭐ {correction}")

        print(f"\n📈 24-hour Growth Curve:")
        print(f"Hour | Density (CFU/mL)")
        print(f"-----|------------------")

        for i, density in enumerate(result['growth_curve']):
            print(f"{i:4d} | {density:.2e}")
    
    def plot_growth_curve(self, result):
        """绘制生长曲线"""

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 8))

        time_points = result['time_points']
        densities = result['growth_curve']

        plt.subplot(2, 1, 1)
        plt.plot(time_points, densities, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Time (hours)')
        plt.ylabel('Density (CFU/mL)')
        plt.title(f"{result['bacteria']} Growth Curve\npH={result['conditions']['ph']}, T={result['conditions']['temperature']}°C")
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        plt.subplot(2, 1, 2)
        plt.semilogy(time_points, densities, 'r-o', linewidth=2, markersize=4)
        plt.xlabel('Time (hours)')
        plt.ylabel('Density (CFU/mL, Log Scale)')
        plt.title('Log Scale Growth Curve')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def main():
    """Main function - Interactive prediction"""

    print("🧬 Advanced Bacterial Growth Prediction Tool")
    print("=" * 50)

    # Initialize predictor
    predictor = ImprovedBacteriaPredictor()

    while True:
        print(f"\n📝 Enter prediction parameters:")

        try:
            bacteria = input("Bacteria name (e.g., 'Escherichia coli'): ").strip()
            if not bacteria:
                break

            ph = float(input("pH value (e.g., 7.0): "))
            if not (0 <= ph <= 14):
                print("❌ pH must be between 0-14!")
                continue

            temperature = float(input("Temperature (°C, e.g., 37.0): "))
            if not (-10 <= temperature <= 80):
                print("❌ Temperature must be between -10 to 80°C!")
                continue

            print(f"\n🔄 Predicting growth of {bacteria}...")

            # Make prediction
            result, error = predictor.predict_growth(bacteria, ph, temperature)

            if error:
                print(f"❌ {error}")
                continue

            # Display results
            predictor.display_prediction(result)

            # Ask if plot is needed
            plot_choice = input(f"\n📊 Plot growth curve? (y/n): ").strip().lower()
            if plot_choice == 'y':
                predictor.plot_growth_curve(result)

        except KeyboardInterrupt:
            break
        except ValueError:
            print("❌ Input format error!")
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n👋 Thank you for using the Advanced Bacterial Growth Prediction Tool!")

if __name__ == "__main__":
    main()
