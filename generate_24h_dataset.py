#!/usr/bin/env python3
"""
Generate 24-hour bacteria growth dataset with 24 time points
"""

import numpy as np
import pandas as pd
import random

class Bacteria24HDataGenerator:
    def __init__(self):
        """Initialize with realistic bacterial parameters"""
        
        # Bacterial species with their optimal conditions and growth parameters
        # 基于真实代时数据优化增长参数
        self.bacteria_params = {
            'Escherichia coli': {
                'optimal_ph': 7.0, 'ph_range': (6.0, 8.0),
                'optimal_temp': 37.0, 'temp_range': (20.0, 45.0),
                'max_density': 1e9, 'growth_rate': 1.2, 'lag_time': 2.0,
                'doubling_time_optimal': 0.33  # 20分钟代时
            },
            'Staphylococcus aureus': {
                'optimal_ph': 7.4, 'ph_range': (6.5, 8.5),
                'optimal_temp': 25.0, 'temp_range': (15.0, 45.0),  # 25°C为最适温度
                'max_density': 8e8, 'growth_rate': 1.0, 'lag_time': 1.5,
                'doubling_time_optimal': 2.5  # 25°C时2.5小时代时，20°C时约3-4小时
            },
            'Bacillus subtilis': {
                'optimal_ph': 7.0, 'ph_range': (6.0, 8.5),
                'optimal_temp': 30.0, 'temp_range': (15.0, 50.0),
                'max_density': 1.2e9, 'growth_rate': 1.1, 'lag_time': 1.0,
                'doubling_time_optimal': 0.4  # 24分钟代时
            },
            'Pseudomonas aeruginosa': {
                'optimal_ph': 7.0, 'ph_range': (6.0, 8.0),
                'optimal_temp': 37.0, 'temp_range': (20.0, 42.0),
                'max_density': 9e8, 'growth_rate': 1.3, 'lag_time': 1.8,
                'doubling_time_optimal': 0.35  # 21分钟代时
            },
            'Streptococcus pyogenes': {
                'optimal_ph': 7.4, 'ph_range': (6.5, 8.0),
                'optimal_temp': 37.0, 'temp_range': (25.0, 42.0),
                'max_density': 7e8, 'growth_rate': 0.8, 'lag_time': 2.5,
                'doubling_time_optimal': 0.67  # 40分钟代时
            }
        }
        
        # 24 time points (0-24 hours)
        self.time_points = np.linspace(0, 24, 24)
        
    def calculate_growth_factors(self, bacteria, ph, temperature):
        """Calculate growth factors based on environmental conditions"""
        params = self.bacteria_params[bacteria]
        
        # pH factor (Gaussian-like response)
        ph_optimal = params['optimal_ph']
        ph_min, ph_max = params['ph_range']
        if ph_min <= ph <= ph_max:
            ph_factor = np.exp(-0.5 * ((ph - ph_optimal) / 0.8) ** 2)
        else:
            ph_factor = 0.01  # Very poor growth outside range
            
        # Temperature factor (Gaussian-like response)
        temp_optimal = params['optimal_temp']
        temp_min, temp_max = params['temp_range']
        if temp_min <= temperature <= temp_max:
            temp_factor = np.exp(-0.5 * ((temperature - temp_optimal) / 8.0) ** 2)
        else:
            temp_factor = 0.01  # Very poor growth outside range
            
        return ph_factor * temp_factor
    
    def calculate_doubling_time(self, bacteria, temperature):
        """计算基于温度的代时"""
        params = self.bacteria_params[bacteria]
        optimal_temp = params['optimal_temp']
        optimal_doubling_time = params['doubling_time_optimal']

        # 温度对代时的影响 (Q10 = 2, 即温度每升高10°C，代时减半)
        temp_factor = 2 ** ((optimal_temp - temperature) / 10.0)
        doubling_time = optimal_doubling_time * temp_factor

        # 限制代时范围 (最快15分钟，最慢8小时)
        doubling_time = np.clip(doubling_time, 0.25, 8.0)

        return doubling_time

    def enhanced_growth_model(self, t, K, doubling_time, lag, N0=1e5):
        """
        增强的细菌生长模型，更好地体现指数增长特征

        Parameters:
        - t: time array
        - K: carrying capacity (max density)
        - doubling_time: 代时 (小时)
        - lag: lag time
        - N0: initial density
        """
        N = np.zeros_like(t)

        # 计算增长速率常数 k = ln(2) / doubling_time
        k = np.log(2) / doubling_time

        for i, time in enumerate(t):
            if time <= lag:
                # 延滞期：轻微波动，可能略有下降
                fluctuation = 0.95 + 0.1 * np.random.random()
                N[i] = N0 * fluctuation
            else:
                # 生长期开始
                t_eff = time - lag

                if t_eff <= 12.0:  # 延长对数期到12小时，确保充分的指数增长
                    # 纯指数增长公式: N(t) = N0 * e^(kt)
                    N_exp = N0 * np.exp(k * t_eff)

                    # 只有当接近载量时才开始限制增长
                    if N_exp < K * 0.8:  # 80%载量以下保持纯指数增长
                        N[i] = N_exp
                    else:
                        # 平滑过渡到载量限制
                        excess_factor = (N_exp - K * 0.8) / (K * 0.2)
                        limitation = 1.0 / (1.0 + excess_factor)
                        N[i] = K * 0.8 + (N_exp - K * 0.8) * limitation
                else:
                    # 稳定期：基于前一时间点的密度，加入微小波动
                    if i > 0:
                        base_density = N[i-1]
                    else:
                        base_density = K * 0.9

                    # 稳定期的动态平衡：±1.5%波动 + 缓慢趋向载量
                    trend_factor = 1.0 + 0.01 * (K - base_density) / K  # 缓慢趋向载量
                    fluctuation = 1.0 + 0.03 * (np.random.random() - 0.5)  # ±1.5%
                    N[i] = base_density * trend_factor * fluctuation

                    # 确保在合理范围内
                    N[i] = np.clip(N[i], K * 0.85, K * 1.02)

        # 添加测量噪声，确保连续性
        for i in range(1, len(N)):
            if N[i] > 0:  # 只对非零值添加噪声
                # 噪声水平随密度和生长阶段调整
                if t[i] <= lag:
                    noise_level = 0.02  # 延滞期小噪声
                elif t[i] <= lag + 12:
                    noise_level = 0.03  # 对数期中等噪声
                else:
                    noise_level = 0.015  # 稳定期小噪声

                noise = np.random.normal(0, noise_level * N[i])
                N[i] = max(N[i] + noise, N[i-1] * 0.98)  # 防止不合理的下降

                # 确保最小值不低于初始密度的50%
                N[i] = max(N[i], N0 * 0.5)

        return N
    
    def generate_growth_curve(self, bacteria, ph, temperature):
        """Generate a realistic growth curve for given conditions"""

        if bacteria not in self.bacteria_params:
            raise ValueError(f"Unknown bacteria: {bacteria}")

        params = self.bacteria_params[bacteria]
        growth_factor = self.calculate_growth_factors(bacteria, ph, temperature)

        # 计算温度相关的代时
        doubling_time = self.calculate_doubling_time(bacteria, temperature)

        # 环境因子对代时的进一步影响
        if growth_factor < 0.7:
            doubling_time *= (1.5 / max(growth_factor, 0.1))  # 次优条件延长代时

        # 调整最大密度
        max_density = params['max_density'] * (0.1 + 0.9 * growth_factor)

        # 延滞期调整
        if growth_factor > 0.5:
            lag_time = params['lag_time'] / max(growth_factor, 0.5)
        else:
            lag_time = params['lag_time'] * (2.0 / max(growth_factor, 0.1))

        # 初始密度 (典型接种量)
        initial_density = random.uniform(1e4, 5e5)

        # 生成增强的生长曲线
        if growth_factor < 0.05:  # 极端恶劣条件 - 死亡/衰减
            densities = np.zeros_like(self.time_points)
            for i, t in enumerate(self.time_points):
                # 逐渐衰减
                decline_rate = 0.02 * (1 - growth_factor)
                densities[i] = initial_density * np.exp(-decline_rate * t)
                # 添加噪声
                noise = np.random.normal(0, 0.1 * densities[i])
                densities[i] = max(densities[i] + noise, initial_density * 0.1)
        else:
            densities = self.enhanced_growth_model(
                self.time_points,
                max_density,
                doubling_time,
                lag_time,
                initial_density
            )

        return densities
    
    def generate_dataset(self, n_samples=5000):
        """Generate a realistic dataset with 24 time points"""
        
        data = []
        bacteria_list = list(self.bacteria_params.keys())
        
        print(f"Generating {n_samples} bacterial growth curves (24 time points)...")
        
        for i in range(n_samples):
            if i % 500 == 0:
                print(f"Progress: {i}/{n_samples}")
                
            # Random bacteria
            bacteria = random.choice(bacteria_list)
            
            # Random conditions (weighted towards optimal ranges)
            params = self.bacteria_params[bacteria]
            
            # pH: 70% within optimal range, 30% outside
            if random.random() < 0.7:
                ph_min, ph_max = params['ph_range']
                ph = random.uniform(ph_min, ph_max)
            else:
                ph = random.uniform(4.0, 10.0)
                
            # Temperature: 70% within optimal range, 30% outside  
            if random.random() < 0.7:
                temp_min, temp_max = params['temp_range']
                temperature = random.uniform(temp_min, temp_max)
            else:
                temperature = random.uniform(5.0, 60.0)
            
            # Generate growth curve
            try:
                densities = self.generate_growth_curve(bacteria, ph, temperature)
                
                # Create row
                row = {
                    'bacteria_name': bacteria,
                    'ph': round(ph, 1),
                    'temperature': round(temperature, 1)
                }
                
                # Add density values for 24 time points
                for j, density in enumerate(densities):
                    row[f'density_hour_{j}'] = density
                    
                data.append(row)
                
            except Exception as e:
                print(f"Error generating curve for {bacteria}: {e}")
                continue
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} valid growth curves (24 time points)")
        
        return df

def print_doubling_time_info():
    """打印温度对代时影响的信息"""
    print("\n🧬 温度对细菌代时的影响:")
    print("=" * 50)

    generator = Bacteria24HDataGenerator()

    for bacteria, params in generator.bacteria_params.items():
        print(f"\n📊 {bacteria}:")
        print(f"   最适温度: {params['optimal_temp']}°C")
        print(f"   最适代时: {params['doubling_time_optimal']*60:.0f}分钟")

        # 显示不同温度下的代时
        temps = [15, 20, 25, 30, 37, 42]
        print(f"   温度-代时关系:")
        for temp in temps:
            if params['temp_range'][0] <= temp <= params['temp_range'][1]:
                dt = generator.calculate_doubling_time(bacteria, temp)
                print(f"     {temp}°C: {dt*60:.0f}分钟")
            else:
                print(f"     {temp}°C: 生长受限")

def main():
    """Generate 24-hour bacterial growth dataset"""

    generator = Bacteria24HDataGenerator()

    # 显示温度-代时关系
    print_doubling_time_info()

    # Generate dataset
    print(f"\n🔄 生成增强的24小时细菌生长数据集...")
    df = generator.generate_dataset(n_samples=5000)

    # Save dataset
    filename = 'bacteria_24h_enhanced_dataset.csv'
    df.to_csv(filename, index=False)
    print(f"✅ 数据集已保存为 '{filename}'")

    # Print statistics
    print(f"\n📊 数据集统计:")
    print(f"总样本数: {len(df)}")
    print(f"细菌种类: {df['bacteria_name'].nunique()}")
    print(f"pH范围: {df['ph'].min():.1f} - {df['ph'].max():.1f}")
    print(f"温度范围: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}°C")

    # Show density ranges
    density_cols = [col for col in df.columns if col.startswith('density_hour_')]
    all_densities = df[density_cols].values.flatten()
    print(f"密度范围: {all_densities.min():.2e} - {all_densities.max():.2e} CFU/mL")
    print(f"时间点数: {len(density_cols)} (0-23小时)")

    # 分析对数期增长特征
    print(f"\n🧬 对数期增长特征分析:")
    sample_curves = df.sample(3)
    for _, row in sample_curves.iterrows():
        bacteria = row['bacteria_name']
        temp = row['temperature']
        ph = row['ph']

        densities = [row[col] for col in density_cols]

        # 找到对数期 (密度增长最快的阶段)
        growth_rates = []
        for i in range(1, len(densities)):
            if densities[i] > densities[i-1]:
                rate = (densities[i] - densities[i-1]) / densities[i-1]
                growth_rates.append(rate)

        if growth_rates:
            max_growth_rate = max(growth_rates) * 100
            print(f"  {bacteria} (T={temp}°C, pH={ph}): 最大增长率 {max_growth_rate:.1f}%/小时")

if __name__ == "__main__":
    main()
