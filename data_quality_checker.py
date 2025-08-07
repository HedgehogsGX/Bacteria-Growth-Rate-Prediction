#!/usr/bin/env python3
"""
Data Quality Checker - 检查训练数据中的生物学不合理模式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_data_quality(csv_file):
    """检查数据质量，识别生物学不合理的样本"""
    
    print(f"🔍 检查数据文件: {csv_file}")
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"📊 总样本数: {len(df)}")
    
    # 提取密度数据列（假设从第4列开始是24个时间点的密度）
    density_columns = df.columns[3:27]  # 24个时间点
    print(f"📈 时间点数: {len(density_columns)}")
    
    problematic_samples = []
    
    for idx, row in df.iterrows():
        bacteria = row.iloc[0]
        ph = row.iloc[1] 
        temp = row.iloc[2]
        densities = row.iloc[3:27].values.astype(float)
        
        # 检查异常模式
        issues = []
        
        # 1. 检查零值或极小值
        zero_count = np.sum(densities == 0)
        very_small_count = np.sum(densities < 1000)
        
        if zero_count > 0:
            issues.append(f"包含{zero_count}个零值")
        
        if very_small_count > 5:  # 超过5个时间点密度过低
            issues.append(f"包含{very_small_count}个极小值(<1000)")
        
        # 2. 检查"死亡-复活"模式
        # 寻找密度从高到极低再到高的模式
        for i in range(1, len(densities)-1):
            if (densities[i-1] > 1e5 and  # 前一个时间点密度较高
                densities[i] < 1e3 and    # 当前时间点密度极低
                i < len(densities)-5):    # 确保后面还有足够的时间点
                
                # 检查后续是否有显著恢复
                future_max = np.max(densities[i+1:])
                if future_max > densities[i-1] * 0.1:  # 后续恢复到之前的10%以上
                    issues.append(f"时间点{i}出现'死亡-复活'模式")
        
        # 3. 检查剧烈波动
        for i in range(1, len(densities)):
            ratio = densities[i] / max(densities[i-1], 1)
            if ratio > 100:  # 增长超过100倍
                issues.append(f"时间点{i-1}到{i}增长过快({ratio:.1f}倍)")
            elif ratio < 0.01:  # 下降超过99%
                issues.append(f"时间点{i-1}到{i}下降过快({ratio:.3f}倍)")
        
        if issues:
            problematic_samples.append({
                'index': idx,
                'bacteria': bacteria,
                'ph': ph,
                'temperature': temp,
                'issues': issues,
                'densities': densities
            })
    
    print(f"\n⚠️  发现 {len(problematic_samples)} 个有问题的样本")
    
    # 显示前10个问题样本
    for i, sample in enumerate(problematic_samples[:10]):
        print(f"\n样本 {sample['index']}: {sample['bacteria']} (pH={sample['ph']}, T={sample['temperature']}°C)")
        for issue in sample['issues']:
            print(f"  - {issue}")
    
    if len(problematic_samples) > 10:
        print(f"  ... 还有 {len(problematic_samples)-10} 个问题样本")
    
    return problematic_samples

def clean_data(csv_file, output_file):
    """清理数据，移除或修复有问题的样本"""
    
    print(f"\n🧹 开始清理数据...")
    
    # 检查问题样本
    problematic_samples = check_data_quality(csv_file)
    problematic_indices = [s['index'] for s in problematic_samples]
    
    # 读取原始数据
    df = pd.read_csv(csv_file)
    
    print(f"📊 原始数据: {len(df)} 样本")
    print(f"❌ 问题样本: {len(problematic_indices)} 个")
    
    # 移除问题样本
    clean_df = df.drop(problematic_indices)
    
    print(f"✅ 清理后数据: {len(clean_df)} 样本")
    print(f"🗑️  移除了 {len(df) - len(clean_df)} 个问题样本")
    
    # 保存清理后的数据
    clean_df.to_csv(output_file, index=False)
    print(f"💾 清理后的数据已保存到: {output_file}")
    
    return clean_df

def visualize_problematic_sample(sample):
    """可视化问题样本的生长曲线"""

    # 设置字体以避免中文显示问题
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 6))

    time_points = list(range(24))
    densities = sample['densities']

    plt.plot(time_points, densities, 'b-o', linewidth=2, markersize=4)
    plt.yscale('log')
    plt.xlabel('Time (hours)')
    plt.ylabel('Density (CFU/mL)')
    plt.title(f"{sample['bacteria']} - pH={sample['ph']}, T={sample['temperature']}°C\nIssues: {', '.join(sample['issues'])}")
    plt.grid(True, alpha=0.3)

    # 标记零值点
    zero_points = np.where(densities == 0)[0]
    if len(zero_points) > 0:
        plt.scatter(zero_points, [1] * len(zero_points), color='red', s=100, marker='x', label='Zero values')
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 检查当前数据质量
    problematic = check_data_quality("bacteria_24h_enhanced_dataset.csv")
    
    if problematic:
        print(f"\n🔧 是否要清理数据？(y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            clean_data("bacteria_24h_enhanced_dataset.csv", "bacteria_24h_cleaned_dataset.csv")
        
        # 可视化第一个问题样本
        if problematic:
            print(f"\n📊 可视化第一个问题样本...")
            visualize_problematic_sample(problematic[0])
    else:
        print("✅ 数据质量良好，未发现明显问题！")
