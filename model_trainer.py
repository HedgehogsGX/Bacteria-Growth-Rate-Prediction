#!/usr/bin/env python3
"""
Improved Model Trainer - 使用清理后的数据和生物学约束训练改进模型
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ImprovedBacteriaGrowthModel:
    def __init__(self):
        self.model = None
        self.genus_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def biological_continuity_loss(self, y_true, y_pred):
        """生物学连续性损失函数 - 惩罚相邻时间点的剧烈变化"""

        # 基础MSE损失
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # 连续性损失 - 惩罚相邻时间点的剧烈变化
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        continuity_loss = tf.reduce_mean(tf.square(diff_true - diff_pred))

        # 单调性损失 - 在早期阶段（前8小时）惩罚剧烈下降
        early_pred = y_pred[:, :8]  # 前8个时间点
        early_diff = early_pred[:, 1:] - early_pred[:, :-1]
        # 惩罚负的变化（下降）
        monotonic_penalty = tf.reduce_mean(tf.maximum(-early_diff, 0.0))

        # 零值惩罚 - 强烈惩罚预测为零或极小值
        zero_penalty = tf.reduce_mean(tf.maximum(3.0 - y_pred, 0.0))  # log10(1000) ≈ 3.0

        # 组合损失
        total_loss = mse_loss + 0.1 * continuity_loss + 0.2 * monotonic_penalty + 0.5 * zero_penalty

        return total_loss
    
    def create_improved_model(self, num_genera, num_species):
        """创建改进的模型架构"""
        
        # 输入层
        genus_input = Input(shape=(1,), name='genus_input')
        species_input = Input(shape=(1,), name='species_input')
        env_input = Input(shape=(2,), name='env_input')  # pH, temperature
        
        # 嵌入层 - 更大的嵌入维度
        genus_embedding = Embedding(num_genera, 16, name='genus_embedding')(genus_input)
        species_embedding = Embedding(num_species, 16, name='species_embedding')(species_input)
        
        # 展平嵌入
        genus_flat = Flatten()(genus_embedding)
        species_flat = Flatten()(species_embedding)
        
        # 合并所有特征
        merged = Concatenate()([genus_flat, species_flat, env_input])
        
        # 更深的网络结构，加入批归一化和dropout
        x = Dense(256, activation='relu')(merged)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # 输出层 - 24个时间点的密度预测
        # 使用sigmoid激活确保输出在合理范围内，然后缩放到log10范围
        output = Dense(24, activation='sigmoid', name='growth_curve')(x)
        # 将sigmoid输出(0-1)映射到合理的log10范围(3-9)，对应1e3到1e9 CFU/mL
        output = tf.keras.layers.Lambda(lambda x: x * 6.0 + 3.0)(output)
        
        model = Model(inputs=[genus_input, species_input, env_input], outputs=output)
        
        return model
    
    def prepare_data(self, csv_file):
        """准备训练数据"""
        
        print(f"📂 加载数据: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # 解析细菌名称
        bacteria_names = df.iloc[:, 0].str.split(' ', expand=True)
        genera = bacteria_names[0]
        species = bacteria_names[1]
        
        # 编码细菌信息
        genus_encoded = self.genus_encoder.fit_transform(genera)
        species_encoded = self.species_encoder.fit_transform(species)
        
        # 环境特征
        env_features = df[['ph', 'temperature']].values
        env_features_scaled = self.scaler.fit_transform(env_features)
        
        # 密度数据 - 转换为log10
        density_data = df.iloc[:, 3:27].values  # 24个时间点
        # 确保没有零值或负值
        density_data = np.maximum(density_data, 1.0)
        density_log = np.log10(density_data)
        
        print(f"📊 数据统计:")
        print(f"   样本数: {len(df)}")
        print(f"   细菌属: {len(self.genus_encoder.classes_)}")
        print(f"   细菌种: {len(self.species_encoder.classes_)}")
        print(f"   密度范围: {np.min(density_data):.2e} - {np.max(density_data):.2e} CFU/mL")
        print(f"   Log10范围: {np.min(density_log):.2f} - {np.max(density_log):.2f}")
        
        return {
            'genus': genus_encoded,
            'species': species_encoded,
            'env': env_features_scaled,
            'density_log': density_log
        }
    
    def train_model(self, csv_file, epochs=100, batch_size=32):
        """训练改进的模型"""
        
        # 准备数据
        data = self.prepare_data(csv_file)
        
        # 创建模型
        num_genera = len(self.genus_encoder.classes_)
        num_species = len(self.species_encoder.classes_)
        self.model = self.create_improved_model(num_genera, num_species)
        
        # 编译模型 - 使用自定义损失函数
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.biological_continuity_loss,
            metrics=['mse']
        )
        
        print(f"\n🏗️  模型架构:")
        self.model.summary()
        
        # 准备训练数据
        X = {
            'genus_input': data['genus'],
            'species_input': data['species'],
            'env_input': data['env']
        }
        y = data['density_log']
        
        # 分割训练和验证数据
        X_train = {}
        X_val = {}
        for key in X.keys():
            X_train[key], X_val[key], y_train, y_val = train_test_split(
                X[key], y, test_size=0.2, random_state=42
            )
        
        print(f"\n📊 数据分割:")
        print(f"   训练集: {len(y_train)} 样本")
        print(f"   验证集: {len(y_val)} 样本")
        
        # 回调函数
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6, monitor='val_loss')
        ]
        
        # 训练模型
        print(f"\n🚀 开始训练...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, model_path):
        """保存模型"""
        self.model.save(model_path)
        print(f"💾 模型已保存到: {model_path}")

if __name__ == "__main__":
    # 创建改进的模型训练器
    trainer = ImprovedBacteriaGrowthModel()
    
    # 使用清理后的数据训练模型
    print("🧠 训练改进的细菌生长预测模型...")
    history = trainer.train_model("bacteria_24h_cleaned_dataset.csv", epochs=150, batch_size=64)
    
    # 保存改进的模型
    trainer.save_model("bacteria_growth_model.h5")
    
    print("✅ 改进模型训练完成！")
