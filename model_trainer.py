#!/usr/bin/env python3

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
import json
import os

class ImprovedBacteriaGrowthModel:
    def __init__(self):
        self.model = None
        self.genus_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.data_split_config = "data_split_config.json"
        
    def biological_continuity_loss(self, y_true, y_pred):
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

    def create_data_split(self, data, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
        print(f"📊 Creating data split configuration...")
        print(f"   Train: {train_size*100:.0f}%, Validation: {val_size*100:.0f}%, Test: {test_size*100:.0f}%")
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"

        total_samples = len(data['genus'])
        indices = np.arange(total_samples)

        # First split: separate test set (stratified by genus)
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=data['genus']
        )

        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (train_size + val_size)  # Adjust validation size
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=data['genus'][train_val_indices]
        )

        # Create split configuration
        split_config = {
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'test_indices': test_indices.tolist(),
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'total_size': total_samples,
            'random_state': random_state,
            'created_timestamp': pd.Timestamp.now().isoformat()
        }

        # Save configuration
        with open(self.data_split_config, 'w') as f:
            json.dump(split_config, f, indent=2)

        print(f"✅ Data split configuration saved to: {self.data_split_config}")
        print(f"   Training set: {len(train_indices)} samples ({len(train_indices)/total_samples*100:.1f}%)")
        print(f"   Validation set: {len(val_indices)} samples ({len(val_indices)/total_samples*100:.1f}%)")
        print(f"   Test set: {len(test_indices)} samples ({len(test_indices)/total_samples*100:.1f}%)")

        return split_config

    def load_data_split(self):
        """Load existing data split configuration"""

        if not os.path.exists(self.data_split_config):
            return None

        with open(self.data_split_config, 'r') as f:
            split_config = json.load(f)

        print(f"📂 Loaded existing data split configuration:")
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

        # Create data splits
        splits = {
            'train': {
                'genus': data['genus'][train_indices],
                'species': data['species'][train_indices],
                'env': data['env'][train_indices],
                'density_log': data['density_log'][train_indices]
            },
            'val': {
                'genus': data['genus'][val_indices],
                'species': data['species'][val_indices],
                'env': data['env'][val_indices],
                'density_log': data['density_log'][val_indices]
            },
            'test': {
                'genus': data['genus'][test_indices],
                'species': data['species'][test_indices],
                'env': data['env'][test_indices],
                'density_log': data['density_log'][test_indices]
            }
        }

        return splits

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
    
    def train_model(self, csv_file, epochs=100, batch_size=32, force_new_split=False):
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
        
        # 检查是否存在数据划分配置
        split_config = self.load_data_split()

        if split_config is None or force_new_split:
            print(f"🔄 Creating new data split...")
            split_config = self.create_data_split(data)
        else:
            print(f"📂 Using existing data split configuration")

        # 应用数据划分
        splits = self.apply_data_split(data, split_config)

        # 准备训练和验证数据
        X_train = {
            'genus_input': splits['train']['genus'],
            'species_input': splits['train']['species'],
            'env_input': splits['train']['env']
        }
        y_train = splits['train']['density_log']

        X_val = {
            'genus_input': splits['val']['genus'],
            'species_input': splits['val']['species'],
            'env_input': splits['val']['env']
        }
        y_val = splits['val']['density_log']

        print(f"\n📊 Final data split for training:")
        print(f"   Training set: {len(y_train)} samples ({len(y_train)/len(data['genus'])*100:.1f}%)")
        print(f"   Validation set: {len(y_val)} samples ({len(y_val)/len(data['genus'])*100:.1f}%)")
        print(f"   Test set: {split_config['test_size']} samples ({split_config['test_size']/len(data['genus'])*100:.1f}%) - RESERVED")
        
        # 回调函数
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6, monitor='val_loss')
        ]
        
        # 训练模型
        print(f"\n🚀 开始训练...")
        print(f"⚠️  注意: 测试集在训练过程中完全不可见，仅用于最终评估")
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
