#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前列腺泡腺癌神经网络识别系统
基于深度学习的前列腺癌细胞分类和检测
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
from datetime import datetime

# 深度学习框架
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class ProstateCancerCNN:
    """
    前列腺泡腺癌卷积神经网络分类器
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        # 前列腺癌细胞类型定义
        self.class_names = {
            0: '正常前列腺细胞',
            1: '低级别前列腺癌细胞',
            2: '中级别前列腺癌细胞', 
            3: '高级别前列腺癌细胞',
            4: '免疫浸润细胞'
        }
        
        # Gleason评分对应
        self.gleason_mapping = {
            0: 'Gleason 0 (正常)',
            1: 'Gleason 3+3=6 (低级别)',
            2: 'Gleason 3+4=7 (中级别)',
            3: 'Gleason 4+4=8+ (高级别)',
            4: '免疫细胞浸润'
        }
        
        print("🧠 前列腺泡腺癌神经网络系统初始化完成")
        print(f"📊 输入尺寸: {input_shape}")
        print(f"🎯 分类数量: {num_classes}")
    
    def build_model(self, model_type='custom'):
        """
        构建卷积神经网络模型
        """
        if model_type == 'custom':
            self.model = self._build_custom_cnn()
        elif model_type == 'resnet':
            self.model = self._build_resnet_model()
        elif model_type == 'efficientnet':
            self.model = self._build_efficientnet_model()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        print(f"✅ {model_type} 模型构建完成")
        return self.model
    
    def _build_custom_cnn(self):
        """
        构建自定义CNN模型 - 专门针对前列腺癌细胞特征优化
        """
        model = models.Sequential([
            # 第一个卷积块 - 提取低级特征
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二个卷积块 - 提取中级特征
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三个卷积块 - 提取高级特征
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第四个卷积块 - 细胞形态特征
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 全局平均池化
            layers.GlobalAveragePooling2D(),
            
            # 全连接层
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # 输出层
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_resnet_model(self):
        """
        基于ResNet的迁移学习模型
        """
        # 加载预训练的ResNet50
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # 冻结基础模型的权重
        base_model.trainable = False
        
        # 添加自定义分类头
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_efficientnet_model(self):
        """
        基于EfficientNet的迁移学习模型
        """
        # 加载预训练的EfficientNetB0
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_synthetic_data(self, num_samples_per_class=1000, save_path="synthetic_data"):
        """
        创建合成训练数据 - 模拟前列腺癌细胞图像
        """
        print("🎨 创建合成前列腺癌细胞数据...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 为每个类别创建目录
        for class_id, class_name in self.class_names.items():
            class_dir = os.path.join(save_path, f"class_{class_id}_{class_name}")
            os.makedirs(class_dir, exist_ok=True)
        
        all_images = []
        all_labels = []
        
        for class_id in range(self.num_classes):
            print(f"📸 生成 {self.class_names[class_id]} 样本...")
            
            for i in range(num_samples_per_class):
                # 生成合成细胞图像
                image = self._generate_synthetic_cell_image(class_id)
                
                # 保存图像
                filename = f"synthetic_{class_id}_{i:04d}.png"
                filepath = os.path.join(save_path, f"class_{class_id}_{self.class_names[class_id]}", filename)
                cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                all_images.append(image)
                all_labels.append(class_id)
        
        print(f"✅ 合成数据创建完成: {len(all_images)} 张图像")
        return np.array(all_images), np.array(all_labels)
    
    def _generate_synthetic_cell_image(self, class_id, size=(224, 224)):
        """
        生成合成的前列腺癌细胞图像
        """
        image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # 设置背景色 (模拟H&E染色的组织背景)
        if class_id == 0:  # 正常细胞
            bg_color = [245, 230, 230]  # 浅粉色背景
            cell_color = [200, 150, 180]  # 正常细胞质颜色
            nucleus_color = [100, 80, 120]  # 正常细胞核颜色
            cell_size_range = (15, 25)
            nucleus_size_ratio = 0.3
            cell_density = 0.6
        elif class_id == 1:  # 低级别癌细胞
            bg_color = [240, 225, 225]
            cell_color = [190, 140, 170]
            nucleus_color = [90, 70, 110]
            cell_size_range = (18, 30)
            nucleus_size_ratio = 0.4
            cell_density = 0.7
        elif class_id == 2:  # 中级别癌细胞
            bg_color = [235, 220, 220]
            cell_color = [180, 130, 160]
            nucleus_color = [80, 60, 100]
            cell_size_range = (20, 35)
            nucleus_size_ratio = 0.5
            cell_density = 0.8
        elif class_id == 3:  # 高级别癌细胞
            bg_color = [230, 215, 215]
            cell_color = [170, 120, 150]
            nucleus_color = [70, 50, 90]
            cell_size_range = (25, 45)
            nucleus_size_ratio = 0.6
            cell_density = 0.9
        else:  # 免疫浸润细胞
            bg_color = [240, 230, 235]
            cell_color = [160, 110, 140]
            nucleus_color = [60, 40, 80]
            cell_size_range = (10, 20)
            nucleus_size_ratio = 0.7
            cell_density = 1.0
        
        # 设置背景
        image[:, :] = bg_color
        
        # 添加随机噪声
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 生成细胞
        num_cells = int(cell_density * np.random.randint(20, 50))
        
        for _ in range(num_cells):
            # 随机位置
            center_x = np.random.randint(30, size[1] - 30)
            center_y = np.random.randint(30, size[0] - 30)
            
            # 随机大小
            cell_radius = np.random.randint(cell_size_range[0], cell_size_range[1])
            nucleus_radius = int(cell_radius * nucleus_size_ratio)
            
            # 绘制细胞质
            cv2.circle(image, (center_x, center_y), cell_radius, cell_color, -1)
            
            # 绘制细胞核
            nucleus_x = center_x + np.random.randint(-3, 4)
            nucleus_y = center_y + np.random.randint(-3, 4)
            cv2.circle(image, (nucleus_x, nucleus_y), nucleus_radius, nucleus_color, -1)
            
            # 添加细胞边界
            cv2.circle(image, (center_x, center_y), cell_radius, 
                      [max(0, c-20) for c in cell_color], 2)
        
        # 添加一些纤维结构
        for _ in range(np.random.randint(5, 15)):
            x1 = np.random.randint(0, size[1])
            y1 = np.random.randint(0, size[0])
            x2 = np.random.randint(0, size[1])
            y2 = np.random.randint(0, size[0])
            fiber_color = [min(255, c+20) for c in bg_color]
            cv2.line(image, (x1, y1), (x2, y2), fiber_color, 1)
        
        return image
    
    def prepare_data(self, images, labels, test_size=0.2, validation_size=0.2):
        """
        准备训练数据
        """
        print("📊 准备训练数据...")
        
        # 数据预处理
        images = images.astype('float32') / 255.0
        
        # 转换标签为one-hot编码
        labels_categorical = tf.keras.utils.to_categorical(labels, self.num_classes)
        
        # 分割数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels_categorical, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), random_state=42
        )
        
        print(f"✅ 数据分割完成:")
        print(f"   训练集: {X_train.shape[0]} 样本")
        print(f"   验证集: {X_val.shape[0]} 样本")
        print(f"   测试集: {X_test.shape[0]} 样本")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        创建数据增强生成器
        """
        # 训练数据增强
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # 验证数据不进行增强
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def train_model(self, train_data, val_data, epochs=50, batch_size=32, save_path="models"):
        """
        训练模型
        """
        if self.model is None:
            raise ValueError("请先构建模型!")
        
        print("🚀 开始训练模型...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # 创建数据生成器
        train_gen, val_gen = self.create_data_generators(X_train, y_train, X_val, y_val, batch_size)
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 回调函数
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(save_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 训练模型
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=len(X_val) // batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("✅ 模型训练完成!")
        
        # 保存训练历史
        with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
            json.dump(self.history.history, f)
        
        return self.history
    
    def evaluate_model(self, test_data):
        """
        评估模型性能
        """
        if self.model is None:
            raise ValueError("请先训练模型!")
        
        X_test, y_test = test_data
        
        print("📊 评估模型性能...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # 计算指标
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"测试准确率: {test_acc:.4f}")
        print(f"测试精确率: {test_precision:.4f}")
        print(f"测试召回率: {test_recall:.4f}")
        
        # 分类报告
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=list(self.class_names.values()),
                                     output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_true_classes,
            'pred_labels': y_pred_classes
        }
    
    def plot_training_history(self, save_path="plots"):
        """
        绘制训练历史
        """
        if self.history is None:
            print("❌ 没有训练历史数据")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 准确率
        axes[0, 0].plot(self.history.history['accuracy'], label='训练准确率')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='验证准确率')
        axes[0, 0].set_title('模型准确率')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].legend()
        
        # 损失
        axes[0, 1].plot(self.history.history['loss'], label='训练损失')
        axes[0, 1].plot(self.history.history['val_loss'], label='验证损失')
        axes[0, 1].set_title('模型损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].legend()
        
        # 精确率
        axes[1, 0].plot(self.history.history['precision'], label='训练精确率')
        axes[1, 0].plot(self.history.history['val_precision'], label='验证精确率')
        axes[1, 0].set_title('模型精确率')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('精确率')
        axes[1, 0].legend()
        
        # 召回率
        axes[1, 1].plot(self.history.history['recall'], label='训练召回率')
        axes[1, 1].plot(self.history.history['val_recall'], label='验证召回率')
        axes[1, 1].set_title('模型召回率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('召回率')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 训练历史图表已保存到: {save_path}/training_history.png")
    
    def plot_confusion_matrix(self, cm, save_path="plots"):
        """
        绘制混淆矩阵
        """
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.class_names.values()),
                   yticklabels=list(self.class_names.values()))
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 混淆矩阵已保存到: {save_path}/confusion_matrix.png")
    
    def predict_image(self, image_path):
        """
        预测单张图像
        """
        if self.model is None:
            raise ValueError("请先训练模型!")
        
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 预处理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_shape[:2])
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        # 预测
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        result = {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'gleason_score': self.gleason_mapping[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(prediction[0][i]) 
                for i in range(self.num_classes)
            }
        }
        
        return result
    
    def save_model(self, filepath="models/prostate_cancer_model.h5"):
        """
        保存模型
        """
        if self.model is None:
            raise ValueError("没有可保存的模型!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # 保存类别信息
        metadata = {
            'class_names': self.class_names,
            'gleason_mapping': self.gleason_mapping,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 模型已保存到: {filepath}")
        print(f"✅ 元数据已保存到: {metadata_path}")
    
    def load_model(self, filepath="models/prostate_cancer_model.h5"):
        """
        加载模型
        """
        if not os.path.exists(filepath):
            raise ValueError(f"模型文件不存在: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        
        # 加载元数据
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.class_names = metadata.get('class_names', self.class_names)
            self.gleason_mapping = metadata.get('gleason_mapping', self.gleason_mapping)
            self.input_shape = tuple(metadata.get('input_shape', self.input_shape))
            self.num_classes = metadata.get('num_classes', self.num_classes)
        
        print(f"✅ 模型已加载: {filepath}")
        return self.model


def main():
    """
    主函数 - 演示完整的训练流程
    """
    print("🧠 前列腺泡腺癌神经网络识别系统")
    print("=" * 60)
    
    # 创建模型实例
    cnn = ProstateCancerCNN(input_shape=(224, 224, 3), num_classes=5)
    
    # 构建模型
    model = cnn.build_model(model_type='custom')
    print(f"📋 模型结构:")
    model.summary()
    
    # 创建合成训练数据
    print("\n🎨 创建合成训练数据...")
    images, labels = cnn.create_synthetic_data(num_samples_per_class=500)
    
    # 准备数据
    train_data, val_data, test_data = cnn.prepare_data(images, labels)
    
    # 训练模型
    print("\n🚀 开始训练...")
    history = cnn.train_model(train_data, val_data, epochs=20, batch_size=32)
    
    # 评估模型
    print("\n📊 评估模型...")
    evaluation = cnn.evaluate_model(test_data)
    
    # 绘制结果
    cnn.plot_training_history()
    cnn.plot_confusion_matrix(evaluation['confusion_matrix'])
    
    # 保存模型
    cnn.save_model()
    
    # 测试预测
    print("\n🔍 测试预测功能...")
    test_files = ["test_image.png", "test_image.jpg"]
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                result = cnn.predict_image(test_file)
                print(f"\n📁 {test_file}:")
                print(f"   预测类别: {result['class_name']}")
                print(f"   Gleason评分: {result['gleason_score']}")
                print(f"   置信度: {result['confidence']:.4f}")
            except Exception as e:
                print(f"❌ 预测失败: {e}")
    
    print("\n✅ 前列腺癌神经网络系统训练完成!")
    print("💡 现在可以使用训练好的模型进行前列腺癌细胞识别了!")


if __name__ == "__main__":
    main()