#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前列腺癌神经网络训练脚本
快速训练和部署前列腺泡腺癌识别模型
"""

import os
import sys
import numpy as np
from pathlib import Path

# 导入自定义模块
from prostate_cancer_cnn import ProstateCancerCNN

def quick_train():
    """
    快速训练前列腺癌识别模型
    """
    print("🚀 快速训练前列腺癌识别模型")
    print("=" * 50)
    
    try:
        # 创建模型实例
        cnn = ProstateCancerCNN(input_shape=(224, 224, 3), num_classes=5)
        
        # 构建模型
        print("🏗️  构建神经网络模型...")
        model = cnn.build_model(model_type='custom')
        
        # 创建合成训练数据
        print("🎨 生成合成训练数据...")
        images, labels = cnn.create_synthetic_data(num_samples_per_class=200)
        
        # 准备数据
        print("📊 准备训练数据...")
        train_data, val_data, test_data = cnn.prepare_data(images, labels)
        
        # 训练模型
        print("🚀 开始训练模型...")
        history = cnn.train_model(train_data, val_data, epochs=10, batch_size=16)
        
        # 评估模型
        print("📊 评估模型性能...")
        evaluation = cnn.evaluate_model(test_data)
        
        print(f"\n✅ 训练完成!")
        print(f"📈 最终准确率: {evaluation['accuracy']:.4f}")
        print(f"📈 精确率: {evaluation['precision']:.4f}")
        print(f"📈 召回率: {evaluation['recall']:.4f}")
        
        # 保存模型
        print("💾 保存模型...")
        cnn.save_model()
        
        # 绘制结果
        print("📊 生成训练图表...")
        cnn.plot_training_history()
        cnn.plot_confusion_matrix(evaluation['confusion_matrix'])
        
        print("\n🎉 前列腺癌神经网络训练成功完成!")
        print("💡 模型已保存，可以开始进行前列腺癌细胞识别了!")
        
        return cnn, evaluation
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_model():
    """
    测试训练好的模型
    """
    print("🔍 测试前列腺癌识别模型")
    print("=" * 50)
    
    try:
        # 加载模型
        cnn = ProstateCancerCNN()
        cnn.load_model()
        
        # 测试图像列表
        test_images = [
            "test_image.png",
            "test_image.jpg", 
            "test_image.tiff",
            "测试图像.png"
        ]
        
        print("📸 测试图像识别...")
        for img_path in test_images:
            if os.path.exists(img_path):
                try:
                    result = cnn.predict_image(img_path)
                    print(f"\n📁 {img_path}:")
                    print(f"   🎯 预测类别: {result['class_name']}")
                    print(f"   📊 Gleason评分: {result['gleason_score']}")
                    print(f"   🎲 置信度: {result['confidence']:.4f}")
                    
                    # 显示所有类别的概率
                    print("   📈 各类别概率:")
                    for class_name, prob in result['all_probabilities'].items():
                        print(f"      {class_name}: {prob:.4f}")
                        
                except Exception as e:
                    print(f"❌ 预测 {img_path} 失败: {e}")
            else:
                print(f"⚠️  文件不存在: {img_path}")
        
        print("\n✅ 模型测试完成!")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        print("💡 请先运行训练脚本创建模型")

def main():
    """
    主函数
    """
    print("🧠 前列腺泡腺癌神经网络系统")
    print("🔬 基于深度学习的前列腺癌细胞识别")
    print("=" * 60)
    
    while True:
        print("\n📋 请选择操作:")
        print("1. 🚀 快速训练模型")
        print("2. 🔍 测试现有模型")
        print("3. 📊 查看模型信息")
        print("4. 🚪 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            quick_train()
        elif choice == '2':
            test_model()
        elif choice == '3':
            show_model_info()
        elif choice == '4':
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重新输入")

def show_model_info():
    """
    显示模型信息
    """
    print("📊 模型信息")
    print("=" * 30)
    
    model_path = "models/prostate_cancer_model.h5"
    metadata_path = "models/prostate_cancer_model_metadata.json"
    
    if os.path.exists(model_path):
        print(f"✅ 模型文件: {model_path}")
        
        # 获取文件大小
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📦 文件大小: {size_mb:.2f} MB")
        
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"🎯 分类数量: {metadata.get('num_classes', 'N/A')}")
            print(f"📐 输入尺寸: {metadata.get('input_shape', 'N/A')}")
            print(f"📅 创建时间: {metadata.get('created_at', 'N/A')}")
            
            print("\n🏷️  类别定义:")
            class_names = metadata.get('class_names', {})
            for class_id, class_name in class_names.items():
                print(f"   {class_id}: {class_name}")
        
    else:
        print("❌ 未找到训练好的模型")
        print("💡 请先运行训练脚本创建模型")

if __name__ == "__main__":
    main()