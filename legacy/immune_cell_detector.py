import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from skimage import measure, morphology, filters
from skimage.feature import local_binary_pattern
import pandas as pd

class ImmuneCellDetector:
    """
    免疫细胞检测器 - 专门用于H&E染色图像中的肿瘤浸润免疫细胞识别
    """
    
    def __init__(self, config=None):
        # 保存配置
        self.config = config if config else {}
        
        # 免疫细胞类型定义
        self.cell_types = {
            'lymphocyte': '淋巴细胞',
            'macrophage': '巨噬细胞', 
            'neutrophil': '中性粒细胞',
            'plasma_cell': '浆细胞',
            'eosinophil': '嗜酸性粒细胞',
            'tumor_cell': '肿瘤细胞',
            'stromal_cell': '基质细胞'
        }
        
        # 细胞特征阈值
        self.cell_features = {
            'lymphocyte': {
                'size_range': (20, 80),      # 细胞大小范围(像素)
                'nucleus_ratio': (0.7, 0.9), # 核质比
                'circularity': (0.8, 1.0),   # 圆形度
                'intensity_range': (50, 120), # 灰度值范围
                'texture_contrast': (15, 40)  # 纹理对比度
            },
            'macrophage': {
                'size_range': (80, 200),
                'nucleus_ratio': (0.3, 0.6),
                'circularity': (0.6, 0.9),
                'intensity_range': (80, 150),
                'texture_contrast': (20, 60)
            },
            'neutrophil': {
                'size_range': (60, 120),
                'nucleus_ratio': (0.6, 0.8),
                'circularity': (0.7, 0.95),
                'intensity_range': (70, 130),
                'texture_contrast': (25, 50)
            },
            'plasma_cell': {
                'size_range': (70, 150),
                'nucleus_ratio': (0.4, 0.7),
                'circularity': (0.7, 0.9),
                'intensity_range': (60, 120),
                'texture_contrast': (30, 70)
            },
            'tumor_cell': {
                'size_range': (100, 300),
                'nucleus_ratio': (0.5, 0.8),
                'circularity': (0.5, 0.8),
                'intensity_range': (40, 100),
                'texture_contrast': (40, 100)
            }
        }
    
    def preprocess_he_image(self, image):
        """
        H&E图像预处理
        """
        # 颜色空间转换和增强
        if len(image.shape) == 3:
            # 转换到LAB颜色空间以更好地分离H&E染色
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # 分离H&E染色通道
            hematoxylin_channel = self._extract_hematoxylin(image)
            eosin_channel = self._extract_eosin(image)
            
            # 合成增强图像
            enhanced = cv2.merge([hematoxylin_channel, eosin_channel, lab[:,:,2]])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 转换为灰度图用于细胞检测
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 高斯滤波去噪
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray, enhanced
    
    def _extract_hematoxylin(self, image):
        """
        提取苏木精(细胞核)染色通道
        """
        # 颜色去卷积矩阵 - H&E染色
        he_matrix = np.array([
            [0.65, 0.70, 0.29],  # 苏木精
            [0.07, 0.99, 0.11],  # 伊红
            [0.27, 0.57, 0.78]   # 残留
        ])
        
        # 归一化图像
        img_norm = image.astype(np.float32) / 255.0
        img_norm = np.maximum(img_norm, 1e-6)
        
        # 对数变换
        log_img = -np.log(img_norm)
        
        # 颜色去卷积
        stains = np.dot(log_img.reshape(-1, 3), np.linalg.pinv(he_matrix))
        hematoxylin = stains[:, 0].reshape(image.shape[:2])
        
        # 归一化到0-255
        hematoxylin = np.clip(hematoxylin * 255, 0, 255).astype(np.uint8)
        
        return hematoxylin
    
    def _extract_eosin(self, image):
        """
        提取伊红(细胞质)染色通道
        """
        # 类似苏木精提取，但使用伊红通道
        he_matrix = np.array([
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78]
        ])
        
        img_norm = image.astype(np.float32) / 255.0
        img_norm = np.maximum(img_norm, 1e-6)
        log_img = -np.log(img_norm)
        
        stains = np.dot(log_img.reshape(-1, 3), np.linalg.pinv(he_matrix))
        eosin = stains[:, 1].reshape(image.shape[:2])
        
        eosin = np.clip(eosin * 255, 0, 255).astype(np.uint8)
        
        return eosin
    
    def segment_cells(self, image, min_size=20, max_size=500):
        """
        细胞分割
        """
        # 预处理
        gray, enhanced = self.preprocess_he_image(image)
        
        # 多尺度分割
        cells = []
        
        # 方法1: 自适应阈值分割
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # 方法2: Otsu阈值分割
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 方法3: 基于梯度的分割
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        _, thresh3 = cv2.threshold(gradient.astype(np.uint8), 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 组合分割结果
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, thresh3)
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 距离变换和分水岭分割
        dist_transform = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
        
        # 提取有效细胞
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_size <= area <= max_size:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                
                # 创建细胞掩码
                cell_mask = (labels == i).astype(np.uint8) * 255
                
                # 提取细胞区域
                cell_roi = gray[y:y+h, x:x+w]
                cell_mask_roi = cell_mask[y:y+h, x:x+w]
                
                if cell_roi.size > 0:
                    cells.append({
                        'id': i,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'centroid': centroids[i],
                        'roi': cell_roi,
                        'mask': cell_mask_roi,
                        'full_mask': cell_mask
                    })
        
        return cells, enhanced
    
    def extract_cell_features(self, cell_data, original_image):
        """
        提取细胞特征
        """
        features = {}
        
        roi = cell_data['roi']
        mask = cell_data['mask']
        
        if roi.size == 0 or mask.size == 0:
            return None
        
        # 基本几何特征
        features['area'] = cell_data['area']
        features['perimeter'] = cv2.arcLength(cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                                               cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
        features['circularity'] = 4 * np.pi * features['area'] / (features['perimeter']**2) if features['perimeter'] > 0 else 0
        features['aspect_ratio'] = cell_data['bbox'][2] / cell_data['bbox'][3] if cell_data['bbox'][3] > 0 else 1
        
        # 强度特征
        masked_roi = cv2.bitwise_and(roi, mask)
        non_zero_pixels = masked_roi[masked_roi > 0]
        
        if len(non_zero_pixels) > 0:
            features['mean_intensity'] = np.mean(non_zero_pixels)
            features['std_intensity'] = np.std(non_zero_pixels)
            features['min_intensity'] = np.min(non_zero_pixels)
            features['max_intensity'] = np.max(non_zero_pixels)
        else:
            features['mean_intensity'] = 0
            features['std_intensity'] = 0
            features['min_intensity'] = 0
            features['max_intensity'] = 0
        
        # 纹理特征 - 局部二值模式
        if roi.shape[0] > 8 and roi.shape[1] > 8:
            lbp = local_binary_pattern(roi, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
            features['lbp_uniformity'] = np.sum(lbp_hist**2) / np.sum(lbp_hist)**2 if np.sum(lbp_hist) > 0 else 0
            features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)) / len(lbp_hist)
        else:
            features['lbp_uniformity'] = 0
            features['lbp_entropy'] = 0
        
        # 梯度特征
        if roi.shape[0] > 3 and roi.shape[1] > 3:
            sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            features['mean_gradient'] = np.mean(gradient_magnitude)
            features['std_gradient'] = np.std(gradient_magnitude)
        else:
            features['mean_gradient'] = 0
            features['std_gradient'] = 0
        
        # 形状特征
        if len(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) > 0:
            contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                features['ellipse_ratio'] = ellipse[1][0] / ellipse[1][1] if ellipse[1][1] > 0 else 1
                
                # 凸包特征
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                features['solidity'] = features['area'] / hull_area if hull_area > 0 else 0
            else:
                features['ellipse_ratio'] = 1
                features['solidity'] = 1
        else:
            features['ellipse_ratio'] = 1
            features['solidity'] = 1
        
        # 核质比估算 (基于强度分布)
        if len(non_zero_pixels) > 0:
            # 假设较暗的区域是细胞核
            nucleus_threshold = np.percentile(non_zero_pixels, 30)
            nucleus_pixels = non_zero_pixels[non_zero_pixels <= nucleus_threshold]
            features['nucleus_ratio'] = len(nucleus_pixels) / len(non_zero_pixels)
        else:
            features['nucleus_ratio'] = 0
        
        return features
    
    def classify_cell(self, features):
        """
        基于特征对细胞进行分类
        """
        if features is None:
            return 'unknown', 0.0
        
        scores = {}
        
        for cell_type, thresholds in self.cell_features.items():
            score = 0
            total_criteria = 0
            
            # 大小评分
            if thresholds['size_range'][0] <= features['area'] <= thresholds['size_range'][1]:
                score += 2
            elif features['area'] < thresholds['size_range'][0] * 1.5 or features['area'] > thresholds['size_range'][1] * 0.7:
                score += 1
            total_criteria += 2
            
            # 核质比评分
            if thresholds['nucleus_ratio'][0] <= features['nucleus_ratio'] <= thresholds['nucleus_ratio'][1]:
                score += 2
            elif abs(features['nucleus_ratio'] - np.mean(thresholds['nucleus_ratio'])) < 0.2:
                score += 1
            total_criteria += 2
            
            # 圆形度评分
            if thresholds['circularity'][0] <= features['circularity'] <= thresholds['circularity'][1]:
                score += 1
            total_criteria += 1
            
            # 强度评分
            if thresholds['intensity_range'][0] <= features['mean_intensity'] <= thresholds['intensity_range'][1]:
                score += 1
            total_criteria += 1
            
            # 纹理对比度评分
            if thresholds['texture_contrast'][0] <= features['std_intensity'] <= thresholds['texture_contrast'][1]:
                score += 1
            total_criteria += 1
            
            # 计算归一化得分
            scores[cell_type] = score / total_criteria if total_criteria > 0 else 0
        
        # 选择得分最高的细胞类型
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # 设置最低置信度阈值
        if confidence < 0.4:
            return 'unknown', confidence
        
        return best_type, confidence
    
    def detect_immune_cells(self, image, min_size=20, max_size=500):
        """
        检测和分类免疫细胞
        """
        # 分割细胞
        cells, enhanced_image = self.segment_cells(image, min_size, max_size)
        
        # 分类结果
        results = {
            'cells': [],
            'counts': {cell_type: 0 for cell_type in self.cell_types.keys()},
            'counts': {**{cell_type: 0 for cell_type in self.cell_types.keys()}, 'unknown': 0},
            'enhanced_image': enhanced_image,
            'total_cells': len(cells)
        }
        
        # 对每个细胞进行特征提取和分类
        for cell in cells:
            features = self.extract_cell_features(cell, image)
            cell_type, confidence = self.classify_cell(features)
            
            cell_result = {
                'id': cell['id'],
                'type': cell_type,
                'confidence': confidence,
                'centroid': cell['centroid'],
                'bbox': cell['bbox'],
                'area': cell['area'],
                'features': features
            }
            
            results['cells'].append(cell_result)
            results['counts'][cell_type] += 1
        
        return results
    
    def visualize_results(self, image, results, show_labels=True):
        """
        可视化检测结果
        """
        # 创建结果图像
        result_img = image.copy()
        if len(result_img.shape) == 2:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        
        # 定义颜色映射
        colors = {
            'lymphocyte': (255, 0, 0),      # 红色
            'macrophage': (0, 255, 0),      # 绿色
            'neutrophil': (0, 0, 255),      # 蓝色
            'plasma_cell': (255, 255, 0),   # 青色
            'eosinophil': (255, 0, 255),    # 品红色
            'tumor_cell': (128, 0, 128),    # 紫色
            'stromal_cell': (0, 255, 255),  # 黄色
            'unknown': (128, 128, 128)      # 灰色
        }
        
        # 绘制检测结果
        for cell in results['cells']:
            centroid = cell['centroid']
            cell_type = cell['type']
            confidence = cell['confidence']
            
            # 获取颜色
            color = colors.get(cell_type, (128, 128, 128))
            
            # 绘制圆点
            cv2.circle(result_img, (int(centroid[0]), int(centroid[1])), 5, color, -1)
            
            # 绘制边框
            x, y, w, h = cell['bbox']
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 1)
            
            # 添加标签
            if show_labels:
                label = f"{self.cell_types.get(cell_type, cell_type)[:2]}"
                cv2.putText(result_img, label, (int(centroid[0])+8, int(centroid[1])+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result_img
    
    def generate_report(self, results):
        """
        生成分析报告
        """
        report = {
            'summary': {},
            'detailed_counts': results['counts'],
            'immune_infiltration_score': 0,
            'dominant_immune_cells': [],
            'spatial_distribution': {}
        }
        
        total_cells = results['total_cells']
        immune_cells = sum([count for cell_type, count in results['counts'].items() 
                           if cell_type in ['lymphocyte', 'macrophage', 'neutrophil', 
                                          'plasma_cell', 'eosinophil']])
        
        # 基本统计
        report['summary']['total_cells'] = total_cells
        report['summary']['immune_cells'] = immune_cells
        report['summary']['tumor_cells'] = results['counts'].get('tumor_cell', 0)
        report['summary']['stromal_cells'] = results['counts'].get('stromal_cell', 0)
        
        # 免疫浸润评分
        if total_cells > 0:
            immune_ratio = immune_cells / total_cells
            report['immune_infiltration_score'] = immune_ratio * 100
            
            # 判断浸润程度
            if immune_ratio > 0.3:
                report['infiltration_level'] = '高度浸润'
            elif immune_ratio > 0.15:
                report['infiltration_level'] = '中度浸润'
            elif immune_ratio > 0.05:
                report['infiltration_level'] = '轻度浸润'
            else:
                report['infiltration_level'] = '无明显浸润'
        
        # 主要免疫细胞类型
        immune_counts = {k: v for k, v in results['counts'].items() 
                        if k in ['lymphocyte', 'macrophage', 'neutrophil', 'plasma_cell', 'eosinophil'] and v > 0}
        
        if immune_counts:
            sorted_immune = sorted(immune_counts.items(), key=lambda x: x[1], reverse=True)
            report['dominant_immune_cells'] = [
                {'type': self.cell_types[cell_type], 'count': count, 'percentage': count/immune_cells*100}
                for cell_type, count in sorted_immune[:3]
            ]
        
        return report