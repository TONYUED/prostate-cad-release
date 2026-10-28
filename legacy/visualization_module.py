import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import ListedColormap
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class VisualizationModule:
    """
    可视化模块 - 生成肿瘤浸润免疫细胞分析的各种图表和可视化
    """
    
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 细胞类型颜色映射
        self.cell_colors = {
            'lymphocyte': '#FF0000',      # 红色 - 淋巴细胞
            'macrophage': '#00FF00',      # 绿色 - 巨噬细胞
            'neutrophil': '#0000FF',      # 蓝色 - 中性粒细胞
            'plasma_cell': '#FFFF00',     # 黄色 - 浆细胞
            'eosinophil': '#FF00FF',      # 品红色 - 嗜酸性粒细胞
            'tumor_cell': '#800080',      # 紫色 - 肿瘤细胞
            'stromal_cell': '#00FFFF',    # 青色 - 基质细胞
            'unknown': '#808080'          # 灰色 - 未知
        }
        
        # 细胞类型中文名称
        self.cell_names = {
            'lymphocyte': '淋巴细胞',
            'macrophage': '巨噬细胞',
            'neutrophil': '中性粒细胞',
            'plasma_cell': '浆细胞',
            'eosinophil': '嗜酸性粒细胞',
            'tumor_cell': '肿瘤细胞',
            'stromal_cell': '基质细胞',
            'unknown': '未知细胞'
        }
        
        # 浸润模式颜色
        self.pattern_colors = {
            'inflamed': '#FF6B6B',    # 红色 - 炎症型
            'excluded': '#4ECDC4',    # 青色 - 排斥型
            'desert': '#45B7D1',      # 蓝色 - 沙漠型
            'mixed': '#96CEB4'        # 绿色 - 混合型
        }
    
    def create_cell_detection_overlay(self, original_image, detection_results, 
                                    show_labels=True, show_confidence=False):
        """
        创建细胞检测结果叠加图
        """
        # 创建图像副本
        overlay_image = original_image.copy()
        if len(overlay_image.shape) == 2:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)
        
        cells = detection_results['cells']
        
        # 绘制检测结果
        for cell in cells:
            centroid = cell['centroid']
            cell_type = cell['type']
            confidence = cell['confidence']
            bbox = cell['bbox']
            
            # 获取颜色
            color_hex = self.cell_colors.get(cell_type, '#808080')
            color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            
            # 绘制中心点
            cv2.circle(overlay_image, (int(centroid[0]), int(centroid[1])), 
                      6, color_rgb, -1)
            cv2.circle(overlay_image, (int(centroid[0]), int(centroid[1])), 
                      8, (255, 255, 255), 2)
            
            # 绘制边界框
            x, y, w, h = bbox
            cv2.rectangle(overlay_image, (x, y), (x+w, y+h), color_rgb, 2)
            
            # 添加标签
            if show_labels:
                label = self.cell_names.get(cell_type, cell_type)
                if show_confidence:
                    label += f" ({confidence:.2f})"
                
                # 计算文本位置
                text_x = int(centroid[0]) + 10
                text_y = int(centroid[1]) - 10
                
                # 添加文本背景
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay_image, (text_x-2, text_y-text_h-2), 
                            (text_x+text_w+2, text_y+2), (255, 255, 255), -1)
                
                # 添加文本
                cv2.putText(overlay_image, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 1)
        
        return overlay_image
    
    def create_density_heatmap(self, spatial_analysis, image_shape):
        """
        创建细胞密度热图
        """
        density_grid = spatial_analysis['density']['grid']
        
        if density_grid.size == 0:
            # 创建空的热图
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, '无细胞检测到', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('细胞密度分布热图')
            return fig
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 创建热图
        im = ax.imshow(density_grid, cmap='hot', interpolation='bilinear')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('相对密度', fontsize=12)
        
        # 标记热点区域
        hotspots = spatial_analysis['density']['hotspots']
        for hotspot in hotspots:
            center_x, center_y = hotspot['center']
            circle = Circle((center_x, center_y), hotspot['area']**0.5, 
                          fill=False, color='cyan', linewidth=2)
            ax.add_patch(circle)
            ax.text(center_x, center_y, f"{hotspot['intensity']:.2f}", 
                   ha='center', va='center', color='cyan', fontweight='bold')
        
        ax.set_title('细胞密度分布热图', fontsize=14)
        ax.set_xlabel('X坐标 (网格单位)', fontsize=12)
        ax.set_ylabel('Y坐标 (网格单位)', fontsize=12)
        
        # 添加统计信息
        stats_text = f"最大密度: {spatial_analysis['density']['max_density']:.1f}\n"
        stats_text += f"平均密度: {spatial_analysis['density']['mean_density']:.2f}\n"
        stats_text += f"密度标准差: {spatial_analysis['density']['std_density']:.2f}\n"
        stats_text += f"热点区域: {len(hotspots)}个"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_cluster_visualization(self, spatial_analysis, detection_results):
        """
        创建细胞聚类可视化
        """
        cells = detection_results['cells']
        clusters = spatial_analysis['clusters']['clusters']
        labels = spatial_analysis['clusters'].get('labels', [])
        
        if len(cells) == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, '无细胞检测到', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('细胞聚类分析')
            return fig
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 提取细胞位置
        positions = np.array([cell['centroid'] for cell in cells])
        cell_types = [cell['type'] for cell in cells]
        
        # 绘制细胞点
        if len(labels) == len(positions):
            # 使用聚类标签着色
            unique_labels = set(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # 噪声点用黑色表示
                    mask = np.array(labels) == label
                    ax.scatter(positions[mask, 0], positions[mask, 1], 
                             c='black', marker='x', s=50, alpha=0.6, label='噪声点')
                else:
                    mask = np.array(labels) == label
                    ax.scatter(positions[mask, 0], positions[mask, 1], 
                             c=[color], s=60, alpha=0.7, label=f'聚类 {label}')
        else:
            # 使用细胞类型着色
            for cell_type in set(cell_types):
                mask = np.array(cell_types) == cell_type
                color = self.cell_colors.get(cell_type, '#808080')
                ax.scatter(positions[mask, 0], positions[mask, 1], 
                         c=color, s=60, alpha=0.7, label=self.cell_names.get(cell_type, cell_type))
        
        # 绘制聚类中心和边界
        for cluster in clusters:
            center = cluster['center']
            radius = cluster['radius']
            
            # 绘制聚类中心
            ax.plot(center[0], center[1], 'k+', markersize=15, markeredgewidth=3)
            
            # 绘制聚类边界
            circle = Circle(center, radius, fill=False, color='red', 
                          linewidth=2, linestyle='--', alpha=0.7)
            ax.add_patch(circle)
            
            # 添加聚类信息
            ax.text(center[0], center[1]-radius-20, 
                   f"聚类{cluster['id']}\n大小:{cluster['size']}", 
                   ha='center', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'细胞聚类分析 (共{len(clusters)}个聚类)', fontsize=14)
        ax.set_xlabel('X坐标 (像素)', fontsize=12)
        ax.set_ylabel('Y坐标 (像素)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 反转Y轴以匹配图像坐标
        ax.invert_yaxis()
        
        # 添加聚类质量信息
        silhouette_score = spatial_analysis['clusters'].get('silhouette_score', 0)
        quality_text = f"聚类数量: {len(clusters)}\n"
        quality_text += f"轮廓系数: {silhouette_score:.3f}\n"
        quality_text += f"聚类质量: {'优秀' if silhouette_score > 0.7 else '良好' if silhouette_score > 0.5 else '一般' if silhouette_score > 0.25 else '较差'}"
        
        ax.text(0.02, 0.02, quality_text, transform=ax.transAxes, 
               verticalalignment='bottom', bbox=dict(boxstyle='round', 
               facecolor='lightblue', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_cell_composition_chart(self, detection_results):
        """
        创建细胞组成饼图和柱状图
        """
        cells = detection_results['cells']
        
        if len(cells) == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.text(0.5, 0.5, '无细胞检测到', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=16)
            ax2.text(0.5, 0.5, '无细胞检测到', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=16)
            ax1.set_title('细胞类型分布')
            ax2.set_title('细胞数量统计')
            return fig
        
        # 统计细胞类型
        cell_counts = {}
        for cell in cells:
            cell_type = cell['type']
            cell_counts[cell_type] = cell_counts.get(cell_type, 0) + 1
        
        # 准备数据
        types = list(cell_counts.keys())
        counts = list(cell_counts.values())
        colors = [self.cell_colors.get(t, '#808080') for t in types]
        labels = [self.cell_names.get(t, t) for t in types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 饼图
        wedges, texts, autotexts = ax1.pie(counts, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('细胞类型分布', fontsize=14)
        
        # 美化饼图文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 柱状图
        bars = ax2.bar(labels, counts, color=colors, alpha=0.8)
        ax2.set_title('细胞数量统计', fontsize=14)
        ax2.set_xlabel('细胞类型', fontsize=12)
        ax2.set_ylabel('数量', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 添加总计信息
        total_cells = len(cells)
        immune_cells = sum([count for cell_type, count in cell_counts.items() 
                           if cell_type in ['lymphocyte', 'macrophage', 'neutrophil', 
                                          'plasma_cell', 'eosinophil']])
        
        stats_text = f"总细胞数: {total_cells}\n"
        stats_text += f"免疫细胞: {immune_cells}\n"
        stats_text += f"免疫细胞比例: {immune_cells/total_cells*100:.1f}%"
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='lightgreen', alpha=0.8), fontsize=11)
        
        plt.tight_layout()
        return fig