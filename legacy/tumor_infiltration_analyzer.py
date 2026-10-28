import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull, Voronoi
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TumorInfiltrationAnalyzer:
    """
    肿瘤浸润分析器 - 分析免疫细胞在肿瘤组织中的空间分布和浸润模式
    """
    
    def __init__(self, config=None):
        # 保存配置
        self.config = config if config else {}
        
        # 免疫细胞类型权重 (用于计算浸润评分)
        self.immune_cell_weights = {
            'lymphocyte': 1.0,      # 淋巴细胞 - 主要抗肿瘤免疫细胞
            'macrophage': 0.8,      # 巨噬细胞 - 可促进或抑制肿瘤
            'neutrophil': 0.6,      # 中性粒细胞 - 炎症反应
            'plasma_cell': 0.9,     # 浆细胞 - 体液免疫
            'eosinophil': 0.5,      # 嗜酸性粒细胞 - 过敏反应
        }
        
        # 浸润模式定义
        self.infiltration_patterns = {
            'excluded': '免疫排斥型',
            'inflamed': '免疫炎症型', 
            'desert': '免疫沙漠型',
            'mixed': '混合型'
        }
        
        # 空间分析参数
        self.spatial_params = {
            'neighborhood_radius': 100,  # 邻域半径(像素)
            'cluster_eps': 50,          # DBSCAN聚类参数
            'min_samples': 3,           # 最小样本数
            'grid_size': 50             # 网格大小
        }
    
    def analyze_spatial_distribution(self, detection_results, image_shape):
        """
        分析免疫细胞的空间分布
        """
        cells = detection_results['cells']
        
        if len(cells) == 0:
            return self._empty_spatial_analysis()
        
        # 提取细胞位置和类型
        positions = np.array([cell['centroid'] for cell in cells])
        cell_types = [cell['type'] for cell in cells]
        
        # 空间分析结果
        spatial_analysis = {
            'total_cells': len(cells),
            'cell_positions': positions,
            'cell_types': cell_types,
            'image_shape': image_shape
        }
        
        # 1. 密度分析
        spatial_analysis['density'] = self._analyze_cell_density(positions, image_shape)
        
        # 2. 聚类分析
        spatial_analysis['clusters'] = self._analyze_cell_clusters(positions, cell_types)
        
        # 3. 邻域分析
        spatial_analysis['neighborhoods'] = self._analyze_neighborhoods(positions, cell_types)
        
        # 4. 空间自相关分析
        spatial_analysis['spatial_autocorr'] = self._analyze_spatial_autocorrelation(positions, cell_types)
        
        # 5. 距离分析
        spatial_analysis['distances'] = self._analyze_cell_distances(positions, cell_types)
        
        return spatial_analysis
    
    def _analyze_cell_density(self, positions, image_shape):
        """
        分析细胞密度分布
        """
        h, w = image_shape[:2]
        grid_size = self.spatial_params['grid_size']
        
        # 创建密度网格
        grid_h = h // grid_size
        grid_w = w // grid_size
        density_grid = np.zeros((grid_h, grid_w))
        
        # 计算每个网格的细胞密度
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            grid_x = min(x // grid_size, grid_w - 1)
            grid_y = min(y // grid_size, grid_h - 1)
            density_grid[grid_y, grid_x] += 1
        
        # 归一化密度
        max_density = np.max(density_grid)
        if max_density > 0:
            density_grid = density_grid / max_density
        
        return {
            'grid': density_grid,
            'max_density': max_density,
            'mean_density': np.mean(density_grid),
            'std_density': np.std(density_grid),
            'hotspots': self._find_density_hotspots(density_grid)
        }
    
    def _find_density_hotspots(self, density_grid, threshold=0.7):
        """
        找到密度热点区域
        """
        hotspots = []
        hotspot_mask = density_grid > threshold
        
        # 连通组件分析
        num_labels, labels = cv2.connectedComponents(hotspot_mask.astype(np.uint8))
        
        for i in range(1, num_labels):
            component_mask = (labels == i)
            y_coords, x_coords = np.where(component_mask)
            
            if len(y_coords) > 0:
                center_y = np.mean(y_coords)
                center_x = np.mean(x_coords)
                area = len(y_coords)
                intensity = np.mean(density_grid[component_mask])
                
                hotspots.append({
                    'center': (center_x, center_y),
                    'area': area,
                    'intensity': intensity
                })
        
        return hotspots
    
    def _analyze_cell_clusters(self, positions, cell_types):
        """
        分析细胞聚类
        """
        if len(positions) < 3:
            return {'clusters': [], 'n_clusters': 0, 'silhouette_score': 0}
        
        # DBSCAN聚类
        eps = self.spatial_params['cluster_eps']
        min_samples = self.spatial_params['min_samples']
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        
        # 计算聚类质量
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            # 排除噪声点计算轮廓系数
            mask = labels != -1
            if np.sum(mask) > 1:
                silhouette_avg = silhouette_score(positions[mask], labels[mask])
            else:
                silhouette_avg = 0
        else:
            silhouette_avg = 0
        
        # 分析每个聚类
        clusters = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # 跳过噪声点
                continue
                
            cluster_mask = labels == cluster_id
            cluster_positions = positions[cluster_mask]
            cluster_cell_types = [cell_types[i] for i in range(len(cell_types)) if cluster_mask[i]]
            
            # 计算聚类特征
            center = np.mean(cluster_positions, axis=0)
            radius = np.max(pdist(cluster_positions)) / 2 if len(cluster_positions) > 1 else 0
            
            # 细胞类型分布
            type_counts = {}
            for cell_type in cluster_cell_types:
                type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
            
            clusters.append({
                'id': cluster_id,
                'center': center,
                'radius': radius,
                'size': len(cluster_positions),
                'cell_types': type_counts,
                'positions': cluster_positions
            })
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'labels': labels
        }
    
    def _analyze_neighborhoods(self, positions, cell_types):
        """
        分析细胞邻域
        """
        if len(positions) < 2:
            return {'neighborhoods': [], 'interaction_matrix': {}}
        
        radius = self.spatial_params['neighborhood_radius']
        
        # 构建邻域图
        nbrs = NearestNeighbors(radius=radius).fit(positions)
        distances, indices = nbrs.radius_neighbors(positions)
        
        neighborhoods = []
        interaction_counts = {}
        
        for i, (neighbor_dists, neighbor_indices) in enumerate(zip(distances, indices)):
            center_type = cell_types[i]
            
            # 统计邻域内的细胞类型
            neighbor_types = [cell_types[j] for j in neighbor_indices if j != i]
            type_counts = {}
            for neighbor_type in neighbor_types:
                type_counts[neighbor_type] = type_counts.get(neighbor_type, 0) + 1
            
            neighborhoods.append({
                'center_cell': i,
                'center_type': center_type,
                'neighbor_count': len(neighbor_indices) - 1,
                'neighbor_types': type_counts,
                'avg_distance': np.mean(neighbor_dists[1:]) if len(neighbor_dists) > 1 else 0
            })
            
            # 统计细胞类型间的相互作用
            for neighbor_type in neighbor_types:
                interaction_key = f"{center_type}-{neighbor_type}"
                interaction_counts[interaction_key] = interaction_counts.get(interaction_key, 0) + 1
        
        return {
            'neighborhoods': neighborhoods,
            'interaction_matrix': interaction_counts
        }
    
    def _analyze_spatial_autocorrelation(self, positions, cell_types):
        """
        分析空间自相关性 - 相同类型细胞是否倾向于聚集
        """
        if len(positions) < 3:
            return {'moran_i': {}, 'geary_c': {}}
        
        # 构建空间权重矩阵
        distances = squareform(pdist(positions))
        
        # 使用反距离权重
        weights = 1.0 / (distances + 1e-10)
        np.fill_diagonal(weights, 0)
        
        # 行标准化
        row_sums = np.sum(weights, axis=1)
        weights = weights / (row_sums[:, np.newaxis] + 1e-10)
        
        moran_i = {}
        geary_c = {}
        
        # 对每种细胞类型计算空间自相关
        unique_types = list(set(cell_types))
        for cell_type in unique_types:
            # 创建二进制指示变量
            y = np.array([1 if ct == cell_type else 0 for ct in cell_types])
            
            if np.sum(y) > 1:  # 至少需要2个该类型的细胞
                # Moran's I
                n = len(y)
                y_mean = np.mean(y)
                numerator = np.sum(weights * np.outer(y - y_mean, y - y_mean))
                denominator = np.sum((y - y_mean) ** 2)
                
                if denominator > 0:
                    moran_i[cell_type] = (n / np.sum(weights)) * (numerator / denominator)
                else:
                    moran_i[cell_type] = 0
                
                # Geary's C
                numerator = np.sum(weights * (np.subtract.outer(y, y) ** 2))
                if denominator > 0:
                    geary_c[cell_type] = ((n - 1) / (2 * np.sum(weights))) * (numerator / denominator)
                else:
                    geary_c[cell_type] = 1
            else:
                moran_i[cell_type] = 0
                geary_c[cell_type] = 1
        
        return {
            'moran_i': moran_i,  # > 0: 正相关(聚集), < 0: 负相关(分散)
            'geary_c': geary_c   # < 1: 正相关(聚集), > 1: 负相关(分散)
        }
    
    def _analyze_cell_distances(self, positions, cell_types):
        """
        分析细胞间距离
        """
        if len(positions) < 2:
            return {'nearest_neighbor': {}, 'type_distances': {}}
        
        # 最近邻距离
        nbrs = NearestNeighbors(n_neighbors=2).fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        
        nearest_neighbor = {}
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            cell_type = cell_types[i]
            neighbor_type = cell_types[idx[1]]  # idx[0]是自己
            nn_distance = dist[1]
            
            if cell_type not in nearest_neighbor:
                nearest_neighbor[cell_type] = []
            nearest_neighbor[cell_type].append({
                'distance': nn_distance,
                'neighbor_type': neighbor_type
            })
        
        # 计算不同细胞类型间的平均距离
        type_distances = {}
        unique_types = list(set(cell_types))
        
        for type1 in unique_types:
            for type2 in unique_types:
                if type1 <= type2:  # 避免重复计算
                    indices1 = [i for i, ct in enumerate(cell_types) if ct == type1]
                    indices2 = [i for i, ct in enumerate(cell_types) if ct == type2]
                    
                    if indices1 and indices2:
                        pos1 = positions[indices1]
                        pos2 = positions[indices2]
                        
                        if type1 == type2:
                            # 同类型细胞间距离
                            if len(pos1) > 1:
                                dists = pdist(pos1)
                                type_distances[f"{type1}-{type1}"] = {
                                    'mean': np.mean(dists),
                                    'std': np.std(dists),
                                    'min': np.min(dists)
                                }
                        else:
                            # 不同类型细胞间距离
                            dists = []
                            for p1 in pos1:
                                for p2 in pos2:
                                    dists.append(np.linalg.norm(p1 - p2))
                            
                            if dists:
                                type_distances[f"{type1}-{type2}"] = {
                                    'mean': np.mean(dists),
                                    'std': np.std(dists),
                                    'min': np.min(dists)
                                }
        
        return {
            'nearest_neighbor': nearest_neighbor,
            'type_distances': type_distances
        }
    
    def classify_infiltration_pattern(self, spatial_analysis, detection_results):
        """
        分类肿瘤浸润模式
        """
        cells = detection_results['cells']
        total_cells = len(cells)
        
        if total_cells == 0:
            return 'desert', 0.0, "无细胞检测到"
        
        # 统计免疫细胞
        immune_cells = [cell for cell in cells if cell['type'] in self.immune_cell_weights]
        tumor_cells = [cell for cell in cells if cell['type'] == 'tumor_cell']
        
        immune_count = len(immune_cells)
        tumor_count = len(tumor_cells)
        
        # 计算免疫细胞比例
        immune_ratio = immune_count / total_cells
        
        # 分析空间分布特征
        density_info = spatial_analysis.get('density', {})
        cluster_info = spatial_analysis.get('clusters', {})
        
        # 决策逻辑
        pattern_scores = {
            'desert': 0,
            'excluded': 0,
            'inflamed': 0,
            'mixed': 0
        }
        
        # 免疫沙漠型 - 免疫细胞很少
        if immune_ratio < 0.05:
            pattern_scores['desert'] += 3
        elif immune_ratio < 0.1:
            pattern_scores['desert'] += 1
        
        # 免疫排斥型 - 免疫细胞存在但与肿瘤细胞分离
        if immune_count > 0 and tumor_count > 0:
            # 分析免疫细胞和肿瘤细胞的空间分离程度
            immune_positions = np.array([cell['centroid'] for cell in immune_cells])
            tumor_positions = np.array([cell['centroid'] for cell in tumor_cells])
            
            # 计算免疫细胞到最近肿瘤细胞的平均距离
            if len(immune_positions) > 0 and len(tumor_positions) > 0:
                min_distances = []
                for immune_pos in immune_positions:
                    distances = [np.linalg.norm(immune_pos - tumor_pos) for tumor_pos in tumor_positions]
                    min_distances.append(min(distances))
                
                avg_min_distance = np.mean(min_distances)
                
                # 如果平均距离较大，说明可能是排斥型
                if avg_min_distance > 100:  # 像素距离阈值
                    pattern_scores['excluded'] += 2
                elif avg_min_distance > 50:
                    pattern_scores['excluded'] += 1
        
        # 免疫炎症型 - 免疫细胞丰富且与肿瘤细胞混合
        if immune_ratio > 0.2:
            pattern_scores['inflamed'] += 2
            
            # 检查聚类情况
            n_clusters = cluster_info.get('n_clusters', 0)
            if n_clusters > 2:  # 多个聚类可能表示炎症反应
                pattern_scores['inflamed'] += 1
        
        # 混合型 - 部分区域有浸润，部分区域没有
        hotspots = density_info.get('hotspots', [])
        if len(hotspots) > 1 and immune_ratio > 0.1:
            pattern_scores['mixed'] += 2
        
        # 根据密度分布的标准差判断
        density_std = density_info.get('std_density', 0)
        if density_std > 0.3:  # 密度变化较大
            pattern_scores['mixed'] += 1
        
        # 选择得分最高的模式
        best_pattern = max(pattern_scores, key=pattern_scores.get)
        confidence = pattern_scores[best_pattern] / 5.0  # 归一化到0-1
        
        # 生成解释
        explanations = {
            'desert': f"免疫细胞比例很低({immune_ratio:.1%})",
            'excluded': f"免疫细胞与肿瘤细胞空间分离",
            'inflamed': f"免疫细胞丰富({immune_ratio:.1%})且分布均匀",
            'mixed': f"免疫细胞分布不均匀，存在{len(hotspots)}个热点区域"
        }
        
        explanation = explanations.get(best_pattern, "")
        
        return best_pattern, confidence, explanation
    
    def calculate_infiltration_score(self, detection_results, spatial_analysis):
        """
        计算肿瘤浸润评分
        """
        cells = detection_results['cells']
        
        if len(cells) == 0:
            return 0.0, {}
        
        # 基础评分 - 基于免疫细胞数量和类型
        base_score = 0
        cell_type_scores = {}
        
        for cell in cells:
            cell_type = cell['type']
            confidence = cell['confidence']
            
            if cell_type in self.immune_cell_weights:
                weight = self.immune_cell_weights[cell_type]
                score = weight * confidence
                base_score += score
                
                if cell_type not in cell_type_scores:
                    cell_type_scores[cell_type] = 0
                cell_type_scores[cell_type] += score
        
        # 空间分布调整
        spatial_bonus = 0
        
        # 密度分布奖励
        density_info = spatial_analysis.get('density', {})
        hotspots = density_info.get('hotspots', [])
        if len(hotspots) > 0:
            spatial_bonus += len(hotspots) * 0.1
        
        # 聚类质量奖励
        cluster_info = spatial_analysis.get('clusters', {})
        silhouette = cluster_info.get('silhouette_score', 0)
        if silhouette > 0.5:
            spatial_bonus += 0.2
        
        # 细胞类型多样性奖励
        immune_types = len([ct for ct in cell_type_scores.keys() if ct in self.immune_cell_weights])
        diversity_bonus = min(immune_types * 0.1, 0.5)
        
        # 最终评分
        total_score = base_score + spatial_bonus + diversity_bonus
        
        # 归一化到0-100
        max_possible_score = len(cells) * max(self.immune_cell_weights.values()) + 1.0
        normalized_score = min(100, (total_score / max_possible_score) * 100)
        
        score_breakdown = {
            'base_score': base_score,
            'spatial_bonus': spatial_bonus,
            'diversity_bonus': diversity_bonus,
            'total_score': total_score,
            'normalized_score': normalized_score,
            'cell_type_scores': cell_type_scores
        }
        
        return normalized_score, score_breakdown
    
    def _empty_spatial_analysis(self):
        """
        返回空的空间分析结果
        """
        return {
            'total_cells': 0,
            'cell_positions': np.array([]),
            'cell_types': [],
            'density': {'grid': np.array([]), 'max_density': 0, 'mean_density': 0, 'std_density': 0, 'hotspots': []},
            'clusters': {'clusters': [], 'n_clusters': 0, 'silhouette_score': 0, 'labels': []},
            'neighborhoods': {'neighborhoods': [], 'interaction_matrix': {}},
            'spatial_autocorr': {'moran_i': {}, 'geary_c': {}},
            'distances': {'nearest_neighbor': {}, 'type_distances': {}}
        }
    
    def generate_infiltration_report(self, detection_results, spatial_analysis, image_path=None):
        """
        生成肿瘤浸润分析报告
        """
        # 分类浸润模式
        pattern, pattern_confidence, pattern_explanation = self.classify_infiltration_pattern(
            spatial_analysis, detection_results)
        
        # 计算浸润评分
        infiltration_score, score_breakdown = self.calculate_infiltration_score(
            detection_results, spatial_analysis)
        
        # 统计信息
        cells = detection_results['cells']
        total_cells = len(cells)
        
        cell_counts = {}
        for cell in cells:
            cell_type = cell['type']
            cell_counts[cell_type] = cell_counts.get(cell_type, 0) + 1
        
        immune_cells = sum([count for cell_type, count in cell_counts.items() 
                           if cell_type in self.immune_cell_weights])
        
        # 生成报告
        report = {
            'metadata': {
                'image_path': image_path,
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_cells_detected': total_cells
            },
            
            'infiltration_pattern': {
                'pattern': pattern,
                'pattern_name': self.infiltration_patterns.get(pattern, pattern),
                'confidence': pattern_confidence,
                'explanation': pattern_explanation
            },
            
            'infiltration_score': {
                'score': infiltration_score,
                'level': self._get_infiltration_level(infiltration_score),
                'breakdown': score_breakdown
            },
            
            'cell_composition': {
                'total_cells': total_cells,
                'immune_cells': immune_cells,
                'immune_ratio': immune_cells / total_cells if total_cells > 0 else 0,
                'cell_counts': cell_counts,
                'cell_percentages': {ct: count/total_cells*100 for ct, count in cell_counts.items()} if total_cells > 0 else {}
            },
            
            'spatial_features': {
                'density_hotspots': len(spatial_analysis.get('density', {}).get('hotspots', [])),
                'cell_clusters': spatial_analysis.get('clusters', {}).get('n_clusters', 0),
                'cluster_quality': spatial_analysis.get('clusters', {}).get('silhouette_score', 0),
                'spatial_autocorrelation': spatial_analysis.get('spatial_autocorr', {})
            },
            
            'clinical_implications': self._get_clinical_implications(pattern, infiltration_score, cell_counts)
        }
        
        return report
    
    def _get_infiltration_level(self, score):
        """
        根据评分确定浸润程度
        """
        if score >= 70:
            return "高度浸润"
        elif score >= 40:
            return "中度浸润"
        elif score >= 15:
            return "轻度浸润"
        else:
            return "无明显浸润"
    
    def _get_clinical_implications(self, pattern, score, cell_counts):
        """
        生成临床意义解释
        """
        implications = []
        
        # 基于浸润模式的解释
        if pattern == 'inflamed':
            implications.append("免疫炎症型肿瘤通常对免疫治疗反应较好")
        elif pattern == 'excluded':
            implications.append("免疫排斥型肿瘤可能需要联合治疗策略")
        elif pattern == 'desert':
            implications.append("免疫沙漠型肿瘤对免疫治疗反应可能较差")
        elif pattern == 'mixed':
            implications.append("混合型浸润模式提示肿瘤异质性")
        
        # 基于评分的解释
        if score >= 70:
            implications.append("高浸润评分提示良好的免疫反应")
        elif score < 15:
            implications.append("低浸润评分可能与免疫逃逸相关")
        
        # 基于细胞类型的解释
        lymphocyte_count = cell_counts.get('lymphocyte', 0)
        macrophage_count = cell_counts.get('macrophage', 0)
        
        if lymphocyte_count > macrophage_count * 2:
            implications.append("淋巴细胞占优势，提示适应性免疫反应活跃")
        elif macrophage_count > lymphocyte_count * 2:
            implications.append("巨噬细胞占优势，需要进一步分析M1/M2极化状态")
        
        return implications