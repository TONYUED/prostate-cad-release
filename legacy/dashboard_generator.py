import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import pandas as pd
from pathlib import Path
import json

class DashboardGenerator:
    """
    仪表板生成器 - 创建综合分析仪表板
    """
    
    def __init__(self, visualization_module):
        self.viz = visualization_module
    
    def create_infiltration_dashboard(self, infiltration_report):
        """
        创建肿瘤浸润分析仪表板
        """
        # 提取数据
        pattern_info = infiltration_report['infiltration_pattern']
        score_info = infiltration_report['infiltration_score']
        composition = infiltration_report['cell_composition']
        spatial_features = infiltration_report['spatial_features']
        
        fig = plt.figure(figsize=(20, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 浸润模式显示
        ax1 = fig.add_subplot(gs[0, 0])
        pattern = pattern_info['pattern']
        pattern_name = pattern_info['pattern_name']
        confidence = pattern_info['confidence']
        
        # 创建浸润模式可视化
        pattern_color = self.viz.pattern_colors.get(pattern, '#808080')
        circle = Circle((0.5, 0.5), 0.4, color=pattern_color, alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(0.5, 0.5, pattern_name, ha='center', va='center', 
                fontsize=12, fontweight='bold')
        ax1.text(0.5, 0.1, f'置信度: {confidence:.2f}', ha='center', va='center', 
                fontsize=10)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('浸润模式', fontsize=14, fontweight='bold')
        
        # 2. 浸润评分仪表
        ax2 = fig.add_subplot(gs[0, 1])
        score = score_info['score']
        level = score_info['level']
        
        # 创建半圆仪表
        theta = np.linspace(0, np.pi, 100)
        r = 0.8
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # 背景弧
        ax2.plot(x, y, 'lightgray', linewidth=10)
        
        # 评分弧
        score_theta = np.linspace(0, np.pi * score / 100, int(score) if score > 0 else 1)
        score_x = r * np.cos(score_theta)
        score_y = r * np.sin(score_theta)
        
        # 根据评分选择颜色
        if score >= 70:
            score_color = 'green'
        elif score >= 40:
            score_color = 'orange'
        else:
            score_color = 'red'
        
        ax2.plot(score_x, score_y, score_color, linewidth=10)
        
        # 添加文本
        ax2.text(0, -0.3, f'{score:.1f}', ha='center', va='center', 
                fontsize=20, fontweight='bold')
        ax2.text(0, -0.5, level, ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-0.6, 1)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('浸润评分', fontsize=14, fontweight='bold')
        
        # 3. 细胞组成
        ax3 = fig.add_subplot(gs[0, 2:])
        cell_counts = composition['cell_counts']
        
        if cell_counts:
            types = list(cell_counts.keys())
            counts = list(cell_counts.values())
            colors = [self.viz.cell_colors.get(t, '#808080') for t in types]
            labels = [self.viz.cell_names.get(t, t) for t in types]
            
            bars = ax3.bar(labels, counts, color=colors, alpha=0.8)
            ax3.set_title('细胞组成', fontsize=14, fontweight='bold')
            ax3.set_ylabel('数量')
            ax3.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, '无细胞检测到', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=16)
            ax3.set_title('细胞组成', fontsize=14, fontweight='bold')
        
        # 4. 空间特征雷达图
        ax4 = fig.add_subplot(gs[1, :2], projection='polar')
        
        # 准备雷达图数据
        features = ['密度热点', '聚类数量', '聚类质量', '免疫比例']
        values = [
            min(spatial_features['density_hotspots'] / 5, 1),  # 归一化到0-1
            min(spatial_features['cell_clusters'] / 10, 1),
            spatial_features['cluster_quality'],
            composition['immune_ratio']
        ]
        
        # 闭合雷达图
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax4.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax4.fill(angles, values, alpha=0.25, color='blue')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(features)
        ax4.set_ylim(0, 1)
        ax4.set_title('空间特征分析', fontsize=14, fontweight='bold', pad=20)
        
        # 5. 统计摘要
        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis('off')
        
        summary_text = f"""
        分析摘要
        ────────────────────
        总细胞数: {composition['total_cells']}
        免疫细胞数: {composition['immune_cells']}
        免疫细胞比例: {composition['immune_ratio']*100:.1f}%
        
        空间特征:
        • 密度热点: {spatial_features['density_hotspots']}个
        • 细胞聚类: {spatial_features['cell_clusters']}个
        • 聚类质量: {spatial_features['cluster_quality']:.3f}
        
        浸润模式: {pattern_name}
        浸润程度: {level}
        浸润评分: {score:.1f}/100
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
                verticalalignment='top', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 6. 临床意义
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        implications = infiltration_report['clinical_implications']
        implications_text = "临床意义:\n" + "\n".join([f"• {imp}" for imp in implications])
        
        ax6.text(0.05, 0.95, implications_text, transform=ax6.transAxes, 
                verticalalignment='top', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('肿瘤浸润免疫细胞分析报告', fontsize=18, fontweight='bold')
        
        return fig
    
    def save_visualization_report(self, figures, output_dir="visualization_output"):
        """
        保存可视化报告
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        
        for name, fig in figures.items():
            if fig is not None:
                file_path = output_path / f"{name}.png"
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                saved_files.append(str(file_path))
                plt.close(fig)  # 关闭图形以释放内存
        
        return saved_files
    
    def generate_html_report(self, infiltration_report, figure_paths, output_path="report.html"):
        """
        生成HTML报告
        """
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>肿瘤浸润免疫细胞分析报告</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .summary-card h3 {{ margin: 0 0 10px 0; font-size: 1.2em; }}
                .summary-card .value {{ font-size: 2em; font-weight: bold; }}
                .figure {{ text-align: center; margin: 30px 0; }}
                .figure img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                .implications {{ background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 20px; margin: 20px 0; }}
                .implications h3 {{ color: #856404; margin-top: 0; }}
                .implications ul {{ margin: 10px 0; }}
                .implications li {{ margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                .pattern-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; }}
                .pattern-inflamed {{ background-color: #FF6B6B; }}
                .pattern-excluded {{ background-color: #4ECDC4; }}
                .pattern-desert {{ background-color: #45B7D1; }}
                .pattern-mixed {{ background-color: #96CEB4; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>肿瘤浸润免疫细胞分析报告</h1>
                
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>总细胞数</h3>
                        <div class="value">{total_cells}</div>
                    </div>
                    <div class="summary-card">
                        <h3>免疫细胞数</h3>
                        <div class="value">{immune_cells}</div>
                    </div>
                    <div class="summary-card">
                        <h3>免疫细胞比例</h3>
                        <div class="value">{immune_ratio:.1f}%</div>
                    </div>
                    <div class="summary-card">
                        <h3>浸润评分</h3>
                        <div class="value">{infiltration_score:.1f}</div>
                    </div>
                </div>
                
                <h2>浸润模式分析</h2>
                <p>检测到的浸润模式: <span class="pattern-badge pattern-{pattern}">{pattern_name}</span></p>
                <p>置信度: {pattern_confidence:.2f}</p>
                <p>解释: {pattern_explanation}</p>
                
                <h2>细胞组成分析</h2>
                <table>
                    <tr><th>细胞类型</th><th>数量</th><th>比例</th></tr>
                    {cell_composition_table}
                </table>
                
                <h2>空间分布特征</h2>
                <ul>
                    <li>密度热点区域: {density_hotspots}个</li>
                    <li>细胞聚类数量: {cell_clusters}个</li>
                    <li>聚类质量评分: {cluster_quality:.3f}</li>
                </ul>
                
                <h2>可视化结果</h2>
                {figures_html}
                
                <div class="implications">
                    <h3>临床意义</h3>
                    <ul>
                        {implications_list}
                    </ul>
                </div>
                
                <p style="text-align: center; color: #7f8c8d; margin-top: 40px;">
                    报告生成时间: {timestamp}
                </p>
            </div>
        </body>
        </html>
        """
        
        # 准备数据
        pattern_info = infiltration_report['infiltration_pattern']
        score_info = infiltration_report['infiltration_score']
        composition = infiltration_report['cell_composition']
        spatial_features = infiltration_report['spatial_features']
        implications = infiltration_report['clinical_implications']
        
        # 生成细胞组成表格
        cell_composition_table = ""
        for cell_type, count in composition['cell_counts'].items():
            percentage = composition['cell_percentages'].get(cell_type, 0)
            cell_name = self.viz.cell_names.get(cell_type, cell_type)
            cell_composition_table += f"<tr><td>{cell_name}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        # 生成图片HTML
        figures_html = ""
        for name, path in figure_paths.items():
            figures_html += f"""
            <div class="figure">
                <h3>{name}</h3>
                <img src="{path}" alt="{name}">
            </div>
            """
        
        # 生成临床意义列表
        implications_list = "\n".join([f"<li>{imp}</li>" for imp in implications])
        
        # 填充模板
        html_content = html_template.format(
            total_cells=composition['total_cells'],
            immune_cells=composition['immune_cells'],
            immune_ratio=composition['immune_ratio'] * 100,
            infiltration_score=score_info['score'],
            pattern=pattern_info['pattern'],
            pattern_name=pattern_info['pattern_name'],
            pattern_confidence=pattern_info['confidence'],
            pattern_explanation=pattern_info['explanation'],
            cell_composition_table=cell_composition_table,
            density_hotspots=spatial_features['density_hotspots'],
            cell_clusters=spatial_features['cell_clusters'],
            cluster_quality=spatial_features['cluster_quality'],
            figures_html=figures_html,
            implications_list=implications_list,
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path