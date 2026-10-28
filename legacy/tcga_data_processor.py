import os
import json
import pandas as pd
import numpy as np
import cv2
import requests
from pathlib import Path
import zipfile
import tarfile
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class TCGADataProcessor:
    """
    TCGA数据处理器 - 用于下载、处理和管理TCGA H&E图像数据
    """
    
    def __init__(self, config=None):
        # 如果传入的是配置字典，提取data_dir；否则使用默认值
        if isinstance(config, dict):
            data_dir = config.get('data_dir', 'tcga_data')
        elif isinstance(config, str):
            data_dir = config
        else:
            data_dir = 'tcga_data'
            
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # TCGA API端点
        self.tcga_api_base = "https://api.gdc.cancer.gov"
        self.cases_endpoint = f"{self.tcga_api_base}/cases"
        self.files_endpoint = f"{self.tcga_api_base}/files"
        self.data_endpoint = f"{self.tcga_api_base}/data"
        
        # 前列腺癌项目ID
        self.prad_project = "TCGA-PRAD"
        
        # 支持的图像格式
        self.image_formats = ['.svs', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 元数据存储
        self.metadata_file = self.data_dir / "metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """加载本地元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'cases': {},
                'files': {},
                'download_history': [],
                'last_updated': None
            }
    
    def save_metadata(self):
        """保存元数据"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def query_tcga_cases(self, project_id=None, sample_type="Primary Tumor", limit=50):
        """
        查询TCGA病例
        """
        if project_id is None:
            project_id = self.prad_project
        
        # 构建查询参数
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [project_id]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.samples.sample_type",
                        "value": [sample_type]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "format": "json",
            "size": str(limit),
            "expand": "samples,samples.portions,samples.portions.analytes,samples.portions.analytes.aliquots"
        }
        
        try:
            response = requests.get(self.cases_endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            cases = data.get('data', {}).get('cases', [])
            
            self.logger.info(f"找到 {len(cases)} 个 {project_id} 病例")
            
            # 更新元数据
            for case in cases:
                case_id = case['id']
                self.metadata['cases'][case_id] = {
                    'submitter_id': case.get('submitter_id'),
                    'project_id': project_id,
                    'samples': case.get('samples', []),
                    'query_date': datetime.now().isoformat()
                }
            
            self.save_metadata()
            return cases
            
        except requests.RequestException as e:
            self.logger.error(f"查询TCGA病例失败: {e}")
            return []
    
    def query_he_images(self, case_ids=None, data_category="Biospecimen", 
                       data_type="Slide Image", limit=100):
        """
        查询H&E染色图像文件
        """
        # 构建查询过滤器
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_category",
                        "value": [data_category]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_type",
                        "value": [data_type]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.experimental_strategy",
                        "value": ["Tissue Slide"]
                    }
                }
            ]
        }
        
        # 如果指定了病例ID，添加过滤条件
        if case_ids:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "cases.id",
                    "value": case_ids
                }
            })
        
        params = {
            "filters": json.dumps(filters),
            "format": "json",
            "size": str(limit),
            "expand": "cases,cases.samples,cases.samples.portions,cases.samples.portions.slides"
        }
        
        try:
            response = requests.get(self.files_endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('data', {}).get('files', [])
            
            # 过滤H&E图像
            he_files = []
            for file_info in files:
                file_name = file_info.get('file_name', '').lower()
                
                # 检查是否为H&E染色
                if any(keyword in file_name for keyword in ['he', 'h&e', 'hematoxylin', 'eosin']):
                    he_files.append(file_info)
                # 或者检查文件扩展名
                elif any(file_name.endswith(ext) for ext in self.image_formats):
                    he_files.append(file_info)
            
            self.logger.info(f"找到 {len(he_files)} 个H&E图像文件")
            
            # 更新元数据
            for file_info in he_files:
                file_id = file_info['id']
                self.metadata['files'][file_id] = {
                    'file_name': file_info.get('file_name'),
                    'file_size': file_info.get('file_size'),
                    'data_category': file_info.get('data_category'),
                    'data_type': file_info.get('data_type'),
                    'cases': [case['id'] for case in file_info.get('cases', [])],
                    'query_date': datetime.now().isoformat()
                }
            
            self.save_metadata()
            return he_files
            
        except requests.RequestException as e:
            self.logger.error(f"查询H&E图像失败: {e}")
            return []
    
    def download_file(self, file_id, file_name=None, chunk_size=8192):
        """
        下载TCGA文件
        """
        if file_name is None:
            file_name = f"{file_id}.unknown"
        
        file_path = self.data_dir / "images" / file_name
        file_path.parent.mkdir(exist_ok=True)
        
        # 检查文件是否已存在
        if file_path.exists():
            self.logger.info(f"文件已存在: {file_path}")
            return str(file_path)
        
        try:
            # 下载文件
            url = f"{self.data_endpoint}/{file_id}"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r下载进度: {progress:.1f}%", end='', flush=True)
            
            print()  # 换行
            self.logger.info(f"文件下载完成: {file_path}")
            
            # 记录下载历史
            self.metadata['download_history'].append({
                'file_id': file_id,
                'file_name': file_name,
                'file_path': str(file_path),
                'download_date': datetime.now().isoformat(),
                'file_size': downloaded_size
            })
            
            self.save_metadata()
            return str(file_path)
            
        except requests.RequestException as e:
            self.logger.error(f"下载文件失败 {file_id}: {e}")
            return None
    
    def extract_archive(self, archive_path):
        """
        解压缩文件
        """
        archive_path = Path(archive_path)
        extract_dir = archive_path.parent / archive_path.stem
        extract_dir.mkdir(exist_ok=True)
        
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                self.logger.warning(f"不支持的压缩格式: {archive_path.suffix}")
                return None
            
            self.logger.info(f"文件解压完成: {extract_dir}")
            return str(extract_dir)
            
        except Exception as e:
            self.logger.error(f"解压文件失败 {archive_path}: {e}")
            return None
    
    def load_image(self, image_path, target_size=None):
        """
        增强的图像加载函数 - 支持多种格式和中文路径
        """
        image_path = Path(image_path)
    
        if not image_path.exists():
            if hasattr(self, 'logger'):
                self.logger.error(f"图像文件不存在: {image_path}")
            else:
                print(f"❌ 图像文件不存在: {image_path}")
            return None
    
        try:
            # 获取文件信息
            file_size = image_path.stat().st_size
            suffix = image_path.suffix.lower()
        
            if hasattr(self, 'logger'):
                self.logger.info(f"加载图像: {image_path.name} ({file_size / (1024*1024):.2f} MB)")
        
            image = None
        
            # 标准图像格式
            if suffix in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                # 方法1: 直接使用OpenCV
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # 方法2: 使用numpy处理中文路径
                    try:
                        image_data = np.fromfile(str(image_path), dtype=np.uint8)
                        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
            
                # 方法3: 使用PIL作为备用
                if image is None:
                    try:
                        from PIL import Image as PILImage
                        pil_image = PILImage.open(image_path)
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        image = np.array(pil_image)
                    except Exception:
                        pass
        
            # SVS医学图像格式
            elif suffix == '.svs':
                try:
                    import openslide
                    slide = openslide.OpenSlide(str(image_path))
                
                    # 选择合适的级别
                    level = slide.level_count - 1  # 默认最小分辨率
                    if target_size:
                        for i, size in enumerate(slide.level_dimensions):
                            if max(size) <= max(target_size) * 2:
                                level = i
                                break
                
                    # 读取图像
                    region = slide.read_region((0, 0), level, slide.level_dimensions[level])
                    image = np.array(region.convert('RGB'))
                
                except ImportError:
                    error_msg = "需要安装openslide-python来处理SVS文件: pip install openslide-python"
                    if hasattr(self, 'logger'):
                        self.logger.error(error_msg)
                    else:
                        print(f"❌ {error_msg}")
                    return None
                except Exception as e:
                    error_msg = f"加载SVS文件失败: {e}"
                    if hasattr(self, 'logger'):
                        self.logger.error(error_msg)
                    else:
                        print(f"❌ {error_msg}")
                    return None
        
            # 其他格式尝试用PIL
            else:
                try:
                    from PIL import Image as PILImage
                    pil_image = PILImage.open(image_path)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    image = np.array(pil_image)
                except Exception:
                    error_msg = f"不支持的图像格式: {suffix}"
                    if hasattr(self, 'logger'):
                        self.logger.error(error_msg)
                    else:
                        print(f"❌ {error_msg}")
                    return None
        
            if image is None:
                error_msg = f"无法加载图像: {image_path}"
                if hasattr(self, 'logger'):
                    self.logger.error(error_msg)
                else:
                    print(f"❌ {error_msg}")
                return None
        
            # 调整图像大小
            if target_size and len(target_size) == 2:
                h, w = image.shape[:2]
                target_h, target_w = target_size
                if h != target_h or w != target_w:
                    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
            success_msg = f"图像加载成功: {image_path.name}, 尺寸: {image.shape}"
            if hasattr(self, 'logger'):
                self.logger.info(success_msg)
            else:
                print(f"✅ {success_msg}")
        
            return image
        
        except Exception as e:
            error_msg = f"加载图像失败 {image_path}: {e}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            else:
                print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_image_for_analysis(self, image, tile_size=(1024, 1024), overlap=0.1):
        """
        预处理图像用于分析 - 将大图像分割成小块
        """
        if image is None:
            return []
        
        h, w = image.shape[:2]
        tile_h, tile_w = tile_size
        
        # 计算重叠像素
        overlap_h = int(tile_h * overlap)
        overlap_w = int(tile_w * overlap)
        
        # 计算步长
        step_h = tile_h - overlap_h
        step_w = tile_w - overlap_w
        
        tiles = []
        
        for y in range(0, h - tile_h + 1, step_h):
            for x in range(0, w - tile_w + 1, step_w):
                # 提取图像块
                tile = image[y:y+tile_h, x:x+tile_w]
                
                # 检查图像块质量
                if self._is_valid_tile(tile):
                    tiles.append({
                        'image': tile,
                        'position': (x, y),
                        'size': (tile_w, tile_h)
                    })
        
        self.logger.info(f"图像分割完成: {len(tiles)} 个有效图像块")
        return tiles
    
    def _is_valid_tile(self, tile, min_tissue_ratio=0.3):
        """
        检查图像块是否包含足够的组织内容
        """
        # 转换为灰度图
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile
        
        # 使用Otsu阈值分割
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 计算组织区域比例
        tissue_pixels = np.sum(binary == 0)  # 假设暗色区域是组织
        total_pixels = binary.size
        tissue_ratio = tissue_pixels / total_pixels
        
        return tissue_ratio >= min_tissue_ratio
    
    def get_case_clinical_data(self, case_id):
        """
        获取病例临床数据
        """
        params = {
            "filters": json.dumps({
                "op": "in",
                "content": {
                    "field": "cases.id",
                    "value": [case_id]
                }
            }),
            "expand": "diagnoses,demographic,exposures",
            "format": "json"
        }
        
        try:
            response = requests.get(self.cases_endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            cases = data.get('data', {}).get('cases', [])
            
            if cases:
                return cases[0]
            else:
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"获取临床数据失败 {case_id}: {e}")
            return None
    
    def create_dataset_manifest(self, output_file="dataset_manifest.csv"):
        """
        创建数据集清单
        """
        manifest_data = []
        
        for file_id, file_info in self.metadata['files'].items():
            # 查找对应的病例信息
            case_ids = file_info.get('cases', [])
            for case_id in case_ids:
                case_info = self.metadata['cases'].get(case_id, {})
                
                manifest_data.append({
                    'file_id': file_id,
                    'file_name': file_info.get('file_name'),
                    'file_size': file_info.get('file_size'),
                    'case_id': case_id,
                    'case_submitter_id': case_info.get('submitter_id'),
                    'project_id': case_info.get('project_id'),
                    'data_category': file_info.get('data_category'),
                    'data_type': file_info.get('data_type')
                })
        
        # 保存为CSV文件
        df = pd.DataFrame(manifest_data)
        manifest_path = self.data_dir / output_file
        df.to_csv(manifest_path, index=False)
        
        self.logger.info(f"数据集清单已保存: {manifest_path}")
        return str(manifest_path)
    
    def get_downloaded_images(self):
        """
        获取已下载的图像列表
        """
        images = []
        images_dir = self.data_dir / "images"
        
        if images_dir.exists():
            for file_path in images_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.image_formats:
                    images.append(str(file_path))
        
        return images
    
    def batch_download_he_images(self, max_files=10, project_id=None):
        """
        批量下载H&E图像
        """
        if project_id is None:
            project_id = self.prad_project
        
        # 查询病例
        cases = self.query_tcga_cases(project_id=project_id, limit=50)
        if not cases:
            self.logger.warning("未找到病例")
            return []
        
        # 提取病例ID
        case_ids = [case['id'] for case in cases[:20]]  # 限制病例数量
        
        # 查询H&E图像
        he_files = self.query_he_images(case_ids=case_ids, limit=max_files)
        if not he_files:
            self.logger.warning("未找到H&E图像")
            return []
        
        # 下载文件
        downloaded_files = []
        for i, file_info in enumerate(he_files[:max_files]):
            file_id = file_info['id']
            file_name = file_info.get('file_name', f"{file_id}.unknown")
            
            self.logger.info(f"下载文件 {i+1}/{min(max_files, len(he_files))}: {file_name}")
            
            file_path = self.download_file(file_id, file_name)
            if file_path:
                downloaded_files.append(file_path)
        
        self.logger.info(f"批量下载完成: {len(downloaded_files)} 个文件")
        return downloaded_files