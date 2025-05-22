"""
微生物图像数据集加载模块
"""
import os
import json
import numpy as np
import paddle
from PIL import Image
from paddle.io import Dataset
from ..utils.data_augmentation import MicroAugment, AgriAugment, process_multispectral

class MicroDataset(Dataset):
    """微生物图像数据集类"""
    def __init__(self, data_dir, transforms=None, split='train', use_multispectral=False):
        """初始化数据集
        Args:
            data_dir: 数据集根目录
            transforms: 数据增强列表
            split: 数据集分割('train', 'val', 'test')
            use_multispectral: 是否使用多光谱数据
        """
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms or []
        self.split = split
        self.use_multispectral = use_multispectral
        
        # 加载标注文件
        anno_file = os.path.join(data_dir, f'annotations/{split}.json')
        with open(anno_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
            
        # 构建图像ID到文件路径的映射
        self.id2path = {
            img['id']: os.path.join(data_dir, 'images', img['file_name'])
            for img in self.annotations['images']
        }
        
        # 按图像ID组织标注数据
        self.id2annos = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id2annos:
                self.id2annos[img_id] = []
            self.id2annos[img_id].append(ann)
            
        self.img_ids = list(self.id2path.keys())
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """获取单个数据样本
        Returns:
            image: 图像数据，shape为[C, H, W]
            target: 标注数据字典，包含boxes、labels等
        """
        img_id = self.img_ids[idx]
        img_path = self.id2path[img_id]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # 处理多光谱数据
        if self.use_multispectral:
            # 获取对应的UV和IR通道图像路径
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            uv_path = os.path.join(
                os.path.dirname(img_path), 
                f'{base_name}_uv.jpg'
            )
            ir_path = os.path.join(
                os.path.dirname(img_path),
                f'{base_name}_ir.jpg'
            )
            
            # 加载UV和IR通道
            uv_img = np.array(Image.open(uv_path).convert('L'))
            ir_img = np.array(Image.open(ir_path).convert('L'))
            
            # 融合多光谱数据
            image = process_multispectral(image, uv_img, ir_img)
        
        # 准备标注数据
        target = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': []
        }
        
        # 处理标注
        for anno in self.id2annos[img_id]:
            # 边界框
            x, y, w, h = anno['bbox']
            target['boxes'].append([x, y, x + w, y + h])
            # 类别标签
            target['labels'].append(anno['category_id'])
            # 区域面积
            target['area'].append(anno['area'])
            # 是否群体
            target['iscrowd'].append(anno['iscrowd'])
            
        # 转换为numpy数组
        for k, v in target.items():
            target[k] = np.array(v, dtype=np.float32)
            
        # 应用数据增强
        if self.split == 'train':
            for transform in self.transforms:
                if isinstance(transform, (MicroAugment, AgriAugment)):
                    image = transform(image)
                else:
                    # 其他自定义增强
                    image = transform(image)
        
        # 转换为Tensor
        image = paddle.vision.transforms.to_tensor(image)
        return image, target
    
    @staticmethod
    def collate_fn(batch):
        """数据批处理函数"""
        images = []
        targets = []
        for image, target in batch:
            images.append(image)
            targets.append(target)
        images = paddle.stack(images, axis=0)
        return images, targets