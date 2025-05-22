"""
模型迁移转换工具
用于将已有的PyTorch模型（Faster R-CNN ResNet50）转换为PaddlePaddle格式
"""
import os
import paddle
import numpy as np
import torch
from collections import OrderedDict

def convert_state_dict(torch_state_dict, paddle_model):
    """将PyTorch的权重转换为PaddlePaddle格式
    Args:
        torch_state_dict: PyTorch模型状态字典
        paddle_model: PaddlePaddle模型实例
    Returns:
        paddle_state_dict: 转换后的PaddlePaddle状态字典
    """
    paddle_state_dict = OrderedDict()
    
    # 权重名称映射（PyTorch -> PaddlePaddle）
    name_mapping = {
        # 主干网络
        'backbone.': 'backbone.',
        'layer': 'stage',
        'running_mean': '_mean',
        'running_var': '_variance',
        # RPN网络
        'rpn.': 'rpn_head.',
        'rpn_conv': 'rpn_conv',
        'rpn_cls': 'rpn_cls',
        'rpn_reg': 'rpn_reg',
        # 检测头
        'bbox_head.': 'bbox_head.',
        'fc6': 'fc_6',
        'fc7': 'fc_7'
    }
    
    for key, value in torch_state_dict.items():
        paddle_key = key
        
        # 转换层名称
        for torch_name, paddle_name in name_mapping.items():
            if torch_name in paddle_key:
                paddle_key = paddle_key.replace(torch_name, paddle_name)
        
        # 转换参数数据类型和格式
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            
        # 处理维度顺序差异 (NCHW vs NHWC)
        if len(value.shape) == 4:
            value = np.transpose(value, [2, 3, 1, 0])
            
        paddle_state_dict[paddle_key] = value
        
    return paddle_state_dict

def convert_colony_detector(torch_path, save_path):
    """转换菌落检测模型
    Args:
        torch_path: PyTorch模型路径
        save_path: 保存PaddlePaddle模型路径
    """
    from ..models.colony_detector import ColonyDetector
    
    # 加载PyTorch权重
    torch_state_dict = torch.load(torch_path, map_location='cpu')
    
    # 创建PaddlePaddle模型
    paddle_model = ColonyDetector(num_classes=2, pretrained=False)
    
    # 转换权重
    paddle_state_dict = convert_state_dict(torch_state_dict, paddle_model)
    
    # 加载转换后的权重
    paddle_model.set_state_dict(paddle_state_dict)
    
    # 保存PaddlePaddle模型
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    paddle.save(paddle_model.state_dict(), save_path)
    
def convert_inhibition_detector(torch_path, save_path):
    """转换抑菌圈检测模型
    Args:
        torch_path: PyTorch模型路径
        save_path: 保存PaddlePaddle模型路径
    """
    # 使用相同的ColonyDetector架构，但修改类别数
    from ..models.colony_detector import ColonyDetector
    
    # 加载PyTorch权重
    torch_state_dict = torch.load(torch_path, map_location='cpu')
    
    # 创建PaddlePaddle模型（抑菌圈检测为单类别）
    paddle_model = ColonyDetector(num_classes=1, pretrained=False)
    
    # 转换权重
    paddle_state_dict = convert_state_dict(torch_state_dict, paddle_model)
    
    # 加载转换后的权重
    paddle_model.set_state_dict(paddle_state_dict)
    
    # 保存PaddlePaddle模型
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    paddle.save(paddle_model.state_dict(), save_path)

class OpenCVHelpers:
    """OpenCV辅助函数，用于抑菌圈测量"""
    @staticmethod
    def measure_inhibition_zone(image, bbox):
        """测量抑菌圈直径
        Args:
            image: OpenCV格式图像
            bbox: 检测框 [x1, y1, x2, y2]
        Returns:
            diameter: 抑菌圈直径(像素)
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        
        # 图像预处理
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu阈值分割
        _, binary = cv2.threshold(
            blur, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # 轮廓检测
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # 找到最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            diameter = radius * 2
            
            return diameter
        return 0