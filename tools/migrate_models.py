"""
模型和数据迁移工具
"""
import os
import shutil
import yaml
from pathlib import Path
import paddle

def setup_data_structure(train_dir, output_dir):
    """设置数据目录结构
    Args:
        train_dir: 原始训练数据目录
        output_dir: 输出目录
    """
    # 创建必要的目录
    dirs = [
        'data/images',
        'data/annotations',
        'models/pretrained',
        'models/converted'
    ]
    
    for d in dirs:
        Path(os.path.join(output_dir, d)).mkdir(parents=True, exist_ok=True)
        
    return {name: os.path.join(output_dir, d) for name, d in zip(
        ['images', 'annotations', 'pretrained', 'converted'], dirs
    )}

def create_dataset_config(train_dir, output_path):
    """创建数据集配置文件
    Args:
        train_dir: 原始训练数据目录
        output_path: 配置文件输出路径
    """
    config = {
        'data': {
            'train_dir': train_dir,
            'image_dir': 'images',
            'annotation_dir': 'annotations',
            'classes': ['colony', 'inhibition_zone'],
            'split_ratio': {
                'train': 0.8,
                'val': 0.1,
                'test': 0.1
            }
        },
        'model': {
            'colony_detector': {
                'weights': 'models/pretrained/colony_detector.pdparams',
                'num_classes': 1
            },
            'inhibition_detector': {
                'weights': 'models/pretrained/inhibition_detector.pdparams',
                'num_classes': 1
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

def main():
    # 源目录
    train_dir = r"D:\train"
    legacy_code_dir = r"D:\-Users-\Documents\GitHub\CNN-MicroAI-Colony"
    
    # 目标目录（当前项目）
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 设置目录结构
    dirs = setup_data_structure(train_dir, project_dir)
    
    # 创建数据集配置
    config_path = os.path.join(project_dir, 'configs/dataset.yaml')
    create_dataset_config(train_dir, config_path)
    
    print("迁移工具设置完成！")
    print("\n后续迁移步骤：")
    print("1. 将图像数据从 D:\\train\\images 复制到 data/images/")
    print("2. 将已有模型权重转换并保存到 models/pretrained/")
    print("3. 根据需要修改 configs/dataset.yaml 中的配置")
    print("\n使用方法：")
    print("from microai_core.utils.model_converter import convert_colony_detector")
    print("convert_colony_detector(")
    print("    torch_path='legacy_model.pth',")
    print("    save_path='models/pretrained/colony_detector.pdparams'")
    print(")")

if __name__ == '__main__':
    main()