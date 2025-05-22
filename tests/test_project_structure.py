"""
项目结构完整性检查
"""
import os
import unittest
import importlib
import yaml

class TestProjectStructure(unittest.TestCase):
    """测试项目结构完整性"""
    
    def test_core_files_exist(self):
        """检查核心文件是否存在"""
        required_files = [
            'microai_core/__init__.py',
            'microai_core/models/colony_detector.py',
            'microai_core/data/dataset.py',
            'microai_core/utils/data_augmentation.py',
            'microai_core/utils/metrics.py',
            'configs/colony_detection.yaml',
            'requirements.txt',
            'setup.py',
            'README.md'
        ]
        
        for file_path in required_files:
            self.assertTrue(
                os.path.exists(file_path),
                f"缺少必要文件: {file_path}"
            )
            
    def test_package_imports(self):
        """测试包导入"""
        modules = [
            'microai_core.models.colony_detector',
            'microai_core.data.dataset',
            'microai_core.utils.data_augmentation',
            'microai_core.utils.metrics'
        ]
        
        for module in modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                self.fail(f"无法导入模块 {module}: {e}")
                
    def test_config_validity(self):
        """检查配置文件有效性"""
        config_file = 'configs/colony_detection.yaml'
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 检查必要的配置项
            required_sections = [
                'data', 'model', 'training', 
                'evaluation', 'device'
            ]
            for section in required_sections:
                self.assertIn(
                    section, 
                    config,
                    f"配置文件缺少必要部分: {section}"
                )
                
            # 检查模型相关配置
            self.assertIn('backbone', config['model'])
            self.assertIn('num_classes', config['model'])
            self.assertIn('quantization', config['model'])
            
            # 检查数据相关配置
            self.assertIn('multispectral', config['data'])
            self.assertIn('augmentation', config['data'])
            
        except Exception as e:
            self.fail(f"配置文件检查失败: {e}")
            
    def test_readme_content(self):
        """检查README.md内容"""
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
            
        required_sections = [
            '主要功能',
            '环境要求',
            '安装说明',
            '项目结构',
            '快速开始'
        ]
        
        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"README.md缺少必要部分: {section}"
            )

if __name__ == '__main__':
    unittest.main()