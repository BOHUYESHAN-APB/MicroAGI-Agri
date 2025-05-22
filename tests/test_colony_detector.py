"""
菌落检测模型测试
"""
import unittest
import numpy as np
import paddle
from microai_core.models.colony_detector import ColonyDetector, CBAM
from microai_core.data.dataset import MicroDataset

class TestColonyDetector(unittest.TestCase):
    """菌落检测模型测试类"""
    def setUp(self):
        self.model = ColonyDetector(num_classes=2, pretrained=False)
        # 创建模拟输入数据
        self.dummy_input = paddle.randn([2, 3, 512, 512])
        
    def test_model_output_shape(self):
        """测试模型输出形状"""
        self.model.eval()
        with paddle.no_grad():
            output = self.model(self.dummy_input)
            self.assertEqual(len(output), 2)  # 批大小为2
            self.assertIn('boxes', output[0])
            self.assertIn('scores', output[0])
            self.assertIn('labels', output[0])
            
    def test_cbam_attention(self):
        """测试CBAM注意力机制"""
        cbam = CBAM(channel=64)
        x = paddle.randn([2, 64, 32, 32])
        out = cbam(x)
        self.assertEqual(out.shape, x.shape)
        
    def test_predict_method(self):
        """测试单图预测方法"""
        self.model.eval()
        # 创建模拟图像
        image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        results = self.model.predict(image, threshold=0.5)
        # 检查返回结果格式
        for box, score, label in results:
            self.assertEqual(len(box), 4)  # [x1, y1, x2, y2]
            self.assertTrue(0 <= score <= 1)  # 置信度在[0,1]范围
            self.assertTrue(label in [0, 1])  # 二分类
            
class TestMicroDataset(unittest.TestCase):
    """数据集测试类"""
    def test_dataset_loading(self):
        """测试数据集加载"""
        try:
            dataset = MicroDataset(
                data_dir="tests/data",
                use_multispectral=True
            )
        except Exception as e:
            self.fail(f"Dataset initialization failed: {e}")
            
    def test_multispectral_processing(self):
        """测试多光谱数据处理"""
        dataset = MicroDataset(
            data_dir="tests/data",
            use_multispectral=True
        )
        if len(dataset) > 0:
            image, target = dataset[0]
            self.assertEqual(image.shape[0], 3)  # 3通道输出
            self.assertTrue('boxes' in target)
            self.assertTrue('labels' in target)
            
if __name__ == '__main__':
    unittest.main()