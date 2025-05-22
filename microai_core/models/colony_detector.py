"""
菌落检测模型模块
"""
import paddle
import paddle.vision as vision
from paddle import nn

class CBAM(nn.Layer):
    """CBAM注意力机制模块"""
    def __init__(self, channel, ratio=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, channel // ratio, 1),
            nn.ReLU(),
            nn.Conv2D(channel // ratio, channel, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2D(2, 1, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        # 空间注意力
        max_pool = paddle.max(x, axis=1, keepdim=True)
        avg_pool = paddle.mean(x, axis=1, keepdim=True)
        pool = paddle.concat([max_pool, avg_pool], axis=1)
        sa = self.spatial_attention(pool)
        return x * sa

class CustomBBoxHead(nn.Layer):
    """自定义边界框检测头，集成CBAM注意力机制"""
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, 256, 3, padding=1)
        self.cbam1 = CBAM(256)
        self.conv2 = nn.Conv2D(256, 128, 3, padding=1)
        self.cbam2 = CBAM(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.cbam1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.cbam2(x)
        x = self.flatten(x)
        return self.fc(x)

class ColonyDetector(vision.models.FasterRCNN):
    """菌落检测模型，基于Faster R-CNN"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__(
            backbone=vision.models.ResNet50_vd(pretrained=pretrained),
            num_classes=num_classes,
            rpn_channel=512,
            bbox_head=CustomBBoxHead()
        )
        
    def forward(self, x):
        # 多尺度特征融合
        features = self.backbone(x)
        features = [features[f] for f in ['res2', 'res3', 'res4', 'res5']]
        return super().forward(features)
    
    def predict(self, image, threshold=0.5):
        """单张图像预测
        Args:
            image: 输入图像，shape为[H, W, 3]
            threshold: 检测阈值
        Returns:
            检测结果列表，每个元素为(bbox, score, class_id)
        """
        self.eval()
        with paddle.no_grad():
            # 预处理图像
            image = paddle.vision.transforms.to_tensor(image)
            image = paddle.unsqueeze(image, axis=0)
            
            # 模型推理
            pred = self(image)[0]
            
            # 后处理
            boxes = pred['boxes'].numpy()
            scores = pred['scores'].numpy()
            labels = pred['labels'].numpy()
            
            # 阈值过滤
            keep = scores > threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            return list(zip(boxes, scores, labels))