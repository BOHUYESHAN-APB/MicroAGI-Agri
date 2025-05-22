import albumentations as A
import random
import cv2
import numpy as np

class MicroAugment:
    """微生物图像数据增强工具类"""
    def __init__(self):
        self.transform = A.Compose([
            A.CLAHE(p=0.8),
            A.RandomRotate90(),
            A.GridDistortion(p=0.3),
            A.GaussNoise(var_limit=(10,50)),
            A.Cutout(num_holes=8, max_h_size=32)  # 模拟培养基缺陷
        ])
        
    def __call__(self, img):
        return self.transform(image=img)["image"]

class AgriAugment:
    """农业场景专用增强"""
    def __call__(self, img):
        # 模拟田间干扰
        if random.random() < 0.6:
            img = self.add_soil_particles(img) 
        if random.random() < 0.3:
            img = self.add_glare(img)
        return img
    
    def add_soil_particles(self, img):
        """添加土壤颗粒噪声"""
        # 生成随机土壤颗粒
        h, w = img.shape[:2]
        particles = np.zeros((h, w), dtype=np.uint8)
        num_particles = random.randint(100, 500)
        
        for _ in range(num_particles):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            radius = random.randint(1, 3)
            color = random.randint(50, 150)
            cv2.circle(particles, (x, y), radius, color, -1)
        
        # 添加到原图
        particles = cv2.GaussianBlur(particles, (3, 3), 0)
        return cv2.addWeighted(img, 0.9, cv2.cvtColor(particles, cv2.COLOR_GRAY2BGR), 0.1, 0)
        
    def add_glare(self, img):
        """添加反光效果"""
        h, w = img.shape[:2]
        glare = np.zeros((h, w), dtype=np.uint8)
        
        # 生成随机光斑
        num_spots = random.randint(1, 3)
        for _ in range(num_spots):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            radius = random.randint(20, 50)
            center = (x, y)
            
            # 创建径向渐变
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
            mask = dist_from_center <= radius
            intensity = np.zeros((h, w))
            intensity[mask] = 255 * (1 - dist_from_center[mask]/radius)
            glare = np.maximum(glare, intensity.astype(np.uint8))
        
        # 应用高斯模糊使效果更自然
        glare = cv2.GaussianBlur(glare, (9, 9), 0)
        return cv2.addWeighted(img, 1, cv2.cvtColor(glare, cv2.COLOR_GRAY2BGR), 0.3, 0)

def process_multispectral(img_rgb, img_uv, img_ir):
    """处理多光谱数据
    Args:
        img_rgb: RGB通道图像
        img_uv: 紫外通道图像
        img_ir: 红外通道图像
    Returns:
        融合后的图像
    """
    # 确保所有图像大小一致
    h, w = img_rgb.shape[:2]
    img_uv = cv2.resize(img_uv, (w, h))
    img_ir = cv2.resize(img_ir, (w, h))
    
    # 通道加权融合
    fused = cv2.addWeighted(img_rgb, 0.6, img_uv, 0.25, 0)
    fused = cv2.addWeighted(fused, 1.0, img_ir, 0.15, 0)
    
    # 计算UV通道的透明度图
    alpha = cv2.normalize(img_uv, None, 0, 1, cv2.NORM_MINMAX)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    # 根据透明度进行混合
    return cv2.addWeighted(fused, alpha.mean(), img_rgb, 1-alpha.mean(), 0)