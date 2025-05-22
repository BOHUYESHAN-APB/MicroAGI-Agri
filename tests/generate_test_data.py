import numpy as np
import cv2
import os

def generate_test_images(output_dir, num_images=10):
    """生成测试图像数据"""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(f"{output_dir}/test_{i}.png", img)

if __name__ == '__main__':
    generate_test_images("test_data")