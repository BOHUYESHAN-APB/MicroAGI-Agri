# 农智菌通智能微生物分析平台

基于国产AI框架（PaddlePaddle）的微生物智能分析平台，专注于小杂粮作物病虫害生物防控场景。

## 主要功能

- 高精度菌落识别与计数（支持复杂背景干扰）
- 多维度抑菌圈智能评估
- 国产化AI框架迁移与优化
- 农业场景专项数据集构建

## 环境要求

### 硬件要求
- GPU：NVIDIA GPU（支持CUDA 11+）
- 内存：≥16GB
- 存储：≥500GB（原始图像数据集）

### 软件依赖
```bash
Python >= 3.8
PaddlePaddle >= 3.0.0
OpenCV >= 4.6.0
```

## 安装说明

1. 克隆代码仓库
```bash
git clone https://github.com/BOHUYESHAN-APB/MicroAGI-Agri.git
cd MicroAGI-Agri
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 安装开发版本
```bash
pip install -e .
```

## 项目结构
```
microai_core/
├── data/               # 数据管理模块
├── models/            # 模型定义
├── training/         # 训练相关
└── utils/           # 工具类
```

## 快速开始

1. 准备数据集
```bash
python tools/prepare_dataset.py --data_dir /path/to/raw/images
```

2. 训练模型
```bash
python training/train.py \
    --config configs/default.yaml \
    --dataset data/trainval.txt \
    --val_set data/test.txt \
    --batch_size 8 \
    --output_dir outputs/
```

3. 模型预测
```bash
python inference/predict.py \
    --image test.jpg \
    --model outputs/best_model.pdparams
```

## 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m '[类型] 描述 #issue编号'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

## 许可证

本项目采用Apache License 2.0许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目维护者：[BOHUYESHAN-APB](https://github.com/BOHUYESHAN-APB)
- 项目Issues：[GitHub Issues](https://github.com/BOHUYESHAN-APB/MicroAGI-Agri/issues)
