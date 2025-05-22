# MicroAI 智能微生物分析平台

## 项目简介
本项目是一个基于国产AI框架的微生物智能分析平台，专注于小杂粮作物病虫害生物防控场景。核心功能包括高精度菌落识别与计数、多维度抑菌圈智能评估等。

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 运行程序
```bash
python main.py --config configs/default.yaml
```

## 项目结构
```
microai-core/
├── configs/            # 配置文件
├── data/               # 数据目录
├── docs/               # 文档
├── model/              # 模型定义
├── training/           # 训练脚本
├── utils/              # 工具类
└── main.py             # 主程序入口
```

## 贡献指南
欢迎提交Pull Request，请确保代码符合PEP8规范。

## 许可证
MIT