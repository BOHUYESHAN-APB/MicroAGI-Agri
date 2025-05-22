# 模型与数据迁移指南

## 目录结构说明

当前项目使用如下目录结构组织迁移数据：
```
MicroAGI-Agri/
├── data/
│   ├── images/          # 迁移的图像数据
│   └── annotations/     # 标注文件
├── models/
│   ├── pretrained/     # 迁移的预训练模型
│   └── converted/      # 转换后的模型
└── configs/
    └── dataset.yaml    # 数据集配置
```

## 迁移步骤

### 1. 准备工作

执行迁移工具进行初始化：
```bash
python tools/migrate_models.py
```
这将创建必要的目录结构和配置文件。

### 2. 数据迁移

源数据位置：
- 图像数据：`D:\train\images`
- 模型文件：`D:\train\models`
- 原始代码：`D:\-Users-\Documents\GitHub\CNN-MicroAI-Colony`

迁移流程：

1. **图像数据**
   ```bash
   # 将训练图像复制到新项目
   xcopy "D:\train\images\*" "data\images\" /E /H /C /I
   ```

2. **模型权重**
   ```python
   from microai_core.utils.model_converter import convert_colony_detector
   
   # 转换菌落检测模型
   convert_colony_detector(
       torch_path=r"D:\train\models\colony_detector.pth",
       save_path="models/pretrained/colony_detector.pdparams"
   )
   
   # 转换抑菌圈检测模型
   convert_colony_detector(
       torch_path=r"D:\train\models\inhibition_detector.pth",
       save_path="models/pretrained/inhibition_detector.pdparams"
   )
   ```

### 3. 配置调整

检查并根据需要修改 `configs/dataset.yaml` 文件：

```yaml
data:
  train_dir: "data/images"
  classes: ['colony', 'inhibition_zone']
  split_ratio:
    train: 0.8
    val: 0.1
    test: 0.1

model:
  colony_detector:
    weights: "models/pretrained/colony_detector.pdparams"
    num_classes: 1
  inhibition_detector:
    weights: "models/pretrained/inhibition_detector.pdparams"
    num_classes: 1
```

### 4. 验证迁移

运行测试确认迁移成功：
```bash
python -m pytest tests/test_colony_detector.py -v
```

## 注意事项

1. 模型迁移
   - 确保原始模型为PyTorch格式
   - 检查模型结构匹配性
   - 验证转换后的模型输出

2. 数据迁移
   - 保持图像文件名一致性
   - 确保标注格式正确
   - 验证数据完整性

3. 可能的问题
   - 如果遇到模型加载错误，检查权重文件格式
   - 如果出现性能差异，可能需要调整模型参数
   - 数据加载错误可能是路径或格式问题

## 后续工作

1. 进行完整性测试
2. 验证模型性能
3. 根据需要调整模型参数
4. 更新配置文件