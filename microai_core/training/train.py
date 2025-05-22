import paddle
import paddle.vision as vision
import yaml
from utils.data_augment import MicroAugment, AgriAugment, process_multispectral
from utils.metrics import calculate_map

class CustomBBoxHead(paddle.nn.Layer):
    """自定义边界框检测头"""
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(512, 256, 3, padding=1)
        self.conv2 = paddle.nn.Conv2D(256, 128, 3, padding=1)
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(128 * 7 * 7, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x)
        x = self.conv2(x)
        x = paddle.nn.functional.relu(x)
        x = self.flatten(x)
        return self.fc(x)

class AgriFasterRCNN(vision.models.FasterRCNN):
    """农业场景专用Faster R-CNN模型"""
    def __init__(self):
        super().__init__(
            backbone=vision.models.ResNet50_vd(pretrained=True),
            num_classes=2,
            rpn_channel=512,
            bbox_head=CustomBBoxHead()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = [features[f] for f in ['res2', 'res3', 'res4', 'res5']]
        return super().forward(features)

def train_model(args):
    """训练模型主函数"""
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化模型
    model = AgriFasterRCNN()
    if args.pretrain:
        model.set_state_dict(paddle.load(args.pretrain))
    
    # 配置优化器
    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=config['training']['learning_rate'],
        T_max=config['training']['num_epochs']
    )
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters()
    )
    
    # 数据增强
    transforms = [MicroAugment()]
    if args.use_multispectral:
        transforms.append(
            lambda x: process_multispectral(x, args.uv_channel, args.ir_channel)
        )
    transforms.append(AgriAugment())
    
    # 数据加载
    train_dataset = MicroDataset(
        data_dir=args.dataset,
        transforms=transforms,
        split='train'
    )
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    if args.val_set:
        val_dataset = MicroDataset(
            data_dir=args.val_set,
            split='val'
        )
        val_loader = paddle.io.DataLoader(
            val_dataset,
            batch_size=args.batch_size
        )
    
    # 模型量化
    if args.enable_quant:
        if args.quant_type == 'PTQ':
            quant = paddle.quantization.QuantizationAwareTraining(model)
        else:  # QAT
            quant = paddle.quantization.PostTrainingQuantization(model)
        model = quant.quantize()
    
    # 训练循环
    best_map = 0.0
    for epoch in range(config['training']['num_epochs']):
        model.train()
        for batch in train_loader:
            images, targets = batch
            outputs = model(images)
            loss = paddle.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        
        lr_scheduler.step()
        
        # 验证
        if args.val_set and (epoch + 1) % config['training']['val_interval'] == 0:
            model.eval()
            map_score = calculate_map(model, val_loader)
            print(f'Epoch {epoch+1}, mAP: {map_score:.4f}')
            
            # 保存最佳模型
            if map_score > best_map:
                best_map = map_score
                paddle.save(
                    model.state_dict(),
                    f'{args.output_dir}/best_model.pdparams'
                )
        
        # 定期保存检查点
        if (epoch + 1) % config['training']['save_interval'] == 0:
            paddle.save(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_map': best_map
                },
                f'{args.output_dir}/checkpoint_epoch{epoch+1}.pd'
            )