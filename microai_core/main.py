import argparse
from training.train import train_model
from utils.data_augment import MicroAugment

def main():
    parser = argparse.ArgumentParser(description='MicroAI 训练脚本')
    # 基础配置
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, required=True, help='训练集路径')
    parser.add_argument('--val_set', type=str, help='验证集路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--pretrain', type=str, help='预训练权重路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    
    # 多光谱数据处理
    parser.add_argument('--use_multispectral', action='store_true', help='启用多光谱数据处理')
    parser.add_argument('--uv_channel', type=str, help='UV通道数据路径')
    parser.add_argument('--ir_channel', type=str, help='IR通道数据路径')
    
    # 模型量化
    parser.add_argument('--enable_quant', action='store_true', help='启用模型量化')
    parser.add_argument('--quant_type', type=str, choices=['PTQ', 'QAT'], default='PTQ', help='量化类型')
    parser.add_argument('--quant_bits', type=int, default=8, help='量化位数')
    
    # 硬件适配
    parser.add_argument('--device', type=str, choices=['gpu', 'ascend', 'kunlun'], default='gpu', help='运行设备类型')
    parser.add_argument('--device_id', type=int, default=0, help='设备ID')
    args = parser.parse_args()

    print('开始训练...')
    train_model(args)
    print('训练完成!')

if __name__ == '__main__':
    main()