"""训练主脚本

这个脚本是水果分类项目的主要训练入口，包括：
1. 数据加载和预处理
2. 模型创建和初始化
3. 训练循环
4. 模型评估和保存

使用方法：
    python train.py [--config CONFIG] [--resume] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                    [--learning_rate LR] [--optimizer OPTIMIZER] [--scheduler SCHEDULER]
                    [--early_stopping EARLY_STOPPING] [--fruit_weight FRUIT_WEIGHT]
                    [--state_weight STATE_WEIGHT] [--no_pretrained]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict, Tuple, List, Optional, Union, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目其他模块
from models.efficientnet_model import create_model, MultiTaskLoss
from data.loaders import get_data_loaders
from config import load_config
from train.trainer import Trainer, create_trainer
from train.config import get_train_config, update_config_from_args
from train.evaluation import evaluate_and_report
from train.checkpoint import resume_training, find_best_checkpoint


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子，确保结果可复现
    
    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='水果分类训练脚本')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--optimizer', type=str, default=None, help='优化器 (adam, sgd, adamw)')
    parser.add_argument('--scheduler', type=str, default=None, help='学习率调度器 (step, cosine, plateau, none)')
    parser.add_argument('--early_stopping', type=int, default=None, help='早停轮数')
    parser.add_argument('--fruit_weight', type=float, default=None, help='水果类型损失权重')
    parser.add_argument('--state_weight', type=float, default=None, help='腐烂状态损失权重')
    parser.add_argument('--no_pretrained', action='store_true', help='不使用预训练模型')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = get_train_config(args.config)
    
    # 根据命令行参数更新配置
    config = update_config_from_args(config, vars(args))
    
    # 打印配置信息
    print("训练配置:")
    for key, value in config['train'].items():
        print(f"  {key}: {value}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader, (fruit_classes, state_classes) = get_data_loaders(config)
    print(f"水果类型: {fruit_classes}")
    print(f"腐烂状态: {state_classes}")
    
    # 创建模型
    pretrained = not args.no_pretrained
    model = create_model(
        num_fruit_classes=len(fruit_classes),
        num_state_classes=len(state_classes),
        pretrained=pretrained
    )
    
    # 创建训练器
    trainer = create_trainer(model, train_loader, val_loader, config)
    
    # 如果需要从检查点恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint_dir = config['train'].get('save_dir', os.path.join(project_root, 'checkpoints'))
        start_epoch, _, history = resume_training(model, trainer.optimizer, trainer.scheduler, checkpoint_dir, device)
        if history:
            trainer.history = history
    
    # 训练模型
    epochs = config['train'].get('epochs', 50)
    early_stopping = config['train'].get('early_stopping', 10)
    save_best = config['train'].get('save_best', True)
    
    trainer.train(
        num_epochs=epochs,
        save_best=save_best,
        early_stopping=early_stopping
    )
    
    # 评估最佳模型
    checkpoint_dir = config['train'].get('save_dir', os.path.join(project_root, 'checkpoints'))
    best_checkpoint = find_best_checkpoint(checkpoint_dir)
    
    if best_checkpoint:
        # 加载最佳模型
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估模型
        print("\n评估最佳模型:")
        evaluate_and_report(
            model=model,
            data_loader=val_loader,
            device=device,
            fruit_class_names=fruit_classes,
            state_class_names=state_classes,
            save_dir=checkpoint_dir
        )
    
    print("训练完成！")


if __name__ == '__main__':
    main()
