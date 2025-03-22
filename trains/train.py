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
                    [--state_weight STATE_WEIGHT] [--no_pretrained] [--backbone BACKBONE]
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
from models.model import create_model, MultiTaskLoss
from data.loaders import get_data_loaders, FruitDataset, get_transforms
from config import load_config
from trains.trainer import Trainer, create_trainer
from trains.evaluation import evaluate_and_report
from trains.checkpoint import resume_training, find_best_checkpoint



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
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=None, help='动量 (仅用于SGD)')
    parser.add_argument('--optimizer', type=str, default=None, help='优化器 (adam, sgd, adamw)')
    parser.add_argument('--scheduler', type=str, default=None, help='学习率调度器 (step, cosine, plateau, none)')
    parser.add_argument('--early_stopping', type=int, default=None, help='早停轮数')
    parser.add_argument('--fruit_weight', type=float, default=None, help='水果类型损失权重')
    parser.add_argument('--state_weight', type=float, default=None, help='腐烂状态损失权重')
    
    # 模型参数
    parser.add_argument('--no_pretrained', action='store_true', help='不使用预训练模型')
    parser.add_argument('--backbone', type=str, default=None, help='骨干网络类型 (efficientnet_b3, efficientnet_b4, resnet18, resnet34, resnet50, resnet101)')
    
    # 检查点保存参数
    parser.add_argument('--save_best', action='store_true', dest='save_best', help='保存最佳模型')
    parser.add_argument('--no_save_best', action='store_false', dest='save_best', help='不保存最佳模型')
    parser.add_argument('--save_latest', action='store_true', dest='save_latest', help='保存最新模型')
    parser.add_argument('--no_save_latest', action='store_false', dest='save_latest', help='不保存最新模型')
    parser.add_argument('--save_checkpoints', action='store_true', dest='save_checkpoints', help='启用周期性保存检查点')
    parser.add_argument('--no_save_checkpoints', action='store_false', dest='save_checkpoints', help='禁用周期性保存检查点')
    parser.add_argument('--save_freq', type=int, default=None, help='检查点保存频率（每多少轮保存一次）')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 设置save_best、save_latest和save_checkpoints的默认值为None，这样在update_config_from_args中可以判断是否指定了这些参数
    parser.set_defaults(save_best=None, save_latest=None, save_checkpoints=None)
    
    return parser.parse_args()


def get_train_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取训练配置
    
    Args:
        config_path (str, optional): 配置文件路径，如果为None则使用默认路径
        
    Returns:
        Dict[str, Any]: 训练配置字典
    """
    # 如果未指定配置文件路径，使用默认路径
    if config_path is None:
        config_path = os.path.join(project_root, 'config', 'config.yaml')
    
    # 加载配置文件
    config = load_config(config_path)
    
    # 创建必要的配置结构
    if 'train' not in config:
        config['train'] = {}
    if 'model' not in config:
        config['model'] = {}
    if 'device' not in config:
        config['device'] = {}
    
    train_config = config['train']
    device_config = config['device']
    
    # 定义默认配置字典
    default_train_config = {
        # 训练基本参数
        'epochs': 50,
        'save_dir': os.path.join(project_root, 'checkpoints'),
        'early_stopping': 10,
        'save_best': True,
        'save_latest': True,
        'save_checkpoints': True,
        'save_freq': 5,
        
        # 优化器参数
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'momentum': 0.9,  # 仅用于SGD
        
        # 学习率调度器参数
        'scheduler': 'cosine',
        'step_size': 10,  # 仅用于StepLR
        'gamma': 0.1,     # 仅用于StepLR
        'patience': 5,    # 仅用于ReduceLROnPlateau
        'factor': 0.1,    # 仅用于ReduceLROnPlateau
        
        # 多任务损失权重
        'fruit_weight': 1.0,
        'state_weight': 1.0,
        
        # 批次大小，默认值，如果设备配置中有批次大小则会被覆盖
        'batch_size': 32,
    }
    
    # 定义默认数据增强配置
    default_augmentation_config = {
        'horizontal_flip': True,
        'rotation_angle': 15,
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.05,
        'random_erasing': 0.2,
        'gaussian_noise': 0.05,
        'random_resized_crop': True,
        'mixup': {
            'enabled': False,
            'alpha': 0.2
        }
    }
    
    # 只为配置文件中缺失的参数设置默认值
    for key, value in default_train_config.items():
        if key not in train_config:
            train_config[key] = value
    
    # 处理批次大小，优先使用设备配置中的批次大小
    if 'batch_size' in device_config:
        train_config['batch_size'] = device_config['batch_size']
    
    # 处理混合精度训练设置
    if 'mixed_precision' in device_config:
        train_config['use_amp'] = device_config['mixed_precision']
    else:
        train_config['use_amp'] = False
    
    # 处理数据增强配置
    if 'augmentation' in config:
        # 如果配置文件中有augmentation部分，将其复制到train_config中
        train_config['augmentation'] = config['augmentation'].copy()
    elif 'augmentation' not in train_config:
        # 如果配置文件和train_config中都没有augmentation，创建默认配置
        train_config['augmentation'] = {}
    
    # 设置默认的数据增强参数
    aug_config = train_config['augmentation']
    for key, value in default_augmentation_config.items():
        if key not in aug_config:
            aug_config[key] = value
    
    return config


def update_config_from_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据命令行参数更新配置
    
    Args:
        config (Dict[str, Any]): 原始配置
        args (Dict[str, Any]): 命令行参数
        
    Returns:
        Dict[str, Any]: 更新后的配置
    """
    # 复制配置，避免修改原始配置
    updated_config = config.copy()
    train_config = updated_config.get('train', {}).copy()
    updated_config['train'] = train_config
    model_config = updated_config.get('model', {}).copy()
    updated_config['model'] = model_config
    device_config = updated_config.get('device', {}).copy()
    updated_config['device'] = device_config
    
    # 更新训练参数
    if 'epochs' in args and args['epochs'] is not None:
        train_config['epochs'] = args['epochs']
    
    if 'learning_rate' in args and args['learning_rate'] is not None:
        train_config['learning_rate'] = args['learning_rate']
    
    if 'weight_decay' in args and args['weight_decay'] is not None:
        train_config['weight_decay'] = args['weight_decay']
    
    if 'momentum' in args and args['momentum'] is not None:
        train_config['momentum'] = args['momentum']
    
    if 'optimizer' in args and args['optimizer'] is not None:
        train_config['optimizer'] = args['optimizer']
    
    if 'scheduler' in args and args['scheduler'] is not None:
        train_config['scheduler'] = args['scheduler']
    
    # 将batch_size从设备配置移到训练配置中，保持一致性
    if 'batch_size' in args and args['batch_size'] is not None:
        train_config['batch_size'] = args['batch_size']
        # 同时更新device_config中的batch_size，保持兼容性
        device_config['batch_size'] = args['batch_size']
    
    if 'early_stopping' in args and args['early_stopping'] is not None:
        train_config['early_stopping'] = args['early_stopping']
    
    if 'fruit_weight' in args and args['fruit_weight'] is not None:
        train_config['fruit_weight'] = args['fruit_weight']
    
    if 'state_weight' in args and args['state_weight'] is not None:
        train_config['state_weight'] = args['state_weight']
    
    # 更新检查点保存相关参数
    if 'save_best' in args and args['save_best'] is not None:
        train_config['save_best'] = args['save_best']
    
    if 'save_latest' in args and args['save_latest'] is not None:
        train_config['save_latest'] = args['save_latest']
    
    if 'save_checkpoints' in args and args['save_checkpoints'] is not None:
        train_config['save_checkpoints'] = args['save_checkpoints']
    
    if 'save_freq' in args and args['save_freq'] is not None:
        train_config['save_freq'] = args['save_freq']
        
    # 更新模型参数
    if 'backbone' in args and args['backbone'] is not None:
        model_config['backbone'] = args['backbone']
        
    if 'no_pretrained' in args and args['no_pretrained']:
        model_config['pretrained'] = False
    
    return updated_config


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
    pretrained = not args.no_pretrained if args.no_pretrained else config.get('model', {}).get('pretrained', True)
    backbone = config.get('model', {}).get('backbone', 'efficientnet_b4')
    print(f"使用骨干网络: {backbone}")
    print(f"是否使用预训练模型: {pretrained}")
    
    model = create_model(
        num_fruit_classes=len(fruit_classes),
        num_state_classes=len(state_classes),
        backbone=backbone,
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
    save_latest = config['train'].get('save_latest', True)
    save_checkpoints = config['train'].get('save_checkpoints', True)
    save_freq = config['train'].get('save_freq', 5)
    
    # 打印训练设置
    print(f"训练轮数: {epochs}")
    print(f"早停轮数: {early_stopping}")
    print(f"保存最佳模型: {save_best}")
    print(f"保存最新模型: {save_latest}")
    print(f"启用周期性检查点: {save_checkpoints}")
    print(f"检查点保存频率: 每{save_freq}轮")
    
    trainer.train(
        num_epochs=epochs,
        save_best=save_best,
        save_latest=save_latest,
        save_checkpoints=save_checkpoints,
        save_freq=save_freq,
        early_stopping=early_stopping
    )
    
    # 评估最佳模型
    checkpoint_dir = config['train'].get('save_dir', os.path.join(project_root, 'checkpoints'))
    best_checkpoint = find_best_checkpoint(checkpoint_dir)
    
    if best_checkpoint:
        # 加载最佳模型
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 在验证集上评估模型
        print("\n在验证集上评估最佳模型:")
        evaluate_and_report(
            model=model,
            data_loader=val_loader,
            device=device,
            fruit_class_names=fruit_classes,
            state_class_names=state_classes,
            save_dir=os.path.join(checkpoint_dir, 'validation_results')
        )
        
        # 创建测试集数据加载器进行最终评估
        print("\n在测试集上评估最佳模型:")
        # 创建一个临时的FruitDataLoader来获取测试集
        test_dataset = FruitDataset(
            csv_file=os.path.join(project_root, 
                              config['data'].get('processed_dir', 'data'), 
                              config['data']['dataset_csv']),
            transform=get_transforms(mode='test', img_size=config['model']['img_size']),
            split='test'  # 明确指定使用测试集
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['device']['batch_size'],
            shuffle=False,  # 测试集不需要打乱
            num_workers=config['device']['num_workers'],
            pin_memory=True
        )
        
        print(f"测试集大小: {len(test_dataset)}")
        
        # 在测试集上进行最终评估
        evaluate_and_report(
            model=model,
            data_loader=test_loader,
            device=device,
            fruit_class_names=fruit_classes,
            state_class_names=state_classes,
            save_dir=os.path.join(checkpoint_dir, 'test_results')
        )
    
    print("训练完成！")


if __name__ == '__main__':
    main()
