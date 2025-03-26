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

import os              # 提供操作系统功能，如文件路径处理
import sys             # 提供系统特定的参数和函数
import argparse        # 命令行参数解析库
import torch           # PyTorch深度学习库
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as optim  # PyTorch优化器模块
from torch.utils.data import DataLoader  # 数据加载工具
import numpy as np     # 科学计算库
import random          # 随机数生成库
from typing import Dict, Tuple, List, Optional, Union, Any  # 类型提示

# 添加项目根目录到Python路径，确保可以导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
project_root = os.path.dirname(current_dir)               # 获取项目根目录
if project_root not in sys.path:                          # 如果项目根目录不在系统路径中
    sys.path.append(project_root)                         # 添加到系统路径

# 导入项目其他模块
from models.model import create_model, MultiTaskLoss      # 导入模型创建和多任务损失函数
from data.loaders import get_data_loaders, FruitDataset, get_transforms  # 数据加载和预处理
from config import load_config                            # 配置加载功能
from trains.trainer import Trainer, create_trainer        # 训练器实现
from trains.evaluation import evaluate_and_report         # 评估和报告功能
from trains.checkpoint import resume_training, find_best_checkpoint  # 检查点管理功能



def set_seed(seed: int = 42) -> None:
    """
    设置随机种子，确保结果可复现
    
    Args:
        seed (int): 随机种子
    """
    random.seed(seed)                  # 设置Python内置随机模块的种子
    np.random.seed(seed)               # 设置NumPy随机种子
    torch.manual_seed(seed)            # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)       # 设置PyTorch GPU随机种子
    torch.cuda.manual_seed_all(seed)   # 设置所有GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False     # 禁用cudnn自动寻找最适合当前配置的高效算法
    os.environ['PYTHONHASHSEED'] = str(seed)   # 设置Python哈希种子


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='水果分类训练脚本')  # 创建参数解析器
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')  # 配置文件路径参数
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')  # 恢复训练标志
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')  # 训练轮数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')  # 批次大小
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')  # 学习率
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')  # L2正则化系数
    parser.add_argument('--momentum', type=float, default=None, help='动量 (仅用于SGD)')  # SGD优化器的动量参数
    parser.add_argument('--optimizer', type=str, default=None, help='优化器 (adam, sgd, adamw)')  # 优化器类型
    parser.add_argument('--scheduler', type=str, default=None, help='学习率调度器 (step, cosine, plateau, none)')  # 学习率调度器
    parser.add_argument('--early_stopping', type=int, default=None, help='早停轮数')  # 早停策略的容忍轮数
    parser.add_argument('--fruit_weight', type=float, default=None, help='水果类型损失权重')  # 水果类型任务的损失权重
    parser.add_argument('--state_weight', type=float, default=None, help='腐烂状态损失权重')  # 腐烂状态任务的损失权重
    
    # 模型参数
    parser.add_argument('--no_pretrained', action='store_true', help='不使用预训练模型')  # 是否使用预训练模型
    parser.add_argument('--backbone', type=str, default=None, 
                        help='骨干网络类型 (efficientnet_b3, efficientnet_b4, resnet18, resnet34, resnet50, resnet101)')  # 骨干网络类型
    
    # 检查点保存参数
    parser.add_argument('--save_best', action='store_true', dest='save_best', help='保存最佳模型')  # 是否保存最佳模型
    parser.add_argument('--no_save_best', action='store_false', dest='save_best', help='不保存最佳模型')  # 不保存最佳模型
    parser.add_argument('--save_latest', action='store_true', dest='save_latest', help='保存最新模型')  # 是否保存最新模型
    parser.add_argument('--no_save_latest', action='store_false', dest='save_latest', help='不保存最新模型')  # 不保存最新模型
    parser.add_argument('--save_checkpoints', action='store_true', dest='save_checkpoints', help='启用周期性保存检查点')  # 是否周期性保存检查点
    parser.add_argument('--no_save_checkpoints', action='store_false', dest='save_checkpoints', help='禁用周期性保存检查点')  # 禁用周期性保存检查点
    parser.add_argument('--save_freq', type=int, default=None, help='检查点保存频率（每多少轮保存一次）')  # 检查点保存频率
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')  # 随机种子设置
    
    # 设置save_best、save_latest和save_checkpoints的默认值为None，这样在update_config_from_args中可以判断是否指定了这些参数
    parser.set_defaults(save_best=None, save_latest=None, save_checkpoints=None)
    
    return parser.parse_args()  # 返回解析后的参数


def get_train_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取训练配置
    
    加载配置文件并设置默认值，确保所有必要的训练参数都有合理的设置。
    如果配置文件中缺少某些参数，将使用预定义的默认值填充。
    
    Args:
        config_path (str, optional): 配置文件路径，如果为None则使用默认路径
        
    Returns:
        Dict[str, Any]: 包含完整训练配置的字典
    """
    # 如果未指定配置文件路径，使用默认路径
    if config_path is None:
        config_path = os.path.join(project_root, 'config', 'config.yaml')
    
    # 加载配置文件
    config = load_config(config_path)
    
    # 创建必要的配置结构，确保基本配置节点存在
    if 'train' not in config:
        config['train'] = {}  # 训练相关配置
    if 'model' not in config:
        config['model'] = {}  # 模型相关配置
    if 'device' not in config:
        config['device'] = {}  # 设备相关配置
    
    # 获取配置引用，方便后续操作
    train_config = config['train']
    device_config = config['device']
    
    # 定义默认配置字典，包含所有训练所需的基本参数
    default_train_config = {
        # 训练基本参数
        'epochs': 50,                                        # 训练总轮数
        'save_dir': os.path.join(project_root, 'checkpoints'), # 模型保存目录
        'early_stopping': 10,                                # 早停策略的容忍轮数
        'save_best': True,                                   # 是否保存验证集上表现最佳的模型
        'save_latest': True,                                 # 是否保存最新的模型
        'save_checkpoints': True,                            # 是否定期保存检查点
        'save_freq': 5,                                      # 检查点保存频率（每多少轮保存一次）
        
        # 优化器参数
        'optimizer': 'adam',                                 # 优化器类型（adam, sgd, adamw等）
        'learning_rate': 0.001,                              # 学习率
        'weight_decay': 0.0001,                              # L2正则化系数
        'momentum': 0.9,                                     # 动量参数（仅用于SGD优化器）
        
        # 学习率调度器参数
        'scheduler': 'cosine',                               # 学习率调度器类型（step, cosine, plateau, none）
        'step_size': 10,                                     # StepLR的步长（每多少轮降低学习率）
        'gamma': 0.1,                                        # StepLR的衰减率
        'patience': 5,                                       # ReduceLROnPlateau的容忍轮数
        'factor': 0.1,                                       # ReduceLROnPlateau的衰减因子
        
        # 多任务损失权重
        'fruit_weight': 1.0,                                 # 水果类型任务的损失权重
        'state_weight': 1.0,                                 # 腐烂状态任务的损失权重
        
        # 批次大小，默认值，如果设备配置中有批次大小则会被覆盖
        'batch_size': 32,                                    # 训练批次大小
    }
    
    # 定义默认数据增强配置，控制训练时的数据增强策略
    default_augmentation_config = {
        'horizontal_flip': True,                             # 是否启用水平翻转
        'rotation_angle': 15,                                # 随机旋转的最大角度
        'brightness': 0.1,                                   # 亮度调整范围
        'contrast': 0.1,                                     # 对比度调整范围
        'saturation': 0.1,                                   # 饱和度调整范围
        'hue': 0.05,                                         # 色调调整范围
        'random_erasing': 0.2,                               # 随机擦除的概率
        'gaussian_noise': 0.05,                              # 高斯噪声的强度
        'random_resized_crop': True,                         # 是否启用随机裁剪并调整大小
        'mixup': {                                           # Mixup数据增强设置
            'enabled': False,                                # 是否启用Mixup
            'alpha': 0.2                                     # Mixup的alpha参数
        }
    }
    
    # 只为配置文件中缺失的参数设置默认值，保留用户在配置文件中的自定义设置
    for key, value in default_train_config.items():
        if key not in train_config:
            train_config[key] = value
    
    # 处理批次大小，优先使用设备配置中的批次大小（兼容旧版本配置）
    if 'batch_size' in device_config:
        train_config['batch_size'] = device_config['batch_size']
    
    # 处理混合精度训练设置，提高训练效率
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
    
    # 设置默认的数据增强参数，保留用户自定义的增强设置
    aug_config = train_config['augmentation']
    for key, value in default_augmentation_config.items():
        if key not in aug_config:
            aug_config[key] = value
    
    return config


def update_config_from_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据命令行参数更新配置
    
    允许用户通过命令行参数覆盖配置文件中的设置，提供更灵活的训练控制方式。
    只有显式指定的参数（非None）才会覆盖原始配置。
    
    Args:
        config (Dict[str, Any]): 原始配置字典
        args (Dict[str, Any]): 命令行参数字典
        
    Returns:
        Dict[str, Any]: 更新后的配置字典
    """
    # 复制配置，避免修改原始配置
    updated_config = config.copy()
    train_config = updated_config.get('train', {}).copy()  # 获取训练配置的副本
    updated_config['train'] = train_config
    model_config = updated_config.get('model', {}).copy()  # 获取模型配置的副本
    updated_config['model'] = model_config
    device_config = updated_config.get('device', {}).copy()  # 获取设备配置的副本
    updated_config['device'] = device_config
    
    # 更新训练参数 - 只更新命令行中明确指定的参数（非None值）
    if 'epochs' in args and args['epochs'] is not None:
        train_config['epochs'] = args['epochs']  # 更新训练轮数
    
    if 'learning_rate' in args and args['learning_rate'] is not None:
        train_config['learning_rate'] = args['learning_rate']  # 更新学习率
    
    if 'weight_decay' in args and args['weight_decay'] is not None:
        train_config['weight_decay'] = args['weight_decay']  # 更新权重衰减
    
    if 'momentum' in args and args['momentum'] is not None:
        train_config['momentum'] = args['momentum']  # 更新动量参数
    
    if 'optimizer' in args and args['optimizer'] is not None:
        train_config['optimizer'] = args['optimizer']  # 更新优化器类型
    
    if 'scheduler' in args and args['scheduler'] is not None:
        train_config['scheduler'] = args['scheduler']  # 更新学习率调度器
    
    # 将batch_size从设备配置移到训练配置中，保持一致性
    if 'batch_size' in args and args['batch_size'] is not None:
        train_config['batch_size'] = args['batch_size']  # 更新训练批次大小
        # 同时更新device_config中的batch_size，保持兼容性（支持旧版本代码）
        device_config['batch_size'] = args['batch_size']
    
    if 'early_stopping' in args and args['early_stopping'] is not None:
        train_config['early_stopping'] = args['early_stopping']  # 更新早停轮数
    
    if 'fruit_weight' in args and args['fruit_weight'] is not None:
        train_config['fruit_weight'] = args['fruit_weight']  # 更新水果类型损失权重
    
    if 'state_weight' in args and args['state_weight'] is not None:
        train_config['state_weight'] = args['state_weight']  # 更新腐烂状态损失权重
    
    # 更新检查点保存相关参数
    if 'save_best' in args and args['save_best'] is not None:
        train_config['save_best'] = args['save_best']  # 更新是否保存最佳模型
    
    if 'save_latest' in args and args['save_latest'] is not None:
        train_config['save_latest'] = args['save_latest']  # 更新是否保存最新模型
    
    if 'save_checkpoints' in args and args['save_checkpoints'] is not None:
        train_config['save_checkpoints'] = args['save_checkpoints']  # 更新是否保存周期性检查点
    
    if 'save_freq' in args and args['save_freq'] is not None:
        train_config['save_freq'] = args['save_freq']  # 更新检查点保存频率
        
    # 更新模型参数
    if 'backbone' in args and args['backbone'] is not None:
        model_config['backbone'] = args['backbone']  # 更新骨干网络类型
        
    if 'no_pretrained' in args and args['no_pretrained']:
        model_config['pretrained'] = False  # 禁用预训练模型
    
    return updated_config


def main():
    """
    主函数 - 训练和评估水果分类模型的入口点
    
    流程包括:
    1. 解析命令行参数
    2. 加载和更新配置
    3. 准备数据集和数据加载器
    4. 创建并训练模型
    5. 评估最佳模型在验证集和测试集上的性能
    """
    # 解析命令行参数 - 获取用户通过命令行指定的训练参数
    args = parse_args()
    
    # 设置随机种子 - 确保实验的可重复性
    set_seed(args.seed)
    
    # 加载配置 - 从配置文件中读取训练设置
    config = get_train_config(args.config)
    
    # 根据命令行参数更新配置 - 命令行参数优先级高于配置文件
    config = update_config_from_args(config, vars(args))
    
    # 打印配置信息 - 显示当前使用的训练参数
    print("训练配置:")
    for key, value in config['train'].items():
        print(f"  {key}: {value}")
    
    # 设置设备 - 优先使用GPU，如果不可用则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据 - 准备训练集和验证集的数据加载器
    train_loader, val_loader, (fruit_classes, state_classes) = get_data_loaders(config)
    print(f"水果类型: {fruit_classes}")  # 打印所有水果类别
    print(f"腐烂状态: {state_classes}")  # 打印所有腐烂状态类别
    
    # 创建模型 - 设置骨干网络和是否使用预训练权重
    # 优先使用命令行参数中的设置，如果未指定则使用配置文件中的设置
    pretrained = not args.no_pretrained if args.no_pretrained else config.get('model', {}).get('pretrained', True)
    backbone = config.get('model', {}).get('backbone', 'efficientnet_b4')
    print(f"使用骨干网络: {backbone}")
    print(f"是否使用预训练模型: {pretrained}")
    
    # 实例化模型 - 创建用于水果分类和腐烂状态检测的多任务模型
    model = create_model(
        num_fruit_classes=len(fruit_classes),  # 水果类别数量
        num_state_classes=len(state_classes),  # 腐烂状态类别数量
        backbone=backbone,                     # 骨干网络类型
        pretrained=pretrained                  # 是否使用预训练权重
    )
    
    # 创建训练器 - 封装了训练过程中需要的优化器、调度器和损失函数等组件
    trainer = create_trainer(model, train_loader, val_loader, config)
    
    # 如果需要从检查点恢复训练 - 加载之前训练的模型状态和训练历史
    start_epoch = 0  # 默认从第0轮开始训练
    if args.resume:
        checkpoint_dir = config['train'].get('save_dir', os.path.join(project_root, 'checkpoints'))
        # 恢复训练状态，包括模型权重、优化器状态和训练历史
        start_epoch, _, history = resume_training(model, trainer.optimizer, trainer.scheduler, checkpoint_dir, device)
        if history:
            trainer.history = history  # 更新训练器的历史记录
    
    # 获取训练相关参数
    epochs = config['train'].get('epochs', 50)                  # 训练总轮数
    early_stopping = config['train'].get('early_stopping', 10)  # 早停策略的容忍轮数
    save_best = config['train'].get('save_best', True)          # 是否保存验证集上表现最佳的模型
    save_latest = config['train'].get('save_latest', True)      # 是否保存最新的模型
    save_checkpoints = config['train'].get('save_checkpoints', True)  # 是否定期保存检查点
    save_freq = config['train'].get('save_freq', 5)             # 检查点保存频率
    
    # 打印训练设置 - 显示关键训练参数
    print(f"训练轮数: {epochs}")
    print(f"早停轮数: {early_stopping}")
    print(f"保存最佳模型: {save_best}")
    print(f"保存最新模型: {save_latest}")
    print(f"启用周期性检查点: {save_checkpoints}")
    print(f"检查点保存频率: 每{save_freq}轮")
    
    # 开始训练模型 - 执行训练循环
    trainer.train(
        num_epochs=epochs,                # 训练轮数
        save_best=save_best,              # 是否保存最佳模型
        save_latest=save_latest,          # 是否保存最新模型
        save_checkpoints=save_checkpoints,# 是否启用周期性保存
        save_freq=save_freq,              # 检查点保存频率
        early_stopping=early_stopping     # 早停策略的容忍轮数
    )
    
    # 评估最佳模型 - 在训练完成后评估模型性能
    checkpoint_dir = config['train'].get('save_dir', os.path.join(project_root, 'checkpoints'))
    best_checkpoint = find_best_checkpoint(checkpoint_dir)  # 查找验证集上性能最佳的模型检查点
    
    if best_checkpoint:
        # 加载最佳模型 - 恢复模型参数
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 在验证集上评估模型 - 计算准确率、混淆矩阵等指标
        print("\n在验证集上评估最佳模型:")
        evaluate_and_report(
            model=model,
            data_loader=val_loader,
            device=device,
            fruit_class_names=fruit_classes,
            state_class_names=state_classes,
            save_dir=os.path.join(checkpoint_dir, 'validation_results')  # 保存评估结果的目录
        )
        
        # 创建测试集数据加载器进行最终评估 - 使用未参与训练和验证的数据评估模型
        print("\n在测试集上评估最佳模型:")
        # 创建一个临时的FruitDataset来获取测试集
        test_dataset = FruitDataset(
            csv_file=os.path.join(project_root, 
                              config['data'].get('processed_dir', 'data'), 
                              config['data']['dataset_csv']),
            transform=get_transforms(mode='test', img_size=config['model']['img_size']),  # 测试集使用的数据变换
            split='test'  # 明确指定使用测试集
        )
        
        # 创建测试集的数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['device']['batch_size'],
            shuffle=False,  # 测试集不需要打乱顺序
            num_workers=config['device']['num_workers'],  # 数据加载的工作线程数
            pin_memory=True  # 将数据加载到固定内存，加速GPU训练
        )
        
        print(f"测试集大小: {len(test_dataset)}")
        
        # 在测试集上进行最终评估 - 计算模型在未见过数据上的性能
        evaluate_and_report(
            model=model,
            data_loader=test_loader,
            device=device,
            fruit_class_names=fruit_classes,
            state_class_names=state_classes,
            save_dir=os.path.join(checkpoint_dir, 'test_results')  # 测试结果保存目录
        )
    
    print("训练完成！")  # 训练流程结束提示


if __name__ == '__main__':
    main()  # 当脚本作为主程序运行时执行main函数
