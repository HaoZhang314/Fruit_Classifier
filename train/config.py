"""训练配置模块

这个模块负责管理训练相关的配置，包括：
1. 训练参数配置
2. 优化器配置
3. 学习率调度器配置
4. 数据增强配置

主要函数：
- get_train_config: 获取训练配置
"""

import os
import sys
from typing import Dict, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入配置模块
from config import load_config


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
    
    # 加载基础配置
    config = load_config(config_path)
    
    # 如果配置中没有train部分，添加默认训练配置
    if 'train' not in config:
        config['train'] = {}
    
    # 设置默认训练参数
    train_config = config['train']
    
    # 训练基本参数
    train_config.setdefault('epochs', 50)
    train_config.setdefault('save_dir', os.path.join(project_root, 'checkpoints'))
    train_config.setdefault('early_stopping', 10)
    train_config.setdefault('save_best', True)
    
    # 优化器参数
    train_config.setdefault('optimizer', 'adam')
    train_config.setdefault('learning_rate', 0.001)
    train_config.setdefault('weight_decay', 0.0001)
    train_config.setdefault('momentum', 0.9)  # 仅用于SGD
    
    # 学习率调度器参数
    train_config.setdefault('scheduler', 'cosine')
    train_config.setdefault('step_size', 10)  # 仅用于StepLR
    train_config.setdefault('gamma', 0.1)     # 仅用于StepLR
    train_config.setdefault('patience', 5)    # 仅用于ReduceLROnPlateau
    train_config.setdefault('factor', 0.1)    # 仅用于ReduceLROnPlateau
    
    # 多任务损失权重
    train_config.setdefault('fruit_weight', 1.0)
    train_config.setdefault('state_weight', 1.0)
    
    # 数据增强参数
    if 'augmentation' not in train_config:
        train_config['augmentation'] = {}
    
    aug_config = train_config['augmentation']
    aug_config.setdefault('horizontal_flip', True)
    aug_config.setdefault('rotation_angle', 15)
    aug_config.setdefault('brightness', 0.1)
    aug_config.setdefault('contrast', 0.1)
    aug_config.setdefault('saturation', 0.1)
    aug_config.setdefault('hue', 0.05)
    
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
    
    # 更新训练参数
    if 'epochs' in args and args['epochs'] is not None:
        train_config['epochs'] = args['epochs']
    
    if 'learning_rate' in args and args['learning_rate'] is not None:
        train_config['learning_rate'] = args['learning_rate']
    
    if 'optimizer' in args and args['optimizer'] is not None:
        train_config['optimizer'] = args['optimizer']
    
    if 'scheduler' in args and args['scheduler'] is not None:
        train_config['scheduler'] = args['scheduler']
    
    if 'batch_size' in args and args['batch_size'] is not None:
        updated_config['device']['batch_size'] = args['batch_size']
    
    if 'weight_decay' in args and args['weight_decay'] is not None:
        train_config['weight_decay'] = args['weight_decay']
    
    if 'early_stopping' in args and args['early_stopping'] is not None:
        train_config['early_stopping'] = args['early_stopping']
    
    if 'fruit_weight' in args and args['fruit_weight'] is not None:
        train_config['fruit_weight'] = args['fruit_weight']
    
    if 'state_weight' in args and args['state_weight'] is not None:
        train_config['state_weight'] = args['state_weight']
    
    return updated_config


if __name__ == '__main__':
    # 测试获取配置
    config = get_train_config()
    print("训练配置:")
    for key, value in config['train'].items():
        print(f"{key}: {value}")
