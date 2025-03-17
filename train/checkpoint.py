"""模型检查点模块

这个模块实现了模型检查点的保存和加载功能，包括：
1. 模型权重保存
2. 训练状态保存
3. 训练恢复

主要函数：
- save_checkpoint: 保存检查点
- load_checkpoint: 加载检查点
- resume_training: 恢复训练
"""

import os
import sys
import json
import torch
from typing import Dict, Optional, Any, Tuple

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int, metrics: Dict[str, float], history: Dict[str, list],
                   save_path: str, is_best: bool = False) -> None:
    """
    保存检查点
    
    Args:
        model (torch.nn.Module): 模型
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器
        epoch (int): 当前训练轮数
        metrics (Dict[str, float]): 当前性能指标
        history (Dict[str, list]): 训练历史记录
        save_path (str): 保存路径
        is_best (bool): 是否为最佳模型
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建检查点字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'history': history
    }
    
    # 保存检查点
    torch.save(checkpoint, save_path)
    
    # 如果是最佳模型，复制一份
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(checkpoint, best_path)
    
    print(f"检查点已保存到: {save_path}")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   checkpoint_path: str, device: torch.device) -> Tuple[int, Dict[str, float], Dict[str, list]]:
    """
    加载检查点
    
    Args:
        model (torch.nn.Module): 模型
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器
        checkpoint_path (str): 检查点路径
        device (torch.device): 设备
        
    Returns:
        Tuple[int, Dict[str, float], Dict[str, list]]: (当前轮数, 性能指标, 训练历史记录)
    """
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, {}, {}
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载学习率调度器状态
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 获取当前轮数和性能指标
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    history = checkpoint.get('history', {})
    
    print(f"从轮数 {epoch} 加载检查点成功")
    
    return epoch, metrics, history


def resume_training(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   checkpoint_dir: str, device: torch.device) -> Tuple[int, Dict[str, float], Dict[str, list]]:
    """
    恢复训练
    
    Args:
        model (torch.nn.Module): 模型
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器
        checkpoint_dir (str): 检查点目录
        device (torch.device): 设备
        
    Returns:
        Tuple[int, Dict[str, float], Dict[str, list]]: (当前轮数, 性能指标, 训练历史记录)
    """
    # 检查目录是否存在
    if not os.path.exists(checkpoint_dir):
        print(f"检查点目录不存在: {checkpoint_dir}")
        return 0, {}, {}
    
    # 查找最新的检查点
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
    
    if not checkpoints:
        print(f"未找到检查点文件")
        return 0, {}, {}
    
    # 按照轮数排序
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    
    print(f"找到最新检查点: {latest_checkpoint}")
    
    # 加载检查点
    return load_checkpoint(model, optimizer, scheduler, latest_checkpoint, device)


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    查找最佳检查点
    
    Args:
        checkpoint_dir (str): 检查点目录
        
    Returns:
        Optional[str]: 最佳检查点路径，如果不存在则返回None
    """
    # 检查目录是否存在
    if not os.path.exists(checkpoint_dir):
        print(f"检查点目录不存在: {checkpoint_dir}")
        return None
    
    # 查找最佳检查点
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if not os.path.exists(best_path):
        print(f"未找到最佳检查点文件")
        return None
    
    return best_path


def save_training_history(history: Dict[str, list], save_path: str) -> None:
    """
    保存训练历史记录
    
    Args:
        history (Dict[str, list]): 训练历史记录
        save_path (str): 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将numpy数组转换为列表
    serializable_history = {}
    for key, value in history.items():
        serializable_history[key] = [float(v) for v in value]
    
    # 保存为JSON
    with open(save_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    
    print(f"训练历史记录已保存到: {save_path}")


def load_training_history(history_path: str) -> Dict[str, list]:
    """
    加载训练历史记录
    
    Args:
        history_path (str): 历史记录路径
        
    Returns:
        Dict[str, list]: 训练历史记录
    """
    # 检查文件是否存在
    if not os.path.exists(history_path):
        print(f"历史记录文件不存在: {history_path}")
        return {}
    
    # 加载JSON
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history
