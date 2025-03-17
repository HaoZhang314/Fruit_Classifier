"""模型评估模块

这个模块实现了模型评估的功能，包括：
1. 模型性能评估
2. 混淆矩阵计算和可视化
3. 分类报告生成

主要函数：
- evaluate_model: 评估模型性能
- plot_confusion_matrix: 绘制混淆矩阵
- generate_classification_report: 生成分类报告
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Union, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, 
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    评估模型性能
    
    Args:
        model (torch.nn.Module): 待评估的模型
        data_loader (DataLoader): 数据加载器
        device (torch.device): 计算设备
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (水果类型预测, 水果类型真实标签, 腐烂状态预测, 腐烂状态真实标签)
    """
    model.eval()
    all_fruit_preds = []
    all_state_preds = []
    all_fruit_targets = []
    all_state_targets = []
    
    with torch.no_grad():
        for images, fruit_targets, state_targets in data_loader:
            # 将数据移动到设备
            images = images.to(device)
            
            # 前向传播
            fruit_logits, state_logits = model(images)
            
            # 计算预测结果
            fruit_preds = torch.argmax(fruit_logits, dim=1)
            state_preds = torch.argmax(state_logits, dim=1)
            
            # 收集预测和目标
            all_fruit_preds.append(fruit_preds.cpu().numpy())
            all_state_preds.append(state_preds.cpu().numpy())
            all_fruit_targets.append(fruit_targets.numpy())
            all_state_targets.append(state_targets.numpy())
    
    # 合并所有批次的预测和目标
    all_fruit_preds = np.concatenate(all_fruit_preds)
    all_state_preds = np.concatenate(all_state_preds)
    all_fruit_targets = np.concatenate(all_fruit_targets)
    all_state_targets = np.concatenate(all_state_targets)
    
    return all_fruit_preds, all_fruit_targets, all_state_preds, all_state_targets


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], title: str, 
                         save_path: str) -> None:
    """
    绘制混淆矩阵
    
    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        class_names (List[str]): 类别名称
        title (str): 图表标题
        save_path (str): 保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    
    # 设置刻度标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)
    
    # 添加文本注释
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:{fmt}})",
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black",
                     fontsize=8)
    
    # 设置标签
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str], task_name: str,
                                  save_dir: str) -> Dict:
    """
    生成分类报告
    
    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        class_names (List[str]): 类别名称
        task_name (str): 任务名称
        save_dir (str): 保存目录
        
    Returns:
        Dict: 分类报告字典
    """
    # 生成分类报告
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 保存报告为文本文件
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"{task_name}_classification_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"{task_name} 分类报告\n")
        f.write("="*50 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    # 绘制混淆矩阵
    cm_path = os.path.join(save_dir, f"{task_name}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, f"{task_name} 混淆矩阵", cm_path)
    
    return report


def evaluate_and_report(model: torch.nn.Module, data_loader: DataLoader, 
                       device: torch.device, fruit_class_names: List[str], 
                       state_class_names: List[str], save_dir: str) -> Dict:
    """
    评估模型并生成报告
    
    Args:
        model (torch.nn.Module): 待评估的模型
        data_loader (DataLoader): 数据加载器
        device (torch.device): 计算设备
        fruit_class_names (List[str]): 水果类型名称
        state_class_names (List[str]): 腐烂状态名称
        save_dir (str): 保存目录
        
    Returns:
        Dict: 评估结果字典
    """
    # 评估模型
    fruit_preds, fruit_targets, state_preds, state_targets = evaluate_model(model, data_loader, device)
    
    # 生成水果类型分类报告
    fruit_report = generate_classification_report(
        fruit_targets, fruit_preds, fruit_class_names, "水果类型", save_dir
    )
    
    # 生成腐烂状态分类报告
    state_report = generate_classification_report(
        state_targets, state_preds, state_class_names, "腐烂状态", save_dir
    )
    
    # 计算总体准确率
    fruit_acc = (fruit_preds == fruit_targets).mean()
    state_acc = (state_preds == state_targets).mean()
    
    # 打印总体准确率
    print(f"水果类型准确率: {fruit_acc:.4f}")
    print(f"腐烂状态准确率: {state_acc:.4f}")
    
    # 返回评估结果
    return {
        'fruit_accuracy': fruit_acc,
        'state_accuracy': state_acc,
        'fruit_report': fruit_report,
        'state_report': state_report
    }
