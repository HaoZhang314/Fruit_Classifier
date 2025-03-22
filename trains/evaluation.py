#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""模型评估模块

这个模块实现了模型评估的功能，包括：
1. 模型性能评估
2. 混淆矩阵计算和可视化
3. 分类报告生成
4. 错误预测样本可视化

主要函数：
- evaluate_model: 评估模型性能
- plot_confusion_matrix: 绘制混淆矩阵
- generate_classification_report: 生成分类报告
- visualize_error_samples: 可视化错误预测的样本

使用方法：
    作为模块导入：
        from trains.evaluation import evaluate_and_report
    
    作为脚本执行：
        python -m trains.evaluation [--config CONFIG] [--checkpoint CHECKPOINT] [--output OUTPUT]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Union, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目其他模块
from models.model import create_model, FruitClassifier
from data.loaders import get_data_loaders, FruitDataset, get_transforms
from config import load_config
from trains.checkpoint import find_best_checkpoint


def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, 
                  device: torch.device, collect_errors: bool = False) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        model (torch.nn.Module): 待评估的模型
        data_loader (DataLoader): 数据加载器
        device (torch.device): 计算设备
        collect_errors (bool, optional): 是否收集错误预测的样本
        
    Returns:
        Dict[str, Any]: 评估结果字典，包含预测结果、真实标签和错误样本（如果collect_errors=True）
    """
    model.eval()
    all_fruit_preds = []
    all_state_preds = []
    all_fruit_targets = []
    all_state_targets = []
    
    # 收集错误预测的样本
    error_samples = []
    
    with torch.no_grad():
        for images, fruit_targets, state_targets in tqdm(data_loader, desc="评估模型"):
            # 将数据移动到设备
            images = images.to(device)
            fruit_targets_device = fruit_targets.to(device)
            state_targets_device = state_targets.to(device)
            
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
            
            # 收集错误预测的样本
            if collect_errors:
                for i in range(len(images)):
                    if fruit_preds[i] != fruit_targets_device[i] or state_preds[i] != state_targets_device[i]:
                        error_samples.append({
                            'image': images[i].cpu(),
                            'image_index': len(error_samples),  # 使用错误样本的索引来标识
                            'fruit_pred': fruit_preds[i].item(),
                            'fruit_target': fruit_targets_device[i].item(),
                            'state_pred': state_preds[i].item(),
                            'state_target': state_targets_device[i].item(),
                            'fruit_prob': torch.softmax(fruit_logits[i], dim=0).cpu().numpy(),
                            'state_prob': torch.softmax(state_logits[i], dim=0).cpu().numpy()
                        })
    
    # 合并所有批次的预测和目标
    all_fruit_preds = np.concatenate(all_fruit_preds)
    all_state_preds = np.concatenate(all_state_preds)
    all_fruit_targets = np.concatenate(all_fruit_targets)
    all_state_targets = np.concatenate(all_state_targets)
    
    # 计算准确率
    fruit_accuracy = (all_fruit_preds == all_fruit_targets).mean()
    state_accuracy = (all_state_preds == all_state_targets).mean()
    
    # 返回结果字典
    results = {
        'fruit_preds': all_fruit_preds,
        'fruit_targets': all_fruit_targets,
        'state_preds': all_state_preds,
        'state_targets': all_state_targets,
        'fruit_accuracy': fruit_accuracy,
        'state_accuracy': state_accuracy
    }
    
    # 如果收集了错误样本，添加到结果中
    if collect_errors:
        results['error_samples'] = error_samples
    
    return results


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
                       state_class_names: List[str], save_dir: str,
                       visualize_errors: bool = False, top_k_errors: int = 10) -> Dict:
    """
    评估模型并生成报告
    
    Args:
        model (torch.nn.Module): 待评估的模型
        data_loader (DataLoader): 数据加载器
        device (torch.device): 计算设备
        fruit_class_names (List[str]): 水果类型名称
        state_class_names (List[str]): 腐烂状态名称
        save_dir (str): 保存目录
        visualize_errors (bool, optional): 是否可视化错误预测的样本
        top_k_errors (int, optional): 可视化前K个错误预测的样本
        
    Returns:
        Dict: 评估结果字典
    """
    # 评估模型
    results = evaluate_model(model, data_loader, device, collect_errors=visualize_errors)
    
    # 生成水果类型分类报告
    fruit_report = generate_classification_report(
        results['fruit_targets'], results['fruit_preds'], fruit_class_names, "水果类型", save_dir
    )
    
    # 生成腐烂状态分类报告
    state_report = generate_classification_report(
        results['state_targets'], results['state_preds'], state_class_names, "腐烂状态", save_dir
    )
    
    # 打印总体准确率
    print(f"水果类型准确率: {results['fruit_accuracy']:.4f}")
    print(f"腐烂状态准确率: {results['state_accuracy']:.4f}")
    
    # 添加分类报告到结果中
    results['fruit_report'] = fruit_report
    results['state_report'] = state_report
    
    # 如果需要可视化错误样本
    if visualize_errors and 'error_samples' in results:
        class_names = (fruit_class_names, state_class_names)
        visualize_error_samples(results, class_names, save_dir, top_k_errors)
    
    return results


def visualize_error_samples(results: Dict[str, Any], class_names: Tuple[List[str], List[str]], 
                          output_dir: str, top_k: int = 10) -> None:
    """
    可视化错误预测的样本
    
    Args:
        results (Dict[str, Any]): 评估结果
        class_names (Tuple[List[str], List[str]]): 类别名称
        output_dir (str): 输出目录
        top_k (int): 可视化前K个错误预测的样本
    """
    if 'error_samples' not in results or not results['error_samples']:
        print("没有错误预测的样本可供可视化")
        return
    
    # 解包类别名称
    fruit_class_names, state_class_names = class_names
    
    # 创建输出目录
    error_dir = os.path.join(output_dir, 'error_samples')
    os.makedirs(error_dir, exist_ok=True)
    
    # 按照错误类型对样本进行分组
    fruit_errors = [s for s in results['error_samples'] if s['fruit_pred'] != s['fruit_target']]
    state_errors = [s for s in results['error_samples'] if s['state_pred'] != s['state_target']]
    both_errors = [s for s in results['error_samples'] if s['fruit_pred'] != s['fruit_target'] and s['state_pred'] != s['state_target']]
    
    # 打印错误统计信息
    print(f"总样本数: {len(results['fruit_targets'])}")
    print(f"水果类型错误数: {len(fruit_errors)}")
    print(f"腐烂状态错误数: {len(state_errors)}")
    print(f"两者都错误数: {len(both_errors)}")
    
    # 可视化前K个错误样本
    error_samples = results['error_samples'][:min(top_k, len(results['error_samples']))]
    
    # 创建图表
    n_samples = len(error_samples)
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    
    for i, sample in enumerate(error_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # 显示图像
        img = sample['image'].permute(1, 2, 0).numpy()
        # 如果图像是归一化的，需要反归一化
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        
        # 获取预测和真实标签
        fruit_pred = fruit_class_names[sample['fruit_pred']]
        fruit_target = fruit_class_names[sample['fruit_target']]
        state_pred = state_class_names[sample['state_pred']]
        state_target = state_class_names[sample['state_target']]
        
        # 设置标题
        title = f"预测: {fruit_pred} ({state_pred})\n真实: {fruit_target} ({state_target})"
        plt.title(title, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, 'error_samples.png'))
    plt.close()
    
    # 生成详细的错误分析报告
    report_path = os.path.join(error_dir, 'error_analysis.txt')
    with open(report_path, 'w') as f:
        f.write("错误预测样本分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总样本数: {len(results['fruit_targets'])}\n")
        f.write(f"水果类型错误数: {len(fruit_errors)} ({len(fruit_errors)/len(results['fruit_targets']):.2%})\n")
        f.write(f"腐烂状态错误数: {len(state_errors)} ({len(state_errors)/len(results['fruit_targets']):.2%})\n")
        f.write(f"两者都错误数: {len(both_errors)} ({len(both_errors)/len(results['fruit_targets']):.2%})\n\n")
        
        # 分析水果类型错误
        f.write("水果类型错误分析:\n")
        f.write("-" * 30 + "\n")
        fruit_confusion = {}
        for e in fruit_errors:
            key = (fruit_class_names[e['fruit_target']], fruit_class_names[e['fruit_pred']])
            fruit_confusion[key] = fruit_confusion.get(key, 0) + 1
        
        for (true, pred), count in sorted(fruit_confusion.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {true} 被错误分类为 {pred}: {count} 次\n")
        
        # 分析腐烂状态错误
        f.write("\n腐烂状态错误分析:\n")
        f.write("-" * 30 + "\n")
        state_confusion = {}
        for e in state_errors:
            key = (state_class_names[e['state_target']], state_class_names[e['state_pred']])
            state_confusion[key] = state_confusion.get(key, 0) + 1
        
        for (true, pred), count in sorted(state_confusion.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {true} 被错误分类为 {pred}: {count} 次\n")


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='模型评估脚本')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径，如果为None则自动寻找最佳检查点')
    parser.add_argument('--output', type=str, default='evaluation_results', help='评估结果输出目录')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小，如果为None则使用配置文件中的设置')
    parser.add_argument('--visualize_errors', action='store_true', help='是否可视化错误预测的样本')
    parser.add_argument('--top_k_errors', type=int, default=10, help='可视化前K个错误预测的样本')
    
    return parser.parse_args()


def load_model(config: Dict[str, Any], data_loader: DataLoader = None, checkpoint_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    加载模型和检查点
    
    Args:
        config (Dict[str, Any]): 配置字典
        data_loader (DataLoader, optional): 数据加载器，用于获取类别数量
        checkpoint_path (str, optional): 检查点路径，如果为None则自动寻找最佳检查点
        
    Returns:
        Tuple[nn.Module, Dict[str, Any]]: 模型和检查点信息
    """
    # 获取水果类型和腐烂状态的数量
    if data_loader is not None:
        # 从数据集获取类别数量
        dataset = data_loader.dataset
        # 如果是Subset对象，需要访问.dataset属性
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        
        # 获取类别数量
        num_fruit_classes = len(dataset.data_frame['fruit_type'].unique())
        num_state_classes = len(dataset.data_frame['state'].unique())
    else:
        # 如果没有提供数据加载器，使用默认值或从配置中获取
        num_fruit_classes = 5  # 默认值，根据实际情况调整
        num_state_classes = 2  # 默认值，根据实际情况调整
    
    # 获取骨干网络类型和预训练设置
    backbone = config['model'].get('backbone', 'resnet50')
    pretrained = config['model'].get('pretrained', True)
    
    # 创建模型
    model = create_model(num_fruit_classes, num_state_classes, backbone, pretrained)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 如果未指定检查点路径，寻找最佳检查点
    if checkpoint_path is None:
        save_dir = config['train'].get('save_dir', os.path.join(project_root, 'checkpoints'))
        checkpoint_path = find_best_checkpoint(save_dir)
        
        if checkpoint_path is None:
            raise ValueError("未找到有效的检查点，请先训练模型或指定检查点路径")
    
    # 加载检查点
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def get_class_names(dataset: FruitDataset) -> Tuple[List[str], List[str]]:
    """
    获取类别名称
    
    Args:
        dataset (FruitDataset): 数据集
        
    Returns:
        Tuple[List[str], List[str]]: 水果类型名称列表和腐烂状态名称列表
    """
    # 获取水果类型名称
    fruit_types = dataset.data_frame['fruit_type'].unique()
    fruit_type_names = sorted(fruit_types)
    
    # 获取腐烂状态名称
    state_types = dataset.data_frame['state'].unique()
    state_type_names = sorted(state_types)
    
    return fruit_type_names, state_type_names


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    if args.config is None:
        config_path = os.path.join(project_root, 'config', 'config.yaml')
    else:
        config_path = args.config
    
    config = load_config(config_path)
    
    # 设置输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据加载器
    # 如果命令行指定了batch_size，则临时修改配置中的值
    if args.batch_size:
        config['device']['batch_size'] = args.batch_size
    
    # 获取数据加载器
    _, test_loader, class_info = get_data_loaders(config)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, checkpoint = load_model(config, test_loader, args.checkpoint)
    
    # 获取类别名称
    dataset = test_loader.dataset
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    fruit_class_names, state_class_names = get_class_names(dataset)
    
    # 评估模型并生成报告
    results = evaluate_and_report(
        model, test_loader, device,
        fruit_class_names, state_class_names, output_dir,
        visualize_errors=args.visualize_errors,
        top_k_errors=args.top_k_errors
    )
    
    # 保存评估结果摘要
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    summary = {
        'fruit_accuracy': float(results['fruit_accuracy']),
        'state_accuracy': float(results['state_accuracy']),
        'checkpoint': checkpoint.get('epoch', 'unknown'),
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"评估完成，结果已保存到 {output_dir}")


if __name__ == '__main__':
    main()
