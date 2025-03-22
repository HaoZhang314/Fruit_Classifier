#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估脚本

这个脚本用于评估模型在测试集上的性能，包括：
1. 加载最佳模型
2. 对测试集进行预测
3. 计算各种性能指标
4. 生成混淆矩阵和分类报告
5. 可视化错误预测的样本

使用方法：
    python evaluate_model.py [--config CONFIG] [--checkpoint CHECKPOINT] [--output OUTPUT]
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
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目其他模块
from models.model import create_model, FruitClassifier
from data.loaders import get_data_loaders, FruitDataset, get_transforms
from config import load_config
from trains.checkpoint import find_best_checkpoint


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


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        model (nn.Module): 模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 设备
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    model.eval()
    
    # 收集预测和目标
    all_fruit_preds = []
    all_state_preds = []
    all_fruit_targets = []
    all_state_targets = []
    
    # 收集错误预测的样本
    error_samples = []
    
    with torch.no_grad():
        for images, fruit_targets, state_targets in tqdm(test_loader, desc="评估模型"):
            # 我们将在这里不获取图像路径，因为很难准确地知道每个批次对应的数据集索引
            # 直接使用图像内容来存储错误样本
            # 如果需要图像路径，可以修改FruitDataset类来返回路径
            
            # 将数据移动到设备
            images = images.to(device)
            fruit_targets = fruit_targets.to(device)
            state_targets = state_targets.to(device)
            
            # 前向传播
            fruit_logits, state_logits = model(images)
            
            # 计算预测结果
            fruit_preds = torch.argmax(fruit_logits, dim=1)
            state_preds = torch.argmax(state_logits, dim=1)
            
            # 收集预测和目标
            all_fruit_preds.append(fruit_preds.cpu().numpy())
            all_state_preds.append(state_preds.cpu().numpy())
            all_fruit_targets.append(fruit_targets.cpu().numpy())
            all_state_targets.append(state_targets.cpu().numpy())
            
            # 收集错误预测的样本
            for i in range(len(images)):
                if fruit_preds[i] != fruit_targets[i] or state_preds[i] != state_targets[i]:
                    error_samples.append({
                        'image': images[i].cpu(),
                        'image_index': len(error_samples),  # 使用错误样本的索引来标识
                        'fruit_pred': fruit_preds[i].item(),
                        'fruit_target': fruit_targets[i].item(),
                        'state_pred': state_preds[i].item(),
                        'state_target': state_targets[i].item(),
                        'fruit_prob': torch.softmax(fruit_logits[i], dim=0).cpu().numpy(),
                        'state_prob': torch.softmax(state_logits[i], dim=0).cpu().numpy()
                    })
    
    # 合并所有批次的预测和目标
    all_fruit_preds = np.concatenate(all_fruit_preds)
    all_state_preds = np.concatenate(all_state_preds)
    all_fruit_targets = np.concatenate(all_fruit_targets)
    all_state_targets = np.concatenate(all_state_targets)
    
    # 计算准确率
    fruit_acc = accuracy_score(all_fruit_targets, all_fruit_preds)
    state_acc = accuracy_score(all_state_targets, all_state_preds)
    
    # 计算总体准确率 - 同时正确预测水果类型和腐烂状态的比例
    # 手动计算，而不使用accuracy_score函数
    correct_both = sum((all_fruit_targets[i] == all_fruit_preds[i]) and 
                       (all_state_targets[i] == all_state_preds[i])
                      for i in range(len(all_fruit_targets)))
    overall_acc = correct_both / len(all_fruit_targets)
    
    # 计算精确率、召回率和F1分数
    fruit_precision, fruit_recall, fruit_f1, _ = precision_recall_fscore_support(
        all_fruit_targets, all_fruit_preds, average='weighted'
    )
    state_precision, state_recall, state_f1, _ = precision_recall_fscore_support(
        all_state_targets, all_state_preds, average='weighted'
    )
    
    # 返回评估结果
    return {
        'fruit_acc': fruit_acc,
        'state_acc': state_acc,
        'overall_acc': overall_acc,
        'fruit_precision': fruit_precision,
        'fruit_recall': fruit_recall,
        'fruit_f1': fruit_f1,
        'state_precision': state_precision,
        'state_recall': state_recall,
        'state_f1': state_f1,
        'all_fruit_preds': all_fruit_preds,
        'all_state_preds': all_state_preds,
        'all_fruit_targets': all_fruit_targets,
        'all_state_targets': all_state_targets,
        'error_samples': error_samples
    }


def generate_confusion_matrices(results: Dict[str, Any], class_names: Tuple[List[str], List[str]], output_dir: str) -> None:
    """
    生成混淆矩阵
    
    Args:
        results (Dict[str, Any]): 评估结果
        class_names (Tuple[List[str], List[str]]): 类别名称
        output_dir (str): 输出目录
    """
    fruit_type_names, state_type_names = class_names
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成水果类型混淆矩阵
    plt.figure(figsize=(10, 8))
    fruit_cm = confusion_matrix(results['all_fruit_targets'], results['all_fruit_preds'])
    sns.heatmap(fruit_cm, annot=True, fmt='d', cmap='Blues', xticklabels=fruit_type_names, yticklabels=fruit_type_names)
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('水果类型混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fruit_confusion_matrix.png'))
    plt.close()
    
    # 生成腐烂状态混淆矩阵
    plt.figure(figsize=(8, 6))
    state_cm = confusion_matrix(results['all_state_targets'], results['all_state_preds'])
    sns.heatmap(state_cm, annot=True, fmt='d', cmap='Blues', xticklabels=state_type_names, yticklabels=state_type_names)
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('腐烂状态混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'state_confusion_matrix.png'))
    plt.close()


def generate_classification_reports(results: Dict[str, Any], class_names: Tuple[List[str], List[str]], output_dir: str) -> None:
    """
    生成分类报告
    
    Args:
        results (Dict[str, Any]): 评估结果
        class_names (Tuple[List[str], List[str]]): 类别名称
        output_dir (str): 输出目录
    """
    fruit_type_names, state_type_names = class_names
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成水果类型分类报告
    fruit_report = classification_report(
        results['all_fruit_targets'], results['all_fruit_preds'],
        target_names=fruit_type_names, output_dict=True
    )
    
    # 生成腐烂状态分类报告
    state_report = classification_report(
        results['all_state_targets'], results['all_state_preds'],
        target_names=state_type_names, output_dict=True
    )
    
    # 保存分类报告
    with open(os.path.join(output_dir, 'classification_reports.json'), 'w') as f:
        json.dump({
            'fruit_report': fruit_report,
            'state_report': state_report
        }, f, indent=4)
    
    # 生成分类报告表格
    fruit_df = pd.DataFrame(fruit_report).transpose()
    state_df = pd.DataFrame(state_report).transpose()
    
    # 保存为CSV
    fruit_df.to_csv(os.path.join(output_dir, 'fruit_classification_report.csv'))
    state_df.to_csv(os.path.join(output_dir, 'state_classification_report.csv'))


def visualize_error_samples(results: Dict[str, Any], class_names: Tuple[List[str], List[str]], output_dir: str, top_k: int = 10) -> None:
    """
    可视化错误预测的样本
    
    Args:
        results (Dict[str, Any]): 评估结果
        class_names (Tuple[List[str], List[str]]): 类别名称
        output_dir (str): 输出目录
        top_k (int): 可视化前K个错误预测的样本
    """
    fruit_type_names, state_type_names = class_names
    error_samples = results['error_samples']
    
    # 创建输出目录
    error_dir = os.path.join(output_dir, 'error_samples')
    os.makedirs(error_dir, exist_ok=True)
    
    # 按照预测概率的差异对错误样本进行排序
    for sample in error_samples:
        fruit_prob_diff = sample['fruit_prob'][sample['fruit_target']] - sample['fruit_prob'][sample['fruit_pred']]
        state_prob_diff = sample['state_prob'][sample['state_target']] - sample['state_prob'][sample['state_pred']]
        sample['prob_diff'] = fruit_prob_diff + state_prob_diff
    
    error_samples.sort(key=lambda x: x['prob_diff'])
    
    # 可视化前K个错误预测的样本
    k = min(top_k, len(error_samples))
    if k == 0:
        print("没有错误预测的样本！")
        return
    
    # 创建图表
    fig, axs = plt.subplots(k, 1, figsize=(12, 5 * k))
    if k == 1:
        axs = [axs]
    
    for i in range(k):
        sample = error_samples[i]
        
        # 获取图像
        image = sample['image'].permute(1, 2, 0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # 显示图像
        axs[i].imshow(image)
        
        # 设置标题
        title = f"错误样本 #{sample['image_index'] + 1}\n"
        title += f"水果类型: 真实={fruit_type_names[sample['fruit_target']]} ({sample['fruit_prob'][sample['fruit_target']]:.2f}), "
        title += f"预测={fruit_type_names[sample['fruit_pred']]} ({sample['fruit_prob'][sample['fruit_pred']]:.2f})\n"
        title += f"腐烂状态: 真实={state_type_names[sample['state_target']]} ({sample['state_prob'][sample['state_target']]:.2f}), "
        title += f"预测={state_type_names[sample['state_pred']]} ({sample['state_prob'][sample['state_pred']]:.2f})"
        axs[i].set_title(title)
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, f'top_{k}_error_samples.png'))
    plt.close()
    
    # 保存错误预测的详细信息
    error_df = pd.DataFrame([
        {
            'error_sample_id': sample['image_index'] + 1,
            'fruit_target': fruit_type_names[sample['fruit_target']],
            'fruit_pred': fruit_type_names[sample['fruit_pred']],
            'fruit_target_prob': sample['fruit_prob'][sample['fruit_target']],
            'fruit_pred_prob': sample['fruit_prob'][sample['fruit_pred']],
            'state_target': state_type_names[sample['state_target']],
            'state_pred': state_type_names[sample['state_pred']],
            'state_target_prob': sample['state_prob'][sample['state_target']],
            'state_pred_prob': sample['state_prob'][sample['state_pred']]
        }
        for sample in error_samples
    ])
    
    error_df.to_csv(os.path.join(error_dir, 'error_samples.csv'), index=False)


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(project_root, 'config', 'config.yaml')
    
    config = load_config(config_path)
    
    # 设置批次大小
    if args.batch_size is not None:
        config['device']['batch_size'] = args.batch_size
    
    # 加载数据
    train_loader, test_loader, class_info = get_data_loaders(config)
    
    # 加载模型
    model, checkpoint = load_model(config, test_loader, args.checkpoint)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取类别名称
    class_names = class_info
    
    # 评估模型
    print("开始评估模型...")
    results = evaluate_model(model, test_loader, device)
    
    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成评估报告
    print("生成评估报告...")
    
    # 保存评估指标
    metrics = {
        'fruit_accuracy': results['fruit_acc'],
        'state_accuracy': results['state_acc'],
        'overall_accuracy': results['overall_acc'],
        'fruit_precision': results['fruit_precision'],
        'fruit_recall': results['fruit_recall'],
        'fruit_f1': results['fruit_f1'],
        'state_precision': results['state_precision'],
        'state_recall': results['state_recall'],
        'state_f1': results['state_f1']
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 生成混淆矩阵
    print("生成混淆矩阵...")
    generate_confusion_matrices(results, class_names, output_dir)
    
    # 生成分类报告
    print("生成分类报告...")
    generate_classification_reports(results, class_names, output_dir)
    
    # 可视化错误预测的样本
    if args.visualize_errors and len(results['error_samples']) > 0:
        print("可视化错误预测的样本...")
        visualize_error_samples(results, class_names, output_dir, args.top_k_errors)
    
    # 打印评估结果
    print("\n===== 评估结果 =====")
    print(f"水果类型准确率: {results['fruit_acc']:.4f}")
    print(f"腐烂状态准确率: {results['state_acc']:.4f}")
    print(f"总体准确率: {results['overall_acc']:.4f}")
    print(f"水果类型精确率: {results['fruit_precision']:.4f}")
    print(f"水果类型召回率: {results['fruit_recall']:.4f}")
    print(f"水果类型F1分数: {results['fruit_f1']:.4f}")
    print(f"腐烂状态精确率: {results['state_precision']:.4f}")
    print(f"腐烂状态召回率: {results['state_recall']:.4f}")
    print(f"腐烂状态F1分数: {results['state_f1']:.4f}")
    print(f"错误预测样本数: {len(results['error_samples'])}")
    print(f"总样本数: {len(results['all_fruit_targets'])}")
    print(f"错误率: {len(results['error_samples']) / len(results['all_fruit_targets']):.4f}")
    print("=====================")
    
    print(f"\n评估结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
