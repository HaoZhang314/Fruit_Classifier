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

# 导入必要的库
# 系统和文件操作相关库
import os                  # 操作系统接口，用于文件路径处理和目录创建
import sys                 # 系统相关功能，用于修改Python路径
import argparse            # 命令行参数解析工具
import json                # JSON数据处理，用于保存评估结果

# 数据处理和科学计算库
import numpy as np         # 科学计算库，提供高效的数组操作
import pandas as pd        # 数据分析库，用于数据处理和分析
from tqdm import tqdm      # 进度条库，用于显示评估进度
from pathlib import Path   # 面向对象的文件系统路径处理

# 可视化相关库
import matplotlib.pyplot as plt  # 绘图库，用于创建混淆矩阵和错误样本可视化
import seaborn as sns            # 基于matplotlib的高级可视化库，提供更美观的图表

# 机器学习评估指标库
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# PyTorch相关库
import torch               # PyTorch深度学习库
import torch.nn as nn      # PyTorch神经网络模块
from torch.utils.data import DataLoader  # 数据加载器，用于批量加载数据

# 类型注解库
from typing import Dict, Tuple, List, Optional, Union, Any  # 类型提示，增强代码可读性和IDE支持

# 添加项目根目录到Python路径
# 这样可以使用绝对导入，避免相对导入的复杂性
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
project_root = os.path.dirname(current_dir)               # 获取项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)                         # 将项目根目录添加到Python路径

# 导入项目其他模块
# 这些是项目内部的自定义模块，用于模型创建、数据加载和配置管理
from models.model import create_model, FruitClassifier    # 模型定义和创建函数
from data.loaders import get_data_loaders, FruitDataset, get_transforms  # 数据加载和处理
from config import load_config                            # 配置加载函数
from trains.checkpoint import find_best_checkpoint        # 检查点管理函数


def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, 
                  device: torch.device, collect_errors: bool = False) -> Dict[str, Any]:
    """
    评估模型性能，计算多项评估指标包括准确率、精确率、召回率和F1分数
    
    该函数对模型在给定数据集上的性能进行全面评估，主要步骤包括：
    1. 使用模型对测试数据进行预测
    2. 收集所有预测结果和真实标签
    3. 计算水果类型和腐烂状态分类的各项评估指标
    4. 如果需要，收集错误预测的样本用于后续分析
    
    评估指标包括：
    - 准确率(Accuracy)：正确预测的样本比例
    - 精确率(Precision)：在预测为正类的样本中，真正为正类的比例
    - 召回率(Recall)：在所有真正为正类的样本中，被正确预测为正类的比例
    - F1分数：精确率和召回率的调和平均值，综合考虑两者
    
    Args:
        model (torch.nn.Module): 待评估的模型，通常是FruitClassifier实例
        data_loader (DataLoader): 数据加载器，提供测试数据的批次迭代
        device (torch.device): 计算设备，如CPU或GPU
        collect_errors (bool, optional): 是否收集错误预测的样本，用于后续错误分析和可视化
        
    Returns:
        Dict[str, Any]: 评估结果字典，包含以下内容：
            - 预测结果和真实标签
            - 水果类型和腐烂状态的各项评估指标（准确率、精确率、召回率、F1分数）
            - 错误预测的样本（如果collect_errors=True）
    """
    # 设置模型为评估模式，禁用dropout和批归一化的训练行为
    model.eval()
    
    # 初始化列表，用于收集所有批次的预测结果和真实标签
    all_fruit_preds = []    # 水果类型预测结果列表
    all_state_preds = []    # 腐烂状态预测结果列表
    all_fruit_targets = []  # 水果类型真实标签列表
    all_state_targets = []  # 腐烂状态真实标签列表
    
    # 收集错误预测的样本
    error_samples = []      # 错误预测样本列表，用于后续分析
    
    # 使用torch.no_grad()上下文管理器禁用梯度计算，减少内存使用并加速推理
    with torch.no_grad():
        # 遍历数据加载器中的每个批次，使用tqdm显示进度条
        for images, fruit_targets, state_targets in tqdm(data_loader, desc="评估模型"):
            # 将数据移动到指定设备(CPU/GPU)
            images = images.to(device)                      # 图像数据
            fruit_targets_device = fruit_targets.to(device) # 水果类型标签
            state_targets_device = state_targets.to(device) # 腐烂状态标签
            
            # 前向传播 - 将图像输入模型获取预测结果
            # 模型输出两组logits，分别对应水果类型和腐烂状态
            fruit_logits, state_logits = model(images)
            
            # 计算预测结果 - 对logits应用argmax获取预测的类别索引
            # dim=1表示在第二个维度（类别维度）上取最大值的索引
            fruit_preds = torch.argmax(fruit_logits, dim=1)  # 水果类型预测
            state_preds = torch.argmax(state_logits, dim=1)  # 腐烂状态预测
            
            # 收集预测和目标 - 将当前批次的结果添加到列表中
            # 注意将GPU张量转移到CPU并转换为NumPy数组，以便后续处理
            all_fruit_preds.append(fruit_preds.cpu().numpy())      # 收集水果类型预测
            all_state_preds.append(state_preds.cpu().numpy())      # 收集腐烂状态预测
            all_fruit_targets.append(fruit_targets.numpy())        # 收集水果类型真实标签
            all_state_targets.append(state_targets.numpy())        # 收集腐烂状态真实标签
            
            # 收集错误预测的样本
            # 当collect_errors=True时，该部分代码负责识别并收集模型预测错误的样本，以便后续进行可视化和分析
            if collect_errors:
                # 遍历当前批次中的每个图像样本
                for i in range(len(images)):
                    # 判断条件：如果水果类型预测错误或者腐烂状态预测错误（或两者都错误）
                    if fruit_preds[i] != fruit_targets_device[i] or state_preds[i] != state_targets_device[i]:
                        # 收集错误样本的详细信息
                        error_samples.append({
                            'image': images[i].cpu(),  # 将图像从GPU转移到CPU，以便后续处理和可视化
                            'image_index': len(error_samples),  # 使用error_samples列表的长度作为错误样本的索引
                            'fruit_pred': fruit_preds[i].item(),  # 模型预测的水果类型（转换为Python标量）
                            'fruit_target': fruit_targets_device[i].item(),  # 真实的水果类型
                            'state_pred': state_preds[i].item(),  # 模型预测的腐烂状态
                            'state_target': state_targets_device[i].item(),  # 真实的腐烂状态
                            'fruit_prob': torch.softmax(fruit_logits[i], dim=0).cpu().numpy(),  # 水果类型的预测概率分布
                            'state_prob': torch.softmax(state_logits[i], dim=0).cpu().numpy()  # 腐烂状态的预测概率分布
                        })
    
    # 合并所有批次的预测和目标数组
    # np.concatenate将列表中的所有数组沿着第一个维度（样本维度）连接起来
    all_fruit_preds = np.concatenate(all_fruit_preds)       # 合并所有批次的水果类型预测
    all_state_preds = np.concatenate(all_state_preds)       # 合并所有批次的腐烂状态预测
    all_fruit_targets = np.concatenate(all_fruit_targets)   # 合并所有批次的水果类型真实标签
    all_state_targets = np.concatenate(all_state_targets)   # 合并所有批次的腐烂状态真实标签
    
    # 计算准确率 (Accuracy) - 正确预测的样本比例
    # 通过比较预测值和真实值是否相等，然后计算平均值得到准确率
    # 这是最基本的评估指标，表示模型预测正确的样本占总样本的比例
    fruit_accuracy = (all_fruit_preds == all_fruit_targets).mean()  # 水果类型准确率
    state_accuracy = (all_state_preds == all_state_targets).mean()  # 腐烂状态准确率
    
    # 计算精确率 (Precision)、召回率 (Recall) 和 F1 分数
    # 使用sklearn的precision_recall_fscore_support函数计算这些指标
    # precision: 在预测为正类的样本中，真正为正类的比例，评估预测的可靠性
    # recall: 在所有真正为正类的样本中，被正确预测为正类的比例，评估模型的完整性
    # f1: 精确率和召回率的调和平均值(2*P*R/(P+R))，综合考虑两者，平衡评估模型性能
    # average='weighted': 考虑类别不平衡问题，按照各类别样本数量加权平均
    fruit_precision, fruit_recall, fruit_f1, _ = precision_recall_fscore_support(
        all_fruit_targets, all_fruit_preds, average='weighted')  # 水果类型的精确率、召回率和F1
    state_precision, state_recall, state_f1, _ = precision_recall_fscore_support(
        all_state_targets, all_state_preds, average='weighted')  # 腐烂状态的精确率、召回率和F1
    
    # 构建并返回结果字典 - 包含所有预测结果和评估指标
    results = {
        # 原始预测结果和目标值 - 用于后续分析和可视化
        'fruit_preds': all_fruit_preds,      # 水果类型预测结果数组
        'fruit_targets': all_fruit_targets,  # 水果类型真实标签数组
        'state_preds': all_state_preds,      # 腐烂状态预测结果数组
        'state_targets': all_state_targets,  # 腐烂状态真实标签数组
        
        # 水果类型评估指标 - 全面评估水果分类性能
        'fruit_accuracy': fruit_accuracy,    # 准确率 - 所有样本中预测正确的比例
        'fruit_precision': fruit_precision,  # 精确率 - 预测为某类的样本中真正属于该类的比例
        'fruit_recall': fruit_recall,        # 召回率 - 某类样本中被正确预测的比例
        'fruit_f1': fruit_f1,                # F1分数 - 精确率和召回率的调和平均值
        
        # 腐烂状态评估指标 - 全面评估腐烂检测性能
        'state_accuracy': state_accuracy,    # 准确率
        'state_precision': state_precision,  # 精确率
        'state_recall': state_recall,        # 召回率
        'state_f1': state_f1                 # F1分数
    }
    
    # 如果收集了错误样本，将其添加到结果字典中
    if collect_errors:
        results['error_samples'] = error_samples  # 错误样本列表，包含图像和详细信息
    
    return results  # 返回包含所有评估结果的字典


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], title: str, 
                         save_path: str) -> None:
    """
    绘制混淆矩阵并保存为图像文件
    
    混淆矩阵是评估分类模型性能的重要工具，它展示了模型预测结果与真实标签之间的关系。
    矩阵的每一行代表一个真实类别，每一列代表一个预测类别。
    矩阵中的每个单元格(i,j)表示真实类别为i但被预测为类别j的样本数量。
    
    该函数完成以下任务：
    1. 使用sklearn的confusion_matrix函数计算原始混淆矩阵
    2. 对混淆矩阵进行归一化处理，使每行（真实类别）的总和为1，便于比较不同类别的错误模式
    3. 使用matplotlib创建可视化图表，包括颜色映射、标题和坐标轴标签
    4. 在每个单元格中显示原始样本数和归一化后的比例
    5. 将生成的混淆矩阵图保存到指定路径
    
    混淆矩阵的分析可以帮助我们：
    - 识别模型容易混淆的类别对
    - 发现特定类别的识别问题
    - 评估模型在各个类别上的表现平衡性
    - 指导后续的模型改进方向
    
    Args:
        y_true (np.ndarray): 真实标签数组，每个元素代表一个样本的真实类别
        y_pred (np.ndarray): 预测标签数组，每个元素代表一个样本的预测类别
        class_names (List[str]): 类别名称列表，用于标记混淆矩阵的行和列
        title (str): 图表标题，通常包含任务名称，如"水果类型混淆矩阵"
        save_path (str): 混淆矩阵图像的保存路径，通常为PNG格式
    """
    # 计算混淆矩阵 - 使用sklearn的confusion_matrix函数
    # 混淆矩阵是一个二维数组，其中cm[i,j]表示真实类别为i但被预测为类别j的样本数量
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化混淆矩阵 - 使每行（真实类别）的总和为1
    # 这样可以更好地比较不同类别的错误模式，特别是在类别不平衡的情况下
    # 归一化后的值表示某个真实类别被预测为各个类别的概率分布
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建图表 - 设置图表大小和基本属性
    plt.figure(figsize=(10, 8))  # 设置图表大小为10x8英寸
    # 使用imshow函数显示混淆矩阵，使用蓝色色谱(Blues)，颜色深浅表示数值大小
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)  # 设置图表标题和字体大小
    plt.colorbar()  # 添加颜色条，显示颜色与数值的对应关系
    
    # 设置刻度标签 - 使用类别名称标记坐标轴
    tick_marks = np.arange(len(class_names))  # 创建刻度位置
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=10)  # 设置x轴刻度标签，旋转45度
    plt.yticks(tick_marks, class_names, fontsize=10)  # 设置y轴刻度标签
    
    # 添加文本注释 - 在每个单元格中显示原始样本数和归一化后的比例
    fmt = '.2f'  # 设置浮点数格式，保留两位小数
    thresh = cm_normalized.max() / 2.  # 设置阈值，用于决定文本颜色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # 在每个单元格中添加文本，显示原始样本数和归一化后的比例
            # 如果单元格颜色较深，使用白色文本；如果单元格颜色较浅，使用黑色文本
            plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:{fmt}})",
                     horizontalalignment="center",  # 水平居中对齐
                     color="white" if cm_normalized[i, j] > thresh else "black",  # 根据背景颜色选择文本颜色
                     fontsize=8)  # 设置字体大小
    
    # 设置坐标轴标签
    plt.ylabel('真实标签', fontsize=12)  # 设置y轴标签和字体大小
    plt.xlabel('预测标签', fontsize=12)  # 设置x轴标签和字体大小
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域
    
    # 保存图表 - 将混淆矩阵图保存到指定路径
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存目录存在
    plt.savefig(save_path)  # 保存图表
    plt.close()  # 关闭图表，释放资源


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
    # 生成分类报告 - 使用sklearn的classification_report函数
    # 该报告包含每个类别的精确率、召回率、F1分数和支持度（样本数）
    # output_dict=True表示返回字典格式的报告，便于后续处理
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 保存报告为文本文件 - 便于人工查看和分析
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    report_path = os.path.join(save_dir, f"{task_name}_classification_report.txt")  # 构建报告文件路径
    
    # 将分类报告写入文本文件
    with open(report_path, 'w') as f:
        f.write(f"{task_name} 分类报告\n")  # 写入标题
        f.write("="*50 + "\n")  # 写入分隔线
        # 写入格式化的分类报告，包含每个类别的详细指标
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    # 绘制混淆矩阵 - 调用之前定义的plot_confusion_matrix函数
    cm_path = os.path.join(save_dir, f"{task_name}_confusion_matrix.png")  # 构建混淆矩阵图像路径
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_true, y_pred, class_names, f"{task_name} 混淆矩阵", cm_path)
    
    # 返回分类报告字典 - 包含每个类别的详细指标
    return report

def evaluate_and_report(model: torch.nn.Module, data_loader: DataLoader, 
                       device: torch.device, fruit_class_names: List[str], 
                       state_class_names: List[str], save_dir: str,
                       visualize_errors: bool = False, top_k_errors: int = 10) -> Dict:
    """
    评估模型并生成详细的评估报告，包括多项指标和可视化结果
    
    该函数是模型评估的高级封装，完成以下任务：
    1. 调用evaluate_model函数进行模型评估，获取各项指标
    2. 生成水果类型和腐烂状态的分类报告，包括混淆矩阵
    3. 打印并输出全面的评估指标，包括准确率、精确率、召回率和F1分数
    4. 可选地可视化错误预测的样本，帮助分析模型的失败模式
    
    该函数特别关注计算和输出以下指标：
    - 准确率(Accuracy)：所有样本中预测正确的比例
    - 精确率(Precision)：评估模型预测的可靠性，预测为正类中真正为正类的比例
    - 召回率(Recall)：评估模型的完整性，所有真正为正类的样本中被正确预测的比例
    - F1分数：精确率和召回率的调和平均值，在类别不平衡情况下特别有用
    
    Args:
        model (torch.nn.Module): 待评估的模型，通常是FruitClassifier实例
        data_loader (DataLoader): 数据加载器，提供测试数据
        device (torch.device): 计算设备，如CPU或GPU
        fruit_class_names (List[str]): 水果类型名称列表，用于标记混淆矩阵和报告
        state_class_names (List[str]): 腐烂状态名称列表，用于标记混淆矩阵和报告
        save_dir (str): 评估结果和可视化图表的保存目录
        visualize_errors (bool, optional): 是否可视化错误预测的样本，默认为False
        top_k_errors (int, optional): 当visualize_errors=True时，可视化前K个错误预测的样本，默认为10
        
    Returns:
        Dict: 评估结果字典，包含预测结果、真实标签、各项评估指标和分类报告
    """
    # 调用evaluate_model函数评估模型性能
    # collect_errors参数决定是否收集错误样本信息，用于后续可视化
    results = evaluate_model(model, data_loader, device, collect_errors=visualize_errors)
    
    # 为水果类型生成分类报告和混淆矩阵
    # 传入真实标签、预测标签、类别名称列表、任务名称和保存目录
    fruit_report = generate_classification_report(
        results['fruit_targets'],        # 水果类型的真实标签
        results['fruit_preds'],          # 水果类型的预测标签
        fruit_class_names,               # 水果类型名称列表
        "水果类型",                       # 任务名称
        save_dir                         # 保存目录
    )
    
    # 为腐烂状态生成分类报告和混淆矩阵
    # 与上面类似，但使用腐烂状态相关的数据
    state_report = generate_classification_report(
        results['state_targets'],        # 腐烂状态的真实标签
        results['state_preds'],          # 腐烂状态的预测标签
        state_class_names,               # 腐烂状态名称列表
        "腐烂状态",                       # 任务名称
        save_dir                         # 保存目录
    )
    
    # 打印水果类型分类任务的评估指标
    # 包括准确率、精确率、召回率和F1分数，保留4位小数
    print(f"水果类型评估指标:")
    print(f"  准确率: {results['fruit_accuracy']:.4f}")  # 正确预测的水果类型比例
    print(f"  精确率: {results['fruit_precision']:.4f}") # 预测为某类水果中真正属于该类的比例
    print(f"  召回率: {results['fruit_recall']:.4f}")    # 某类水果中被正确预测的比例
    print(f"  F1分数: {results['fruit_f1']:.4f}")        # 精确率和召回率的调和平均值
    
    # 打印腐烂状态分类任务的评估指标
    # 同样包括准确率、精确率、召回率和F1分数
    print(f"腐烂状态评估指标:")
    print(f"  准确率: {results['state_accuracy']:.4f}")  # 正确预测的腐烂状态比例
    print(f"  精确率: {results['state_precision']:.4f}") # 预测为某种状态中真正属于该状态的比例
    print(f"  召回率: {results['state_recall']:.4f}")    # 某种状态中被正确预测的比例
    print(f"  F1分数: {results['state_f1']:.4f}")        # 精确率和召回率的调和平均值
    
    # 将生成的分类报告添加到结果字典中
    # 便于后续分析或保存完整评估结果
    results['fruit_report'] = fruit_report  # 水果类型分类报告
    results['state_report'] = state_report  # 腐烂状态分类报告
    
    # 如果需要可视化错误样本且结果中包含错误样本信息
    # 则调用visualize_error_samples函数进行可视化
    if visualize_errors and 'error_samples' in results:
        class_names = (fruit_class_names, state_class_names)  # 组合两种分类任务的类别名称
        visualize_error_samples(results, class_names, save_dir, top_k_errors)  # 可视化错误样本
    
    # 返回包含所有评估结果的字典
    return results


def visualize_error_samples(results: Dict[str, Any], class_names: Tuple[List[str], List[str]], 
                          output_dir: str, top_k: int = 10) -> None:
    """
    可视化错误预测的样本
    
    该函数对模型预测错误的样本进行可视化分析，帮助研究人员和开发者理解模型的失败模式。
    通过直观地展示错误样本及其真实标签和预测标签，可以发现模型的系统性错误和改进方向。
    
    该函数完成以下任务：
    1. 检查并统计不同类型的错误（水果类型错误、腐烂状态错误、两者都错误）
    2. 创建专门的目录保存错误样本可视化结果
    3. 生成包含多个错误样本的图表，每个样本显示图像及其真实和预测标签
    4. 保存高分辨率的错误样本可视化图表，便于后续分析
    
    错误分析对于模型改进至关重要，它可以帮助：
    - 识别数据集中的问题样本或标注错误
    - 发现模型对特定类别的系统性偏见
    - 指导数据增强和模型架构改进的方向
    - 评估模型在实际应用中可能面临的挑战
    
    Args:
        results (Dict[str, Any]): 评估结果字典，包含预测结果、真实标签和错误样本信息
        class_names (Tuple[List[str], List[str]]): 类别名称元组，第一个元素是水果类型名称列表，
                                                 第二个元素是腐烂状态名称列表
        output_dir (str): 输出目录路径，用于保存错误样本可视化结果
        top_k (int, optional): 要可视化的错误样本数量上限，默认为10个
    """
    # 检查是否有错误样本可供可视化 - 如果没有错误样本，则提前返回
    if 'error_samples' not in results or not results['error_samples']:
        print("没有错误预测的样本可供可视化")
        return
    
    # 解包类别名称 - 将元组拆分为水果类型名称列表和腐烂状态名称列表
    # 这些名称将用于将数字标签转换为可读的类别名称
    fruit_class_names, state_class_names = class_names
    
    # 创建输出目录 - 在指定的输出目录下创建专门用于存放错误样本可视化结果的子目录
    # exist_ok=True 确保目录已存在时不会引发错误
    error_dir = os.path.join(output_dir, 'error_samples')
    os.makedirs(error_dir, exist_ok=True)
    
    # 按照错误类型对样本进行分组 - 这有助于理解不同类型错误的分布和特点
    # 水果类型错误：模型正确预测了腐烂状态，但错误预测了水果类型
    fruit_errors = [s for s in results['error_samples'] if s['fruit_pred'] != s['fruit_target']]
    # 腐烂状态错误：模型正确预测了水果类型，但错误预测了腐烂状态
    state_errors = [s for s in results['error_samples'] if s['state_pred'] != s['state_target']]
    # 两者都错误：模型同时错误预测了水果类型和腐烂状态
    both_errors = [s for s in results['error_samples'] if s['fruit_pred'] != s['fruit_target'] and s['state_pred'] != s['state_target']]
    # 打印错误统计信息 - 输出各类错误的数量，帮助了解模型的整体错误模式
    # 总样本数用于计算错误率，评估模型的整体性能
    print(f"总样本数: {len(results['fruit_targets'])}")
    # 水果类型错误数量，反映模型在水果类型分类任务上的表现
    print(f"水果类型错误数: {len(fruit_errors)}")
    # 腐烂状态错误数量，反映模型在腐烂状态分类任务上的表现
    print(f"腐烂状态错误数: {len(state_errors)}")
    # 两者都错误的数量，反映模型在同时处理两个任务时的困难样本
    print(f"两者都错误数: {len(both_errors)}")
    
    # 可视化前K个错误样本 - 限制可视化的样本数量，避免图表过大难以查看
    # min函数确保当错误样本数量少于top_k时，只使用实际可用的样本数量
    error_samples = results['error_samples'][:min(top_k, len(results['error_samples']))]
    
    # 创建图表 - 设置图表布局，根据样本数量确定行列数
    # 计算需要展示的样本总数
    n_samples = len(error_samples)
    # 设置每行最多显示5个样本，保证每个样本有足够的显示空间
    n_cols = min(5, n_samples)
    # 根据样本数量和每行列数计算需要的行数，使用整除并向上取整
    n_rows = (n_samples + n_cols - 1) // n_cols
    # 创建足够大的图表以容纳所有错误样本 - 每个样本占据4x4英寸的空间
    # 这确保了每个样本图像有足够的大小和清晰度
    plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    
    # 遍历错误样本并可视化 - 为每个错误样本创建一个子图
    for i, sample in enumerate(error_samples):
        # 在图表的指定位置创建子图，i+1是因为subplot的索引从1开始
        plt.subplot(n_rows, n_cols, i + 1)
        
        # 显示图像 - 将PyTorch张量转换为NumPy数组以便使用matplotlib显示
        # permute函数将张量的维度从(C,H,W)重排为(H,W,C)，符合matplotlib的要求
        img = sample['image'].permute(1, 2, 0).numpy()
        # 显示图像，matplotlib会自动处理归一化的图像数据
        plt.imshow(img)
        
        # 获取真实和预测的类别名称 - 将数字索引转换为可读的类别名称
        # 使用索引从类别名称列表中获取对应的类别名称
        true_fruit = fruit_class_names[sample['fruit_target']]
        pred_fruit = fruit_class_names[sample['fruit_pred']]
        true_state = state_class_names[sample['state_target']]
        pred_state = state_class_names[sample['state_pred']]
        
        # 设置标题，显示真实和预测的类别 - 包括水果类型和腐烂状态
        # 使用较小的字体大小确保标题不会过大
        plt.title(f"真实: {true_fruit}({true_state})\n预测: {pred_fruit}({pred_state})", 
                 fontsize=9)
        # 隐藏坐标轴，使图像更加清晰
        plt.axis('off')
    
    # 调整子图之间的间距 - 确保图表布局紧凑且美观
    plt.tight_layout()
    
    # 保存错误样本可视化图表 - 将图表保存为PNG文件
    # 在error_dir目录下创建一个描述性文件名
    error_viz_path = os.path.join(error_dir, 'error_samples_visualization.png')
    # 保存图表，设置较高的DPI以确保图像质量
    plt.savefig(error_viz_path)
    # 关闭当前图表，释放内存资源
    plt.close()
    
    # 打印保存路径信息 - 通知用户可视化结果的保存位置
    print(f"错误样本可视化已保存至: {error_viz_path}")

def parse_args():
    """
    解析命令行参数
    
    该函数定义并解析评估脚本所需的命令行参数，包括配置文件路径、模型检查点路径、
    输出目录等重要设置。通过命令行参数，用户可以灵活地控制评估过程的各个方面。
    
    Returns:
        argparse.Namespace: 解析后的参数对象，包含用户通过命令行指定的所有设置
    """
    # 创建参数解析器对象，并设置描述信息
    parser = argparse.ArgumentParser(description='模型评估脚本')
    
    # 添加各种命令行参数选项
    # 配置文件路径参数 - 允许用户指定自定义配置文件
    parser.add_argument('--config', type=str, default=None, 
                        help='配置文件路径，默认使用项目根目录下的标准配置')
    
    # 模型检查点路径参数 - 指定要评估的模型权重文件
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='模型检查点路径，如果为None则自动寻找最佳检查点')
    
    # 输出目录参数 - 指定评估结果的保存位置
    parser.add_argument('--output', type=str, default='evaluation_results', 
                        help='评估结果输出目录，包括报告和可视化内容')
    
    # 批次大小参数 - 控制评估时的内存使用和处理速度
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='批次大小，如果为None则使用配置文件中的设置')
    
    # 错误样本可视化开关 - 决定是否生成错误预测的可视化分析
    parser.add_argument('--visualize_errors', action='store_true', 
                        help='是否可视化错误预测的样本，便于错误分析')
    
    # 错误样本数量参数 - 控制可视化的错误样本数量
    parser.add_argument('--top_k_errors', type=int, default=10, 
                        help='可视化前K个错误预测的样本，数量过多可能导致图表过大')
    
    # 解析命令行参数并返回
    return parser.parse_args()


def load_model(config: Dict[str, Any], data_loader: DataLoader = None, 
               checkpoint_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    加载模型和检查点
    
    该函数负责根据配置创建模型实例，并从指定的检查点文件加载预训练权重。
    如果未指定检查点路径，函数会自动寻找最佳检查点。该函数是评估过程的
    关键前置步骤，确保使用正确的模型和权重进行评估。
    
    Args:
        config (Dict[str, Any]): 配置字典，包含模型结构和训练设置
        data_loader (DataLoader, optional): 数据加载器，用于获取类别数量信息
        checkpoint_path (str, optional): 检查点路径，如果为None则自动寻找最佳检查点
        
    Returns:
        Tuple[nn.Module, Dict[str, Any]]: 加载好权重的模型和检查点信息字典
    """
    # 获取水果类型和腐烂状态的类别数量
    if data_loader is not None:
        # 从数据加载器获取底层数据集
        dataset = data_loader.dataset
        # 如果是Subset对象（如在验证集中），需要访问.dataset属性获取原始数据集
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        
        # 从数据集的DataFrame中获取唯一类别的数量
        num_fruit_classes = len(dataset.data_frame['fruit_type'].unique())  # 水果类型数量
        num_state_classes = len(dataset.data_frame['state'].unique())       # 腐烂状态数量
    else:
        # 如果没有提供数据加载器，使用默认值或从配置中获取
        # 这些默认值应该根据实际数据集情况设置
        num_fruit_classes = 5  # 默认假设有5种水果类型
        num_state_classes = 2  # 默认假设有2种腐烂状态（好/坏）
    
    # 从配置中获取骨干网络类型和预训练设置
    backbone = config['model'].get('backbone', 'resnet50')     # 默认使用ResNet50
    pretrained = config['model'].get('pretrained', True)       # 默认使用预训练权重
    
    # 创建模型实例
    model = create_model(num_fruit_classes, num_state_classes, backbone, pretrained)
    
    # 配置计算设备 - 优先使用GPU，如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # 将模型移动到指定设备
    
    # 处理检查点路径 - 如果未指定，则自动寻找最佳检查点
    if checkpoint_path is None:
        # 从配置中获取检查点保存目录，或使用默认目录
        save_dir = config['train'].get('save_dir', os.path.join(project_root, 'checkpoints'))
        # 在保存目录中查找具有最佳性能的检查点
        checkpoint_path = find_best_checkpoint(save_dir)
        
        # 如果仍未找到有效检查点，抛出错误
        if checkpoint_path is None:
            raise ValueError("未找到有效的检查点，请先训练模型或指定检查点路径")
    
    # 加载检查点文件
    print(f"加载检查点: {checkpoint_path}")
    # 加载时考虑设备兼容性，确保CPU和GPU之间可以正确转换
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 将保存的模型权重加载到模型中
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 返回加载好权重的模型和检查点信息
    return model, checkpoint


def get_class_names(dataset: FruitDataset) -> Tuple[List[str], List[str]]:
    """
    获取类别名称
    
    该函数从数据集对象中提取水果类型和腐烂状态的名称列表。这些名称用于
    在评估报告和可视化中标记类别，使结果更易于理解和解释。类别名称按字母
    顺序排序，确保在不同运行之间保持一致的顺序。
    
    Args:
        dataset (FruitDataset): 水果数据集对象，包含类别信息
        
    Returns:
        Tuple[List[str], List[str]]: 包含两个列表的元组：
            1. 水果类型名称列表（按字母顺序排序）
            2. 腐烂状态名称列表（按字母顺序排序）
    """
    # 从数据集的DataFrame中获取所有唯一的水果类型
    fruit_types = dataset.data_frame['fruit_type'].unique()
    # 对水果类型名称进行排序，确保顺序一致性
    fruit_type_names = sorted(fruit_types)
    
    # 从数据集的DataFrame中获取所有唯一的腐烂状态
    state_types = dataset.data_frame['state'].unique()
    # 对腐烂状态名称进行排序，确保顺序一致性
    state_type_names = sorted(state_types)
    
    # 返回排序后的水果类型名称列表和腐烂状态名称列表
    return fruit_type_names, state_type_names


def main():
    """
    主函数 - 评估模型的整体流程控制
    
    该函数作为评估脚本的入口点，负责协调整个评估过程，包括参数解析、
    配置加载、数据准备、模型加载、评估执行和结果保存等关键步骤。
    """
    # 解析命令行参数 - 获取用户通过命令行指定的各项设置
    args = parse_args()
    
    # 加载配置 - 确定配置文件路径并加载配置信息
    # 如果未指定配置文件，则使用项目默认配置路径
    if args.config is None:
        config_path = os.path.join(project_root, 'config', 'config.yaml')
    else:
        config_path = args.config
    
    # 从配置文件读取配置信息到字典对象
    config = load_config(config_path)
    
    # 设置输出目录 - 创建用于存储评估结果的目录
    # exist_ok=True 确保目录已存在时不会引发错误
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据加载器 - 准备用于评估的数据
    # 如果命令行指定了batch_size，则临时修改配置中的值
    # 这允许用户根据硬件资源灵活调整批次大小
    if args.batch_size:
        config['device']['batch_size'] = args.batch_size
    
    # 获取数据加载器（使用测试集数据）
    # mode='eval'表示我们需要验证和测试集数据加载器
    _, test_loader, _ = get_data_loaders(config, mode='eval')
    
    # 获取计算设备 - 确定使用CPU还是GPU进行评估
    # 优先使用GPU以加速评估过程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型 - 创建模型实例并加载预训练权重
    # 从指定检查点或自动找到的最佳检查点加载模型权重
    model, checkpoint = load_model(config, test_loader, args.checkpoint)
    
    # 获取类别名称 - 提取数据集中的水果类型和腐烂状态名称
    # 这些名称用于在报告和可视化中标记类别
    dataset = test_loader.dataset
    # 如果数据集是Subset类型（常见于验证集），需要获取其包装的原始数据集
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    # 获取排序后的水果类型和腐烂状态名称列表
    fruit_class_names, state_class_names = get_class_names(dataset)
    
    # 评估模型并生成报告 - 执行模型评估并生成各种性能指标
    # 包括准确率、精确率、召回率、F1分数等，以及可选的错误样本可视化
    results = evaluate_and_report(
        model, test_loader, device,
        fruit_class_names, state_class_names, output_dir,
        visualize_errors=args.visualize_errors,  # 是否可视化错误样本
        top_k_errors=args.top_k_errors          # 可视化的错误样本数量
    )
    
    # 保存评估结果摘要 - 将关键评估指标保存为JSON文件
    # 这便于后续分析和比较不同模型或配置的性能
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    summary = {
        # 水果类型评估指标 - 记录模型在水果类型分类任务上的性能
        'fruit_accuracy': float(results['fruit_accuracy']),   # 准确率
        'fruit_precision': float(results['fruit_precision']), # 精确率
        'fruit_recall': float(results['fruit_recall']),       # 召回率
        'fruit_f1': float(results['fruit_f1']),               # F1分数
        
        # 腐烂状态评估指标 - 记录模型在腐烂状态分类任务上的性能
        'state_accuracy': float(results['state_accuracy']),   # 准确率
        'state_precision': float(results['state_precision']), # 精确率
        'state_recall': float(results['state_recall']),       # 召回率
        'state_f1': float(results['state_f1']),               # F1分数
        
        # 其他信息 - 记录评估相关的元数据
        'checkpoint': checkpoint.get('epoch', 'unknown'),     # 使用的检查点（训练轮次）
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # 评估时间戳
    }
    
    # 将评估摘要写入JSON文件
    # indent=4确保生成的JSON文件具有良好的可读性
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # 输出完成信息 - 通知用户评估已完成并指明结果位置
    print(f"评估完成，结果已保存到 {output_dir}")


if __name__ == '__main__':
    main()
