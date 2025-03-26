"""训练器模块

这个模块实现了模型训练的核心功能，包括：
1. 训练循环的实现 - 包含完整的训练流程，支持混合精度训练和数据增强
2. 验证循环的实现 - 用于评估模型性能和防止过拟合
3. 训练过程中的指标计算和记录 - 跟踪损失和准确率等指标
4. 模型保存和加载 - 支持保存最佳模型、定期检查点和训练历史
5. 可视化训练过程 - 绘制训练曲线以便分析模型性能

主要类：
- Trainer: 模型训练器，管理整个训练过程，包括训练、验证、保存和加载模型
- 辅助函数: create_trainer用于简化训练器的创建过程
"""

# 标准库导入
import os          # 操作系统接口，用于文件和目录操作
import sys         # 系统相关功能，用于修改Python路径
import time        # 时间相关功能，用于计时
import json        # JSON数据处理，用于保存训练历史
from pathlib import Path  # 面向对象的文件系统路径，提供更现代的路径操作

# 第三方库导入
import torch                       # PyTorch深度学习框架
import torch.nn as nn              # 神经网络模块
import torch.optim as optim        # 优化器
from torch.utils.data import DataLoader  # 数据加载器
from typing import Dict, Tuple, List, Optional, Union, Any, Callable  # 类型注解
import numpy as np                 # 数值计算库
from tqdm import tqdm              # 进度条
import matplotlib.pyplot as plt    # 绘图库，用于可视化训练过程

# 混合精度训练相关导入
from torch.amp import autocast     # 自动混合精度，减少内存使用并加速训练
from torch.cuda.amp import GradScaler  # 梯度缩放器，处理混合精度训练中的数值溢出问题

# 数据增强模块导入
from trains.augmentation import MixUp  # MixUp数据增强技术，混合两个样本及其标签

# 添加项目根目录到Python路径，确保可以导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件目录
project_root = os.path.dirname(current_dir)               # 获取项目根目录
if project_root not in sys.path:                          # 如果项目根目录不在Python路径中
    sys.path.append(project_root)                         # 添加到Python路径

# 导入项目其他模块
from models.model import FruitClassifier, MultiTaskLoss, calculate_metrics  # 导入模型相关类和函数
from config import load_config  # 导入配置加载函数


class Trainer:
    """
    模型训练器
    
    管理模型的训练、验证和测试过程，包括以下核心功能：
    1. 支持混合精度训练 - 使用torch.cuda.amp加速训练并减少内存占用
    2. 支持数据增强 - 如MixUp技术，提高模型泛化能力
    3. 支持学习率调度 - 动态调整学习率优化训练过程
    4. 支持早停机制 - 防止过拟合
    5. 支持模型检查点保存和加载 - 方便训练中断后恢复
    6. 支持训练过程可视化 - 绘制损失和准确率曲线
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: torch.device = None, config: Dict = None):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 待训练的模型，通常是FruitClassifier的实例，负责水果类型和腐烂状态的多任务分类
            train_loader (DataLoader): 训练数据加载器，提供批次化的训练数据，包含图像和对应的标签
            val_loader (DataLoader): 验证数据加载器，提供批次化的验证数据，用于评估模型性能
            criterion (nn.Module): 损失函数，通常是MultiTaskLoss的实例，计算水果类型和腐烂状态的综合损失
            optimizer (torch.optim.Optimizer): 优化器，负责更新模型参数，如Adam、SGD等
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 
                学习率调度器，动态调整学习率，如StepLR、CosineAnnealingLR等，默认为None
            device (torch.device): 
                训练设备，指定模型和数据应该放在CPU还是GPU上，默认自动选择可用的GPU或CPU
            config (Dict): 
                配置字典，包含各种训练参数，如是否使用混合精度训练、数据增强设置等，默认为空字典
        """
        # 基本组件设置
        self.model = model  # 神经网络模型
        self.train_loader = train_loader  # 训练数据加载器
        self.val_loader = val_loader  # 验证数据加载器
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化器
        self.scheduler = scheduler  # 学习率调度器
        
        # 设备设置 - 优先使用GPU（如果可用）
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config is not None else {}
        
        # 将模型移动到指定设备（CPU或GPU）
        self.model.to(self.device)
        
        # 混合精度训练设置 - 可以减少内存使用并加速训练
        # 只有在配置中启用且CUDA可用的情况下才使用
        self.use_amp = self.config.get('use_amp', False)
        if self.use_amp and torch.cuda.is_available():
            # 创建梯度缩放器，用于处理混合精度训练中的数值溢出问题
            self.scaler = GradScaler()
            print("启用混合精度训练 (AMP)")
        else:
            self.scaler = None
            if self.use_amp and not torch.cuda.is_available():
                print("警告: 混合精度训练需要CUDA支持，但当前设备不支持CUDA。已禁用混合精度训练。")
                self.use_amp = False
                
        # 批次级别的数据增强设置 - 在训练过程中动态增强数据
        # 初始化MixUp相关变量
        self.use_mixup = False  # 是否使用MixUp数据增强的标志
        self.mixup = None       # MixUp增强器对象
        
        # 检查配置中是否启用MixUp数据增强
        # MixUp是一种有效的数据增强技术，通过线性插值两个样本及其标签来创建新的训练样本
        # 这有助于提高模型的泛化能力，减少过拟合，尤其是在数据集较小时效果显著
        if 'augmentation' in self.config and 'mixup' in self.config['augmentation']:
            # 获取MixUp的具体配置
            mixup_config = self.config['augmentation']['mixup']
            if mixup_config.get('enabled', False):
                # 如果配置中启用了MixUp，则创建MixUp增强器
                self.use_mixup = True
                # alpha参数控制混合程度：较小的值使混合更接近原始样本，较大的值使混合更均匀
                alpha = mixup_config.get('alpha', 0.2)
                self.mixup = MixUp(alpha=alpha)
                print(f"启用MixUp数据增强 (alpha={alpha})")
        
        # 训练历史记录 - 用于跟踪训练过程中的各种指标
        # 这些指标将用于绘制训练曲线和分析模型性能
        self.history = {
            'train_loss': [],      # 训练集上的总体损失
            'val_loss': [],        # 验证集上的总体损失
            'train_fruit_acc': [],  # 训练集上的水果类型分类准确率
            'val_fruit_acc': [],    # 验证集上的水果类型分类准确率
            'train_state_acc': [],  # 训练集上的腐烂状态分类准确率
            'val_state_acc': [],    # 验证集上的腐烂状态分类准确率
            'learning_rate': []     # 每个epoch的学习率
        }
        
        # 设置模型保存路径 - 用于存储检查点和训练历史
        # 默认保存在项目根目录下的'checkpoints'文件夹中
        self.save_dir = self.config.get('save_dir', os.path.join(project_root, 'checkpoints'))
        # 确保保存目录存在，如果不存在则创建
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 记录最佳模型性能 - 用于保存验证集上性能最好的模型
        self.best_val_loss = float('inf')  # 初始化为无穷大，这样任何有效的损失值都会更新它
        self.best_epoch = 0               # 记录最佳模型对应的epoch索引
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch（完整训练数据集遍历一次）
        
        该方法实现了模型训练的核心流程，包括：
        1. 将模型设置为训练模式
        2. 遍历训练数据加载器中的所有批次
        3. 可选的数据增强（如MixUp）
        4. 前向和反向传播，支持混合精度训练
        5. 计算并返回训练指标
        
        Returns:
            Dict[str, float]: 训练指标字典，包含损失和准确率
        """
        # 将模型设置为训练模式 - 启用梯度计算和批归一化等
        self.model.train()
        
        # 初始化累计损失和预测结果存储列表
        running_loss = 0.0  # 累计损失，用于计算平均损失
        # 创建列表来存储所有批次的预测和目标，用于计算最终指标
        all_fruit_preds = []     # 存储所有批次的水果类型预测
        all_state_preds = []     # 存储所有批次的腐烂状态预测
        all_fruit_targets = []   # 存储所有批次的水果类型真实标签
        all_state_targets = []   # 存储所有批次的腐烂状态真实标签
        
        # 使用tqdm显示进度条 - 直观地展示训练进度和实时指标
        pbar = tqdm(self.train_loader, desc="Training")
        
        # 遍历训练数据加载器中的所有批次
        for batch_idx, (images, fruit_targets, state_targets) in enumerate(pbar):
            # 将数据移动到指定设备（CPU或GPU）
            images = images.to(self.device)                 # 图像数据
            fruit_targets = fruit_targets.to(self.device)   # 水果类型标签
            state_targets = state_targets.to(self.device)   # 腐烂状态标签
            
            # 应用批次级别的数据增强（如MixUp）
            if self.use_mixup:
                # MixUp增强需要在模型前向传播前应用
                # 混合两个样本及其标签，生成新的训练样本
                images, mixed_fruit_targets, mixed_state_targets = self.mixup((images, fruit_targets, state_targets))
                # 使用混合后的目标（软标签）替换原始标签
                fruit_targets = mixed_fruit_targets    # 混合后的水果类型标签
                state_targets = mixed_state_targets    # 混合后的腐烂状态标签
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播和损失计算（使用混合精度）
            if self.use_amp:
                with autocast(device_type='cuda'):
                    fruit_logits, state_logits = self.model(images)
                    # 如果使用了MixUp，需要修改损失函数的调用方式
                    if self.use_mixup:
                        # MixUp使用软标签，需要直接计算交叉熵损失
                        fruit_loss = nn.CrossEntropyLoss(reduction='mean')(fruit_logits, fruit_targets)
                        state_loss = nn.CrossEntropyLoss(reduction='mean')(state_logits, state_targets)
                        loss = fruit_loss + state_loss
                        loss_dict = {'fruit_loss': fruit_loss, 'state_loss': state_loss}
                    else:
                        loss, loss_dict = self.criterion(fruit_logits, state_logits, fruit_targets, state_targets)
                
                # 使用GradScaler进行反向传播和参数更新
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 常规前向传播和反向传播
                fruit_logits, state_logits = self.model(images)
                # 如果使用了MixUp，需要修改损失函数的调用方式
                if self.use_mixup:
                    # MixUp使用软标签，需要直接计算交叉熵损失
                    fruit_loss = nn.CrossEntropyLoss(reduction='mean')(fruit_logits, fruit_targets)
                    state_loss = nn.CrossEntropyLoss(reduction='mean')(state_logits, state_targets)
                    loss = fruit_loss + state_loss
                    loss_dict = {'fruit_loss': fruit_loss, 'state_loss': state_loss}
                else:
                    loss, loss_dict = self.criterion(fruit_logits, state_logits, fruit_targets, state_targets)
                loss.backward()
                self.optimizer.step()
            
            # 记录当前批次的损失值到累计损失中，用于计算整个epoch的平均损失
            running_loss += loss.item()  # item()方法将单个元素张量转换为标量值
            
            # 计算模型预测结果 - 对logits应用argmax获取最可能的类别索引
            fruit_preds = torch.argmax(fruit_logits, dim=1)  # 水果类型预测结果
            state_preds = torch.argmax(state_logits, dim=1)  # 腐烂状态预测结果
            
            # 收集预测和目标用于计算指标（如准确率、精确率、召回率等）
            # 当使用MixUp时需要特殊处理，因为MixUp产生的是软标签（概率分布）
            if self.use_mixup:
                # 对于MixUp增强的数据，我们需要从软标签中提取原始的硬标签用于评估
                # 如果标签是多维的（软标签），则取概率最大的类别作为硬标签
                original_fruit_targets = torch.argmax(fruit_targets, dim=1) if len(fruit_targets.shape) > 1 else fruit_targets
                original_state_targets = torch.argmax(state_targets, dim=1) if len(state_targets.shape) > 1 else state_targets
                
                # 将预测结果和原始标签转移到CPU并转换为numpy数组以便于后续处理
                all_fruit_preds.append(fruit_preds.cpu().numpy())  # 水果类型预测
                all_state_preds.append(state_preds.cpu().numpy())  # 腐烂状态预测
                all_fruit_targets.append(original_fruit_targets.cpu().numpy())  # 原始水果类型标签
                all_state_targets.append(original_state_targets.cpu().numpy())  # 原始腐烂状态标签
            else:
                # 对于没有使用MixUp的数据，直接收集预测和标签
                all_fruit_preds.append(fruit_preds.cpu().numpy())  # 水果类型预测
                all_state_preds.append(state_preds.cpu().numpy())  # 腐烂状态预测
                all_fruit_targets.append(fruit_targets.cpu().numpy())  # 水果类型标签
                all_state_targets.append(state_targets.cpu().numpy())  # 腐烂状态标签
            
            # 更新进度条显示的实时指标，包括总体损失和各任务的损失
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",  # 总体损失
                'fruit_loss': f"{loss_dict['fruit_loss'].item():.4f}",  # 水果类型分类损失
                'state_loss': f"{loss_dict['state_loss'].item():.4f}"   # 腐烂状态分类损失
            })
        
        # 计算整个epoch的平均损失
        epoch_loss = running_loss / len(self.train_loader)  # 除以批次总数得到平均值
        
        # 合并所有批次的预测和目标，用于计算整体指标
        # 使用numpy的concatenate函数将列表中的所有数组合并为一个大数组
        all_fruit_preds = np.concatenate(all_fruit_preds)      # 合并所有水果类型预测
        all_state_preds = np.concatenate(all_state_preds)      # 合并所有腐烂状态预测
        all_fruit_targets = np.concatenate(all_fruit_targets)  # 合并所有水果类型标签
        all_state_targets = np.concatenate(all_state_targets)  # 合并所有腐烂状态标签
        
        # 计算分类准确率
        # 准确率 = 正确预测的样本数 / 总样本数
        fruit_correct = (all_fruit_preds == all_fruit_targets).sum()  # 水果类型正确预测数
        state_correct = (all_state_preds == all_state_targets).sum()  # 腐烂状态正确预测数
        fruit_acc = fruit_correct / len(all_fruit_targets)  # 水果类型分类准确率
        state_acc = state_correct / len(all_state_targets)  # 腐烂状态分类准确率
        
        # 返回训练指标字典，包含平均损失和各任务的准确率
        return {
            'loss': epoch_loss,        # 整个epoch的平均损失
            'fruit_acc': fruit_acc,    # 水果类型分类准确率
            'state_acc': state_acc     # 腐烂状态分类准确率
            # 注意：可以在这里添加更多指标，如精确率、召回率和F1分数等
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch（对验证集进行一次完整评估）
        
        该方法在不计算梯度的情况下对模型进行评估，主要目的是：
        1. 监控模型在验证集上的性能，防止过拟合
        2. 评估模型的泛化能力
        3. 为早停和模型保存提供依据
        
        Returns:
            Dict[str, float]: 验证指标字典，包含损失和准确率
        """
        # 将模型设置为评估模式 - 禁用梯度计算和批归一化等
        self.model.eval()
        
        # 初始化累计损失和预测结果存储列表
        running_loss = 0.0  # 累计损失，用于计算平均损失
        # 创建列表来存储所有批次的预测和目标，用于计算最终指标
        all_fruit_preds = []     # 存储所有批次的水果类型预测
        all_state_preds = []     # 存储所有批次的腐烂状态预测
        all_fruit_targets = []   # 存储所有批次的水果类型真实标签
        all_state_targets = []   # 存储所有批次的腐烂状态真实标签
        
        # 使用torch.no_grad()上下文管理器禁用梯度计算，减少内存使用并加速计算
        with torch.no_grad():
            # 使用tqdm显示进度条 - 直观地展示验证进度和实时指标
            pbar = tqdm(self.val_loader, desc="Validation")
            
            # 遍历验证数据加载器中的所有批次
            for batch_idx, (images, fruit_targets, state_targets) in enumerate(pbar):
                # 将数据移动到指定设备（CPU或GPU）
                images = images.to(self.device)                 # 图像数据
                fruit_targets = fruit_targets.to(self.device)   # 水果类型标签
                state_targets = state_targets.to(self.device)   # 腐烂状态标签
                
                # 前向传播 - 验证时也可以使用混合精度以提高计算效率
                # 但不需要梯度缩放，因为验证时不计算梯度也不更新参数
                if self.use_amp:
                    # 如果启用了混合精度，使用autocast上下文管理器
                    with autocast(device_type='cuda'):
                        fruit_logits, state_logits = self.model(images)  # 模型前向传播
                        loss, loss_dict = self.criterion(fruit_logits, state_logits, fruit_targets, state_targets)  # 计算损失
                else:
                    # 常规前向传播
                    fruit_logits, state_logits = self.model(images)  # 模型前向传播
                    loss, loss_dict = self.criterion(fruit_logits, state_logits, fruit_targets, state_targets)  # 计算损失
                
                # 记录当前批次的损失值到累计损失中
                running_loss += loss.item()  # item()方法将单个元素张量转换为标量值
                
                # 计算模型预测结果 - 对logits应用argmax获取最可能的类别索引
                fruit_preds = torch.argmax(fruit_logits, dim=1)  # 水果类型预测结果
                state_preds = torch.argmax(state_logits, dim=1)  # 腐烂状态预测结果
                
                # 收集预测和目标用于计算指标（如准确率、精确率、召回率等）
                # 将预测和目标转移到CPU并转换为numpy数组以便于后续处理
                all_fruit_preds.append(fruit_preds.cpu().numpy())      # 水果类型预测
                all_state_preds.append(state_preds.cpu().numpy())      # 腐烂状态预测
                all_fruit_targets.append(fruit_targets.cpu().numpy())  # 水果类型标签
                all_state_targets.append(state_targets.cpu().numpy())  # 腐烂状态标签
                
                # 更新进度条显示的实时指标，包括总体损失和各任务的损失
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",                      # 总体损失
                    'fruit_loss': f"{loss_dict['fruit_loss'].item():.4f}",  # 水果类型分类损失
                    'state_loss': f"{loss_dict['state_loss'].item():.4f}"   # 腐烂状态分类损失
                })
        
        # 计算整个epoch的平均验证损失
        epoch_loss = running_loss / len(self.val_loader)  # 除以批次总数得到平均值
        
        # 合并所有批次的预测和目标，用于计算整体指标
        # 使用numpy的concatenate函数将列表中的所有数组合并为一个大数组
        all_fruit_preds = np.concatenate(all_fruit_preds)      # 合并所有水果类型预测
        all_state_preds = np.concatenate(all_state_preds)      # 合并所有腐烂状态预测
        all_fruit_targets = np.concatenate(all_fruit_targets)  # 合并所有水果类型标签
        all_state_targets = np.concatenate(all_state_targets)  # 合并所有腐烂状态标签
        
        # 计算分类准确率
        # 准确率 = 正确预测的样本数 / 总样本数
        fruit_correct = (all_fruit_preds == all_fruit_targets).sum()  # 水果类型正确预测数
        state_correct = (all_state_preds == all_state_targets).sum()  # 腐烂状态正确预测数
        fruit_acc = fruit_correct / len(all_fruit_targets)  # 水果类型分类准确率
        state_acc = state_correct / len(all_state_targets)  # 腐烂状态分类准确率
        
        # 返回验证指标字典，包含平均损失和各任务的准确率
        # 注意：可以根据需要添加更多指标，如精确率、召回率和F1分数等
        return {
            'loss': epoch_loss,        # 整个epoch的平均验证损失
            'fruit_acc': fruit_acc,    # 水果类型分类准确率
            'state_acc': state_acc     # 腐烂状态分类准确率
            # 在这里可以添加更多评估指标，如精确率、召回率和F1分数
        }
    
    def train(self, num_epochs: int, save_best: bool = True, save_latest: bool = True, save_checkpoints: bool = True, save_freq: int = 5, early_stopping: int = 0) -> Dict[str, List[float]]:
        """
        训练模型的主要方法，管理整个训练过程包括训练循环、验证、学习率调整、模型保存和早停策略等
        
        该方法执行以下操作：
        1. 循环进行指定轮数的训练，每轮包含一个完整的训练epoch和验证epoch
        2. 跟踪和记录训练过程中的各种指标
        3. 根据验证结果保存最佳模型和定期检查点
        4. 实现早停机制以防止过拟合
        
        Args:
            num_epochs (int): 要训练的总轮数，每轮包含对整个训练集的一次完整遍历
            save_best (bool): 是否保存验证损失最低的模型，默认为True
            save_latest (bool): 是否在每个epoch结束后保存最新的模型状态，默认为True
            save_checkpoints (bool): 是否启用周期性保存检查点功能，默认为True
            save_freq (int): 检查点保存频率，每多少个epoch保存一次检查点，默认为5，设置为0表示不保存周期性检查点
            early_stopping (int): 早停策略的耐心值，即验证损失连续多少轮没有改善就停止训练，默认为0（不使用早停）
            
        Returns:
            Dict[str, List[float]]: 训练历史记录字典，包含每个epoch的损失、准确率和学习率等指标
        """
        # 打印训练配置信息，包括训练轮数、使用的设备和模型保存设置
        print(f"开始训练，共{num_epochs}个epoch")
        print(f"使用设备: {self.device}")
        print(f"保存设置: 最佳模型={save_best}, 最新模型={save_latest}, 启用周期性检查点={save_checkpoints}, 检查点频率={save_freq}轮")
        
        # 初始化早停计数器，用于跟踪验证损失没有改善的轮数
        no_improve_count = 0
        
        # 主训练循环 - 遍历指定的训练轮数
        for epoch in range(num_epochs):
            # 记录当前轮次的开始时间，用于计算每轮训练的耗时
            start_time = time.time()
            
            # 调用train_epoch方法进行一轮训练，返回训练指标
            train_metrics = self.train_epoch()  # 返回包含损失和准确率等指标的字典
            
            # 调用validate_epoch方法进行一轮验证，返回验证指标
            val_metrics = self.validate_epoch()  # 返回包含验证损失和准确率等指标的字典
            
            # 更新学习率 - 如果配置了学习率调度器，则根据调度策略更新学习率
            if self.scheduler is not None:
                self.scheduler.step()  # 执行学习率调度器的步进
                current_lr = self.scheduler.get_last_lr()[0]  # 获取当前学习率
            else:
                # 如果没有调度器，直接从优化器中获取当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录训练历史 - 将当前轮次的指标添加到历史记录中
            self.history['train_loss'].append(train_metrics['loss'])         # 训练损失
            self.history['val_loss'].append(val_metrics['loss'])             # 验证损失
            self.history['train_fruit_acc'].append(train_metrics['fruit_acc'])  # 训练集水果类型准确率
            self.history['val_fruit_acc'].append(val_metrics['fruit_acc'])      # 验证集水果类型准确率
            self.history['train_state_acc'].append(train_metrics['state_acc'])  # 训练集腐烂状态准确率
            self.history['val_state_acc'].append(val_metrics['state_acc'])      # 验证集腐烂状态准确率
            self.history['learning_rate'].append(current_lr)                    # 当前学习率
            
            # 计算当前轮次的训练耗时
            epoch_time = time.time() - start_time
            
            # 打印详细的训练信息，包括轮次、耗时、损失、准确率和学习率
            print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - "
                  f"Train Loss: {train_metrics['loss']:.4f} - "
                  f"Val Loss: {val_metrics['loss']:.4f} - "
                  f"Train Fruit Acc: {train_metrics['fruit_acc']:.4f} - "
                  f"Val Fruit Acc: {val_metrics['fruit_acc']:.4f} - "
                  f"Train State Acc: {train_metrics['state_acc']:.4f} - "
                  f"Val State Acc: {val_metrics['state_acc']:.4f} - "
                  f"LR: {current_lr:.6f}")
            
            # 保存最佳模型 - 如果当前验证损失低于历史最低值，则保存模型
            if save_best and val_metrics['loss'] < self.best_val_loss:
                # 更新最佳验证损失和对应的epoch
                self.best_val_loss = val_metrics['loss']  # 更新最佳验证损失记录
                self.best_epoch = epoch  # 记录最佳模型的epoch索引
                # 将当前模型保存为最佳模型
                self.save_model(os.path.join(self.save_dir, 'best_model.pth'))
                print(f"保存最佳模型，验证损失: {self.best_val_loss:.4f}")
                # 重置早停计数器，因为模型有改善
                no_improve_count = 0
            else:
                # 如果没有改善，早停计数器增加
                no_improve_count += 1
                
            # 周期性保存检查点 - 根据指定频率保存模型检查点
            if save_checkpoints and save_freq > 0 and (epoch + 1) % save_freq == 0:
                # 构造检查点文件路径，包含epoch编号
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{epoch+1}.pth')
                # 保存当前模型状态作为检查点
                self.save_model(checkpoint_path)
                print(f"保存第{epoch+1}轮检查点")
                
            # 保存最新模型 - 如果启用了该选项，每轮都保存最新的模型状态
            if save_latest:
                # 将当前模型保存为最新模型
                self.save_model(os.path.join(self.save_dir, 'latest_model.pth'))
                print(f"保存最新模型 (第{epoch+1}轮)")
            
            # 早停检查 - 如果启用了早停且连续多轮没有改善，则停止训练
            if early_stopping > 0 and no_improve_count >= early_stopping:
                print(f"早停：{early_stopping}个epoch内验证损失没有改善")
                break  # 跳出训练循环
        
        # 训练结束后的操作
        
        # 保存最后一个epoch的模型状态，无论其性能如何
        self.save_model(os.path.join(self.save_dir, 'last_model.pth'))
        
        # 保存完整的训练历史记录，可用于后续分析和可视化
        self.save_history()
        
        # 打印训练完成信息，包括最佳验证损失和对应的epoch
        print(f"训练完成，最佳验证损失: {self.best_val_loss:.4f}，在第{self.best_epoch+1}个epoch")
        
        # 返回完整的训练历史记录，可用于后续分析和可视化
        return self.history
    
    def save_model(self, path: str) -> None:
        """
        保存模型状态到指定路径。
        
        该方法保存完整的模型检查点，包括：
        1. 模型参数状态
        2. 优化器状态（包含学习率和动量等信息）
        3. 学习率调度器状态（如果存在）
        4. 最佳验证损失和对应的epoch
        
        这使得可以在后续从检查点继续训练，或者用于模型部署。
        
        Args:
            path (str): 模型保存的目标路径，如'models/best_model.pth'
        """
        # 确保目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 构建要保存的检查点字典，包含各种状态信息
        checkpoint = {
            'model_state_dict': self.model.state_dict(),  # 模型参数状态
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,  # 学习率调度器状态（如果存在）
            'best_val_loss': self.best_val_loss,  # 当前最佳验证损失
            'epoch': self.best_epoch,  # 当前最佳epoch
        }
        
        # 使用torch.save将检查点字典保存到指定路径
        torch.save(checkpoint, path)
    
    def load_model(self, path: str) -> None:
        """
        从指定路径加载模型检查点。
        
        该方法从保存的检查点文件中恢复模型状态，包括：
        1. 模型参数
        2. 优化器状态
        3. 学习率调度器状态（如果存在）
        4. 最佳验证损失和对应的epoch
        
        这允许从之前保存的检查点继续训练，或者加载预训练模型进行推理。
        
        Args:
            path (str): 要加载的模型检查点文件路径
        """
        # 首先检查模型文件是否存在
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return
        
        # 使用torch.load加载检查点，并指定设备映射（确保在正确的设备上加载）
        checkpoint = torch.load(path, map_location=self.device)
        
        # 从检查点中恢复模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 从检查点中恢复优化器状态（包含学习率、动量等）
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 如果有学习率调度器且检查点中包含调度器状态，则恢复调度器状态
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复最佳验证损失和对应的epoch
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['epoch']
        
        # 打印加载成功的信息
        print(f"加载模型成功，最佳验证损失: {self.best_val_loss:.4f}，在第{self.best_epoch+1}个epoch")
    
    def save_history(self) -> None:
        """
        保存训练历史记录并生成可视化图表。
        
        该方法执行两个主要操作：
        1. 将训练过程中记录的指标（损失、准确率、学习率等）保存为JSON格式，便于后续分析
        2. 调用plot_history方法生成可视化图表，直观展示训练过程中的指标变化
        
        这些记录对于分析模型训练过程、评估模型性能和调整训练策略非常有价值。
        """
        # 将训练历史记录保存为JSON格式文件
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)  # 将字典序列化为JSON格式并写入文件
        
        # 调用plot_history方法生成可视化图表，直观展示训练过程
        self.plot_history()  # 绘制损失、准确率和学习率等指标的变化曲线
    
    def plot_history(self) -> None:
        """
        绘制训练历史曲线，可视化展示训练过程中的各种指标变化。
        
        该方法创建四个子图，分别展示：
        1. 训练集和验证集的损失变化
        2. 水果类型分类的训练集和验证集准确率
        3. 腐烂状态分类的训练集和验证集准确率
        4. 学习率随训练过程的变化
        
        这些可视化图表有助于：
        - 监控模型训练过程中的收敛情况
        - 发现过拟合或欠拟合问题
        - 评估学习率调度策略的效果
        - 分析不同任务（水果类型和腐烂状态）的学习难度
        """
        # 创建2x2的图表网格，指定图表大小
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2行2列的子图，总大小15x10英寸
        
        # 第一个子图：绘制训练集和验证集的损失曲线
        axs[0, 0].plot(self.history['train_loss'], label='Train Loss')  # 训练损失曲线
        axs[0, 0].plot(self.history['val_loss'], label='Val Loss')      # 验证损失曲线
        axs[0, 0].set_title('Loss')                                     # 设置子图标题
        axs[0, 0].set_xlabel('Epoch')                                   # X轴标签
        axs[0, 0].set_ylabel('Loss')                                    # Y轴标签
        axs[0, 0].legend()                                              # 显示图例
        axs[0, 0].grid(True)                                            # 显示网格线
        
        # 第二个子图：绘制水果类型分类的训练集和验证集准确率曲线
        axs[0, 1].plot(self.history['train_fruit_acc'], label='Train Fruit Acc')  # 训练集水果分类准确率
        axs[0, 1].plot(self.history['val_fruit_acc'], label='Val Fruit Acc')      # 验证集水果分类准确率
        axs[0, 1].set_title('Fruit Classification Accuracy')                      # 设置子图标题
        axs[0, 1].set_xlabel('Epoch')                                            # X轴标签
        axs[0, 1].set_ylabel('Accuracy')                                         # Y轴标签
        axs[0, 1].legend()                                                       # 显示图例
        axs[0, 1].grid(True)                                                     # 显示网格线
        
        # 第三个子图：绘制腐烂状态分类的训练集和验证集准确率曲线
        axs[1, 0].plot(self.history['train_state_acc'], label='Train State Acc')  # 训练集腐烂状态准确率
        axs[1, 0].plot(self.history['val_state_acc'], label='Val State Acc')      # 验证集腐烂状态准确率
        axs[1, 0].set_title('Rot State Accuracy')                                 # 设置子图标题
        axs[1, 0].set_xlabel('Epoch')                                            # X轴标签
        axs[1, 0].set_ylabel('Accuracy')                                         # Y轴标签
        axs[1, 0].legend()                                                       # 显示图例
        axs[1, 0].grid(True)                                                     # 显示网格线
        
        # 第四个子图：绘制学习率变化曲线
        axs[1, 1].plot(self.history['learning_rate'])  # 学习率变化曲线
        axs[1, 1].set_title('Learning Rate')           # 设置子图标题
        axs[1, 1].set_xlabel('Epoch')                  # X轴标签
        axs[1, 1].set_ylabel('Learning Rate')          # Y轴标签
        axs[1, 1].grid(True)                           # 显示网格线
        
        # 自动调整子图之间的间距，使布局更紧凑
        plt.tight_layout()
        
        # 将图表保存为图片文件
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        # 关闭图表，释放资源
        plt.close()


def create_trainer(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                   config: Dict) -> Trainer:
    """
    根据配置创建并初始化训练器实例。
    
    该函数根据提供的配置字典创建一个完整的训练器实例，包括：
    1. 创建多任务损失函数，并配置水果类型和腐烂状态任务的权重
    2. 根据配置选择并初始化适当的优化器（Adam、SGD或AdamW）
    3. 根据配置选择并初始化学习率调度器（阶梯式、余弦退火或基于验证指标的自适应调度）
    4. 自动选择训练设备（GPU或CPU）
    5. 创建并返回配置完成的Trainer实例
    
    Args:
        model (nn.Module): 要训练的模型实例，通常是FruitClassifier实例
        train_loader (DataLoader): 训练数据加载器，提供训练数据的分批迭代
        val_loader (DataLoader): 验证数据加载器，提供验证数据的分批迭代
        config (Dict): 配置字典，包含所有训练相关的参数和设置
        
    Returns:
        Trainer: 完全配置好的训练器实例，可直接用于模型训练
    """
    # 从配置字典中提取训练相关的配置部分
    # 如果不存在，则使用空字典作为默认值
    train_config = config.get('train', {})
    
    # 创建多任务损失函数，用于同时优化水果类型和腐烂状态分类任务
    criterion = MultiTaskLoss(
        # 设置水果类型分类任务的权重，默认为1.0
        fruit_weight=train_config.get('fruit_weight', 1.0),
        # 设置腐烂状态分类任务的权重，默认为1.0
        state_weight=train_config.get('state_weight', 1.0)
    )
    
    # 从配置中获取优化器相关参数
    optimizer_name = train_config.get('optimizer', 'adam').lower()  # 优化器类型，默认为Adam
    lr = train_config.get('learning_rate', 0.001)                  # 学习率，默认为0.001
    weight_decay = train_config.get('weight_decay', 0.0001)         # L2正则化系数，默认为0.0001
    
    # 根据配置的优化器类型创建相应的优化器实例
    if optimizer_name == 'adam':  # Adam优化器
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':  # 随机梯度下降优化器
        momentum = train_config.get('momentum', 0.9)  # 动量参数，默认为0.9
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':  # AdamW优化器（改进版Adam，有更好的权重衰减实现）
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # 不支持的优化器类型
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 从配置中获取学习率调度器类型，默认为余弦退火调度器
    scheduler_name = train_config.get('scheduler', 'cosine').lower()
    
    # 根据配置的调度器类型创建相应的学习率调度器
    if scheduler_name == 'step':  # 阶梯式学习率调度器，每隔一定步数将学习率乘以一个因子
        step_size = train_config.get('step_size', 10)  # 每隔多少个epoch调整学习率，默认为10
        gamma = train_config.get('gamma', 0.1)         # 学习率衰减因子，默认为0.1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':  # 余弦退火学习率调度器，学习率按余弦函数从初始值逐渐减小到接近零
        epochs = train_config.get('epochs', 100)  # 总训练轮数，用于计算学习率衰减周期
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'plateau':  # 基于验证指标的自适应学习率调度器，当指标停止改善时降低学习率
        patience = train_config.get('patience', 5)  # 在降低学习率前等待的轮数，默认为5
        factor = train_config.get('factor', 0.1)    # 学习率降低因子，默认为0.1
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    elif scheduler_name == 'none' or not scheduler_name:  # 不使用学习率调度器
        scheduler = None
    else:  # 不支持的调度器类型
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
    
    # 自动选择可用的训练设备，如果有GPU则使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建并配置训练器实例，传入所有必要的组件和配置
    trainer = Trainer(
        model=model,                # 要训练的模型
        train_loader=train_loader,  # 训练数据加载器
        val_loader=val_loader,      # 验证数据加载器
        criterion=criterion,        # 多任务损失函数
        optimizer=optimizer,        # 优化器
        scheduler=scheduler,        # 学习率调度器
        device=device,              # 训练设备
        config=train_config         # 训练配置
    )
    
    # 返回配置完成的训练器实例
    return trainer
