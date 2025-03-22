"""训练器模块

这个模块实现了模型训练的核心功能，包括：
1. 训练循环的实现
2. 验证循环的实现
3. 训练过程中的指标计算和记录
4. 模型保存和加载

主要类：
- Trainer: 模型训练器，管理整个训练过程
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# 导入混合精度训练所需的模块
from torch.amp import autocast
from torch.cuda.amp import GradScaler
# 导入数据增强模块
from trains.augmentation import MixUp

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目其他模块
from models.model import FruitClassifier, MultiTaskLoss, calculate_metrics
from config import load_config


class Trainer:
    """
    模型训练器
    
    管理模型的训练、验证和测试过程
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: torch.device = None, config: Dict = None):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 待训练的模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            criterion (nn.Module): 损失函数
            optimizer (torch.optim.Optimizer): 优化器
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器
            device (torch.device): 训练设备
            config (Dict): 配置字典
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config is not None else {}
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 混合精度训练设置
        self.use_amp = self.config.get('use_amp', False)
        if self.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler()
            print("启用混合精度训练 (AMP)")
        else:
            self.scaler = None
            if self.use_amp and not torch.cuda.is_available():
                print("警告: 混合精度训练需要CUDA支持，但当前设备不支持CUDA。已禁用混合精度训练。")
                self.use_amp = False
                
        # 批次级别的数据增强设置
        self.use_mixup = False
        self.mixup = None
        
        # 检查配置中是否启用MixUp
        if 'augmentation' in self.config and 'mixup' in self.config['augmentation']:
            mixup_config = self.config['augmentation']['mixup']
            if mixup_config.get('enabled', False):
                self.use_mixup = True
                self.mixup = MixUp(alpha=mixup_config.get('alpha', 0.2))
                print(f"启用MixUp数据增强 (alpha={mixup_config.get('alpha', 0.2)})")
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_fruit_acc': [],
            'val_fruit_acc': [],
            'train_state_acc': [],
            'val_state_acc': [],
            'learning_rate': []
        }
        
        # 设置保存路径
        self.save_dir = self.config.get('save_dir', os.path.join(project_root, 'checkpoints'))
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 记录最佳模型性能
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            Dict[str, float]: 训练指标
        """
        self.model.train()
        running_loss = 0.0
        all_fruit_preds = []
        all_state_preds = []
        all_fruit_targets = []
        all_state_targets = []
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, fruit_targets, state_targets) in enumerate(pbar):
            # 将数据移动到设备
            images = images.to(self.device)
            fruit_targets = fruit_targets.to(self.device)
            state_targets = state_targets.to(self.device)
            
            # 应用批次级别的数据增强（如MixUp）
            if self.use_mixup:
                # MixUp增强需要在模型前向传播前应用
                images, mixed_fruit_targets, mixed_state_targets = self.mixup((images, fruit_targets, state_targets))
                # 使用混合后的目标
                fruit_targets = mixed_fruit_targets
                state_targets = mixed_state_targets
            
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
            
            # 记录损失
            running_loss += loss.item()
            
            # 计算预测结果
            fruit_preds = torch.argmax(fruit_logits, dim=1)
            state_preds = torch.argmax(state_logits, dim=1)
            
            # 收集预测和目标用于计算指标
            # 如果使用了MixUp，我们需要还原原始标签进行评估
            if self.use_mixup:
                # 对于MixUp，我们只收集预测结果，但不使用混合后的标签进行评估
                # 我们将在批次数据中获取原始标签
                original_fruit_targets = torch.argmax(fruit_targets, dim=1) if len(fruit_targets.shape) > 1 else fruit_targets
                original_state_targets = torch.argmax(state_targets, dim=1) if len(state_targets.shape) > 1 else state_targets
                all_fruit_preds.append(fruit_preds.cpu().numpy())
                all_state_preds.append(state_preds.cpu().numpy())
                all_fruit_targets.append(original_fruit_targets.cpu().numpy())
                all_state_targets.append(original_state_targets.cpu().numpy())
            else:
                all_fruit_preds.append(fruit_preds.cpu().numpy())
                all_state_preds.append(state_preds.cpu().numpy())
                all_fruit_targets.append(fruit_targets.cpu().numpy())
                all_state_targets.append(state_targets.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'fruit_loss': f"{loss_dict['fruit_loss'].item():.4f}",
                'state_loss': f"{loss_dict['state_loss'].item():.4f}"
            })
        
        # 计算平均损失
        epoch_loss = running_loss / len(self.train_loader)
        
        # 合并所有批次的预测和目标
        all_fruit_preds = np.concatenate(all_fruit_preds)
        all_state_preds = np.concatenate(all_state_preds)
        all_fruit_targets = np.concatenate(all_fruit_targets)
        all_state_targets = np.concatenate(all_state_targets)
        
        # 计算准确率
        fruit_correct = (all_fruit_preds == all_fruit_targets).sum()
        state_correct = (all_state_preds == all_state_targets).sum()
        fruit_acc = fruit_correct / len(all_fruit_targets)
        state_acc = state_correct / len(all_state_targets)
        
        # 返回训练指标
        return {
            'loss': epoch_loss,
            'fruit_acc': fruit_acc,
            'state_acc': state_acc
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch
        
        Returns:
            Dict[str, float]: 验证指标
        """
        self.model.eval()
        running_loss = 0.0
        all_fruit_preds = []
        all_state_preds = []
        all_fruit_targets = []
        all_state_targets = []
        
        # 不计算梯度
        with torch.no_grad():
            # 使用tqdm显示进度条
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, (images, fruit_targets, state_targets) in enumerate(pbar):
                # 将数据移动到设备
                images = images.to(self.device)
                fruit_targets = fruit_targets.to(self.device)
                state_targets = state_targets.to(self.device)
                
                # 前向传播（验证时也使用混合精度，但不需要梯度缩放）
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        fruit_logits, state_logits = self.model(images)
                        loss, loss_dict = self.criterion(fruit_logits, state_logits, fruit_targets, state_targets)
                else:
                    fruit_logits, state_logits = self.model(images)
                    loss, loss_dict = self.criterion(fruit_logits, state_logits, fruit_targets, state_targets)
                
                # 记录损失
                running_loss += loss.item()
                
                # 计算预测结果
                fruit_preds = torch.argmax(fruit_logits, dim=1)
                state_preds = torch.argmax(state_logits, dim=1)
                
                # 收集预测和目标用于计算指标
                all_fruit_preds.append(fruit_preds.cpu().numpy())
                all_state_preds.append(state_preds.cpu().numpy())
                all_fruit_targets.append(fruit_targets.cpu().numpy())
                all_state_targets.append(state_targets.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'fruit_loss': f"{loss_dict['fruit_loss'].item():.4f}",
                    'state_loss': f"{loss_dict['state_loss'].item():.4f}"
                })
        
        # 计算平均损失
        epoch_loss = running_loss / len(self.val_loader)
        
        # 合并所有批次的预测和目标
        all_fruit_preds = np.concatenate(all_fruit_preds)
        all_state_preds = np.concatenate(all_state_preds)
        all_fruit_targets = np.concatenate(all_fruit_targets)
        all_state_targets = np.concatenate(all_state_targets)
        
        # 计算准确率
        fruit_correct = (all_fruit_preds == all_fruit_targets).sum()
        state_correct = (all_state_preds == all_state_targets).sum()
        fruit_acc = fruit_correct / len(all_fruit_targets)
        state_acc = state_correct / len(all_state_targets)
        
        # 返回验证指标
        return {
            'loss': epoch_loss,
            'fruit_acc': fruit_acc,
            'state_acc': state_acc
        }
    
    def train(self, num_epochs: int, save_best: bool = True, save_latest: bool = True, save_freq: int = 5, early_stopping: int = 0) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            num_epochs (int): 训练轮数
            save_best (bool): 是否保存最佳模型
            save_latest (bool): 是否在每个epoch后保存最新模型
            save_freq (int): 每多少个epoch保存一次检查点，0表示不保存周期性检查点
            early_stopping (int): 早停轮数，0表示不使用早停
            
        Returns:
            Dict[str, List[float]]: 训练历史记录
        """
        print(f"开始训练，共{num_epochs}个epoch")
        print(f"使用设备: {self.device}")
        print(f"保存设置: 最佳模型={save_best}, 最新模型={save_latest}, 检查点频率={save_freq}轮")
        
        # 早停计数器
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证一个epoch
            val_metrics = self.validate_epoch()
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录训练历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_fruit_acc'].append(train_metrics['fruit_acc'])
            self.history['val_fruit_acc'].append(val_metrics['fruit_acc'])
            self.history['train_state_acc'].append(train_metrics['state_acc'])
            self.history['val_state_acc'].append(val_metrics['state_acc'])
            self.history['learning_rate'].append(current_lr)
            
            # 计算epoch耗时
            epoch_time = time.time() - start_time
            
            # 打印训练信息
            print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - "
                  f"Train Loss: {train_metrics['loss']:.4f} - "
                  f"Val Loss: {val_metrics['loss']:.4f} - "
                  f"Train Fruit Acc: {train_metrics['fruit_acc']:.4f} - "
                  f"Val Fruit Acc: {val_metrics['fruit_acc']:.4f} - "
                  f"Train State Acc: {train_metrics['state_acc']:.4f} - "
                  f"Val State Acc: {val_metrics['state_acc']:.4f} - "
                  f"LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.save_model(os.path.join(self.save_dir, 'best_model.pth'))
                print(f"保存最佳模型，验证损失: {self.best_val_loss:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # 保存周期性检查点
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{epoch+1}.pth')
                self.save_model(checkpoint_path)
                print(f"保存第{epoch+1}轮检查点")
                
            # 保存最新模型
            if save_latest:
                self.save_model(os.path.join(self.save_dir, 'latest_model.pth'))
                print(f"保存最新模型 (第{epoch+1}轮)")
            
            # 早停
            if early_stopping > 0 and no_improve_count >= early_stopping:
                print(f"早停：{early_stopping}个epoch内验证损失没有改善")
                break
        
        # 保存最后一个epoch的模型
        self.save_model(os.path.join(self.save_dir, 'last_model.pth'))
        
        # 保存训练历史
        self.save_history()
        
        print(f"训练完成，最佳验证损失: {self.best_val_loss:.4f}，在第{self.best_epoch+1}个epoch")
        
        return self.history
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path (str): 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'epoch': self.best_epoch,
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path (str): 模型路径
        """
        # 检查文件是否存在
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return
        
        # 加载模型
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['epoch']
        
        print(f"加载模型成功，最佳验证损失: {self.best_val_loss:.4f}，在第{self.best_epoch+1}个epoch")
    
    def save_history(self) -> None:
        """
        保存训练历史
        """
        # 保存为JSON
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        
        # 绘制损失曲线
        self.plot_history()
    
    def plot_history(self) -> None:
        """
        绘制训练历史曲线
        """
        # 创建图表
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制损失曲线
        axs[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axs[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # 绘制水果分类准确率曲线
        axs[0, 1].plot(self.history['train_fruit_acc'], label='Train Fruit Acc')
        axs[0, 1].plot(self.history['val_fruit_acc'], label='Val Fruit Acc')
        axs[0, 1].set_title('Fruit Classification Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # 绘制腐烂状态准确率曲线
        axs[1, 0].plot(self.history['train_state_acc'], label='Train State Acc')
        axs[1, 0].plot(self.history['val_state_acc'], label='Val State Acc')
        axs[1, 0].set_title('Rot State Accuracy')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # 绘制学习率曲线
        axs[1, 1].plot(self.history['learning_rate'])
        axs[1, 1].set_title('Learning Rate')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Learning Rate')
        axs[1, 1].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()


def create_trainer(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                  config: Dict) -> Trainer:
    """
    创建训练器
    
    Args:
        model (nn.Module): 模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        config (Dict): 配置字典
        
    Returns:
        Trainer: 训练器实例
    """
    # 获取训练配置
    train_config = config.get('train', {})
    
    # 创建损失函数
    criterion = MultiTaskLoss(
        fruit_weight=train_config.get('fruit_weight', 1.0),
        state_weight=train_config.get('state_weight', 1.0)
    )
    
    # 创建优化器
    optimizer_name = train_config.get('optimizer', 'adam').lower()
    lr = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = train_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 创建学习率调度器
    scheduler_name = train_config.get('scheduler', 'cosine').lower()
    
    if scheduler_name == 'step':
        step_size = train_config.get('step_size', 10)
        gamma = train_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        epochs = train_config.get('epochs', 100)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'plateau':
        patience = train_config.get('patience', 5)
        factor = train_config.get('factor', 0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    elif scheduler_name == 'none' or not scheduler_name:
        scheduler = None
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config
    )
    
    return trainer
