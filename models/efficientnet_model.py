"""多任务水果分类模型

这个模块实现基于EfficientNet-B4的多任务分类模型，用于同时识别水果的类型和腐烂状态。
主要特点：
1. 使用预训练的EfficientNet-B4作为特征提取器
2. 实现两个分支头，分别用于水果类型分类和腐烂状态判断
3. 支持多任务联合训练

主要类：
- FruitClassifier: 水果分类模型，包含特征提取器和两个分类头
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, List, Optional, Union, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入配置模块
from config import load_config

# 加载配置
config_path = os.path.join(project_root, 'config', 'config.yaml')
config = load_config(config_path)


class FruitClassifier(nn.Module):
    """
    多任务水果分类模型
    
    基于EfficientNet-B4实现的多任务分类模型，同时识别水果类型和腐烂状态
    """
    
    def __init__(self, num_fruit_classes: int, num_state_classes: int, pretrained: bool = True):
        """
        初始化水果分类模型
        
        Args:
            num_fruit_classes (int): 水果类型的数量
            num_state_classes (int): 腐烂状态的数量（通常为2，表示新鲜和腐烂）
            pretrained (bool): 是否使用预训练模型
        """
        super(FruitClassifier, self).__init__()
        
        # 记录类别数量
        self.num_fruit_classes = num_fruit_classes
        self.num_state_classes = num_state_classes
        
        # 加载EfficientNet-B4作为特征提取器
        if pretrained:
            self.feature_extractor = models.efficientnet_b4(weights='IMAGENET1K_V1')
        else:
            self.feature_extractor = models.efficientnet_b4(weights=None)
        
        # 获取特征提取器的输出特征维度
        feature_dim = self.feature_extractor.classifier[1].in_features
        
        # 移除原始分类器
        self.feature_extractor.classifier = nn.Identity()
        
        # 创建水果类型分类头
        self.fruit_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_fruit_classes)
        )
        
        # 创建腐烂状态分类头
        self.state_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_state_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, 3, height, width]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 水果类型预测结果，形状为 [batch_size, num_fruit_classes]
                - 腐烂状态预测结果，形状为 [batch_size, num_state_classes]
        """
        # 提取特征
        features = self.feature_extractor(x)
        
        # 分别通过两个分类头
        fruit_logits = self.fruit_classifier(features)
        state_logits = self.state_classifier(features)
        
        return fruit_logits, state_logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        进行预测
        
        Args:
            x (torch.Tensor): 输入图像张量
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 水果类型预测类别，形状为 [batch_size]
                - 腐烂状态预测类别，形状为 [batch_size]
        """
        self.eval()
        with torch.no_grad():
            fruit_logits, state_logits = self.forward(x)
            fruit_preds = torch.argmax(fruit_logits, dim=1)
            state_preds = torch.argmax(state_logits, dim=1)
        return fruit_preds, state_preds


def create_model(num_fruit_classes: int, num_state_classes: int, pretrained: bool = True) -> FruitClassifier:
    """
    创建水果分类模型
    
    Args:
        num_fruit_classes (int): 水果类型的数量
        num_state_classes (int): 腐烂状态的数量
        pretrained (bool): 是否使用预训练模型
        
    Returns:
        FruitClassifier: 初始化后的模型
    """
    model = FruitClassifier(num_fruit_classes, num_state_classes, pretrained)
    return model


class MultiTaskLoss(nn.Module):
    """
    多任务联合损失函数
    
    结合水果类型分类和腐烂状态分类的损失
    """
    
    def __init__(self, fruit_weight: float = 1.0, state_weight: float = 1.0):
        """
        初始化多任务损失函数
        
        Args:
            fruit_weight (float): 水果类型分类损失的权重
            state_weight (float): 腐烂状态分类损失的权重
        """
        super(MultiTaskLoss, self).__init__()
        self.fruit_weight = fruit_weight
        self.state_weight = state_weight
        self.fruit_criterion = nn.CrossEntropyLoss()
        self.state_criterion = nn.CrossEntropyLoss()
    
    def forward(self, fruit_logits: torch.Tensor, state_logits: torch.Tensor, 
                fruit_targets: torch.Tensor, state_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务损失
        
        Args:
            fruit_logits (torch.Tensor): 水果类型预测结果
            state_logits (torch.Tensor): 腐烂状态预测结果
            fruit_targets (torch.Tensor): 水果类型真实标签
            state_targets (torch.Tensor): 腐烂状态真实标签
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 
                - 总损失
                - 包含各个子任务损失的字典
        """
        fruit_loss = self.fruit_criterion(fruit_logits, fruit_targets)
        state_loss = self.state_criterion(state_logits, state_targets)
        
        # 计算加权总损失
        total_loss = self.fruit_weight * fruit_loss + self.state_weight * state_loss
        
        # 返回总损失和各个子任务损失
        return total_loss, {
            'fruit_loss': fruit_loss,
            'state_loss': state_loss,
            'total_loss': total_loss
        }


def calculate_metrics(fruit_preds: torch.Tensor, state_preds: torch.Tensor, 
                      fruit_targets: torch.Tensor, state_targets: torch.Tensor) -> Dict[str, float]:
    """
    计算模型评价指标
    
    Args:
        fruit_preds (torch.Tensor): 水果类型预测结果
        state_preds (torch.Tensor): 腐烂状态预测结果
        fruit_targets (torch.Tensor): 水果类型真实标签
        state_targets (torch.Tensor): 腐烂状态真实标签
        
    Returns:
        Dict[str, float]: 包含各个指标的字典
    """
    # 计算水果类型分类准确率
    fruit_correct = (fruit_preds == fruit_targets).float().sum()
    fruit_accuracy = fruit_correct / fruit_targets.size(0)
    
    # 计算腐烂状态分类准确率
    state_correct = (state_preds == state_targets).float().sum()
    state_accuracy = state_correct / state_targets.size(0)
    
    # 计算两个任务都正确的比例
    both_correct = ((fruit_preds == fruit_targets) & (state_preds == state_targets)).float().sum()
    both_accuracy = both_correct / fruit_targets.size(0)
    
    return {
        'fruit_accuracy': fruit_accuracy.item(),
        'state_accuracy': state_accuracy.item(),
        'both_accuracy': both_accuracy.item()
    }


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(num_fruit_classes=3, num_state_classes=2, pretrained=True)
    model = model.to(device)
    print(f"模型创建成功: {model.__class__.__name__}")
    
    # 生成一个测试批次
    batch_size = 4
    img_size = config['model']['img_size']
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # 前向传播
    fruit_logits, state_logits = model(x)
    print(f"水果类型预测形状: {fruit_logits.shape}")
    print(f"腐烂状态预测形状: {state_logits.shape}")
    
    # 创建多任务损失函数
    criterion = MultiTaskLoss(fruit_weight=1.0, state_weight=1.0)
    
    # 生成测试标签
    fruit_targets = torch.randint(0, 3, (batch_size,)).to(device)
    state_targets = torch.randint(0, 2, (batch_size,)).to(device)
    
    # 计算损失
    total_loss, loss_dict = criterion(fruit_logits, state_logits, fruit_targets, state_targets)
    print(f"总损失: {total_loss.item():.4f}")
    print(f"水果类型损失: {loss_dict['fruit_loss'].item():.4f}")
    print(f"腐烂状态损失: {loss_dict['state_loss'].item():.4f}")
    
    # 进行预测
    fruit_preds, state_preds = model.predict(x)
    print(f"水果类型预测: {fruit_preds}")
    print(f"腐烂状态预测: {state_preds}")
    
    # 计算指标
    metrics = calculate_metrics(fruit_preds, state_preds, fruit_targets, state_targets)
    print(f"水果类型准确率: {metrics['fruit_accuracy']:.4f}")
    print(f"腐烂状态准确率: {metrics['state_accuracy']:.4f}")
    print(f"两个任务都正确的比例: {metrics['both_accuracy']:.4f}")