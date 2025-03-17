"""水果分类模型包

这个包包含了水果分类的模型实现，主要基于EfficientNet-B4架构。
它提供了多任务学习的能力，可以同时识别水果的类型和腐烂状态。

主要类和函数：
- FruitClassifier: 水果分类模型类
- create_model: 创建模型的工厂函数
- MultiTaskLoss: 多任务联合损失函数
- calculate_metrics: 计算模型评价指标的函数
"""

# 从 efficientnet_model.py 导入主要类和函数
from .efficientnet_model import (
    FruitClassifier,
    create_model,
    MultiTaskLoss,
    calculate_metrics
)

# 定义包的公开接口
__all__ = [
    'FruitClassifier',
    'create_model',
    'MultiTaskLoss',
    'calculate_metrics'
]