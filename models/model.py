"""多任务水果分类模型

这个模块实现基于可选骨干网络的多任务分类模型，用于同时识别水果的类型和腐烂状态。
主要特点：
1. 支持多种骨干网络作为特征提取器（EfficientNet-B4、ResNet50等）
2. 实现两个分支头，分别用于水果类型分类和腐烂状态判断
3. 支持多任务联合训练
4. 提供评估指标计算功能，包括准确率等

主要类：
- FruitClassifier: 水果分类模型，包含特征提取器和两个分类头
- MultiTaskLoss: 多任务联合损失函数，结合水果类型和腐烂状态的分类损失

主要函数：
- create_model: 创建水果分类模型的工厂函数
- calculate_metrics: 计算模型评价指标
"""

# 导入必要的库
import os  # 用于文件和目录操作
import sys  # 用于修改Python路径
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from torchvision import models  # PyTorch预训练模型
from typing import Dict, Tuple, List, Optional, Union, Any  # 类型提示

# 添加项目根目录到Python路径，确保可以导入项目中的其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
project_root = os.path.dirname(current_dir)  # 获取项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)  # 将项目根目录添加到Python路径

# 导入配置模块，用于加载项目配置
from config import load_config

# 加载配置文件，获取模型参数和训练设置
config_path = os.path.join(project_root, 'config', 'config.yaml')  # 配置文件路径
config = load_config(config_path)  # 加载YAML配置文件


class FruitClassifier(nn.Module):
    """
    多任务水果分类模型
    
    基于可选骨干网络实现的多任务分类模型，同时识别水果类型和腐烂状态。
    该模型采用共享特征提取器和双头分类器的架构，实现水果类型和腐烂状态的联合预测。
    支持多种骨干网络作为特征提取器，包括EfficientNet和ResNet系列。
    """
    
    def __init__(self, num_fruit_classes: int, num_state_classes: int, backbone: str = 'efficientnet_b4', pretrained: bool = True):
        """
        初始化水果分类模型
        
        构建多任务分类模型的各个组件，包括特征提取器和两个分类头。
        特征提取器负责从图像中提取高级特征，两个分类头分别负责水果类型和腐烂状态的分类。
        
        Args:
            num_fruit_classes (int): 水果类型的数量，对应分类头的输出维度
            num_state_classes (int): 腐烂状态的数量（通常为2，表示新鲜和腐烂）
            backbone (str): 骨干网络类型，支持 'efficientnet_b3', 'efficientnet_b4', 'resnet50', 'resnet18', 'resnet34', 'resnet101'
            pretrained (bool): 是否使用预训练模型，True表示使用ImageNet预训练权重
        """
        super(FruitClassifier, self).__init__()
        
        # 记录类别数量和骨干网络类型，便于后续使用
        self.num_fruit_classes = num_fruit_classes  # 水果类型的数量
        self.num_state_classes = num_state_classes  # 腐烂状态的数量
        self.backbone = backbone  # 骨干网络类型
        
        # 加载特征提取器（骨干网络）并获取其输出特征维度
        self.feature_extractor, feature_dim = self._create_backbone(backbone, pretrained)
        
        # 移除原始分类器，因为我们将使用自定义的分类头
        # 不同类型的网络有不同的分类器属性名
        if 'efficientnet' in backbone:
            self.feature_extractor.classifier = nn.Identity()  # 用Identity层替换原始分类器
        elif 'resnet' in backbone:
            self.feature_extractor.fc = nn.Identity()  # 用Identity层替换原始全连接层
            
        # 创建水果类型分类头，包含两个全连接层和正则化层
        self.fruit_classifier = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(feature_dim, 512),  # 第一个全连接层，降维到512
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.2),  # 再次应用dropout
            nn.Linear(512, num_fruit_classes)  # 输出层，维度为水果类型数量
        )
        
        # 创建腐烂状态分类头，结构与水果类型分类头相同
        self.state_classifier = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(feature_dim, 512),  # 第一个全连接层，降维到512
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.2),  # 再次应用dropout
            nn.Linear(512, num_state_classes)  # 输出层，维度为腐烂状态数量
        )
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        创建骨干网络（特征提取器）
        
        根据指定的骨干网络类型，创建相应的预训练模型作为特征提取器。
        支持多种常用的卷积神经网络架构，如EfficientNet和ResNet系列。
        
        Args:
            backbone (str): 骨干网络类型，指定要使用的预训练模型
            pretrained (bool): 是否使用预训练模型，True则加载ImageNet预训练权重
            
        Returns:
            Tuple[nn.Module, int]: 返回两个值：
                - 创建的骨干网络模型
                - 骨干网络输出特征的维度，用于构建分类头
                
        Raises:
            ValueError: 当指定的骨干网络类型不受支持时抛出
        """
        # 设置预训练权重，如果pretrained为True，则使用ImageNet预训练权重
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        # 根据指定的骨干网络类型创建相应的模型
        if backbone == 'efficientnet_b3':
            model = models.efficientnet_b3(weights=weights)  # 创建EfficientNet-B3模型
            feature_dim = model.classifier[1].in_features  # 获取分类器输入特征维度
        elif backbone == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=weights)  # 创建EfficientNet-B4模型
            feature_dim = model.classifier[1].in_features  # 获取分类器输入特征维度
        elif backbone == 'resnet18':
            model = models.resnet18(weights=weights)  # 创建ResNet18模型
            feature_dim = model.fc.in_features  # 获取全连接层输入特征维度
        elif backbone == 'resnet34':
            model = models.resnet34(weights=weights)  # 创建ResNet34模型
            feature_dim = model.fc.in_features  # 获取全连接层输入特征维度
        elif backbone == 'resnet50':
            model = models.resnet50(weights=weights)  # 创建ResNet50模型
            feature_dim = model.fc.in_features  # 获取全连接层输入特征维度
        elif backbone == 'resnet101':
            model = models.resnet101(weights=weights)  # 创建ResNet101模型
            feature_dim = model.fc.in_features  # 获取全连接层输入特征维度
        else:
            # 如果指定了不支持的骨干网络类型，抛出异常
            raise ValueError(f"不支持的骨干网络类型: {backbone}")
            
        return model, feature_dim  # 返回创建的模型和特征维度
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播方法，定义模型的计算流程
        
        该方法实现了模型的前向计算过程：
        1. 通过特征提取器提取图像特征
        2. 将提取的特征分别输入两个分类头
        3. 返回两个分类任务的预测结果
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, 3, height, width]，
                              其中3表示RGB三个通道，height和width是图像的高和宽
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 返回两个张量：
                - 水果类型预测结果（logits），形状为 [batch_size, num_fruit_classes]
                - 腐烂状态预测结果（logits），形状为 [batch_size, num_state_classes]
        """
        # 通过特征提取器（骨干网络）提取图像特征
        # 输入为图像张量，输出为高维特征表示
        features = self.feature_extractor(x)  # 形状为 [batch_size, feature_dim]
        
        # 将提取的特征分别输入两个分类头进行预测
        fruit_logits = self.fruit_classifier(features)  # 水果类型预测结果
        state_logits = self.state_classifier(features)  # 腐烂状态预测结果
        
        # 返回两个任务的预测结果（未经过softmax的logits）
        return fruit_logits, state_logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        进行预测并返回类别索引
        
        与forward方法不同，该方法返回的是预测的类别索引而非logits。
        该方法会将模型设置为评估模式，并禁用梯度计算以提高效率。
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, 3, height, width]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 返回两个张量：
                - 水果类型预测类别索引，形状为 [batch_size]，每个元素是预测的类别索引
                - 腐烂状态预测类别索引，形状为 [batch_size]，每个元素是预测的类别索引（0或1）
        """
        # 将模型设置为评估模式，这会影响某些层的行为（如Dropout和BatchNorm）
        self.eval()
        
        # 使用torch.no_grad()上下文管理器禁用梯度计算，减少内存使用并加速计算
        with torch.no_grad():
            # 调用forward方法获取logits
            fruit_logits, state_logits = self.forward(x)
            
            # 对logits应用argmax操作，获取预测类别的索引
            # dim=1表示在类别维度上取最大值
            fruit_preds = torch.argmax(fruit_logits, dim=1)  # 水果类型预测
            state_preds = torch.argmax(state_logits, dim=1)  # 腐烂状态预测
            
        # 返回预测的类别索引
        return fruit_preds, state_preds


def create_model(num_fruit_classes: int, num_state_classes: int, backbone: str = 'efficientnet_b4', pretrained: bool = True) -> FruitClassifier:
    """
    创建水果分类模型的工厂函数
    
    这是一个工厂函数，用于创建和配置FruitClassifier实例。
    封装了模型创建过程，使得外部代码可以更简洁地创建模型。
    如果将来模型创建逻辑变得复杂，也可以在这里集中处理。
    
    Args:
        num_fruit_classes (int): 水果类型的数量，对应水果类型分类头的输出维度
        num_state_classes (int): 腐烂状态的数量，对应腐烂状态分类头的输出维度（通常为2）
        backbone (str): 骨干网络类型，支持 'efficientnet_b3', 'efficientnet_b4', 'resnet50', 'resnet18', 'resnet34', 'resnet101'
        pretrained (bool): 是否使用预训练模型，默认为True，即使用ImageNet预训练权重
        
    Returns:
        FruitClassifier: 初始化后的水果分类模型实例
    """
    # 创建并返回FruitClassifier实例，传入所有必要的参数
    model = FruitClassifier(num_fruit_classes, num_state_classes, backbone, pretrained)
    return model


class MultiTaskLoss(nn.Module):
    """
    多任务联合损失函数
    
    结合水果类型分类和腐烂状态分类的损失函数。
    该类实现了一个加权的多任务损失函数，可以同时优化水果类型和腐烂状态两个分类任务。
    通过调整两个任务的权重，可以控制模型对不同任务的关注程度。
    """
    
    def __init__(self, fruit_weight: float = 1.0, state_weight: float = 1.0):
        """
        初始化多任务损失函数
        
        设置两个分类任务的损失权重和对应的损失函数。
        默认情况下，两个任务的权重相等，即都为1.0。
        
        Args:
            fruit_weight (float): 水果类型分类损失的权重，默认为1.0
            state_weight (float): 腐烂状态分类损失的权重，默认为1.0
        """
        # 调用父类的初始化方法
        super(MultiTaskLoss, self).__init__()
        
        # 保存每个任务的损失权重
        self.fruit_weight = fruit_weight  # 水果类型分类任务的权重
        self.state_weight = state_weight  # 腐烂状态分类任务的权重
        
        # 创建每个任务的损失函数，这里使用交叉熏损失
        self.fruit_criterion = nn.CrossEntropyLoss()  # 水果类型分类的损失函数
        self.state_criterion = nn.CrossEntropyLoss()  # 腐烂状态分类的损失函数
    
    def forward(self, fruit_logits: torch.Tensor, state_logits: torch.Tensor, 
                fruit_targets: torch.Tensor, state_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务联合损失
        
        该方法实现了多任务学习的损失计算逻辑：
        1. 分别计算水果类型和腐烂状态的分类损失
        2. 根据设定的权重对两个损失进行加权求和
        3. 返回总损失和各个子任务的损失
        
        Args:
            fruit_logits (torch.Tensor): 水果类型预测结果（logits），形状为 [batch_size, num_fruit_classes]
            state_logits (torch.Tensor): 腐烂状态预测结果（logits），形状为 [batch_size, num_state_classes]
            fruit_targets (torch.Tensor): 水果类型真实标签，形状为 [batch_size]
            state_targets (torch.Tensor): 腐烂状态真实标签，形状为 [batch_size]
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 返回两个值：
                - 加权总损失，用于反向传播和参数更新
                - 包含各个子任务损失的字典，用于监控训练过程
        """
        # 计算水果类型分类的交叉熏损失
        fruit_loss = self.fruit_criterion(fruit_logits, fruit_targets)
        
        # 计算腐烂状态分类的交叉熏损失
        state_loss = self.state_criterion(state_logits, state_targets)
        
        # 计算加权总损失，将两个任务的损失按权重线性组合
        total_loss = self.fruit_weight * fruit_loss + self.state_weight * state_loss
        
        # 返回总损失和包含各个子任务损失的字典
        # 这个字典便于在训练过程中监控每个任务的损失变化
        return total_loss, {
            'fruit_loss': fruit_loss,  # 水果类型分类损失
            'state_loss': state_loss,  # 腐烂状态分类损失
            'total_loss': total_loss   # 加权总损失
        }


def calculate_metrics(fruit_preds: torch.Tensor, state_preds: torch.Tensor, 
                       fruit_targets: torch.Tensor, state_targets: torch.Tensor) -> Dict[str, float]:
    """
    计算模型评价指标
    
    该函数计算模型在水果分类和腐烂状态检测两个任务上的性能指标。
    计算的指标包括：
    1. 水果类型分类准确率 - 正确预测水果类型的样本比例
    2. 腐烂状态分类准确率 - 正确预测腐烂状态的样本比例
    3. 联合准确率 - 同时正确预测水果类型和腐烂状态的样本比例
    
    除了准确率外，还可以根据需要扩展实现精确率、召回率和F1分数等指标。
    
    Args:
        fruit_preds (torch.Tensor): 水果类型预测结果，形状为 [batch_size]，包含预测的类别索引
        state_preds (torch.Tensor): 腐烂状态预测结果，形状为 [batch_size]，包含预测的类别索引
        fruit_targets (torch.Tensor): 水果类型真实标签，形状为 [batch_size]
        state_targets (torch.Tensor): 腐烂状态真实标签，形状为 [batch_size]
        
    Returns:
        Dict[str, float]: 包含各个评价指标的字典，键为指标名称，值为对应的指标值
    """
    # 计算水果类型分类准确率
    # 首先计算预测正确的样本数量，然后除以总样本数得到准确率
    fruit_correct = (fruit_preds == fruit_targets).float().sum()  # 预测正确的样本数量
    fruit_accuracy = fruit_correct / fruit_targets.size(0)  # 准确率 = 正确数量 / 总数量
    
    # 计算腐烂状态分类准确率
    # 同样计算预测正确的样本数量，然后除以总样本数
    state_correct = (state_preds == state_targets).float().sum()  # 预测正确的样本数量
    state_accuracy = state_correct / state_targets.size(0)  # 准确率 = 正确数量 / 总数量
    
    # 计算两个任务都正确的比例（联合准确率）
    # 使用逻辑与运算符(&)找出两个任务都预测正确的样本
    both_correct = ((fruit_preds == fruit_targets) & (state_preds == state_targets)).float().sum()
    both_accuracy = both_correct / fruit_targets.size(0)  # 联合准确率 = 两个任务都正确的数量 / 总数量
    
    # 返回包含所有计算指标的字典
    # 使用item()方法将PyTorch张量转换为Python标量
    return {
        'fruit_accuracy': fruit_accuracy.item(),  # 水果类型分类准确率
        'state_accuracy': state_accuracy.item(),  # 腐烂状态分类准确率
        'both_accuracy': both_accuracy.item()     # 联合准确率（两个任务都正确的比例）
    }


if __name__ == '__main__':
    """
    模型测试主函数
    
    该部分代码用于测试模型的功能和性能，包括：
    1. 测试不同骨干网络的模型创建
    2. 测试模型的前向传播
    3. 测试多任务损失函数的计算
    4. 测试模型的预测功能
    5. 测试评价指标的计算
    
    这是一个独立的测试脚本，可以直接运行该文件来验证模型的正确性。
    """
    # 确定运行设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试不同的骨干网络，验证模型的灵活性
    backbones = ['efficientnet_b3', 'efficientnet_b4', 'resnet50']
    
    # 遍历不同的骨干网络，创建并测试模型
    for backbone in backbones:
        print(f"\n测试骨干网络: {backbone}")
        # 创建模型实例，设置水果类别数为3，腐烂状态数为2
        model = create_model(num_fruit_classes=3, num_state_classes=2, backbone=backbone, pretrained=True)
        # 将模型移动到指定设备（GPU或CPU）
        model = model.to(device)
        print(f"模型创建成功: {model.__class__.__name__} with {backbone}")
    
    # 生成一个随机测试批次，用于验证模型的前向传播
    batch_size = 4  # 批次大小
    img_size = config['model']['img_size']  # 从配置中获取图像尺寸
    # 创建随机输入张量，形状为[batch_size, 3, img_size, img_size]，3表示RGB三通道
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # 执行前向传播，获取两个任务的预测结果
    fruit_logits, state_logits = model(x)
    print(f"水果类型预测形状: {fruit_logits.shape}")  # 应为[batch_size, num_fruit_classes]
    print(f"腐烂状态预测形状: {state_logits.shape}")  # 应为[batch_size, num_state_classes]
    
    # 创建多任务损失函数实例，设置两个任务的权重均为1.0
    criterion = MultiTaskLoss(fruit_weight=1.0, state_weight=1.0)
    
    # 生成随机测试标签，用于计算损失
    # 水果类型标签范围为0-2（共3类）
    fruit_targets = torch.randint(0, 3, (batch_size,)).to(device)
    # 腐烂状态标签范围为0-1（共2类）
    state_targets = torch.randint(0, 2, (batch_size,)).to(device)
    
    # 计算多任务损失
    total_loss, loss_dict = criterion(fruit_logits, state_logits, fruit_targets, state_targets)
    # 打印损失信息
    print(f"总损失: {total_loss.item():.4f}")
    print(f"水果类型损失: {loss_dict['fruit_loss'].item():.4f}")
    print(f"腐烂状态损失: {loss_dict['state_loss'].item():.4f}")
    
    # 使用predict方法进行预测，获取类别索引而非logits
    fruit_preds, state_preds = model.predict(x)
    print(f"水果类型预测: {fruit_preds}")  # 预测的水果类型索引
    print(f"腐烂状态预测: {state_preds}")  # 预测的腐烂状态索引
    
    # 计算评价指标
    metrics = calculate_metrics(fruit_preds, state_preds, fruit_targets, state_targets)
    # 打印各项评价指标
    print(f"水果类型准确率: {metrics['fruit_accuracy']:.4f}")
    print(f"腐烂状态准确率: {metrics['state_accuracy']:.4f}")
    print(f"两个任务都正确的比例: {metrics['both_accuracy']:.4f}")