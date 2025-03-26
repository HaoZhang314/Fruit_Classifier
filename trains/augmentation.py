"""数据增强模块

这个模块实现了高级数据增强策略，用于提高模型的泛化能力，包括：
1. 自定义数据增强变换
2. 增强策略配置
3. 数据增强可视化

主要函数：
- get_augmentation_transforms: 获取数据增强变换
- visualize_augmentations: 可视化数据增强效果
"""

# 导入必要的库
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from typing import Dict, List, Optional, Callable, Tuple

# 添加项目根目录到Python路径，确保可以导入项目中的其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


class GaussianNoise:
    """
    添加高斯噪声
    
    这个类实现了向图像添加随机高斯噪声的变换，可以提高模型对噪声的鲁棒性。
    """
    
    def __init__(self, mean: float = 0., std: float = 0.1):
        """
        初始化高斯噪声变换
        
        Args:
            mean (float): 噪声均值，控制噪声的偏移量
            std (float): 噪声标准差，控制噪声的强度
        """
        self.mean = mean  # 噪声的均值
        self.std = std    # 噪声的标准差
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        应用高斯噪声
        
        将随机生成的高斯噪声添加到输入张量中，噪声强度由std参数控制。
        
        Args:
            tensor (torch.Tensor): 输入图像张量，通常形状为[C, H, W]
            
        Returns:
            torch.Tensor: 添加噪声后的张量，形状与输入相同
        """
        # 生成与输入张量相同大小的随机噪声，并按照指定的均值和标准差进行缩放
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self) -> str:
        """返回该类的字符串表示"""
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class MixUp:
    """
    MixUp数据增强
    
    将两张图像按比例混合，标签也按相同比例混合，是一种有效的正则化技术，
    可以减少过拟合并提高模型在边界样本上的表现。
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        初始化MixUp变换
        
        Args:
            alpha (float): Beta分布的参数，控制混合比例的分布
        """
        self.alpha = alpha  # Beta分布的参数
    
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用MixUp变换
        
        对批次中的图像和标签进行混合，混合比例从Beta(alpha, alpha)分布中采样。
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 
                (图像批次, 水果类型标签, 腐烂状态标签)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (混合后的图像, 混合后的水果类型标签, 混合后的腐烂状态标签)
        """
        # 解包输入批次
        images, fruit_targets, state_targets = batch
        
        # 从Beta分布中生成混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 生成随机打乱的索引，用于确定要混合的图像对
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # 按比例混合图像
        mixed_images = lam * images + (1 - lam) * images[index, :]
        
        # 计算标签的类别数
        num_fruit_classes = torch.max(fruit_targets).item() + 1
        num_state_classes = torch.max(state_targets).item() + 1
        
        # 将标签转换为one-hot编码
        fruit_targets_one_hot = torch.zeros(batch_size, num_fruit_classes)
        fruit_targets_one_hot.scatter_(1, fruit_targets.unsqueeze(1), 1)
        
        state_targets_one_hot = torch.zeros(batch_size, num_state_classes)
        state_targets_one_hot.scatter_(1, state_targets.unsqueeze(1), 1)
        
        # 按相同比例混合标签
        mixed_fruit_targets = lam * fruit_targets_one_hot + (1 - lam) * fruit_targets_one_hot[index]
        mixed_state_targets = lam * state_targets_one_hot + (1 - lam) * state_targets_one_hot[index]
        
        return mixed_images, mixed_fruit_targets, mixed_state_targets
    
    def __repr__(self) -> str:
        """返回该类的字符串表示"""
        return f"{self.__class__.__name__}(alpha={self.alpha})"


def get_augmentation_transforms(config: Dict = None, mode: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    获取数据增强变换
    
    根据配置和模式（训练/测试）创建适当的数据变换管道。
    
    Args:
        config (Dict, optional): 配置字典，包含各种数据增强参数
        mode (str): 模式，'train'表示训练模式（使用增强），'test'表示测试模式（仅基础处理）
        img_size (int): 输出图像大小（宽和高）
        
    Returns:
        transforms.Compose: 组合的变换操作
    """
    # 默认配置参数，如果未提供配置则使用这些值
    if config is None:
        config = {
            'horizontal_flip': True,      # 水平翻转
            'rotation_angle': 15,         # 旋转角度范围
            'brightness': 0.1,            # 亮度变化范围
            'contrast': 0.1,              # 对比度变化范围
            'saturation': 0.1,            # 饱和度变化范围
            'hue': 0.05,                  # 色调变化范围
            'gaussian_noise': 0.05,       # 高斯噪声标准差
            'random_erasing': 0.2         # 随机擦除概率
        }
    
    # 基础变换 - 所有模式（训练和测试）都需要的基本处理
    base_transforms = [
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.ToTensor(),                    # 转换为张量 [0,1]
        # 标准化图像，使用ImageNet的均值和标准差
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # 训练模式额外的数据增强
    if mode == 'train':
        train_transforms = []
        
        # 水平翻转，增加数据多样性
        if config.get('horizontal_flip', True):
            train_transforms.append(transforms.RandomHorizontalFlip())
        
        # 随机旋转，提高旋转不变性
        rotation_angle = config.get('rotation_angle', 15)
        if rotation_angle > 0:
            train_transforms.append(transforms.RandomRotation(rotation_angle))
        
        # 颜色抖动（亮度、对比度、饱和度、色调），提高对光照变化的鲁棒性
        brightness = config.get('brightness', 0.1)
        contrast = config.get('contrast', 0.1)
        saturation = config.get('saturation', 0.1)
        hue = config.get('hue', 0.05)
        if any([brightness > 0, contrast > 0, saturation > 0, hue > 0]):
            train_transforms.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast, 
                                      saturation=saturation, hue=hue)
            )
        
        # 随机仿射变换（平移），提高位置不变性
        train_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        
        # 随机裁剪和调整大小，提高尺度不变性
        train_transforms.append(
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        )
        
        # 随机擦除，模拟遮挡情况
        random_erasing = config.get('random_erasing', 0.2)
        if random_erasing > 0:
            base_transforms.append(transforms.RandomErasing(p=random_erasing))
        
        # 高斯噪声，提高对噪声的鲁棒性
        gaussian_noise = config.get('gaussian_noise', 0.05)
        if gaussian_noise > 0:
            base_transforms.append(GaussianNoise(std=gaussian_noise))
        
        # 组合训练变换和基础变换
        return transforms.Compose(train_transforms + base_transforms)
    else:
        # 测试模式只使用基础变换
        return transforms.Compose(base_transforms)


def visualize_augmentations(image_path: str, config: Dict = None, num_samples: int = 5, 
                           save_path: str = None) -> None:
    """
    可视化数据增强效果
    
    对指定图像应用多次数据增强，并将结果可视化，便于直观评估增强效果。
    
    Args:
        image_path (str): 输入图像的路径
        config (Dict, optional): 数据增强配置
        num_samples (int): 要生成的增强样本数量
        save_path (str, optional): 结果图像的保存路径，如不指定则显示图像
    """
    # 加载原始图像
    image = Image.open(image_path).convert('RGB')
    
    # 获取训练模式的数据增强变换
    transform = get_augmentation_transforms(config, mode='train')
    
    # 创建图表，包含原始图像和num_samples个增强样本
    fig, axs = plt.subplots(1, num_samples + 1, figsize=(3 * (num_samples + 1), 3))
    
    # 显示原始图像
    axs[0].imshow(image)
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    # 生成并显示增强后的图像
    for i in range(num_samples):
        # 应用数据增强变换
        augmented = transform(image)
        
        # 将张量转换回可显示的图像格式
        # 1. 调整通道顺序从[C,H,W]到[H,W,C]
        # 2. 反标准化处理
        # 3. 将像素值限制在[0,1]范围内
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = (augmented * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # 显示增强后的图像
        axs[i + 1].imshow(augmented)
        axs[i + 1].set_title(f'Augmented {i+1}')
        axs[i + 1].axis('off')
    
    # 优化图表布局
    plt.tight_layout()
    
    # 根据参数决定保存或显示图表
    if save_path:
        # 确保保存目录存在
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        # 保存图表到文件
        plt.savefig(save_path)
        plt.close()
    else:
        # 显示图表
        plt.show()


if __name__ == '__main__':
    # 当脚本直接运行时执行的代码块
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='数据增强可视化')
    parser.add_argument('--image', type=str, required=True, help='输入图像的路径')
    parser.add_argument('--save', type=str, default=None, help='结果保存路径')
    parser.add_argument('--samples', type=int, default=5, help='生成的增强样本数量')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用可视化函数
    visualize_augmentations(args.image, num_samples=args.samples, save_path=args.save)
