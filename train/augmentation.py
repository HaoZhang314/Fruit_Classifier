"""数据增强模块

这个模块实现了高级数据增强策略，用于提高模型的泛化能力，包括：
1. 自定义数据增强变换
2. 增强策略配置
3. 数据增强可视化

主要函数：
- get_augmentation_transforms: 获取数据增强变换
- visualize_augmentations: 可视化数据增强效果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from typing import Dict, List, Optional, Callable, Tuple

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


class GaussianNoise:
    """
    添加高斯噪声
    """
    
    def __init__(self, mean: float = 0., std: float = 0.1):
        """
        初始化高斯噪声变换
        
        Args:
            mean (float): 噪声均值
            std (float): 噪声标准差
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        应用高斯噪声
        
        Args:
            tensor (torch.Tensor): 输入张量
            
        Returns:
            torch.Tensor: 添加噪声后的张量
        """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class MixUp:
    """
    MixUp数据增强
    
    将两张图像按比例混合，标签也按相同比例混合
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        初始化MixUp变换
        
        Args:
            alpha (float): Beta分布的参数
        """
        self.alpha = alpha
    
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用MixUp变换
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 
                (图像批次, 水果类型标签, 腐烂状态标签)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (混合后的图像, 混合后的水果类型标签, 混合后的腐烂状态标签)
        """
        images, fruit_targets, state_targets = batch
        
        # 生成混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机打乱索引
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[index, :]
        
        # 混合标签使用one-hot编码
        num_fruit_classes = torch.max(fruit_targets).item() + 1
        num_state_classes = torch.max(state_targets).item() + 1
        
        fruit_targets_one_hot = torch.zeros(batch_size, num_fruit_classes)
        fruit_targets_one_hot.scatter_(1, fruit_targets.unsqueeze(1), 1)
        
        state_targets_one_hot = torch.zeros(batch_size, num_state_classes)
        state_targets_one_hot.scatter_(1, state_targets.unsqueeze(1), 1)
        
        mixed_fruit_targets = lam * fruit_targets_one_hot + (1 - lam) * fruit_targets_one_hot[index]
        mixed_state_targets = lam * state_targets_one_hot + (1 - lam) * state_targets_one_hot[index]
        
        return mixed_images, mixed_fruit_targets, mixed_state_targets
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"


def get_augmentation_transforms(config: Dict = None, mode: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    获取数据增强变换
    
    Args:
        config (Dict, optional): 配置字典
        mode (str): 模式，'train'表示训练模式，'test'表示测试模式
        img_size (int): 图像大小
        
    Returns:
        transforms.Compose: 组合的变换操作
    """
    # 默认配置
    if config is None:
        config = {
            'horizontal_flip': True,
            'rotation_angle': 15,
            'brightness': 0.1,
            'contrast': 0.1,
            'saturation': 0.1,
            'hue': 0.05,
            'gaussian_noise': 0.05,
            'random_erasing': 0.2
        }
    
    # 基础变换 - 所有模式都需要的变换
    base_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # 训练模式额外的数据增强
    if mode == 'train':
        train_transforms = []
        
        # 水平翻转
        if config.get('horizontal_flip', True):
            train_transforms.append(transforms.RandomHorizontalFlip())
        
        # 随机旋转
        rotation_angle = config.get('rotation_angle', 15)
        if rotation_angle > 0:
            train_transforms.append(transforms.RandomRotation(rotation_angle))
        
        # 颜色抖动
        brightness = config.get('brightness', 0.1)
        contrast = config.get('contrast', 0.1)
        saturation = config.get('saturation', 0.1)
        hue = config.get('hue', 0.05)
        if any([brightness > 0, contrast > 0, saturation > 0, hue > 0]):
            train_transforms.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast, 
                                      saturation=saturation, hue=hue)
            )
        
        # 随机仿射变换
        train_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        
        # 随机裁剪和调整大小
        train_transforms.append(
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        )
        
        # 随机擦除
        random_erasing = config.get('random_erasing', 0.2)
        if random_erasing > 0:
            base_transforms.append(transforms.RandomErasing(p=random_erasing))
        
        # 高斯噪声
        gaussian_noise = config.get('gaussian_noise', 0.05)
        if gaussian_noise > 0:
            base_transforms.append(GaussianNoise(std=gaussian_noise))
        
        return transforms.Compose(train_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)


def visualize_augmentations(image_path: str, config: Dict = None, num_samples: int = 5, 
                           save_path: str = None) -> None:
    """
    可视化数据增强效果
    
    Args:
        image_path (str): 图像路径
        config (Dict, optional): 配置字典
        num_samples (int): 生成样本数量
        save_path (str, optional): 保存路径
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 获取数据增强变换
    transform = get_augmentation_transforms(config, mode='train')
    
    # 创建图表
    fig, axs = plt.subplots(1, num_samples + 1, figsize=(3 * (num_samples + 1), 3))
    
    # 显示原始图像
    axs[0].imshow(image)
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    # 显示增强后的图像
    for i in range(num_samples):
        # 应用变换
        augmented = transform(image)
        
        # 将张量转换为图像
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = (augmented * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # 显示图像
        axs[i + 1].imshow(augmented)
        axs[i + 1].set_title(f'Augmented {i+1}')
        axs[i + 1].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # 测试数据增强
    import argparse
    
    parser = argparse.ArgumentParser(description='数据增强可视化')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--save', type=str, default=None, help='保存路径')
    parser.add_argument('--samples', type=int, default=5, help='生成样本数量')
    
    args = parser.parse_args()
    
    # 可视化数据增强效果
    visualize_augmentations(args.image, num_samples=args.samples, save_path=args.save)
