"""数据加载模块

这个模块主要负责水果图像数据集的加载和处理，提供以下功能：
1. 自定义Dataset类，用于加载图像和标签
2. 数据增强和预处理功能
3. 创建返回(图像, 类型标签, 腐烂标签)的DataLoader

主要类：
- FruitDataset: 自定义Dataset类，加载图像和对应的水果类型、腐烂状态标签
- FruitDataLoader: 封装了DataLoader的创建，提供便捷的数据加载接口

主要函数：
- get_data_loaders: 创建训练集和测试集的DataLoader
- get_transforms: 获取数据增强和预处理变换
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple, List, Optional, Union, Callable

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入config包
from config import load_config

# 导入配置文件
config_path = os.path.join(project_root, 'config', 'config.yaml')
config = load_config(config_path)

def get_transforms(mode: str = 'train', img_size: int = config['model']['img_size']) -> transforms.Compose:
    """
    获取数据预处理和增强的变换
    
    Args:
        mode (str): 模式，'train'表示训练模式，'test'表示测试模式
        img_size (int): 训练图像大小
    
    Returns:
        transforms.Compose: 组合的变换操作
    """
    # 基础变换 - 所有模式都需要的变换
    base_transforms = [
        # 调整图像大小
        transforms.Resize((img_size, img_size)),
        # 转换为Tensor
        transforms.ToTensor(),
        # 标准化，根据模型预训练时的均值和标准差
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # 训练模式额外的数据增强
    if mode == 'train':
        train_transforms = [
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 随机旋转
            transforms.RandomRotation(15),
            # 随机颜色抖动
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            # 随机仿射变换
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        return transforms.Compose(train_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)


class FruitDataset(Dataset):
    """
    水果图像数据集
    
    加载水果图像和对应的标签（水果类型和腐烂状态）
    """
    
    def __init__(self, csv_file: str, transform: Optional[Callable] = None, split: str = 'train'):
        """
        初始化数据集
        
        Args:
            csv_file (str): 包含图像路径和标签的CSV文件路径
            transform (Callable, optional): 图像变换函数
            split (str): 数据集划分，'train'或'test'
        """
        self.data_frame = pd.read_csv(csv_file)
        
        # 仅保留指定split的数据
        if split in ['train', 'test']:
            self.data_frame = self.data_frame[self.data_frame['split'] == split]
        
        self.transform = transform
        
        # 创建标签映射
        self.fruit_types = sorted(self.data_frame['fruit_type'].unique())
        self.fruit_type_to_idx = {fruit: idx for idx, fruit in enumerate(self.fruit_types)}
        
        self.states = sorted(self.data_frame['state'].unique())
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        
        print(f"加载了 {len(self.data_frame)} 个{split}样本")
        print(f"水果类型: {self.fruit_types}")
        print(f"腐烂状态: {self.states}")
    
    def __len__(self) -> int:
        """
        返回数据集大小
        
        Returns:
            int: 数据集中样本的数量
        """
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        获取指定索引的样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Tuple[torch.Tensor, int, int]: (图像张量, 水果类型标签, 腐烂状态标签)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 获取图像路径和标签
        img_path = self.data_frame.iloc[idx]['image_path']
        fruit_type = self.data_frame.iloc[idx]['fruit_type']
        state = self.data_frame.iloc[idx]['state']
        
        # 将标签转换为索引
        fruit_type_idx = self.fruit_type_to_idx[fruit_type]
        state_idx = self.state_to_idx[state]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            # 返回一个空白图像作为替代
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, fruit_type_idx, state_idx
    
    def get_class_names(self) -> Tuple[List[str], List[str]]:
        """
        获取类别名称
        
        Returns:
            Tuple[List[str], List[str]]: (水果类型列表, 腐烂状态列表)
        """
        return self.fruit_types, self.states


class FruitDataLoader:
    """
    水果数据加载器

    """
    
    def __init__(self, csv_file: str,
                 batch_size: int = config['device']['batch_size'],
                 img_size: int = config['model']['img_size'], 
                 num_workers: int = config['device']['num_workers'], 
                 shuffle: bool = True):
        """
        初始化数据加载器
        
        Args:
            csv_file (str): 包含图像路径和标签的CSV文件路径
            batch_size (int): 批次大小
            img_size (int): 图像大小
            num_workers (int): 数据加载的工作线程数
            shuffle (bool): 是否打乱数据
        """
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # 创建训练和测试的变换
        self.train_transform = get_transforms(mode='train', img_size=img_size)
        self.test_transform = get_transforms(mode='test', img_size=img_size)
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        获取训练集和测试集的DataLoader
        
        Returns:
            Tuple[DataLoader, DataLoader]: (训练集DataLoader, 测试集DataLoader)
        """
        # 创建训练集
        train_dataset = FruitDataset(
            csv_file=self.csv_file,
            transform=self.train_transform,
            split='train'
        )
        
        # 创建测试集
        test_dataset = FruitDataset(
            csv_file=self.csv_file,
            transform=self.test_transform,
            split='test'
        )
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 测试集不需要打乱
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def get_class_info(self) -> Tuple[List[str], List[str]]:
        """
        获取类别信息
        
        Returns:
            Tuple[List[str], List[str]]: (水果类型列表, 腐烂状态列表)
        """
        # 创建一个临时数据集来获取类别信息
        temp_dataset = FruitDataset(csv_file=self.csv_file, transform=None)
        return temp_dataset.get_class_names()


def get_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, Tuple[List[str], List[str]]]:
    """
    根据配置创建数据加载器
    
    Args:
        config (Dict): 配置字典，包含数据加载相关参数
        
    Returns:
        Tuple[DataLoader, DataLoader, Tuple[List[str], List[str]]]: 
            (训练集DataLoader, 测试集DataLoader, (水果类型列表, 腐烂状态列表))
    """
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 构建CSV文件的完整路径
    csv_file = os.path.join(
        project_root,
        config['data'].get('processed_dir', 'data'),
        config['data']['dataset_csv']
    )
    
    print(f"使用数据集文件: {csv_file}")
    
    # 创建数据加载器
    loader = FruitDataLoader(
        csv_file=csv_file,
        batch_size=config['device']['batch_size'],
        img_size=config['model']['img_size'],
        num_workers=config['device']['num_workers'],
        shuffle=True
    )
    
    # 获取数据加载器和类别信息
    train_loader, test_loader = loader.get_loaders()
    class_info = loader.get_class_info()
    
    return train_loader, test_loader, class_info


if __name__ == '__main__':
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 加载配置文件
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    config = load_config(config_path)
    
    # 设置CSV文件路径
    csv_file = os.path.join(project_root, 
                           config['data']['processed_dir'], 
                           config['data']['dataset_csv'])
    
    # 创建数据加载器
    loader = FruitDataLoader(
        csv_file=csv_file,
        batch_size=config['device']['batch_size'],
        img_size=config['model']['img_size'],
        num_workers=config['device']['num_workers']
    )
    
    # 获取训练集和测试集的DataLoader
    train_loader, test_loader = loader.get_loaders()
    
    # 获取类别信息
    fruit_types, states = loader.get_class_info()
    print(f"水果类型: {fruit_types}")
    print(f"腐烂状态: {states}")
    
    # 查看一个批次的数据
    for images, fruit_type_labels, state_labels in train_loader:
        print(f"批次大小: {images.shape[0]}")
        print(f"图像形状: {images.shape}")
        print(f"水果类型标签: {fruit_type_labels}")
        print(f"腐烂状态标签: {state_labels}")
        break  # 只查看第一个批次