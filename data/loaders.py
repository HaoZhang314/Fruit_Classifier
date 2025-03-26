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

# 导入必要的库
import os  # 用于文件路径操作
import sys  # 用于修改Python路径
import pandas as pd  # 用于读取和处理CSV数据
import torch  # PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader  # PyTorch数据加载工具
from torchvision import transforms  # 图像变换工具
from PIL import Image  # 图像处理库
from typing import Dict, Tuple, List, Optional, Callable  # 类型注解

# 添加项目根目录到Python路径，确保可以导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
project_root = os.path.dirname(current_dir)  # 获取项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)  # 将项目根目录添加到Python路径中

# 导入config包，用于加载配置文件
from config import load_config

# 导入配置文件，获取全局配置参数
config_path = os.path.join(project_root, 'config', 'config.yaml')
config = load_config(config_path)  # 加载YAML配置文件

# 导入数据增强模块，用于图像预处理和增强
from trains.augmentation import get_augmentation_transforms

def get_transforms(mode: str = 'train', img_size: int = config['model']['img_size']) -> transforms.Compose:
    """
    获取数据预处理和增强的变换
    
    Args:
        mode (str): 模式，'train'表示训练模式（包含数据增强），'test'表示测试模式（只进行标准化）
        img_size (int): 训练图像大小，从配置文件中获取默认值
    
    Returns:
        transforms.Compose: 组合的变换操作，包含多个图像处理步骤
    """
    # 使用augmentation模块中的高级数据增强功能
    # 从配置文件中获取数据增强设置，如果配置文件中没有相关设置，则使用默认值
    augmentation_config = config.get('augmentation', {
        'horizontal_flip': True,  # 水平翻转
        'rotation_angle': 15,     # 旋转角度范围
        'brightness': 0.1,        # 亮度调整范围
        'contrast': 0.1,          # 对比度调整范围
        'saturation': 0.1,        # 饱和度调整范围
        'hue': 0.05,              # 色调调整范围
        'gaussian_noise': 0.05,   # 高斯噪声强度
        'random_erasing': 0.2     # 随机擦除概率
    })
    
    # 调用外部函数获取变换，根据模式返回不同的变换组合
    return get_augmentation_transforms(augmentation_config, mode, img_size)


class FruitDataset(Dataset):
    """
    水果图像数据集
    
    加载水果图像和对应的标签（水果类型和腐烂状态）
    继承自PyTorch的Dataset类，实现自定义数据集
    """
    
    def __init__(self, csv_file: str, transform: Optional[Callable] = None, split: str = 'train', custom_indices: Optional[List[int]] = None):
        """
        初始化数据集
        
        Args:
            csv_file (str): 包含图像路径和标签的CSV文件路径
            transform (Callable, optional): 图像变换函数，用于数据预处理和增强
            split (str): 数据集划分，'train'表示训练集，'test'表示测试集
            custom_indices (List[int], optional): 自定义的索引列表，用于从指定split中进一步选择样本，实现数据集的进一步划分
        """
        # 读取CSV文件，包含图像路径和标签信息
        self.data_frame = pd.read_csv(csv_file)
        
        # 仅保留指定split的数据（训练集或测试集）
        if split in ['train', 'test']:
            self.data_frame = self.data_frame[self.data_frame['split'] == split]
        
        # 如果提供了自定义索引，则进一步筛选数据
        # 这在创建训练/验证集划分时很有用
        if custom_indices is not None:
            # 确保索引不超出范围，避免索引错误
            valid_indices = [i for i in custom_indices if i < len(self.data_frame)]
            self.data_frame = self.data_frame.iloc[valid_indices].reset_index(drop=True)
        
        # 保存图像变换函数
        self.transform = transform
        
        # 创建水果类型标签映射（从字符串到索引）
        self.fruit_types = sorted(self.data_frame['fruit_type'].unique())
        self.fruit_type_to_idx = {fruit: idx for idx, fruit in enumerate(self.fruit_types)}
        
        # 创建腐烂状态标签映射（从字符串到索引）
        self.states = sorted(self.data_frame['state'].unique())
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        
        # 打印数据集信息，便于调试
        print(f"加载了 {len(self.data_frame)} 个{split}样本")
        print(f"水果类型: {self.fruit_types}")
        print(f"腐烂状态: {self.states}")
    
    def __len__(self) -> int:
        """
        返回数据集大小
        
        实现Dataset接口的__len__方法，用于确定数据集的大小
        这使得可以使用len(dataset)获取样本数量
        
        Returns:
            int: 数据集中样本的数量
        """
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        获取指定索引的样本
        
        实现Dataset接口的__getitem__方法，用于获取指定索引的样本
        这使得可以使用dataset[idx]获取单个样本
        
        Args:
            idx (int): 样本索引，可以是整数或PyTorch张量
            
        Returns:
            Tuple[torch.Tensor, int, int]: (图像张量, 水果类型标签, 腐烂状态标签)
        """
        # 如果索引是PyTorch张量，则转换为Python列表
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 从数据框中获取图像路径和标签信息
        img_path = self.data_frame.iloc[idx]['image_path']
        fruit_type = self.data_frame.iloc[idx]['fruit_type']
        state = self.data_frame.iloc[idx]['state']
        
        # 将字符串标签转换为数值索引，用于模型训练
        fruit_type_idx = self.fruit_type_to_idx[fruit_type]
        state_idx = self.state_to_idx[state]
        
        # 加载图像，处理可能的错误
        try:
            # 打开图像并转换为RGB格式（确保3通道）
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 如果图像加载失败，打印错误信息并创建一个黑色图像作为替代
            print(f"无法加载图像 {img_path}: {e}")
            # 创建默认大小的黑色图像
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用图像变换（如果提供）
        if self.transform:
            image = self.transform(image)
        
        # 返回图像张量和两种标签
        return image, fruit_type_idx, state_idx
    
    def get_class_names(self) -> Tuple[List[str], List[str]]:
        """
        获取类别名称
        
        提供一个方法来获取所有类别的名称，用于结果解释和可视化
        
        Returns:
            Tuple[List[str], List[str]]: (水果类型列表, 腐烂状态列表)
        """
        # 返回水果类型和腐烂状态的名称列表
        return self.fruit_types, self.states


class FruitDataLoader:
    """
    水果数据加载器
    
    封装了PyTorch的DataLoader创建过程，提供了便捷的接口来加载训练集、验证集和测试集
    处理数据集的划分、变换应用和批次加载等操作
    """
    
    def __init__(self, csv_file: str,
                 batch_size: int = config['device']['batch_size'],
                 img_size: int = config['model']['img_size'], 
                 num_workers: int = config['device']['num_workers'], 
                 shuffle: bool = True):
        """
        初始化数据加载器
        
        Args:
            csv_file (str): 包含图像路径和标签的CSV文件路径，包含所有数据集信息
            batch_size (int): 批次大小，从配置文件中获取默认值
            img_size (int): 图像大小，从配置文件中获取默认值
            num_workers (int): 数据加载的工作线程数，用于并行加载数据
            shuffle (bool): 是否打乱数据，训练时通常设为True
        """
        # 保存初始化参数
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # 创建训练和测试的变换，分别用于不同的数据集
        # 训练变换包含数据增强，测试变换只包含标准化
        self.train_transform = get_transforms(mode='train', img_size=img_size)
        self.test_transform = get_transforms(mode='test', img_size=img_size)
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        获取训练集和验证集的DataLoader
        
        在训练阶段，只使用训练集数据，并从中划分出一部分作为验证集
        测试集数据仅用于最终模型评估
        
        Returns:
            Tuple[DataLoader, DataLoader]: (训练集DataLoader, 验证集DataLoader)
        """
        # 加载训练集数据，从CSV文件中读取所有数据
        train_df = pd.read_csv(self.csv_file)
        # 筛选出标记为训练集的数据
        train_df = train_df[train_df['split'] == 'train']
        
        # 打印训练集数据分布，便于调试
        print(f"加载了 {len(train_df)} 个训练样本")
        
        # 按水果类型和腐烂状态分层，划分训练集和验证集
        # 导入sklearn的train_test_split函数用于数据集划分
        from sklearn.model_selection import train_test_split
        
        # 使用80%的训练数据进行训练，20%用于验证
        # 使用分层抽样确保训练集和验证集中各类别的比例一致
        train_indices, val_indices = train_test_split(
            range(len(train_df)),  # 所有训练样本的索引
            test_size=0.2,  # 验证集占比20%
            random_state=42,  # 固定随机种子，确保结果可复现
            stratify=train_df[['fruit_type', 'state']]  # 按水果类型和腐烂状态进行分层
        )
        
        # 创建训练集，使用训练变换（包含数据增强）
        train_dataset = FruitDataset(
            csv_file=self.csv_file,  # 使用相同的CSV文件
            transform=self.train_transform,  # 应用训练变换
            split='train',  # 只使用训练集数据
            custom_indices=train_indices  # 使用划分出的训练集索引
        )
        
        # 创建验证集，使用测试变换（不包含数据增强）
        val_dataset = FruitDataset(
            csv_file=self.csv_file,  # 使用相同的CSV文件
            transform=self.test_transform,  # 验证集使用测试变换，不需要数据增强
            split='train',  # 仍然从训练集中选择数据
            custom_indices=val_indices  # 使用划分出的验证集索引
        )
        
        # 打印数据集大小信息
        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        
        # 创建训练集DataLoader
        train_loader = DataLoader(
            train_dataset,  # 训练数据集
            batch_size=self.batch_size,  # 批次大小
            shuffle=self.shuffle,  # 是否打乱数据
            num_workers=self.num_workers,  # 数据加载的工作线程数
            pin_memory=True  # 将数据加载到固定内存中，加速GPU训练
        )
        
        # 创建验证集DataLoader
        val_loader = DataLoader(
            val_dataset,  # 验证数据集
            batch_size=self.batch_size,  # 批次大小
            shuffle=False,  # 验证集不需要打乱数据
            num_workers=self.num_workers,  # 数据加载的工作线程数
            pin_memory=True  # 将数据加载到固定内存中，加速GPU训练
        )
        
        # 返回训练集和验证集的DataLoader
        return train_loader, val_loader
    
    def get_test_loader(self) -> DataLoader:
        """
        获取测试集的DataLoader
        
        用于模型评估阶段，加载独立的测试集数据
        测试集数据不参与训练，只用于最终模型性能评估
        
        Returns:
            DataLoader: 测试集DataLoader
        """
        # 加载测试集数据，从CSV文件中读取所有数据
        test_df = pd.read_csv(self.csv_file)
        # 筛选出标记为测试集的数据
        test_df = test_df[test_df['split'] == 'test']
        
        # 打印测试集数据分布，便于调试
        print(f"加载了 {len(test_df)} 个测试样本")
        
        # 创建测试集，使用测试变换（不包含数据增强）
        test_dataset = FruitDataset(
            csv_file=self.csv_file,  # 使用相同的CSV文件
            transform=self.test_transform,  # 应用测试变换
            split='test'  # 只使用测试集数据
        )
        
        # 打印测试集大小信息
        print(f"测试集大小: {len(test_dataset)}")
        
        # 创建测试集DataLoader
        test_loader = DataLoader(
            test_dataset,  # 测试数据集
            batch_size=self.batch_size,  # 批次大小
            shuffle=False,  # 测试集不需要打乱数据，保持顺序便于分析
            num_workers=self.num_workers,  # 数据加载的工作线程数
            pin_memory=True  # 将数据加载到固定内存中，加速GPU处理
        )
        
        # 返回测试集DataLoader
        return test_loader
        
    def get_class_info(self) -> Tuple[List[str], List[str]]:
        """
        获取类别信息
        
        创建一个临时数据集实例，用于获取数据集中的类别信息
        这对于模型输出的解释和可视化很有用
        
        Returns:
            Tuple[List[str], List[str]]: (水果类型列表, 腐烂状态列表)
        """
        # 创建一个临时数据集来获取类别信息，不需要应用变换
        temp_dataset = FruitDataset(csv_file=self.csv_file, transform=None)
        # 调用数据集的方法获取类别名称
        return temp_dataset.get_class_names()


def get_data_loaders(config: Dict, mode: str = 'train') -> Tuple[DataLoader, DataLoader, Tuple[List[str], List[str]]]:
    """
    根据配置创建数据加载器
    
    这是模块的主要入口函数，根据配置和模式创建适当的数据加载器
    提供了统一的接口，方便训练和评估代码调用
    
    Args:
        config (Dict): 配置字典，包含数据加载相关参数，如批次大小、图像大小等
        mode (str): 模式，'train'表示训练模式，'eval'表示评估模式
        
    Returns:
        Tuple[DataLoader, DataLoader, Tuple[List[str], List[str]]]: 
            训练模式: (训练集DataLoader, 验证集DataLoader, (水果类型列表, 腐烂状态列表))
            评估模式: (None, 测试集DataLoader, (水果类型列表, 腐烂状态列表))
    """
    # 获取项目根目录，用于构建文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 构建CSV文件的完整路径，从配置中获取相对路径
    csv_file = os.path.join(
        project_root,
        config['data'].get('processed_dir', 'data'),  # 获取数据目录，默认为'data'
        config['data']['dataset_csv']  # 获取CSV文件名
    )
    
    # 打印使用的数据集文件路径
    print(f"使用数据集文件: {csv_file}")
    
    # 创建FruitDataLoader实例，传入配置参数
    loader = FruitDataLoader(
        csv_file=csv_file,  # CSV文件路径
        batch_size=config['device']['batch_size'],  # 批次大小
        img_size=config['model']['img_size'],  # 图像大小
        num_workers=config['device']['num_workers'],  # 工作线程数
        shuffle=True  # 训练时打乱数据
    )
    
    # 获取类别信息（水果类型和腐烂状态）
    class_info = loader.get_class_info()
    
    # 根据模式返回不同的数据加载器
    if mode == 'eval':
        # 评估模式：只加载测试集数据
        test_loader = loader.get_test_loader()
        # 返回None作为训练集加载器，测试集加载器和类别信息
        return None, test_loader, class_info
    else:
        # 训练模式：加载训练集和验证集数据
        train_loader, val_loader = loader.get_loaders()
        # 返回训练集加载器、验证集加载器和类别信息
        return train_loader, val_loader, class_info


if __name__ == '__main__':
    """
    模块的主执行入口
    
    当直接运行此文件时，执行以下代码块
    用于测试数据加载功能是否正常工作
    """
    # 获取项目根目录，用于构建文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 加载配置文件，获取数据加载参数
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    config = load_config(config_path)
    
    # 设置CSV文件路径，从配置中获取相对路径
    csv_file = os.path.join(project_root, 
                           config['data']['processed_dir'],  # 数据目录 
                           config['data']['dataset_csv'])    # CSV文件名
    
    # 创建数据加载器实例
    loader = FruitDataLoader(
        csv_file=csv_file,  # CSV文件路径
        batch_size=config['device']['batch_size'],  # 批次大小
        img_size=config['model']['img_size'],  # 图像大小
        num_workers=config['device']['num_workers']  # 工作线程数
    )
    
    # 获取训练集和验证集的DataLoader
    train_loader, test_loader = loader.get_loaders()
    
    # 获取类别信息（水果类型和腐烂状态）
    fruit_types, states = loader.get_class_info()
    print(f"水果类型: {fruit_types}")
    print(f"腐烂状态: {states}")
    
    # 查看一个批次的数据，验证数据加载是否正确
    for images, fruit_type_labels, state_labels in train_loader:
        # 打印批次信息
        print(f"批次大小: {images.shape[0]}")  # 批次中的样本数
        print(f"图像形状: {images.shape}")      # 图像的形状（批次大小，通道数，高度，宽度）
        print(f"水果类型标签: {fruit_type_labels}")  # 水果类型标签
        print(f"腐烂状态标签: {state_labels}")      # 腐烂状态标签
        break  # 只查看第一个批次，避免输出过多信息