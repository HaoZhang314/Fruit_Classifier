"""数据预处理模块

这个模块主要负责水果图像数据集的预处理工作，包括：
1. 从原始数据集目录创建结构化的CSV文件，记录图片路径、水果类型和腐烂状态
2. 对数据集进行统计和分析，便于后续的数据加载和模型训练

数据集结构假设：
- 数据集分为训练集（train）和测试集（test）
- 每个类别目录命名格式为：[状态][水果类型]，例如：freshapples或rottenbanana

主要函数：
- create_dataset_csv: 创建数据集的CSV索引文件，包含图像路径、水果类型、腐烂状态和数据集划分信息

"""

# 导入必要的库
import os  # 用于文件和目录操作
import sys  # 用于修改Python路径
import pandas as pd  # 用于数据处理和CSV文件操作
from pathlib import Path  # 提供面向对象的文件系统路径操作

# 添加项目根目录到Python路径，确保可以导入项目中的其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
project_root = os.path.dirname(current_dir)  # 获取项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)  # 将项目根目录添加到Python路径

# 导入配置模块，用于加载项目配置
from config import load_config

def create_dataset_csv(dataset_dir: str, output_csv: str) -> None:
    """
    创建包含图片路径、水果类型和腐烂状态的CSV文件
    
    该函数遍历指定目录下的所有图像文件，根据目录结构提取水果类型和腐烂状态信息，
    并生成一个结构化的CSV文件，供数据加载模块使用。
    
    Args:
        dataset_dir (str): 数据集根目录的路径，应包含训练集和测试集子目录
        output_csv (str): 输出CSV文件的路径，将包含所有图像的元数据
    """
    # 初始化一个空列表，用于存储所有图像的元数据
    data = []
    
    # 遍历训练集和测试集目录
    for split in ['train', 'test']:
        # 构建完整的训练集或测试集路径
        split_dir = os.path.join(dataset_dir, split)
        
        # 检查目录是否存在，如果不存在则跳过
        if not os.path.exists(split_dir):
            continue
            
        # 遍历指定目录下的所有类别目录（例如freshapples，rottenbanana等）
        for class_dir in os.listdir(split_dir):
            # 构建类别目录的完整路径
            class_path = os.path.join(split_dir, class_dir)
            
            # 跳过非目录的文件（可能有一些其他文件混在其中）
            if not os.path.isdir(class_path):
                continue
                
            # 解析类别信息，从目录名中提取水果类型和腐烂状态
            # 目录名格式：状态+水果类型，例如：freshapples, rottenbanana
            # 判断是新鲜水果还是腐烂水果
            state = 'fresh' if class_dir.startswith('fresh') else 'rotten'
            # 提取水果类型，根据状态前缀的长度不同进行切片
            fruit_type = class_dir[5:] if class_dir.startswith('fresh') else class_dir[6:]
            
            # 遍历该类别目录下的所有图片文件
            for img_file in os.listdir(class_path):
                # 检查是否为支持的图片文件格式（png、jpg或jpeg）
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 构建图片文件的完整路径
                    img_path = os.path.join(class_path, img_file)
                    
                    # 将图片信息作为一个字典添加到数据列表中
                    data.append({
                        'image_path': img_path,  # 图片的完整路径
                        'fruit_type': fruit_type,  # 水果类型（例如apples、banana）
                        'state': state,  # 腐烂状态（fresh或rotten）
                        'split': split  # 数据集划分（train或test）
                    })
    
    # 将收集到的数据转换为Pandas DataFrame，便于后续处理
    df = pd.DataFrame(data)
    
    # 确保输出目录存在，如果不存在则创建
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # 将DataFrame保存为CSV文件，不包含索引列
    df.to_csv(output_csv, index=False)
    
    # 打印数据集统计信息，包括文件位置、总样本数和样本分布
    print(f'数据集文件已生成在: {output_csv}')
    print(f'总样本数: {len(df)}')
    print('\n样本分布:')
    # 使用pandas的groupby功能对数据进行分组统计，显示不同划分、水果类型和状态的样本数量
    print(df.groupby(['split', 'fruit_type', 'state']).size().unstack(fill_value=0))

if __name__ == '__main__':
    """
    模块的主执行入口
    
    当直接运行此文件时，执行以下代码块
    用于生成数据集的CSV索引文件，供后续数据加载和模型训练使用
    """
    # 获取项目根目录，用于构建文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
    project_root = os.path.dirname(current_dir)  # 获取项目根目录
    
    # 从配置文件中加载路径和其他设置
    config_path = os.path.join(project_root, 'config', 'config.yaml')  # 配置文件路径
    config = load_config(config_path)  # 加载YAML配置文件
    
    # 根据配置设置数据集路径和输出CSV文件路径
    dataset_dir = os.path.join(project_root, config['data']['dataset_dir'])  # 原始数据集目录
    output_csv = os.path.join(project_root, 
                            config['data']['processed_dir'],  # 处理后数据的目录
                            config['data']['dataset_csv'])  # CSV文件名
    
    # 调用函数生成CSV文件，将原始数据集的信息结构化存储
    create_dataset_csv(dataset_dir, output_csv)
