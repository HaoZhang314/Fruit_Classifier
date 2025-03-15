import os
import yaml
import pandas as pd
from pathlib import Path

def create_dataset_csv(dataset_dir: str, output_csv: str) -> None:
    """
    创建包含图片路径、水果类型和腐烂状态的CSV文件
    
    Args:
        dataset_dir (str): 数据集根目录的路径
        output_csv (str): 输出CSV文件的路径
    """
    # 存储所有数据的列表
    data = []
    
    # 遍历训练集和测试集
    for split in ['train', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        
        # 确保目录存在
        if not os.path.exists(split_dir):
            continue
            
        # 遍历所有类别目录
        for class_dir in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_dir)
            
            # 跳过非目录文件
            if not os.path.isdir(class_path):
                continue
                
            # 解析类别信息
            # 目录名格式：状态+水果类型，例如：freshapples, rottenbanana
            state = 'fresh' if class_dir.startswith('fresh') else 'rotten'
            fruit_type = class_dir[5:] if class_dir.startswith('fresh') else class_dir[6:]
            
            # 遍历该类别下的所有图片
            for img_file in os.listdir(class_path):
                # 检查是否为图片文件
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # 将数据添加到列表中
                    data.append({
                        'image_path': img_path,
                        'fruit_type': fruit_type,
                        'state': state,
                        'split': split
                    })
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV文件
    df.to_csv(output_csv, index=False)
    print(f'数据集文件已生成在: {output_csv}')
    print(f'总样本数: {len(df)}')
    print('\n样本分布:')
    print(df.groupby(['split', 'fruit_type', 'state']).size().unstack(fill_value=0))

def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置内容
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 加载配置文件
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    config = load_config(config_path)
    
    # 设置路径
    dataset_dir = os.path.join(project_root, config['data']['dataset_dir'])
    output_csv = os.path.join(project_root, 
                            config['data']['processed_dir'], 
                            config['data']['dataset_csv'])
    
    # 生成CSV文件
    create_dataset_csv(dataset_dir, output_csv)
