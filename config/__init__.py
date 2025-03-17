"""
配置模块

这个模块负责加载和管理项目配置
"""

import yaml

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
