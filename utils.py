"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: utils.py
@time: 2025/5/13 21:11
"""
import os
import random
import numpy as np
import torch
import json

def set_seed(seed):
    """设置随机种子确保结果可复现"""
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子
    random.seed(seed)  # Python的random模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 禁用cudnn自动优化



# === 通用 I/O 工具 ===
def read_jsonl(file_path):
    """读取 JSONL 文件，返回列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl(file_path, data_list):
    """写入 JSONL 文件，data_list 是一个包含字典的列表"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# === 配置管理 ===
def load_config(config_path='config.json'):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在，请确保config.json在项目根目录")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 验证配置文件结构
    if 'api_keys' not in config:
        raise ValueError("配置文件缺少 'api_keys' 部分")
    if 'model_configs' not in config:
        raise ValueError("配置文件缺少 'model_configs' 部分")
    
    return config

def get_api_key(model_name, config=None):
    """获取指定模型的API密钥"""
    if config is None:
        config = load_config()
    
    model_name_lower = model_name.lower()
    api_key = config['api_keys'].get(model_name_lower)
    
    if not api_key:
        raise ValueError(f"配置文件中缺少模型 '{model_name}' 的API密钥")
    
    return api_key

def get_model_config(model_name, config=None):
    """获取指定模型的配置"""
    if config is None:
        config = load_config()
    
    model_name_lower = model_name.lower()
    model_config = config['model_configs'].get(model_name_lower)
    
    if not model_config:
        raise ValueError(f"配置文件中缺少模型 '{model_name}' 的配置")
    
    return model_config

# === 向后兼容（保留旧接口） ===
api_key = "sk-xxx-your-real-key-here"  # 作为后备使用