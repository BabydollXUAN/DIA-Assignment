import os

class Config:
    # 路径设置
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'data.npz')
    
    # 数据集参数 [cite: 12, 13]
    TOTAL_IMG = 16000
    QUERY_SIZE = 1000
    DB_SIZE = 15000
    NUM_CLASSES = 38
    FEATURE_DIM = 768
    
    # 哈希参数 [cite: 14]
    HASH_BITS = [32, 64, 128]