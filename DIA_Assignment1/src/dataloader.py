import numpy as np
from src.config import Config

class DataLoader:
    def __init__(self):
        print(f"Loading data from {Config.DATA_PATH} ...")
        self.data = np.load(Config.DATA_PATH, allow_pickle=True)
        
        # 加载原始数据 [cite: 26, 27]
        self.raw_feats = self.data['arr_0']  # (16000, 768)
        self.raw_labels = self.data['arr_1'] # (16000,) 可能是对象数组或压缩格式
        
        # 预处理数据
        self._preprocess()
        
    def _preprocess(self):
        """
        划分 Query 和 Database，并将标签转换为二进制矩阵形式
        """
        # 1. 划分特征 [cite: 12]
        # 前1000张为查询图像，后15000张为数据库图像
        self.query_feats = self.raw_feats[:Config.QUERY_SIZE]
        self.db_feats = self.raw_feats[Config.QUERY_SIZE:]
        
        # 2. 处理标签
        # 注意：arr_1 的形状是 (16000,)，通常这意味着它是一个包含列表的对象数组
        # 或者是一个已经是 (16000, 38) 的矩阵。我们需要做一个兼容处理。
        
        # 尝试直接判断形状，如果不是 (N, 38) 则需要转换
        if self.raw_labels.ndim == 1 or self.raw_labels.shape[1] != Config.NUM_CLASSES:
            print("Converting labels to multi-hot binary matrix...")
            self.label_matrix = self._convert_to_multihot(self.raw_labels)
        else:
            self.label_matrix = self.raw_labels.astype(np.float32)

        # 划分标签矩阵
        self.query_labels = self.label_matrix[:Config.QUERY_SIZE]
        self.db_labels = self.label_matrix[Config.QUERY_SIZE:]
        
    def _convert_to_multihot(self, labels):
        """
        将标签列表转换为 (N, 38) 的二进制矩阵
        """
        matrix = np.zeros((Config.TOTAL_IMG, Config.NUM_CLASSES), dtype=np.int32)
        for i, tags in enumerate(labels):
            # 假设 tags 是一个包含类别索引的列表，如 [0, 5, 12]
            # 如果 tags 本身就是 binary vector 则无需此步，视具体数据格式而定
            # 这里为了健壮性，假设它是索引列表
            if isinstance(tags, (list, np.ndarray)):
                 for t in tags:
                     if t < Config.NUM_CLASSES:
                         matrix[i, t] = 1
        return matrix

    def get_ground_truth(self):
        """
        计算相关性矩阵 Ground Truth (GT)
        定义：如果两张图像具有某个相同的标签，则它们相关 
        
        Returns:
            gt (numpy.ndarray): shape (1000, 15000), 值为 0 或 1
        """
        print("Calculating Ground Truth matrix...")
        # 利用矩阵乘法计算交集： (N_q, 38) @ (38, N_db) -> (N_q, N_db)
        # 结果矩阵中，元素 > 0 表示至少有一个共同标签
        dot_product = np.dot(self.query_labels, self.db_labels.T)
        
        # 二值化：大于0设为1 (True)，否则为0 (False)
        gt = (dot_product > 0).astype(np.int32)
        
        print(f"GT Matrix shape: {gt.shape}, Relevant pairs: {np.sum(gt)}")
        return gt

    def get_data(self):
        return self.query_feats, self.db_feats, self.query_labels, self.db_labels