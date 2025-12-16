import numpy as np

class Evaluator:
    @staticmethod
    def calc_hamming_dist(B1, B2):
        # ... (这部分代码保持不变) ...
        q_num = B1.shape[0]
        # bit_len = B1.shape[1] # 如果之前注释掉了，保持原样即可
        dist = np.count_nonzero(B1[:, None, :] != B2[None, :, :], axis=2)
        return dist

    @staticmethod
    def evaluate(q_codes, db_codes, ground_truth):
        """
        返回 mAP 以及平均的 Precision@K 和 Recall@K 曲线
        """
        # 1. 计算距离与排序
        dists = Evaluator.calc_hamming_dist(q_codes, db_codes)
        sorted_indices = np.argsort(dists, axis=1)
        
        nq, ndb = ground_truth.shape
        row_indices = np.arange(nq)[:, None]
        sorted_gt = ground_truth[row_indices, sorted_indices] # (Nq, Ndb)
        
        # 2. 计算 Precision@K 和 Recall@K 曲线
        # hits[i, k] 表示第 i 个查询在前 k 个结果中检索到的相关图像数量
        hits = np.cumsum(sorted_gt, axis=1) 
        
        # 位置索引 1, 2, ..., Ndb
        pos = np.arange(1, ndb + 1).reshape(1, -1)
        
        # Precision 矩阵: (Nq, Ndb)
        precisions = hits / pos
        
        # Recall 矩阵: (Nq, Ndb)
        # 每个查询的总相关数
        total_rel = np.sum(ground_truth, axis=1, keepdims=True)
        total_rel[total_rel == 0] = 1 # 防止除零
        recalls = hits / total_rel
        
        # 计算 mAP
        sum_precisions = np.sum(precisions * sorted_gt, axis=1)
        ap = sum_precisions / total_rel.squeeze()
        mAP = np.mean(ap)
        
        # 计算平均曲线 (所有查询取平均)
        # 为了绘图方便，我们不需要取所有 15000 个点，通常取前 K 个点 (例如 K=1000) 或者全量降采样
        # 这里为了简单，我们计算全量平均，画图时再截取
        mean_precision = np.mean(precisions, axis=0)
        mean_recall = np.mean(recalls, axis=0)
        
        return mAP, mean_precision, mean_recall
        
# import numpy as np

# class Evaluator:
#     @staticmethod
#     def calc_hamming_dist(B1, B2):
#         """
#         计算汉明距离
#         B1: Query codes (Nq, n_bits)
#         B2: DB codes (Ndb, n_bits)
#         Returns: Dist matrix (Nq, Ndb)
#         """
#         # 1. 将 0/1 转换为 -1/1 如果需要，或者直接用异或
#         # 这里使用矩阵乘法技巧加速汉明距离计算：
#         # HammingDist = (L - B1 @ B2.T) / 2  (假设编码是 -1/1)
#         # 如果编码是 0/1：
#         # dist = bit_length - (same_bits)
#         # 简单起见，利用 XOR (异或) 逻辑：
#         # 在 Python 中，利用广播机制可以直接做 != 比较，然后求和
        
#         q_num = B1.shape[0]
#         db_num = B2.shape[0]
#         bit_len = B1.shape[1]
        
#         print(f"Calculating Hamming distances for {bit_len} bits...")
        
#         # 这种广播写法 (Nq, 1, bits) != (1, Ndb, bits) 会生成 (Nq, Ndb, bits) 的大矩阵
#         # 显存/内存消耗 = 1000 * 15000 * 128 / 8 字节 ≈ 240MB，完全没问题
#         dist = np.count_nonzero(B1[:, None, :] != B2[None, :, :], axis=2)
#         return dist

#     @staticmethod
#     def evaluate(q_codes, db_codes, ground_truth, k_values=[50, 100, 200, 500, 1000]):
#         """
#         核心评估函数
#         :param ground_truth: (Nq, Ndb) binary matrix, 1 means relevant
#         """
#         # 1. 计算距离
#         dists = Evaluator.calc_hamming_dist(q_codes, db_codes)
        
#         # 2. 排序 (对每个 Query，按距离从小到大对 DB 索引排序)
#         # argsort 返回的是索引
#         print("Sorting results...")
#         sorted_indices = np.argsort(dists, axis=1)
        
#         # 3. 获取排序后的 GT (重排 ground_truth 矩阵)
#         # 这一步很关键：我们把 GT 矩阵按照检索结果的顺序重排
#         # sorted_gt[i, j] 表示第 i 个查询，第 j 个返回结果是否相关
#         nq, ndb = ground_truth.shape
#         # 利用高级索引重排
#         # row_indices: [[0,0...], [1,1...]]
#         row_indices = np.arange(nq)[:, None]
#         sorted_gt = ground_truth[row_indices, sorted_indices]
        
#         # 4. 计算 mAP
#         print("Calculating mAP...")
#         # Precision @ k = (前k个里相关的数量) / k
#         # 为了计算 AP，我们需要累加相关位置的 Precision
        
#         # 计算累积相关数量 cumsum
#         hits = np.cumsum(sorted_gt, axis=1) # (Nq, Ndb)
        
#         # 计算每个位置的 precision
#         # pos: 1, 2, 3, ..., Ndb
#         pos = np.arange(1, ndb + 1).reshape(1, -1)
#         precision_at_k = hits / pos
        
#         # AP 计算公式: sum(precision * is_relevant) / total_relevant
#         # 注意：只在 is_relevant=1 的位置累加 precision
#         sum_precisions = np.sum(precision_at_k * sorted_gt, axis=1)
        
#         # 每个查询的总相关数
#         total_rel = np.sum(ground_truth, axis=1)
        
#         # 避免除以 0
#         total_rel[total_rel == 0] = 1 
        
#         ap = sum_precisions / total_rel
#         mAP = np.mean(ap)
        
#         return mAP