import time
import numpy as np
import matplotlib.pyplot as plt

from src.dataloader import DataLoader
from src.spherical_hash import SphericalHashing
from src.metrics import Evaluator
from src.config import Config

def main():
    # -------------------------------------------
    # 1. 数据准备
    # -------------------------------------------
    loader = DataLoader()
    q_feats, db_feats, _, _ = loader.get_data()
    ground_truth = loader.get_ground_truth()
    
    # 结果记录
    results = {} # {bits: {'mAP': x, 'time': y}}
    
    # -------------------------------------------
    # 2. 循环测试不同比特数
    # -------------------------------------------
    for bits in Config.HASH_BITS: # [32, 64, 128]
        print(f"\n{'='*10} Start Experiment: {bits} Bits {'='*10}")
        
        # A. 模型初始化与训练
        model = SphericalHashing(n_bits=bits)
        
        t0 = time.time()
        model.train(db_feats) # 用数据库训练
        
        # B. 编码
        q_codes = model.encode(q_feats)
        db_codes = model.encode(db_feats)
        
        # C. 检索与评估
        # 记录检索时间（从拿到特征到算出排序结果）
        # 这里把 编码+距离计算+排序 视为广义的检索过程
        
        mAP = Evaluator.evaluate(q_codes, db_codes, ground_truth)
        
        t_cost = time.time() - t0
        avg_time = t_cost / Config.QUERY_SIZE # 平均每个Query的时间
        
        print(f" >> [Result] {bits} Bits -> mAP: {mAP:.4f}")
        print(f" >> [Time] Total: {t_cost:.2f}s, Avg/Query: {avg_time:.4f}s")
        
        results[bits] = {'mAP': mAP, 'time': avg_time}

    # -------------------------------------------
    # 3. 结果汇总与可视化
    # -------------------------------------------
    print("\n" + "="*30)
    print("Final Summary")
    print("="*30)
    print(f"{'Bits':<10} | {'mAP':<10} | {'Time/Query(s)':<15}")
    print("-" * 40)
    
    bits_list = []
    map_list = []
    
    for bits in Config.HASH_BITS:
        res = results[bits]
        print(f"{bits:<10} | {res['mAP']:.4f}     | {res['time']:.5f}")
        bits_list.append(bits)
        map_list.append(res['mAP'])
        
    # 简单画个图 (可选)
    plt.figure(figsize=(8, 5))
    plt.plot(bits_list, map_list, marker='o', linestyle='-', color='b')
    plt.title('mAP vs Number of Bits (Spherical Hashing)')
    plt.xlabel('Number of Bits')
    plt.ylabel('mAP')
    plt.grid(True)
    
    # 保存图片到 results 文件夹 (需要先创建文件夹，或者直接保存在当前目录)
    plt.savefig('result_map_curve.png')
    print("\nResult plot saved as 'result_map_curve.png'")

if __name__ == "__main__":
    main()
# import time
# import time
# import numpy as np
# from src.dataloader import DataLoader
# from src.config import Config

# def main():
#     # 1. 初始化数据加载器
#     loader = DataLoader()
    
#     # 2. 获取数据
#     q_feats, db_feats, q_labels, db_labels = loader.get_data()
    
#     print("-" * 30)
#     print("Data Loaded Successfully:")
#     print(f"Query Features: {q_feats.shape}")     # 应为 (1000, 768)
#     print(f"Database Features: {db_feats.shape}") # 应为 (15000, 768)
#     print(f"Query Labels: {q_labels.shape}")       # 应为 (1000, 38)
#     print("-" * 30)
    
#     # 3. 计算并验证 Ground Truth
#     start_time = time.time()
#     ground_truth = loader.get_ground_truth()
#     end_time = time.time()
    
#     print(f"Ground Truth Calculation Time: {end_time - start_time:.4f}s")
#     print(f"GT Shape: {ground_truth.shape}") # 应为 (1000, 15000)
    
#     # 简单的统计检查
#     avg_rel = np.mean(np.sum(ground_truth, axis=1))
#     print(f"Average relevant images per query: {avg_rel:.2f}")

# if __name__ == "__main__":
#     main()