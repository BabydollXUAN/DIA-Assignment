import time
import numpy as np
import matplotlib.pyplot as plt

from src.dataloader import DataLoader
from src.spherical_hash import SphericalHashing
from src.metrics import Evaluator
from src.config import Config

def main():
    # 1. 数据准备
    loader = DataLoader()
    q_feats, db_feats, _, _ = loader.get_data()
    ground_truth = loader.get_ground_truth()
    
    results = {} 
    curves = {} # 用于存储曲线数据
    
    # 2. 循环测试
    for bits in Config.HASH_BITS:
        print(f"\n{'='*10} Start Experiment: {bits} Bits {'='*10}")
        model = SphericalHashing(n_bits=bits)
        
        t0 = time.time()
        model.train(db_feats)
        q_codes = model.encode(q_feats)
        db_codes = model.encode(db_feats)
        
        # === 关键修改点 ===
        # 这里必须接收 3 个返回值，否则 mAP 会变成一个元组导致报错
        mAP, P_curve, R_curve = Evaluator.evaluate(q_codes, db_codes, ground_truth)
        
        t_cost = time.time() - t0
        avg_time = t_cost / Config.QUERY_SIZE
        
        print(f" >> [Result] {bits} Bits -> mAP: {mAP:.4f}")
        print(f" >> [Time] Total: {t_cost:.2f}s, Avg/Query: {avg_time:.4f}s")
        
        results[bits] = {'mAP': mAP, 'time': avg_time}
        curves[bits] = {'P': P_curve, 'R': R_curve}

    # 3. 结果汇总
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
        
    # === 画图 1: mAP 曲线 ===
    plt.figure(figsize=(8, 5))
    plt.plot(bits_list, map_list, marker='o', linestyle='-', color='b')
    plt.title('mAP vs Number of Bits')
    plt.xlabel('Number of Bits')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.savefig('result_map_curve.png')
    print("\nSaved result_map_curve.png")

    # === 画图 2 & 3: P/R 曲线 ===
    K_plot = 1000 
    x = np.arange(1, K_plot + 1)
    
    # Precision@K
    plt.figure(figsize=(8, 6))
    for bits in Config.HASH_BITS:
        plt.plot(x, curves[bits]['P'][:K_plot], label=f'{bits} bits')
    plt.title('Precision@K Curve')
    plt.xlabel('K (Number of Retrieved Images)')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig('curve_precision.png')
    print("Saved curve_precision.png")

    # Recall@K
    plt.figure(figsize=(8, 6))
    for bits in Config.HASH_BITS:
        plt.plot(x, curves[bits]['R'][:K_plot], label=f'{bits} bits')
    plt.title('Recall@K Curve')
    plt.xlabel('K (Number of Retrieved Images)')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig('curve_recall.png')
    print("Saved curve_recall.png")

if __name__ == "__main__":
    main()