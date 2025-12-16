import time
from src.dataloader import DataLoader
from src.config import Config

def main():
    # 1. 初始化数据加载器
    loader = DataLoader()
    
    # 2. 获取数据
    q_feats, db_feats, q_labels, db_labels = loader.get_data()
    
    print("-" * 30)
    print("Data Loaded Successfully:")
    print(f"Query Features: {q_feats.shape}")     # 应为 (1000, 768)
    print(f"Database Features: {db_feats.shape}") # 应为 (15000, 768)
    print(f"Query Labels: {q_labels.shape}")       # 应为 (1000, 38)
    print("-" * 30)
    
    # 3. 计算并验证 Ground Truth
    start_time = time.time()
    ground_truth = loader.get_ground_truth()
    end_time = time.time()
    
    print(f"Ground Truth Calculation Time: {end_time - start_time:.4f}s")
    print(f"GT Shape: {ground_truth.shape}") # 应为 (1000, 15000)
    
    # 简单的统计检查
    avg_rel = np.mean(np.sum(ground_truth, axis=1))
    print(f"Average relevant images per query: {avg_rel:.2f}")

if __name__ == "__main__":
    main()