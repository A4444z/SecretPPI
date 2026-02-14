import torch
from src.data.dataset import GlueVAEDataset
import os

def test_dataset():
    """测试 GlueVAEDataset 的加载和特征构建。"""
    root = "test"
    lmdb_path = "test/test_lmdb"
    
    # 如果需要，创建目录结构 (PyG Dataset 期望 root/processed 目录存在)
    os.makedirs(os.path.join(root, "processed"), exist_ok=True)
    
    # 初始化数据集
    dataset = GlueVAEDataset(root=root, split='train', lmdb_path=lmdb_path)
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        # 获取第一个样本
        data = dataset[0]
        print("\n数据对象 (Data Object):")
        print(data)
        
        # 打印节点特征和统计信息
        print(f"\n节点标量特征 (x): {data.x.shape}, 类型: {data.x.dtype}")
        print(f"原子坐标 (pos): {data.pos.shape}, 类型: {data.pos.dtype}")
        print(f"配体标记 (is_ligand): {data.is_ligand.shape}, 配体原子数: {data.is_ligand.sum()}")
        print(f"界面掩码 (mask_interface): {data.mask_interface.shape}, 界面原子数: {data.mask_interface.sum()}")
        print(f"手性向量特征 (vector_features): {data.vector_features.shape}")
        
        # 打印边特征和统计信息
        print(f"\n边索引 (edge_index): {data.edge_index.shape}")
        print(f"边属性 (edge_attr): {data.edge_attr.shape}")
        
        # 检查边类型的分布 (前三维通常是 Covalent, Intra-chain, Inter-chain)
        edge_types = data.edge_attr[:, :3]
        print(f"共价边 (Covalent edges): {edge_types[:, 0].sum().item()}")
        print(f"链内非共价边 (Intra-chain edges): {edge_types[:, 1].sum().item()}")
        print(f"链间边 (Inter-chain edges): {edge_types[:, 2].sum().item()}")

if __name__ == "__main__":
    test_dataset()
