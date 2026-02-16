
import os
import pickle
import lmdb
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_cluster import radius_graph, knn_graph
from torch_scatter import scatter_add
from typing import Optional, List, Tuple

from src.utils.geometry import GaussianRBF, get_random_rotation_matrix, apply_rotation


class GlueVAEDataset(Dataset):
    """
    GlueVAE 蛋白质-蛋白质界面数据集类（优化版）。
    
    该类继承自 PyG 的 Dataset，负责从 LMDB 读取处理好的界面数据，并动态构建几何图结构。
    
    主要优化：
    - Dynamic Patch Sampling: 对大界面进行随机补丁采样，避免显存溢出
    - 优化图构建: 限制每个节点的最大邻居数，防止边数爆炸
    - 鲁棒的向量特征计算: 处理孤立原子情况
    - 增强的元数据: 便于调试和分析
    """
    
    def __init__(
        self, 
        root: str, 
        split: str = 'train', 
        transform=None, 
        pre_transform=None, 
        lmdb_path: Optional[str] = None,
        max_atoms: int = 1000,
        patch_radius: float = 15.0,
        max_num_neighbors: int = 32
    ):
        """
        参数:
            root: 数据集根目录。
            split: 'train', 'val', 或 'test'。用于决定是否进行数据增强。
            lmdb_path: LMDB 数据库路径。如果为 None，则默认使用 root/processed_lmdb。
            max_atoms: 触发补丁采样的最大原子数，默认1000。
            patch_radius: 补丁采样的半径阈值，单位Å，默认15.0。
            max_num_neighbors: 每个节点的最大邻居数，防止边数爆炸，默认32。
        """
        self.lmdb_path = lmdb_path or os.path.join(root, "processed_lmdb")
        self.split = split
        self.max_atoms = max_atoms
        self.patch_radius = patch_radius
        self.max_num_neighbors = max_num_neighbors
        self._keys: Optional[List[bytes]] = None
        self._env: Optional[lmdb.Environment] = None
        
        # 几何计算工具：高斯径向基函数 (RBF)
        self.rbf = GaussianRBF(n_rbf=16, cutoff=4.5, start=0.0)
        
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self) -> List[str]:
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        return []
    
    def download(self):
        pass
    
    def process(self):
        pass
    
    def _connect_db(self):
        """建立与 LMDB 数据库的连接。"""
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, 
                readonly=True, 
                lock=False, 
                readahead=False, 
                meminit=False
            )
    
    def _load_keys(self):
        """从数据库中加载所有数据的键 (Keys)。"""
        self._connect_db()
        if self._keys is None:
            with self._env.begin() as txn:
                self._keys = [k for k, _ in txn.cursor()]
    
    def len(self) -> int:
        """返回数据集样本总数。"""
        self._load_keys()
        return len(self._keys)
    
    def _dynamic_patch_sampling(
        self,
        pos: torch.Tensor,
        z: torch.Tensor,
        residue_index: torch.Tensor,
        is_ligand: torch.Tensor,
        mask_interface: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        动态补丁采样：如果原子数超过 max_atoms，则采样以界面为中心的局部补丁。
        
        策略：先采样 Patch，再旋转 Patch，这样计算量更小。
        
        参数:
            pos: 原子坐标 [N, 3]
            z: 原子序数 [N]
            residue_index: 残基索引 [N]
            is_ligand: 受体/配体标记 [N]
            mask_interface: 界面标记 [N]
        
        返回:
            (pos, z, residue_index, is_ligand, mask_interface, is_patched)
        """
        N = pos.size(0)
        is_patched = False
        
        if N <= self.max_atoms:
            return pos, z, residue_index, is_ligand, mask_interface, is_patched
        
        is_patched = True
        
        # 在界面核心原子中随机选择一个中心
        interface_indices = torch.where(mask_interface == 1)[0]
        
        if len(interface_indices) == 0:
            # 如果没有界面原子，随机选择一个中心
            center_idx = torch.randint(0, N, (1,)).item()
        else:
            # 随机选择一个界面原子作为中心
            center_idx = interface_indices[torch.randint(0, len(interface_indices), (1,))].item()
        
        # 计算所有原子到中心的距离
        center_pos = pos[center_idx:center_idx+1]  # [1, 3]
        dist_to_center = torch.norm(pos - center_pos, dim=1)  # [N]
        
        # 保留距离小于 patch_radius 的原子
        keep_mask = dist_to_center < self.patch_radius
        
        # 确保至少保留一些原子
        if keep_mask.sum() < 100:
            # 如果太少，放宽阈值
            keep_mask = dist_to_center < self.patch_radius * 1.5
        
        # 裁剪所有数组
        pos = pos[keep_mask]
        z = z[keep_mask]
        residue_index = residue_index[keep_mask]
        is_ligand = is_ligand[keep_mask]
        mask_interface = mask_interface[keep_mask]
        
        return pos, z, residue_index, is_ligand, mask_interface, is_patched
    
    def _build_optimized_graph(
        self,
        pos: torch.Tensor
    ) -> torch.Tensor:
        """
        优化的图构建：使用 radius_graph 但限制每个节点的最大邻居数。
        
        参数:
            pos: 原子坐标 [N, 3]
        
        返回:
            edge_index: 优化后的边索引 [2, E]
        """
        # 首先使用 radius_graph 获取所有4.5Å内的边
        edge_index = radius_graph(pos, r=4.5, loop=False)  # [2, E]
        
        if self.max_num_neighbors <= 0:
            return edge_index
        
        row, col = edge_index
        
        # 计算每条边的距离
        dist = torch.norm(pos[row] - pos[col], dim=1)
        
        # 对每个节点，只保留距离最近的max_num_neighbors个邻居
        unique_nodes, inverse_indices = torch.unique(row, return_inverse=True)
        
        # 创建一个mask来选择要保留的边
        keep_mask = torch.zeros_like(row, dtype=torch.bool)
        
        for node in unique_nodes:
            node_mask = (row == node)
            node_dist = dist[node_mask]
            node_cols = col[node_mask]
            
            # 选择距离最近的max_num_neighbors个
            if len(node_dist) > self.max_num_neighbors:
                _, topk_indices = torch.topk(node_dist, k=self.max_num_neighbors, largest=False)
                keep_mask[node_mask] = torch.isin(
                    torch.arange(len(node_dist), device=node_dist.device),
                    topk_indices
                )
            else:
                keep_mask[node_mask] = True
        
        # 应用mask
        edge_index = edge_index[:, keep_mask]
        
        return edge_index
    
    def _compute_robust_vector_features(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        is_covalent: torch.Tensor
    ) -> torch.Tensor:
        """
        鲁棒的向量特征计算：处理孤立原子情况。
        
        参数:
            pos: 原子坐标 [N, 3]
            edge_index: 边索引 [2, E]
            is_covalent: 共价边标记 [E]
        
        返回:
            vector_features: 向量特征 [N, 3]
        """
        row, col = edge_index
        
        # 只保留共价边
        mask_cov = is_covalent
        row_cov = row[mask_cov]
        col_cov = col[mask_cov]
        
        N = pos.size(0)
        vector_features = torch.zeros(N, 3, device=pos.device)
        
        if len(row_cov) > 0:
            # 计算相对向量 (j 指向 i)
            vec_diff = pos[row_cov] - pos[col_cov]  # [Ec, 3]
            
            # 将相对向量累加到目标原子节点上
            vector_features = scatter_add(vec_diff, col_cov, dim=0, dim_size=N)  # [N, 3]
        
        # 处理孤立原子：添加微小随机向量避免零向量
        zero_mask = (vector_features.norm(dim=1) < 1e-8)
        if zero_mask.any():
            # 添加微小的随机向量
            random_vec = torch.randn(zero_mask.sum(), 3, device=pos.device) * 1e-4
            vector_features[zero_mask] = random_vec
        
        return vector_features
    
    def get(self, idx: int) -> Data:
        """
        获取索引为 idx 的数据样本并构建 PyG Data 对象（优化版）。
        
        处理流程：
        1. 从LMDB读取原始数据
        2. 构建基本标记（is_ligand）
        3. 计算界面掩码
        4. Dynamic Patch Sampling（如需要）
        5. 数据增强（随机旋转）
        6. 优化图构建
        7. 鲁棒的向量特征计算
        8. 构建最终的Data对象
        """
        self._connect_db()
        if self._keys is None:
            self._load_keys()
            
        key = self._keys[idx]
        with self._env.begin() as txn:
            byte_data = txn.get(key)
            data_dict = pickle.loads(byte_data)
            
        # 1. 提取基础数组
        pos = torch.from_numpy(data_dict['pos']).float()  # 原子坐标 [N, 3]
        z = torch.from_numpy(data_dict['z']).long()       # 原子序数/元素索引 [N]
        residue_index = torch.from_numpy(data_dict['residue_index']).long()  # 原子所属残基索引 [N]
        res_keys = data_dict['residue_keys']
        meta = data_dict['meta']
        chain_a, chain_b = meta['chains']
        
        # 2. 确定 is_ligand (用于区分受体和配体，0代表受体，1代表配体)
        res_to_batch = []
        for rk in res_keys:
            cid = rk[0]
            if cid == chain_a:
                res_to_batch.append(0)
            else:
                res_to_batch.append(1)
        
        res_to_batch_tensor = torch.tensor(res_to_batch, dtype=torch.long)
        is_ligand = res_to_batch_tensor[residue_index]
        
        # 3. 界面掩码 (Mask Interface)
        mask_interface = torch.zeros(pos.size(0), dtype=torch.float)
        
        mask_a = (is_ligand == 0)
        mask_b = (is_ligand == 1)
        
        if mask_a.any() and mask_b.any():
            pos_a = pos[mask_a]
            pos_b = pos[mask_b]
            
            dist_mat = torch.cdist(pos_a, pos_b)
            min_dist_a, _ = dist_mat.min(dim=1)
            min_dist_b, _ = dist_mat.min(dim=0)
            
            interface_a = (min_dist_a < 4.0).float()
            interface_b = (min_dist_b < 4.0).float()
            
            mask_interface[mask_a] = interface_a
            mask_interface[mask_b] = interface_b
        
        # 4. Dynamic Patch Sampling（在数据增强之前）
        is_patched = False
        original_num_nodes = pos.size(0)
        
        pos, z, residue_index, is_ligand, mask_interface, is_patched = self._dynamic_patch_sampling(
            pos, z, residue_index, is_ligand, mask_interface
        )
        
        # 5. 数据增强 (随机旋转) - 在Patch Sampling之后
        if self.split == 'train':
            rot_mat = get_random_rotation_matrix()
            pos = apply_rotation(pos, rot_mat)
        
        # 6. 优化图结构构建
        edge_index = self._build_optimized_graph(pos)
        row, col = edge_index
        
        # 7. 计算边的欧氏距离和类型
        diff = pos[row] - pos[col]
        dist = torch.norm(diff, p=2, dim=-1)
        
        is_covalent = dist < 1.7
        same_chain = (is_ligand[row] == is_ligand[col])
        
        edge_type = torch.zeros(edge_index.size(1), 3, dtype=torch.float)
        
        mask_type0 = is_covalent
        edge_type[mask_type0, 0] = 1.0
        
        mask_type1 = (~is_covalent) & same_chain
        edge_type[mask_type1, 1] = 1.0
        
        mask_type2 = (~is_covalent) & (~same_chain)
        edge_type[mask_type2, 2] = 1.0
        
        rbf_feat = self.rbf(dist)
        edge_attr = torch.cat([edge_type, rbf_feat], dim=-1)
        
        # 8. 鲁棒的向量特征计算
        vector_features = self._compute_robust_vector_features(pos, edge_index, is_covalent)
        
        # 9. 构建最终的 Data 对象
        data = Data(
            x=z,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            vector_features=vector_features,
            mask_interface=mask_interface,
            is_ligand=is_ligand,
            residue_index=residue_index,
            num_nodes=pos.size(0)
        )
        
        # 添加增强的元数据
        data.pdb_id = meta['pdb_id']
        data.chains = meta['chains']
        data.is_patched = is_patched
        data.original_num_nodes = original_num_nodes
        
        return data
