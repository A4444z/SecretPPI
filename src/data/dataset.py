
import os
import pickle
import lmdb
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_cluster import radius_graph
from torch_scatter import scatter_add
from typing import Optional, List, Tuple

from src.utils.geometry import GaussianRBF, get_random_rotation_matrix, apply_rotation

class GlueVAEDataset(Dataset):
    """
    GlueVAE 蛋白质-蛋白质界面数据集类。
    
    该类继承自 PyG 的 Dataset，负责从 LMDB 读取处理好的界面数据，并动态构建几何图结构。
    """
    def __init__(
        self, 
        root: str, 
        split: str = 'train', 
        transform=None, 
        pre_transform=None, 
        lmdb_path: Optional[str] = None
    ):
        """
        参数:
            root: 数据集根目录。
            split: 'train', 'val', 或 'test'。用于决定是否进行数据增强。
            lmdb_path: LMDB 数据库路径。如果为 None，则默认使用 root/processed_lmdb。
        """
        self.lmdb_path = lmdb_path or os.path.join(root, "processed_lmdb")
        self.split = split
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

    def get(self, idx: int) -> Data:
        """获取索引为 idx 的数据样本并构建 PyG Data 对象。"""
        self._connect_db()
        if self._keys is None:
            self._load_keys()
            
        key = self._keys[idx]
        with self._env.begin() as txn:
            byte_data = txn.get(key)
            data_dict = pickle.loads(byte_data)
            
        # 提取基础数组
        pos = torch.from_numpy(data_dict['pos']).float()  # 原子坐标 [N, 3]
        z = torch.from_numpy(data_dict['z']).long()       # 原子序数/元素索引 [N]
        residue_index = torch.from_numpy(data_dict['residue_index']).long()  # 原子所属残基索引 [N]
        res_keys = data_dict['residue_keys']
        meta = data_dict['meta']
        chain_a, chain_b = meta['chains']
        
        # 1. 确定 is_ligand (用于区分受体和配体，0代表受体，1代表配体)
        # 将残基索引映射到对应的链 ID，进而转换为 0/1 标记
        res_to_batch = []
        for rk in res_keys:
            cid = rk[0]  # 残基键的第一个元素是链 ID
            if cid == chain_a:
                res_to_batch.append(0)  # 受体链
            else:
                res_to_batch.append(1)  # 配体链
        
        res_to_batch_tensor = torch.tensor(res_to_batch, dtype=torch.long)
        is_ligand = res_to_batch_tensor[residue_index]  # 每个原子对应的受体/配体标记 [N]
        
        # 2. 数据增强 (随机旋转)
        # 如果是训练集，则对坐标应用随机的 SO(3) 旋转矩阵
        if self.split == 'train':
            rot_mat = get_random_rotation_matrix()
            pos = apply_rotation(pos, rot_mat)
            
        # 3. 界面掩码 (Mask Interface)
        # 计算每个原子到“对方链”的最短距离，如果小于 4.0A 则标记为界面原子
        mask_interface = torch.zeros(pos.size(0), dtype=torch.float)
        
        mask_a = (is_ligand == 0)
        mask_b = (is_ligand == 1)
        
        if mask_a.any() and mask_b.any():
            pos_a = pos[mask_a]
            pos_b = pos[mask_b]
            
            # 计算 A 链原子与 B 链原子之间的成对距离矩阵 [Na, Nb]
            dist_mat = torch.cdist(pos_a, pos_b)
            
            # 对 A 链每个原子，计算到 B 链所有原子的最短距离
            min_dist_a, _ = dist_mat.min(dim=1)  # [Na]
            # 对 B 链每个原子，计算到 A 链所有原子的最短距离
            min_dist_b, _ = dist_mat.min(dim=0)  # [Nb]
            
            # 距离阈值判定 (4.0 Angstrom)
            interface_a = (min_dist_a < 4.0).float()
            interface_b = (min_dist_b < 4.0).float()
            
            mask_interface[mask_a] = interface_a
            mask_interface[mask_b] = interface_b
            
        # 4. 图结构构建 (Edge Construction)
        # 使用半径图算法，连接距离小于 4.5A 的原子对
        edge_index = radius_graph(pos, r=4.5, loop=False)  # [2, E]
        row, col = edge_index
        
        # 计算边的欧氏距离
        diff = pos[row] - pos[col]
        dist = torch.norm(diff, p=2, dim=-1)  # [E]
        
        # 边分类逻辑
        # Type 0: 共价键 (Covalent, 距离 < 1.7A)
        # Type 1: 链内非共价作用 (Intra-chain, 距离 >= 1.7A 且属于同一条链)
        # Type 2: 链间相互作用 (Inter-chain, 属于不同链，这是模型学习的重点)
        
        is_covalent = dist < 1.7
        same_chain = (is_ligand[row] == is_ligand[col])
        
        # 初始化边的 One-hot 类型编码 [E, 3]
        edge_type = torch.zeros(edge_index.size(1), 3, dtype=torch.float)
        
        # 设置 Type 0
        mask_type0 = is_covalent
        edge_type[mask_type0, 0] = 1.0
        
        # 设置 Type 1
        mask_type1 = (~is_covalent) & same_chain
        edge_type[mask_type1, 1] = 1.0
        
        # 设置 Type 2
        mask_type2 = (~is_covalent) & (~same_chain)
        edge_type[mask_type2, 2] = 1.0
        
        # 拼接边特征：[类型One-hot, 距离RBF编码]
        rbf_feat = self.rbf(dist)  # [E, 16]
        edge_attr = torch.cat([edge_type, rbf_feat], dim=-1)  # [E, 19]
        
        # 5. 手性/向量特征注入 (v_init) - 为 PaiNN 提供方向感知
        # 计算公式：v_i = sum(r_j - r_i)，其中 j 是 i 的所有共价邻居
        
        mask_cov = mask_type0
        row_cov = row[mask_cov]
        col_cov = col[mask_cov]
        
        # 计算相对向量 (j 指向 i)
        vec_diff = pos[row_cov] - pos[col_cov]  # [Ec, 3]
        
        # 将相对向量累加到目标原子节点上
        vector_features = scatter_add(vec_diff, col_cov, dim=0, dim_size=pos.size(0))  # [N, 3]
        
        # 注意：由于向量特征是从 (可能已旋转的) 坐标计算得来的，它天然满足 SE(3) 等变性
        
        # 构建最终的 Data 对象
        data = Data(
            x=z,                          # 元素特征
            pos=pos,                      # 原子坐标
            edge_index=edge_index,        # 图连接索引
            edge_attr=edge_attr,          # 边特征 (类型+距离)
            vector_features=vector_features, # 初始向量特征 (用于手性感知)
            mask_interface=mask_interface,   # 界面标记
            is_ligand=is_ligand,          # 受体/配体标记
            residue_index=residue_index   # 残基索引 (用于后续的残基级 Pooling)
        )
        
        # 存储元数据
        data.pdb_id = meta['pdb_id']
        data.chains = meta['chains']
        
        return data
