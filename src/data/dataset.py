
import os
import pickle
import lmdb
import torch
import numpy as np
import json
import hashlib
from torch_geometric.data import Dataset, Data
from torch_cluster import radius_graph, knn_graph
from torch_scatter import scatter_add
from typing import Optional, List, Tuple, Dict

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
        max_num_neighbors: int = 0,
        num_fps_points: int = 5,
        exclude_pdb_json: Optional[str] = None,
        random_rotation: bool = True,
        max_samples: Optional[int] = None,
        cutoff_radius: float = 8.0,
        split_ratio: Optional[Dict[str, float]] = None
    ):
        """
        参数:
            root: 数据集根目录。
            split: 'train', 'val', 或 'test'。决定数据增强和数据分区。
            lmdb_path: LMDB 数据库路径。如果为 None，则默认使用 root/processed_lmdb。
            max_atoms: 触发补丁采样的最大原子数，默认1000。
            patch_radius: 补丁采样的半径阈值，单位Å，默认15.0。
            max_num_neighbors: 每个节点的最大邻居数，防止边数爆炸，默认0。
            num_fps_points: 使用最远点采样(FPS)生成的候选中心数量，默认5。
            exclude_pdb_json: 包含需要排除的PDB ID的JSON文件路径（如CASF-2016）。
            max_samples: 最多加载多少个样本，None 表示全部加载。
            split_ratio: 各split的比例，如 {'train': 0.9, 'val': 0.05, 'test': 0.05}。
        """
        self.lmdb_path = lmdb_path or os.path.join(root, "processed_lmdb")
        self.split = split
        self.max_atoms = max_atoms
        self.patch_radius = patch_radius
        self.max_num_neighbors = max_num_neighbors
        self.num_fps_points = num_fps_points
        self.max_samples = max_samples
        self._keys: Optional[List[bytes]] = None
        self._env: Optional[lmdb.Environment] = None

        # Deterministic train/val/test split ratios
        self.split_ratio = split_ratio or {'train': 0.9, 'val': 0.05, 'test': 0.05}

        # 加载需要排除的PDB ID
        self.exclude_pdb_ids = set()
        if exclude_pdb_json is not None and os.path.exists(exclude_pdb_json):
            with open(exclude_pdb_json, 'r') as f:
                exclude_data = json.load(f)
                if 'all_pdb_ids' in exclude_data:
                    self.exclude_pdb_ids = set(pdb_id.lower() for pdb_id in exclude_data['all_pdb_ids'])
                    print(f"已加载 {len(self.exclude_pdb_ids)} 个需排除的PDB ID")

        self.random_rotation = random_rotation

        # 用于维护每个样本的采样状态 (capped to prevent memory leak)
        self._sample_states = {}
        self._max_sample_states = 10000

        self.cutoff_radius = cutoff_radius

        # 几何计算工具：高斯径向基函数 (RBF)
        self.rbf = GaussianRBF(n_rbf=16, cutoff=self.cutoff_radius, start=0.0)

        super().__init__(root, transform, pre_transform)
    
    def _process(self):
        """
        覆盖并禁用 PyG 默认的 _process 逻辑。
        因为我们的数据在 LMDB 中直接读取，不需要 PyG 在 processed/ 下生成无用的 .pt 文件。
        这彻底根除了多卡并发读写 NFS 导致的 _pickle.UnpicklingError！
        """
        pass

    def __getstate__(self):
        """
        这个方法极其重要！
        当 PyTorch DataLoader 使用多进程 (num_workers > 0) 时，
        它会把 Dataset 序列化发给子进程。
        如果不把 _env 置为 None，子进程会共享主进程的 LMDB 句柄导致死锁！
        """
        state = self.__dict__.copy()
        state['_env'] = None  # 强制子进程自己重新初始化 LMDB 连接
        return state

    @staticmethod
    def _pdb_id_from_key(key: bytes) -> str:
        """Extract PDB ID from LMDB key (format: 'pdb_id|chainA-chainB')."""
        try:
            return key.decode('utf-8').split('|')[0].lower()
        except Exception:
            return key.hex()

    def _assign_split(self, pdb_id: str) -> str:
        """Deterministic hash-based split assignment by PDB ID.
        All chain-pairs from the same PDB go to the same split."""
        h = int(hashlib.md5(pdb_id.encode()).hexdigest(), 16) % 10000
        h_norm = h / 10000.0
        train_end = self.split_ratio.get('train', 0.9)
        val_end = train_end + self.split_ratio.get('val', 0.05)
        if h_norm < train_end:
            return 'train'
        elif h_norm < val_end:
            return 'val'
        else:
            return 'test'

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
        import os
        current_pid = os.getpid()
        
        # 🚨 核心修复：如果环境未初始化，或者发现自己是子进程（PID 变了）
        if getattr(self, '_env', None) is None or getattr(self, '_env_pid', None) != current_pid:
            import lmdb
            
            # (可选) 如果子进程继承了旧环境，我们不关闭它（防止影响其他进程），直接覆盖
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,       # 抵御 NFS 锁崩溃
                readahead=False,  # 防止预读导致内存泄漏
                meminit=False,    # 加速读取
                max_readers=1024  # 增大并发上限
            )
            self._env_pid = current_pid  # 记住当前环境是谁开的
    
    def _load_keys(self):
        """从数据库中加载 Keys，按 PDB ID 哈希进行 train/val/test 分区。"""
        self._connect_db()

        # Cache filename includes split ratio to auto-invalidate on ratio change
        ratio_tag = f"{self.split_ratio.get('train',0.9):.2f}_{self.split_ratio.get('val',0.05):.2f}"
        cache_path = os.path.join(self.lmdb_path, f"keys_cache_{self.split}_{ratio_tag}.pkl")

        # Fast path: load from cache
        if self._keys is None and self.max_samples is None and os.path.exists(cache_path):
            print(f"[CACHE HIT] Loading {self.split} keys from {cache_path}")
            with open(cache_path, 'rb') as f:
                self._keys = pickle.load(f)
            print(f"  Loaded {len(self._keys)} samples for split='{self.split}'")
            return

        # Slow path: scan LMDB and partition by PDB ID hash
        if self._keys is None:
            all_keys = []
            print(f"[CACHE MISS] Scanning LMDB for split='{self.split}' ...")

            with self._env.begin() as txn:
                cursor = txn.cursor()
                for k, _ in cursor:
                    # Blacklist filtering (e.g. CASF-2016)
                    if self.exclude_pdb_ids:
                        try:
                            pdb_id = k.decode('utf-8').split('|')[0].lower()
                            if pdb_id in self.exclude_pdb_ids:
                                continue
                        except Exception:
                            continue

                    all_keys.append(k)

            # Deterministic split: assign each key by PDB ID hash
            self._keys = [k for k in all_keys
                          if self._assign_split(self._pdb_id_from_key(k)) == self.split]

            print(f"  Total LMDB keys (after blacklist): {len(all_keys)}")
            print(f"  Keys assigned to '{self.split}': {len(self._keys)}")

            # Truncate if max_samples is set (debug mode)
            if self.max_samples is not None and len(self._keys) > self.max_samples:
                self._keys = self._keys[:self.max_samples]
                print(f"  Truncated to max_samples={self.max_samples}")

            # Save cache (only in full mode)
            if self.max_samples is None:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self._keys, f)
                print(f"  Cache saved to {cache_path}")
    def len(self) -> int:
        """返回数据集样本总数。"""
        self._load_keys()
        return len(self._keys)
    
    def _farthest_point_sampling(
        self,
        points: torch.Tensor,
        num_points: int
    ) -> torch.Tensor:
        """
        最远点采样(FPS)：从点集中选择最远的num_points个点。
        
        参数:
            points: 点坐标 [N, 3]
            num_points: 要选择的点数
        
        返回:
            selected_indices: 选中点的索引 [num_points]
        """
        N = points.size(0)
        if N <= num_points:
            return torch.arange(N, device=points.device)
        
        selected_indices = []
        # 随机选择第一个点
        idx = torch.randint(0, N, (1,), device=points.device).item()
        selected_indices.append(idx)
        
        # 计算所有点到已选点的最小距离
        dists = torch.norm(points - points[idx:idx+1], dim=1)
        
        for _ in range(1, num_points):
            # 选择距离最远的点
            idx = torch.argmax(dists).item()
            selected_indices.append(idx)
            
            # 更新最小距离
            new_dists = torch.norm(points - points[idx:idx+1], dim=1)
            dists = torch.min(dists, new_dists)
        
        return torch.tensor(selected_indices, device=points.device)
    
    def _get_or_create_sample_state(
        self,
        key: bytes,
        pos: torch.Tensor,
        mask_interface: torch.Tensor
    ) -> Tuple[List[int], int]:
        """
        获取或创建样本的采样状态。
        【工程防线】：加入极端异常样本兜底保护。
        """
        key_str = key.decode('utf-8')
        
        if key_str not in self._sample_states:
            # 第一次访问该样本，生成FPS候选中心
            interface_indices = torch.where(mask_interface == 1)[0]
            
            # ================= 🚨 工程防线 =================
            # 如果没有任何界面原子，退化为从全体原子中选中心
            if len(interface_indices) == 0:
                interface_indices = torch.arange(pos.size(0), device=pos.device)
            # ===============================================
            
            if len(interface_indices) < self.num_fps_points:
                candidate_indices = interface_indices
            else:
                interface_pos = pos[interface_indices]
                fps_indices_in_interface = self._farthest_point_sampling(
                    interface_pos,
                    self.num_fps_points
                )
                candidate_indices = interface_indices[fps_indices_in_interface]
            
            cand_list = candidate_indices.tolist()
            if not cand_list:
                cand_list = [0] if pos.size(0) > 0 else []
                
            self._sample_states[key_str] = {
                'candidate_centers': cand_list,
                'current_index': 0
            }

            # Prevent unbounded memory growth
            if len(self._sample_states) > self._max_sample_states:
                keys_to_remove = list(self._sample_states.keys())[:len(self._sample_states) // 2]
                for rm_key in keys_to_remove:
                    del self._sample_states[rm_key]

        state = self._sample_states[key_str]
        return state['candidate_centers'], state['current_index']
    
    def _update_sample_state(self, key: bytes):
        """
        更新样本的采样状态，移动到下一个候选中心。
        """
        key_str = key.decode('utf-8')
        if key_str in self._sample_states:
            state = self._sample_states[key_str]
            if len(state['candidate_centers']) > 0:
                state['current_index'] = (state['current_index'] + 1) % len(state['candidate_centers'])

    def _dynamic_patch_sampling(
        self,
        key: bytes,
        pos: torch.Tensor,
        z: torch.Tensor,
        residue_index: torch.Tensor,
        is_ligand: torch.Tensor,
        mask_interface: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, int]:
        """
        动态补丁采样：如果原子数超过 max_atoms，则采样以界面为中心的局部补丁。
        
        策略：使用最远点采样(FPS)系统性地覆盖整个界面。
              先采样 Patch，再旋转 Patch，这样计算量更小。
        
        参数:
            key: 数据键，用于维护采样状态
            pos: 原子坐标 [N, 3]
            z: 原子序数 [N]
            residue_index: 残基索引 [N]
            is_ligand: 受体/配体标记 [N]
            mask_interface: 界面标记 [N]
        
        返回:
            (pos, z, residue_index, is_ligand, mask_interface, is_patched, patch_index)
        """
        N = pos.size(0)
        is_patched = False
        patch_index = 0
        
        if N <= self.max_atoms:
            return pos, z, residue_index, is_ligand, mask_interface, is_patched, patch_index
        
        is_patched = True
        
        # 获取或创建该样本的FPS候选中心
        candidate_centers, current_index = self._get_or_create_sample_state(
            key, pos, mask_interface
        )
        
        patch_index = current_index
        center_idx = candidate_centers[current_index]
        
        # 更新采样状态，下次使用下一个中心
        self._update_sample_state(key)
        
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
        
        return pos, z, residue_index, is_ligand, mask_interface, is_patched, patch_index
    
    def _build_optimized_graph(
        self,
        pos: torch.Tensor
    ) -> torch.Tensor:
        """
        优化的图构建（向量化版）：
        策略：先用 KNN 限制最大邻居数，再用半径筛选。
        这样既限制了边数，又保证了物理距离，且全程 GPU/C++ 加速。
        """
        if self.max_num_neighbors <= 0:
            return radius_graph(pos, r=self.cutoff_radius, loop=False)
        
        # 1. 先找最近的 max_num_neighbors (e.g. 32) 个邻居
        # flow='target_to_source' 是 PyG 默认的消息传递方向
        edge_index = knn_graph(pos, k=self.max_num_neighbors, loop=False, flow='target_to_source')
        
        # 2. 计算这些边的距离
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], p=2, dim=1)
        
        # 3. 再次应用半径阈值 (4.5A) 进行裁剪
        mask = dist < self.cutoff_radius
        edge_index = edge_index[:, mask]
        
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
        获取索引为 idx 的数据样本并构建 PyG Data 对象。
        【科学防线】：针对 Train 集进行异常过滤重采样，针对 Val/Test 记录并执行兜底。
        """
        import random
        self._connect_db()
        if self._keys is None:
            self._load_keys()
            
        # ================= 🚨 科学防线与日志系统 =================
        # Codex 建议 1：只在训练集允许重试，验证/测试集遇到直接跳过重采样，走底层兜底
        max_retries = 10 if self.split == 'train' else 1
        
        for attempt in range(max_retries):
            key = self._keys[idx]
            with self._env.begin() as txn:
                byte_data = txn.get(key)
                data_dict = pickle.loads(byte_data)
                
            # 1. 提取基础数组
            pos = torch.from_numpy(data_dict['pos']).float()
            z = torch.from_numpy(data_dict['z']).long()
            residue_index = torch.from_numpy(data_dict['residue_index']).long()
            res_keys = data_dict['residue_keys']
            meta = data_dict['meta']
            chain_a, chain_b = meta['chains']
            
            # 2. 确定 is_ligand
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
            
            is_valid_sample = True
            if mask_a.any() and mask_b.any():
                pos_a = pos[mask_a]
                pos_b = pos[mask_b]
                
                dist_mat = torch.cdist(pos_a, pos_b)
                min_dist_a, _ = dist_mat.min(dim=1)
                min_dist_b, _ = dist_mat.min(dim=0)
                
                interface_a = (min_dist_a < 4.0).float()
                interface_b = (min_dist_b < 4.0).float()
                
                # 判断是不是奇葩样本（例如距离过远）
                if interface_a.sum() == 0 or interface_b.sum() == 0:
                    is_valid_sample = False
                else:
                    mask_interface[mask_a] = interface_a
                    mask_interface[mask_b] = interface_b
            else:
                is_valid_sample = False  # 残缺 PDB
                
            if is_valid_sample:
                break  # 抽到了好数据，直接打破循环！
            else:
                if attempt < max_retries - 1:
                    # 如果还没耗尽重试次数，重新随机抽卡
                    idx = random.randint(0, len(self._keys) - 1)

        # Codex 建议 2：重试耗尽（或验证集不重试）时，打印详细的追溯日志
        if not is_valid_sample:
            pdb_id = meta.get('pdb_id', 'unknown')
            print(f"\n[WARNING] 触发异常数据兜底机制 | Split: {self.split} | PDB: {pdb_id} | Key: {key.decode('utf-8')} | Attempt: {attempt+1}/{max_retries}")
        # ===============================================================

        # 4. Dynamic Patch Sampling
        is_patched = False
        patch_index = 0
        original_num_nodes = pos.size(0)
        
        # 即使是坏数据（验证集，或者训练集连抽 10 次黑底），底层的工程防线也能保证此函数不崩
        pos, z, residue_index, is_ligand, mask_interface, is_patched, patch_index = self._dynamic_patch_sampling(
            key, pos, z, residue_index, is_ligand, mask_interface
        )

        # =========== 核心修复：坐标去中心化 ===========
        if pos.shape[0] > 0:
            pos_center = pos.mean(dim=0, keepdim=True)
            pos = pos - pos_center
        # ============================================
        # 5. 数据增强 (随机旋转) - 在Patch Sampling之后
        if self.split == 'train' and self.random_rotation:
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
        data.patch_index = patch_index
        data.original_num_nodes = original_num_nodes
        
        return data
