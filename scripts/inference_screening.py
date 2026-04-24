import os
import sys

# 获取当前脚本所在目录的上一级（即项目根目录），并加入系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool import fps
from torch_geometric.nn import radius_graph

from torch_cluster import radius_graph
from torch_scatter import scatter_add
from src.utils.geometry import GaussianRBF

# 导入你的模型
from src.models.glue_cmae import GlueVAE

import warnings
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# 忽略 Biopython 解析 PDB 时的烦人警告
warnings.simplefilter('ignore', PDBConstructionWarning)

# 常见元素符号到原子序数的映射表
ELEMENT_TO_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'CL': 17}

def parse_pdb_to_pyg(pdb_path):
    """
    将 PDB 文件解析为 CMAE 模型所需的 PyG Data 对象。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    pos_list = []
    z_list = []
    residue_indices = []

    res_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                # 跳过水分子和异质分子 (HETATM)
                if residue.id[0] != ' ': 
                    continue 
                for atom in residue:
                    pos_list.append(atom.coord)
                    element = atom.element.strip().upper()
                    # 获取原子序数，如果不认识默认给 6 (碳原子)
                    z_list.append(ELEMENT_TO_Z.get(element, 6))
                    residue_indices.append(res_count)
                res_count += 1

    if len(pos_list) == 0:
        raise ValueError(f"从 {pdb_path} 中没有提取到任何有效原子！")

    # 构建 Tensor
    pos = torch.tensor(pos_list, dtype=torch.float32)
    x = torch.tensor(z_list, dtype=torch.long)
    residue_index = torch.tensor(residue_indices, dtype=torch.long)
    
    # 默认同一条链
    is_ligand = torch.zeros(len(pos), dtype=torch.long)

    # 返回基础数据，向量特征和边特征会在 screening 中自动构建
    return Data(x=x, pos=pos, residue_index=residue_index, is_ligand=is_ligand)


def save_patch_with_attention_to_pdb(patch_data, attention_weights, out_path="best_match_patch.pdb"):
    """
    高阶功能：将选中的蛋白质斑块保存为 PDB 文件，
    并将 Attention 权重写入 B-factor 列，以便在 PyMOL 中进行热力图可视化！
    """
    pos = patch_data.pos.cpu().numpy()
    z_array = patch_data.x.cpu().numpy()
    
    # 逆向映射原子序数到元素名
    Z_TO_ELEMENT = {v: k for k, v in ELEMENT_TO_Z.items()}
    
    # 归一化 attention 权重到 0~100 (B-factor 常用的范围)
    attn = attention_weights
    if attn.max() > attn.min():
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min()) * 100.0
    else:
        attn_norm = np.zeros_like(attn)

    with open(out_path, 'w') as f:
        for i in range(len(pos)):
            x, y, z = pos[i]
            element = Z_TO_ELEMENT.get(z_array[i], 'C')
            b_factor = attn_norm[i]
            # 严格按照 PDB 格式规范写入
            f.write(f"ATOM  {i+1:>5}  CA  ALA A   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00{b_factor:>6.2f}          {element:>2}\n")
    print(f"💾 带有 Attention 热力图的斑块已保存至: {out_path}")

class VirtualScreener:
    def __init__(self, model_path, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # 初始化并加载模型
        print("🧠 正在加载 CMAE 筛选器...")
        self.model = GlueVAE(
            hidden_dim=config['model']['hidden_dim'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            edge_dim=config['model']['edge_dim'],
            vocab_size=config['model']['vocab_size']
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        clean_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        self.model.load_state_dict(clean_state_dict)
        self.model.eval()

        # 在 __init__ 的末尾加上：
        self.cutoff_radius = 8.0  # 对应 dataset 中的默认截断半径
        self.rbf = GaussianRBF(n_rbf=16, cutoff=self.cutoff_radius, start=0.0).to(self.device)

    @torch.no_grad()
    def get_latent_representation(self, patch_data):
        """
        核心推理：绕过 Forward 掩码，直接调用 Encoder 和 AttnPooling 提取 128 维流形特征
        """
        patch_data = patch_data.to(self.device)
        num_graphs = patch_data.batch.max().item() + 1 if hasattr(patch_data, 'batch') else 1
        batch_idx = patch_data.batch if hasattr(patch_data, 'batch') else torch.zeros(patch_data.x.size(0), dtype=torch.long, device=self.device)

        # 1. 提取全原子等变特征与对比投影
        s, z_proj = self.model.encode(
            z=patch_data.x, 
            vector_features=patch_data.vector_features, 
            edge_index=patch_data.edge_index, 
            edge_attr=patch_data.edge_attr, 
            pos=patch_data.pos
        )

        # 2. 多头注意力池化聚合为 Graph 级别向量
        graph_z, attn_w, _ = self.model.attn_pooling(z_proj, batch_idx, num_graphs)
        
        # 3. L2 归一化，进入绝对对比空间
        graph_z = F.normalize(graph_z, p=2, dim=-1)
        
        return graph_z, attn_w

    def extract_patch_manual(self, full_protein_data, center_residue_indices, radius=15.0):
        """
        方法一：【指定界面】基于人工指定的残基提取 Patch
        center_residue_indices: list or tensor, 指定的残基索引
        """
        # 计算指定残基的几何中心
        selected_atoms = torch.isin(full_protein_data.residue_index, torch.tensor(center_residue_indices))
        center_coords = full_protein_data.pos[selected_atoms].mean(dim=0, keepdim=True)
        
        # 截取半径内的所有原子
        dist = torch.norm(full_protein_data.pos - center_coords, dim=-1)
        patch_mask = dist <= radius
        
        if patch_mask.sum() == 0:
            raise ValueError("指定的残基附近没有提取到原子，请检查坐标或半径。")
            
        return self._subgraph_from_mask(full_protein_data, patch_mask)

    def extract_patches_auto(self, full_protein_data, num_patches=20, radius=15.0):
        """
        方法二：【自动选取】利用最远点采样 (FPS) 在蛋白表面均匀撒网，提取多个候选 Patch
        返回: List[Data], 包含所有采样到的局部斑块
        """
        pos = full_protein_data.pos
        
        # 使用 FPS 获取分布在表面/全局的最远点锚点索引
        batch_dummy = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        # fps 的 ratio = num_patches / total_atoms
        ratio = min(1.0, num_patches / pos.size(0))
        anchor_indices = fps(pos, batch_dummy, ratio=ratio)
        
        # 如果获取的锚点多于设定值，截断
        anchor_indices = anchor_indices[:num_patches]
        
        patches = []
        for idx in anchor_indices:
            center_coords = pos[idx].unsqueeze(0)
            dist = torch.norm(pos - center_coords, dim=-1)
            patch_mask = dist <= radius
            
            # 过滤掉太小的碎片 (比如游离的单个氨基酸)
            if patch_mask.sum() > 10:
                patches.append(self._subgraph_from_mask(full_protein_data, patch_mask))
                
        return patches

    def _subgraph_from_mask(self, data, mask):
        """从完整的蛋白图数据中切割出子图，并严格按照训练时的方式重构边与向量特征"""
        device = data.pos.device
        
        # 1. 提取子图节点特征
        subset_data = Data(
            x=data.x[mask],
            pos=data.pos[mask],
            residue_index=data.residue_index[mask] if hasattr(data, 'residue_index') else None,
            is_ligand=data.is_ligand[mask] if hasattr(data, 'is_ligand') else None
        )
        
        # 2. 重构截断半径图 (Radius Graph)
        # 严格对齐 dataset.py 中的 self.cutoff_radius (默认为 8.0)
        edge_index = radius_graph(subset_data.pos, r=self.cutoff_radius, loop=False)
        subset_data.edge_index = edge_index
        
        # 3. 计算边的距离与拓扑类型
        row, col = edge_index
        diff = subset_data.pos[row] - subset_data.pos[col]
        dist = torch.norm(diff, p=2, dim=-1)
        
        is_covalent = dist < 1.7
        
        # 容错处理：如果在单蛋白扫描时没有 is_ligand，则默认都在同一条链
        if hasattr(subset_data, 'is_ligand') and subset_data.is_ligand is not None:
            same_chain = (subset_data.is_ligand[row] == subset_data.is_ligand[col])
        else:
            same_chain = torch.ones_like(is_covalent, dtype=torch.bool)
            
        edge_type = torch.zeros((edge_index.size(1), 3), dtype=torch.float, device=device)
        edge_type[is_covalent, 0] = 1.0
        edge_type[(~is_covalent) & same_chain, 1] = 1.0
        edge_type[(~is_covalent) & (~same_chain), 2] = 1.0
        
        # 4. 拼接边特征 (拓扑特征 + RBF 距离特征)
        rbf_feat = self.rbf(dist.to(self.device)).to(device)
        subset_data.edge_attr = torch.cat([edge_type, rbf_feat], dim=-1)
        
        # 5. 重新计算鲁棒的向量特征 (Vector Features)
        mask_cov = is_covalent
        row_cov = row[mask_cov]
        col_cov = col[mask_cov]
        
        N = subset_data.pos.size(0)
        vector_features = torch.zeros(N, 3, device=device)
        
        if len(row_cov) > 0:
            vec_diff = subset_data.pos[row_cov] - subset_data.pos[col_cov]
            vector_features = scatter_add(vec_diff, col_cov, dim=0, dim_size=N)
            
        # 注意：这里我们故意没有加随机噪声 (randn)，为了保证推断时 SE(3) 的绝对稳定性
        subset_data.vector_features = vector_features
        
        # 记录 batch 索引 (用于后续的 Attention Pooling)
        subset_data.batch = torch.zeros(N, dtype=torch.long, device=device)
        
        return subset_data

    def screen(self, target_patch, candidate_protein_data):
        """
        终极筛选工作流：拿 Target 钥匙去 Candidate 表面开锁
        """
        # 1. 提取目标蛋白 A (Target) 的标准特征
        print("🎯 正在提取蛋白 A (Target) 的流形特征...")
        z_target, _ = self.get_latent_representation(target_patch)  # [1, 128]
        
        # 2. 对候选蛋白 B (Candidate) 进行全景扫描
        print("🌐 正在对蛋白 B (Candidate) 进行表面自动采样与特征编码...")
        candidate_patches = self.extract_patches_auto(candidate_protein_data, num_patches=20)
        print(f"   提取到 {len(candidate_patches)} 个有效候选斑块。")
        
        # 将多个候选斑块拼成一个 Batch 提升计算效率
        batch_candidates = Batch.from_data_list(candidate_patches)
        z_candidates, attn_weights = self.get_latent_representation(batch_candidates) # [N_patches, 128]
        
        # 3. 计算余弦相似度矩阵
        # z_target 和 z_candidates 都已经 L2 归一化，直接矩阵相乘就是余弦相似度
        similarities = torch.matmul(z_candidates, z_target.T).squeeze(-1) # [N_patches]
        
        # 4. 寻找 Best Match
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[best_idx].item()
        best_patch = candidate_patches[best_idx]
        
        # 提取最佳斑块内部的原子级 Attention 权重 (找到具体是谁在起作用)
        # 因为 attn_weights 是 [Total_atoms, num_heads]，我们需要把它拆回对应图
        # 为了简单，这里假定取平均头部的权重
        best_attn_w = attn_weights[batch_candidates.batch == best_idx].mean(dim=-1).cpu().numpy()
        
        print("\n" + "="*40)
        print(f"✅ 筛选完成！")
        print(f"🏆 最高匹配相似度 (Score): {best_score:.4f}")
        print(f"📍 最佳匹配发生在该斑块的质心坐标附近: {best_patch.pos.mean(dim=0).cpu().numpy()}")
        print("="*40)
        
        return {
            'best_score': best_score,
            'best_patch_data': best_patch,
            'best_attention_weights': best_attn_w,
            'all_scores': similarities.cpu().numpy()
        }

if __name__ == "__main__":
    print("This module provides screening utilities. Use run_screening.py as the entry point.")