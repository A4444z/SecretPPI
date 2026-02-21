
"""
Gemini主刀修改的cmae架构，from glue_vae_solo.py
GlueVAE 主模型架构。
完整的变分自编码器，用于蛋白质-蛋白质界面生成。

架构概述：
1. 编码器：多层 PaiNN，提取全原子特征
2. 瓶颈层：原子 -&gt; 残基 Pooling，降维到残基级别
3. 潜在空间：重参数化采样
4. 解码器：条件生成，残基 -&gt; 原子 super-resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from src.models.layers_solo import PaiNNEncoder
from src.utils.loss_solo import CoordinateDecoder

# ================= 🚨 新增 RBF 类 =================
class GaussianSmearing(nn.Module):
    """
    径向基函数 (RBF) 展开，用于将标量距离映射为高维向量。
    """
    def __init__(self, start=0.0, stop=10.0, num_gaussians=16):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # 计算高斯函数的宽度系数
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
# ===================================================

class ResiduePooling(nn.Module):
    """
    原子到残基的 Pooling 层。
    使用 scatter_mean 将同一残基的原子特征聚合为残基特征。
    """
    
    def __init__(self, reduce='mean'):
        super().__init__()
        self.reduce = reduce
        
    def forward(
        self,
        atom_features,
        residue_index
    ):
        """
        参数:
            atom_features: [N, hidden_dim] 原子级特征
            residue_index: [N] 每个原子所属的残基索引
        返回:
            [R, hidden_dim] 残基级特征，R 为残基数
        """
        if self.reduce == 'mean':
            return scatter_mean(atom_features, residue_index, dim=0)
        elif self.reduce == 'sum':
            return scatter_sum(atom_features, residue_index, dim=0)
        else:
            raise ValueError(f"Unknown reduce type: {self.reduce}")


class ResidueToAtomUnpooling(nn.Module):
    """
    残基到原子的 Unpooling 层。
    将残基级特征广播回原子级别。
    """
    
    def forward(
        self,
        residue_features,
        residue_index
    ):
        """
        参数:
            residue_features: [R, hidden_dim] 残基级特征
            residue_index: [N] 每个原子所属的残基索引
        返回:
            [N, hidden_dim] 原子级特征
        """
        return residue_features[residue_index]


class Projector(nn.Module):
    """
    对比学习投影头 (Projection Head)。
    3层 MLP，最后进行 L2 归一化，将特征映射到对比空间。
    """
    def __init__(self, hidden_dim, proj_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        """
        参数:
            x: [R, hidden_dim] 残基特征
        返回:
            z: [R, proj_dim] L2归一化后的对比向量
        """
        z = self.mlp(x)
        # 🚨 极其关键的一步：对特征进行 L2 归一化，使其分布在超球面上
        z = F.normalize(z, p=2, dim=-1)
        return z


class ConditionalPaiNNDecoder(nn.Module):
    """
    条件 PaiNN 解码器。
    结合受体原子和潜在信息，生成配体原子坐标。
    """
    
    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        edge_dim=19,
        vocab_size=101,
        use_gradient_checkpointing=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # PaiNN 编码器作为解码器主干
        self.painn = PaiNNEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_dim,
            vocab_size=vocab_size,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # 坐标预测头
        #self.coord_decoder = CoordinateDecoder(hidden_dim, num_layers=2)
        self.v_proj = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(
        self,
        atom_latent,
        z_atom,
        vector_features,
        edge_index,
        edge_attr,
        pos,
        residue_index
    ):
        """
        参数:
            atom_latent: [N, hidden_dim] 原子级潜在特征
            z_atom: [N] 原子序数
            vector_features: [N, 3] 向量特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            pos: [N, 3] 初始坐标（受体固定，配体可调整）
            residue_index: [N] 残基索引
        返回:
            [N, 3] 预测的坐标偏移/更新
        """
        # 初始化标量特征：嵌入 + 潜在特征
        s_initial = self.painn.embedding(z_atom) + atom_latent
        
        # 通过 PaiNN 提取特征
        s, v = self.painn(z_atom, vector_features, edge_index, edge_attr, pos, initial_s=s_initial)
        
        # 预测坐标偏移
        #delta_pos = self.coord_decoder(s)
        # 它的作用是把 [N, 3, hidden_dim] 的向量特征压缩成 [N, 3, 1] 的物理位移
        #self.v_proj = nn.Linear(hidden_dim, 1, bias=False)

        # ✅ 完美修复：将 [N, 128, 3] 转置为 [N, 3, 128]
        # 这样线性层就会对 128 进行计算，输出 [N, 3, 1]
        # 最后 squeeze(-1) 挤掉最后那个 1，留下完美的 [N, 3] 坐标偏移！
        delta_pos = self.v_proj(v.transpose(1, 2)).squeeze(-1)
        
        return delta_pos


class GlueVAE(nn.Module):
    """
    GlueVAE 主模型。
    完整的变分自编码器架构。
    """
    
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=32,
        num_encoder_layers=6,
        num_decoder_layers=4,
        edge_dim=19,
        vocab_size=101,
        use_gradient_checkpointing=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = PaiNNEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            edge_dim=edge_dim,
            vocab_size=vocab_size,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # 原子 -&gt; 残基 Pooling
        self.residue_pooling = ResiduePooling(reduce='mean')
        
        # ================= 🚨 新增：对比学习投影头 =================
        self.projector = Projector(hidden_dim=hidden_dim, proj_dim=128)
        # =========================================================
        
        # 残基 -&gt; 原子 Unpooling
        self.residue_unpooling = ResidueToAtomUnpooling()
        
        # 解码器
        self.decoder = ConditionalPaiNNDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            edge_dim=edge_dim,
            vocab_size=vocab_size,
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        # ================= 🚨 新增 RBF 层 =================
        # edge_dim (19) - 拓扑特征 (3) = 16 维的高斯特征
        self.rbf = GaussianSmearing(
            start=0.0, 
            stop=10.0, 
            num_gaussians=edge_dim - 3
        )
        # ==================================================
        
    
            
    def encode(
        self,
        z,
        vector_features,
        edge_index,
        edge_attr,
        pos,
        residue_index
    ):
        """
        编码过程：提取残基特征，并投影到对比学习空间。
        """
        # 1. PaiNN 编码器提取原子特征
        s, v = self.encoder(z, vector_features, edge_index, edge_attr, pos)

        # 2. 原子 -> 残基 Pooling
        res_features = self.residue_pooling(s, residue_index)

        # 3. 投影到对比空间得到 Z 
        z_proj = self.projector(res_features)

        # 返回：完整的残基特征(给Decoder重建用) 和 投影后的Z(给InfoNCE算Loss用)
        return res_features, z_proj

        
    def decode(
        self,
        res_features,         # 👈 修改：直接接收完整的残基特征，不再需要 z_latent
        z_atom,
        fake_vector_features, 
        edge_index,
        fake_edge_attr,       
        fake_pos,             
        residue_index
    ):
        """
        解码过程：残基特征 -> 还原坐标。
        """
        # Unpooling：残基特征 -> 原子特征
        atom_latent = self.residue_unpooling(res_features, residue_index)
        
        # 通过解码器 (此时 Decoder 只能看到残缺的 fake_pos 和 fake 特征)
        delta_pos = self.decoder(
            atom_latent, z_atom, fake_vector_features,
            edge_index, fake_edge_attr, fake_pos, residue_index
        )
        
        # 必须是在 fake_pos 的基础上进行偏移！
        pos_pred = fake_pos + delta_pos
        
        return pos_pred

    def forward(
        self,
        z,
        vector_features,
        edge_index,
        edge_attr,
        pos,
        residue_index,
        is_ligand,            # 👈 🚨 必须新增！用于区分受体(0)和配体(1)
        mask_interface=None,  
        batch_idx=None        
    ):
        if batch_idx is None or mask_interface is None:
            raise ValueError("CMAE requires batch_idx and mask_interface.")

        # ================= 🚨 核心修复：全局安全的残基索引压缩 =================
        # 1. 赋予每个 Graph 极大的偏移量 (100000)，彻底隔离不同复合物的残基 ID
        global_residue_index = residue_index + batch_idx * 100000
        # 2. 对这个全局安全的 ID 进行压缩映射，保证绝对不会发生跨 Graph 融合！
        _, residue_index_compact = torch.unique(global_residue_index, sorted=True, return_inverse=True)
        # ======================================================================

        num_graphs = int(batch_idx.max().item()) + 1

        # ================= 1. 构造 View 1 (Mask A) 和 View 2 (Mask B) =================
        # 克隆坐标，防止污染原始真实坐标
        pos_v1 = pos.clone() 
        pos_v2 = pos.clone() 

        mask_v1 = torch.zeros(pos.size(0), dtype=torch.bool, device=pos.device)
        mask_v2 = torch.zeros(pos.size(0), dtype=torch.bool, device=pos.device)

        if self.training:
            for i in range(num_graphs):
                graph_mask = (batch_idx == i)

                # 提取 A 侧 (受体, 0) 和 B 侧 (配体, 1) 的界面原子
                interface_A = torch.where(graph_mask & (is_ligand == 0) & (mask_interface == 1))[0]
                interface_B = torch.where(graph_mask & (is_ligand == 1) & (mask_interface == 1))[0]

                # --- 💥 View 1: 在 A 侧 (受体) 炸出一个 10 埃的大洞，保留 B 侧 ---
                if len(interface_A) > 0:
                    center_idx_A = interface_A[torch.randint(0, len(interface_A), (1,))]
                    dist_to_center_A = torch.norm(pos[graph_mask] - pos[center_idx_A], p=2, dim=-1)
                    # 找出局部 10 埃内的 A 侧原子 (必须同属受体)
                    local_mask_A = (dist_to_center_A < 10.0) & (is_ligand[graph_mask] == 0)
                    global_mask_A = torch.where(graph_mask)[0][local_mask_A]
                    mask_v1[global_mask_A] = True

                # --- 💥 View 2: 在 B 侧 (配体) 炸出一个 10 埃的大洞，保留 A 侧 ---
                if len(interface_B) > 0:
                    center_idx_B = interface_B[torch.randint(0, len(interface_B), (1,))]
                    dist_to_center_B = torch.norm(pos[graph_mask] - pos[center_idx_B], p=2, dim=-1)
                    # 找出局部 10 埃内的 B 侧原子 (必须同属配体)
                    local_mask_B = (dist_to_center_B < 10.0) & (is_ligand[graph_mask] == 1)
                    global_mask_B = torch.where(graph_mask)[0][local_mask_B]
                    mask_v2[global_mask_B] = True

            # 实施物理坐标塌陷 (给被破坏的原子赋予随机高斯噪声，彻底剥夺其局部空间信息)
            if mask_v1.sum() > 0:
                pos_v1[mask_v1] = torch.randn((mask_v1.sum(), 3), device=pos.device) * 0.1
            if mask_v2.sum() > 0:
                pos_v2[mask_v2] = torch.randn((mask_v2.sum(), 3), device=pos.device) * 0.1

        # ================= 2. 重新计算假坐标的边特征 (距离 RBF) =================
        edge_type = edge_attr[:, :3]
        row, col = edge_index
        fake_vector_features = torch.zeros_like(vector_features) # 全零防止泄露

        # View 1 的 RBF 特征
        fake_diff_v1 = pos_v1[row] - pos_v1[col]
        fake_dist_v1 = torch.sqrt((fake_diff_v1 ** 2).sum(dim=-1) + 1e-8)
        fake_edge_attr_v1 = torch.cat([edge_type, self.rbf(fake_dist_v1)], dim=-1)

        # View 2 的 RBF 特征
        fake_diff_v2 = pos_v2[row] - pos_v2[col]
        fake_dist_v2 = torch.sqrt((fake_diff_v2 ** 2).sum(dim=-1) + 1e-8)
        fake_edge_attr_v2 = torch.cat([edge_type, self.rbf(fake_dist_v2)], dim=-1)

        # ================= 3. 双路编码 (Encoder) =================
        # 🚨 必须把压缩后的 residue_index_compact 传进去！
        res_feat_v1, z_proj_v1 = self.encode(z, fake_vector_features, edge_index, fake_edge_attr_v1, pos_v1, residue_index_compact)
        res_feat_v2, z_proj_v2 = self.encode(z, fake_vector_features, edge_index, fake_edge_attr_v2, pos_v2, residue_index_compact)

        # 🚨 核心修复：更安全地构造 res_batch 
        R = int(residue_index_compact.max().item()) + 1
        res_batch = torch.zeros(R, dtype=torch.long, device=pos.device)
        # 因为同残基所有原子的 batch_idx 完全一致，直接 scatter_ 覆盖赋值
        res_batch.scatter_(0, residue_index_compact, batch_idx)

        if self.training:
            # 工业级保险：检查同一 residue 是否出现多个 batch_id
            assert torch.all(res_batch[residue_index_compact] == batch_idx), "Residue spans multiple graphs!"

        # 2. 将同一个 Graph 下的所有残基向量做平均
        graph_z1 = scatter_mean(z_proj_v1, res_batch, dim=0)
        graph_z2 = scatter_mean(z_proj_v2, res_batch, dim=0)

        # 3. 再次 L2 归一化
        graph_z1 = F.normalize(graph_z1, p=2, dim=-1)
        graph_z2 = F.normalize(graph_z2, p=2, dim=-1)

        # ================= 4. 解码重构 (Decoder) =================
        # 为了节约算力且达到辅助重构的目的，我们只挑 View 1 进行解码重构。
        # Decoder 必须通过隐空间，把被炸掉的受体坐标猜出来。
        pos_pred_v1 = self.decode(
            res_feat_v1, z, fake_vector_features,
            edge_index, fake_edge_attr_v1, pos_v1, residue_index_compact
        )

        return graph_z1, graph_z2, pos_pred_v1, mask_v1