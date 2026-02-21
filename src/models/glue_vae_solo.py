
"""
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


class LatentEncoder(nn.Module):
    """
    潜在空间编码器。
    将残基级特征映射到潜在分布的均值和方差。
    """
    
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        参数:
            x: [R, hidden_dim] 残基特征
        返回:
            (mu, logvar): 每个 [R, latent_dim]
        """
        h = self.mlp(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class LatentDecoder(nn.Module):
    """
    潜在空间解码器。
    将潜在向量映射回残基级特征。
    """
    
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, z):
        """
        参数:
            z: [R, latent_dim] 潜在向量
        返回:
            [R, hidden_dim] 残基特征
        """
        return self.mlp(z)


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
        
        # 潜在空间
        self.latent_encoder = LatentEncoder(hidden_dim, latent_dim)
        self.latent_decoder = LatentDecoder(latent_dim, hidden_dim)
        
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
        
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧。
        z = mu + sigma * epsilon, epsilon ~ N(0, 1)
        """
        if self.training:
            # ====== [新增] 防溢出 clamp ======
            logvar = torch.clamp(logvar, min=-20.0, max=20.0)
            # ================================
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
            
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
        编码过程：输入 -&gt; 潜在分布参数。
        """
                # PaiNN 编码器
        s, v = self.encoder(z, vector_features, edge_index, edge_attr, pos)

        # 原子 -> 残基 Pooling
        res_features = self.residue_pooling(s, residue_index)

        # ===== [DEBUG] 一次性打印 =====
        if not hasattr(self, "_debug_once_encode"):
            self._debug_once_encode = False
        if not self._debug_once_encode:
            print("\n[DEBUG][GlueVAE.encode]")
            print("  s finite:", torch.isfinite(s).all().item())
            print("  residue_index min/max:", int(residue_index.min()), int(residue_index.max()))
            print("  residue unique:", int(residue_index.unique().numel()),
                " / max+1:", int(residue_index.max().item()) + 1)
            print("  res_features finite:", torch.isfinite(res_features).all().item(),
                "shape:", tuple(res_features.shape))
        # ============================

        # 潜在分布
        mu, logvar = self.latent_encoder(res_features)

        if not self._debug_once_encode:
            print("  mu finite:", torch.isfinite(mu).all().item())
            print("  logvar finite:", torch.isfinite(logvar).all().item())
            if torch.isfinite(logvar).all():
                print("  logvar range:", float(logvar.min()), float(logvar.max()))
            self._debug_once_encode = True

        return mu, logvar

        
    def decode(
        self,
        z_latent,
        z_atom,
        fake_vector_features, # 👈 接收假的向量
        edge_index,
        fake_edge_attr,       # 👈 接收假的距离
        fake_pos,             # 👈 接收假的起点坐标
        residue_index
    ):
        """
        解码过程：潜在向量 -> 坐标。
        """
        # 潜在 -> 残基特征
        res_features = self.latent_decoder(z_latent)
        
        # Unpooling：残基特征 -> 原子特征
        atom_latent = self.residue_unpooling(res_features, residue_index)
        
        # 通过解码器 (此时 Decoder 只能看到瞎猜的 fake_pos 和 fake 特征)
        delta_pos = self.decoder(
            atom_latent, z_atom, fake_vector_features,
            edge_index, fake_edge_attr, fake_pos, residue_index
        )
        
        # 🚨 终极修复：必须是在 fake_pos 的基础上进行偏移！绝不能加上真实的 pos！
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
        mask_interface=None,  # 👈 新增
        batch_idx=None        # 👈 新增：必须有这个才能区分不同的复合物
    ):
        mu, logvar = self.encode(
            z, vector_features, edge_index, edge_attr, pos, residue_index
        )
        z_latent = self.reparameterize(mu, logvar)
        
        noise_scale = 4.0 
        fake_pos = pos + torch.randn_like(pos) * noise_scale
        
        # ================= 🚨 终极杀招：PyG Batched Interface Block Masking =================
        if self.training and mask_interface is not None and batch_idx is not None:
            # 创建一个全图的空 Mask
            block_mask = torch.zeros(pos.size(0), dtype=torch.bool, device=pos.device)
            
            # 获取 Batch 中总共有多少个独立的图 (比如 16 个)
            num_graphs = int(batch_idx.max().item()) + 1
            
            # 对每一个图执行独立的界面轰炸
            for i in range(num_graphs):
                # 找到属于第 i 个图的所有原子的全局索引
                graph_node_idx = torch.nonzero(batch_idx == i).squeeze(-1)
                
                # 提取这个图的界面掩码
                graph_interface_mask = mask_interface[graph_node_idx]
                graph_interface_nodes = graph_node_idx[torch.nonzero(graph_interface_mask).squeeze(-1)]
                
                # 如果这个图有界面原子
                if graph_interface_nodes.numel() > 0:
                    # 1. 随机选一个爆炸中心
                    center_idx = graph_interface_nodes[torch.randint(0, graph_interface_nodes.numel(), (1,))]
                    center_pos = pos[center_idx]
                    
                    # 2. 算这个图里所有原子到中心的距离
                    dist_to_center = torch.norm(pos[graph_node_idx] - center_pos, p=2, dim=-1)
                    
                    # 3. 找出局部 10 埃内的原子
                    local_block_mask = dist_to_center < 10.0
                    
                    # 4. 把被炸的原子映射回全局的 block_mask 里
                    global_block_mask_idx = graph_node_idx[local_block_mask]
                    block_mask[global_block_mask_idx] = True
            
            # 统计总共被掩码的原子
            num_masked = block_mask.sum()
            if num_masked > 0:
                # 塌陷到各自原子的质心（这里做了简化处理，塌陷到原点附近并施加扰动，彻底破坏其空间结构）
                independent_noise = torch.randn((num_masked, 3), device=pos.device) * 0.1
                fake_pos[block_mask] = independent_noise
        # =========================================================================
            
        edge_type = edge_attr[:, :3]
        
        # 重新计算距离 (防崩溃的 Safe Norm)
        row, col = edge_index
        fake_diff = fake_pos[row] - fake_pos[col]
        dist_sq = (fake_diff ** 2).sum(dim=-1)
        fake_dist = torch.sqrt(dist_sq + 1e-8) 
        
        fake_rbf_feat = self.rbf(fake_dist)
        fake_edge_attr = torch.cat([edge_type, fake_rbf_feat], dim=-1)
        fake_vector_features = torch.zeros_like(vector_features)

        pos_pred = self.decode(
            z_latent, z, fake_vector_features, 
            edge_index, fake_edge_attr, fake_pos, residue_index
        )
        
        return pos_pred, mu, logvar
        
    @torch.no_grad()
    def sample(
        self,
        z,
        vector_features,
        edge_index,
        edge_attr,
        pos,
        residue_index,
        num_samples=1,
        mask_interface=None,  # 👈 新增
        batch_idx=None        # 👈 新增：必须有这个才能区分不同的复合物
    ):
        """
        从潜在空间采样生成多个样本。
        """
        mu, logvar = self.encode(
            z, vector_features, edge_index, edge_attr, pos, residue_index
        )
        
        samples = []
        for _ in range(num_samples):
            z_latent = self.reparameterize(mu, logvar)
            
            # ================= 🚨 修复 sample 方法的数据泄露与维度 =================
            # 🚨 拯救图神经网络的命脉：在真实坐标上施加小幅度扰动，而不是完全抹杀
            noise_scale = 4.0 
            fake_pos = pos + torch.randn_like(pos) * noise_scale  
            edge_type = edge_attr[:, :3]            
            
            row, col = edge_index
            fake_diff = fake_pos[row] - fake_pos[col]
            fake_dist = torch.norm(fake_diff, p=2, dim=-1) + 1e-6 
            
            fake_rbf_feat = self.rbf(fake_dist)
            fake_edge_attr = torch.cat([edge_type, fake_rbf_feat], dim=-1)
            
            # 节点级别初始向量，同样用全零
            fake_vector_features = torch.zeros_like(vector_features)
            # ===============================================================
            # 使用重算后的 fake 特征进行解码
            pos_pred = self.decode(
                z_latent, z, fake_vector_features,
                edge_index, fake_edge_attr, fake_pos, residue_index
            )
            samples.append(pos_pred)
            
        return torch.stack(samples, dim=0)
