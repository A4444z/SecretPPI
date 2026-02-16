
"""
GlueVAE 主模型架构（原子级VAE版本）。
完整的变分自编码器，用于蛋白质-蛋白质界面生成。

关键改进：完全原子级别VAE，保持每个原子的独特特征！
- 不再把同一残基的所有原子用完全一样的特征
- 每个原子都有自己独立的潜在表示
- 同时保留全局信息，捕捉整体结构

架构概述：
1. 编码器：多层 PaiNN，提取全原子特征
2. 全局池化：得到全局结构向量
3. 原子级潜在空间：每个原子独立的mu和logvar
4. 解码器：条件生成，原子级潜在 -&gt; 原子坐标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max

from src.models.layers_solo import PaiNNEncoder
from src.utils.loss_solo import CoordinateDecoder


class GlobalPooling(nn.Module):
    """
    全局池化层。
    将原子级特征聚合为全局特征向量。
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 投影层，用于合并不同池化方式的结果
        self.project = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, atom_features, batch=None):
        """
        参数:
            atom_features: [N, hidden_dim] 原子级特征
            batch: [N] batch索引（如果有多个图）
        返回:
            [B, hidden_dim] 全局特征，B为batch大小
        """
        if batch is None:
            batch = torch.zeros(atom_features.size(0), dtype=torch.long, device=atom_features.device)
        
        # 三种池化方式
        mean_feat = scatter_mean(atom_features, batch, dim=0)
        sum_feat = scatter_sum(atom_features, batch, dim=0)
        max_feat, _ = scatter_max(atom_features, batch, dim=0)
        
        # 拼接并投影
        combined = torch.cat([mean_feat, sum_feat, max_feat], dim=-1)
        global_feat = self.project(combined)
        
        return global_feat


class AtomLevelLatentEncoder(nn.Module):
    """
    原子级潜在空间编码器。
    为每个原子独立预测mu和logvar，同时结合全局信息。
    
    策略：
    - 全局特征：捕捉整体结构
    - 原子特征：捕捉局部环境
    - 两者结合，预测每个原子的潜在分布
    """
    
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 原子特征 + 全局特征 -&gt; 中间特征
        self.atom_global_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # 预测mu和logvar
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, atom_features, global_features, batch=None):
        """
        参数:
            atom_features: [N, hidden_dim] 原子级特征
            global_features: [B, hidden_dim] 全局特征
            batch: [N] batch索引
        返回:
            (mu, logvar): 每个 [N, latent_dim]
        """
        if batch is None:
            batch = torch.zeros(atom_features.size(0), dtype=torch.long, device=atom_features.device)
        
        # 把全局特征广播到每个原子
        global_expanded = global_features[batch]  # [N, hidden_dim]
        
        # 拼接原子特征和全局特征
        combined = torch.cat([atom_features, global_expanded], dim=-1)
        
        # MLP处理
        h = self.atom_global_mlp(combined)
        
        # 预测mu和logvar
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        return mu, logvar


class AtomLevelLatentDecoder(nn.Module):
    """
    原子级潜在空间解码器。
    从原子级潜在向量解码回原子特征。
    """
    
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, z_atom):
        """
        参数:
            z_atom: [N, latent_dim] 原子级潜在向量
        返回:
            [N, hidden_dim] 原子级特征
        """
        return self.mlp(z_atom)


class ConditionalPaiNNDecoder(nn.Module):
    """
    条件 PaiNN 解码器。
    结合原子级潜在信息，生成原子坐标。
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
        self.coord_decoder = CoordinateDecoder(hidden_dim, num_layers=2)
    
    def forward(
        self,
        z_atom,
        atom_features,
        z_atom_type,
        vector_features,
        edge_index,
        edge_attr,
        pos
    ):
        """
        参数:
            z_atom: [N, latent_dim] 原子级潜在向量（暂时未用，保留扩展性）
            atom_features: [N, hidden_dim] 解码后的原子特征
            z_atom_type: [N] 原子序数
            vector_features: [N, 3] 向量特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            pos: [N, 3] 初始坐标
        返回:
            [N, 3] 预测的坐标偏移/更新
        """
        # 通过 PaiNN 提取特征
        # 注意：这里我们用 atom_features 初始化，而不是只从z_atom_type嵌入
        s, v = self.painn(
            z_atom_type, 
            vector_features, 
            edge_index, 
            edge_attr, 
            pos,
            initial_s=atom_features  # 使用解码后的特征初始化
        )
        
        # 预测坐标偏移
        delta_pos = self.coord_decoder(s)
        
        return delta_pos


class GlueVAEAtomLevel(nn.Module):
    """
    GlueVAE 主模型（原子级VAE版本）。
    
    关键改进：
    - 每个原子都有独立的潜在表示
    - 不再将同一残基的所有原子压缩成相同特征
    - 同时保留全局信息，捕捉整体结构
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
        
        # 全局池化
        self.global_pooling = GlobalPooling(hidden_dim)
        
        # 原子级潜在空间
        self.latent_encoder = AtomLevelLatentEncoder(hidden_dim, latent_dim)
        self.latent_decoder = AtomLevelLatentDecoder(latent_dim, hidden_dim)
        
        # 解码器
        self.decoder = ConditionalPaiNNDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            edge_dim=edge_dim,
            vocab_size=vocab_size,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
    
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧。
        z = mu + sigma * epsilon, epsilon ~ N(0, 1)
        """
        if self.training:
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
        batch=None
    ):
        """
        编码过程：输入 -&gt; 潜在分布参数（原子级别！）。
        """
        # PaiNN 编码器
        s, v = self.encoder(z, vector_features, edge_index, edge_attr, pos)
        
        # 全局池化
        global_feat = self.global_pooling(s, batch)
        
        # 原子级潜在分布
        mu, logvar = self.latent_encoder(s, global_feat, batch)
        
        return mu, logvar
    
    def decode(
        self,
        z_latent_atom,
        z_atom,
        vector_features,
        edge_index,
        edge_attr,
        pos
    ):
        """
        解码过程：原子级潜在向量 -&gt; 坐标。
        """
        # 潜在 -&gt; 原子特征
        atom_features = self.latent_decoder(z_latent_atom)
        
        # 通过解码器
        delta_pos = self.decoder(
            z_latent_atom, atom_features, z_atom, 
            vector_features, edge_index, edge_attr, pos
        )
        
        # 更新坐标
        pos_pred = pos + delta_pos
        
        return pos_pred
    
    def forward(
        self,
        z,
        vector_features,
        edge_index,
        edge_attr,
        pos,
        batch=None
    ):
        """
        完整前向传播（原子级VAE版本）。
        
        参数:
            z: [N] 原子序数
            vector_features: [N, 3] 初始向量特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            pos: [N, 3] 真实坐标
            batch: [N] batch索引（可选）
            
        返回:
            (pos_pred, mu, logvar)
            注意：mu和logvar现在是 [N, latent_dim]，每个原子独立！
        """
        # 编码
        mu, logvar = self.encode(
            z, vector_features, edge_index, edge_attr, pos, batch
        )
        
        # 重参数化采样（每个原子独立采样！）
        z_latent_atom = self.reparameterize(mu, logvar)
        
        # 解码
        pos_pred = self.decode(
            z_latent_atom, z, vector_features,
            edge_index, edge_attr, pos
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
        batch=None,
        num_samples=1
    ):
        """
        从潜在空间采样生成多个样本（原子级版本）。
        """
        mu, logvar = self.encode(
            z, vector_features, edge_index, edge_attr, pos, batch
        )
        
        samples = []
        for _ in range(num_samples):
            z_latent_atom = self.reparameterize(mu, logvar)
            pos_pred = self.decode(
                z_latent_atom, z, vector_features,
                edge_index, edge_attr, pos
            )
            samples.append(pos_pred)
        
        return torch.stack(samples, dim=0)
