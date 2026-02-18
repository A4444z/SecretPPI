
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
        self.coord_decoder = CoordinateDecoder(hidden_dim, num_layers=2)
        
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
        delta_pos = self.coord_decoder(s)
        
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
        residue_index
    ):
        """
        编码过程：输入 -&gt; 潜在分布参数。
        """
        # PaiNN 编码器
        s, v = self.encoder(z, vector_features, edge_index, edge_attr, pos)
        
        # 原子 -&gt; 残基 Pooling
        res_features = self.residue_pooling(s, residue_index)
        
        # 潜在分布
        mu, logvar = self.latent_encoder(res_features)
        
        return mu, logvar
        
    def decode(
        self,
        z_latent,
        z_atom,
        vector_features,
        edge_index,
        edge_attr,
        pos,
        residue_index
    ):
        """
        解码过程：潜在向量 -&gt; 坐标。
        """
        # 潜在 -&gt; 残基特征
        res_features = self.latent_decoder(z_latent)
        
        # Unpooling：残基特征 -&gt; 原子特征
        atom_latent = self.residue_unpooling(res_features, residue_index)
        
        # 通过解码器
        delta_pos = self.decoder(
            atom_latent, z_atom, vector_features,
            edge_index, edge_attr, pos, residue_index
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
        residue_index
    ):
        """
        完整前向传播。
        
        参数:
            z: [N] 原子序数
            vector_features: [N, 3] 初始向量特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            pos: [N, 3] 真实坐标
            residue_index: [N] 残基索引
            
        返回:
            (pos_pred, mu, logvar)
        """
        # 编码
        mu, logvar = self.encode(
            z, vector_features, edge_index, edge_attr, pos, residue_index
        )
        
        # 重参数化采样
        z_latent = self.reparameterize(mu, logvar)
        
        # 解码：在这个简单版本中，我们在同一空间预测
        # 更高级的版本会固定受体，只生成配体
        pos_pred = self.decode(
            z_latent, z, vector_features,
            edge_index, edge_attr, pos, residue_index
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
        num_samples=1
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
            pos_pred = self.decode(
                z_latent, z, vector_features,
                edge_index, edge_attr, pos, residue_index
            )
            samples.append(pos_pred)
            
        return torch.stack(samples, dim=0)

