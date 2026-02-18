
"""
损失函数模块。
实现 D-RMSD (Distance-RMSD) 损失和 KL 散度，用于 VAE 训练。

关键特性：
- D-RMSD：基于成对距离矩阵，保证 SE(3) 不变性，无需对齐
- KL 散度：β-VAE 正则化
- β 退火策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_pairwise_distances(pos):
    """
    计算原子坐标的成对距离矩阵。
    
    参数:
        pos: [N, 3] 原子坐标张量
    返回:
        [N, N] 成对距离矩阵
    """
    # 向量化计算: (a - b)^2 = a^2 + b^2 - 2ab
    pos_sq = torch.sum(pos ** 2, dim=-1, keepdim=True)  # [N, 1]
    dist_sq = pos_sq + pos_sq.transpose(-1, -2) - 2.0 * torch.matmul(pos, pos.transpose(-1, -2))
    dist_sq = torch.clamp(dist_sq, min=0.0)  # 防止数值误差导致负数
    dist = torch.sqrt(dist_sq)
    return dist


class DRMSDLoss(nn.Module):
    """
    D-RMSD 损失函数。
    计算预测坐标和真实坐标的成对距离矩阵之间的均方误差。
    
    优势：
    - SE(3) 不变：旋转/平移输入不影响损失值
    - 无需对齐：不需要 Kabsch 算法
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        pos_pred,
        pos_true,
        mask=None
    ):
        """
        参数:
            pos_pred: [N, 3] 预测的原子坐标
            pos_true: [N, 3] 真实的原子坐标
            mask: [N] 可选的原子掩码，1表示有效
        返回:
            D-RMSD 损失标量
        """
        # 计算成对距离矩阵
        D_pred = compute_pairwise_distances(pos_pred)  # [N, N]
        D_true = compute_pairwise_distances(pos_true)  # [N, N]
        
        # 计算 MSE
        mse = (D_pred - D_true) ** 2
        
        # 应用掩码
        if mask is not None:
            mask_2d = mask.unsqueeze(0) * mask.unsqueeze(1)  # [N, N]
            mse = mse * mask_2d
            total_weight = mask_2d.sum()
        else:
            total_weight = mse.numel()
        
        # Reduction
        if self.reduction == 'mean':
            loss = mse.sum() / (total_weight + 1e-8)
        elif self.reduction == 'sum':
            loss = mse.sum()
        else:
            loss = mse
            
        return loss


class KLLoss(nn.Module):
    """
    KL 散度损失。
    计算标准正态分布和学习到的高斯分布之间的 KL 散度。
    
    KL(q(z|x) || p(z)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        mu,
        logvar
    ):
        """
        参数:
            mu: [*, latent_dim] 均值
            logvar: [*, latent_dim] 对数方差
        返回:
            KL 散度
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.reduction == 'mean':
            return kl.mean()
        elif self.reduction == 'sum':
            return kl.sum()
        else:
            return kl


class BetaScheduler:
    """
    β 退火调度器。
    用于 β-VAE，在训练过程中逐步增加 β 的值。
    
    支持多种调度策略：
    - linear: 线性增长
    - cyclic: 周期退火
    - step: 阶梯式增长
    """
    
    def __init__(
        self,
        beta_start=0.0,
        beta_end=1.0,
        warmup_steps=10000,
        schedule_type='linear',
        cycle_length=None
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type
        self.cycle_length = cycle_length if cycle_length is not None else warmup_steps * 2
        self.step = 0
        
    def update(self):
        """更新并返回当前的 β 值。"""
        self.step += 1
        return self.get_beta()
        
    def get_beta(self) -> float:
        """获取当前的 β 值，不更新步数。"""
        if self.schedule_type == 'linear':
            if self.step < self.warmup_steps:
                beta = self.beta_start + (self.beta_end - self.beta_start) * (self.step / self.warmup_steps)
            else:
                beta = self.beta_end
                
        elif self.schedule_type == 'cyclic':
            # 周期退火：每 cycle_length 步重复一次
            cycle_pos = self.step % self.cycle_length
            if cycle_pos < self.warmup_steps:
                beta = self.beta_start + (self.beta_end - self.beta_start) * (cycle_pos / self.warmup_steps)
            else:
                beta = self.beta_end
                
        elif self.schedule_type == 'step':
            # 阶梯式：在固定步数跳跃
            if self.step < self.warmup_steps // 4:
                beta = self.beta_start
            elif self.step < self.warmup_steps // 2:
                beta = self.beta_start + (self.beta_end - self.beta_start) * 0.25
            elif self.step < 3 * self.warmup_steps // 4:
                beta = self.beta_start + (self.beta_end - self.beta_start) * 0.5
            elif self.step < self.warmup_steps:
                beta = self.beta_start + (self.beta_end - self.beta_start) * 0.75
            else:
                beta = self.beta_end
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
            
        return beta


class VAELoss(nn.Module):
    """
    完整的 VAE 损失函数。
    组合 D-RMSD 重建损失和 KL 散度正则化。
    
    Loss = recon_loss + beta * kl_loss
    """
    
    def __init__(
        self,
        beta=1.0,
        recon_reduction='mean',
        kl_reduction='mean'
    ):
        super().__init__()
        self.beta = beta
        self.drmsd_loss = DRMSDLoss(reduction=recon_reduction)
        self.kl_loss = KLLoss(reduction=kl_reduction)
        
    def forward(
        self,
        pos_pred,
        pos_true,
        mu,
        logvar,
        mask=None
    ):
        """
        参数:
            pos_pred: [N, 3] 预测坐标
            pos_true: [N, 3] 真实坐标
            mu: [latent_dim] 潜在空间均值
            logvar: [latent_dim] 潜在空间对数方差
            mask: [N] 可选的原子掩码
        返回:
            (total_loss, recon_loss, kl_loss)
        """
        recon_loss = self.drmsd_loss(pos_pred, pos_true, mask)
        kl_loss = self.kl_loss(mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class CoordinateDecoder(nn.Module):
    """
    简单的坐标解码器，从特征向量预测原子坐标。
    用于 VAE 解码器部分。
    """
    
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 3))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        参数:
            x: [N, hidden_dim] 原子特征
        返回:
            [N, 3] 预测的坐标偏移
        """
        return self.mlp(x)

