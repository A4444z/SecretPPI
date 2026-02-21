

"""
损失函数模块。
实现 D-RMSD (Distance-RMSD) 损失和 KL 散度，用于 VAE 训练。

关键特性：
- D-RMSD：基于成对距离矩阵，保证 SE(3) 不变性，无需对齐
- KL 散度：β-VAE 正则化
- β 退火策略
- 支持 PyG 批量图处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class DRMSDLoss(nn.Module):
    """
    D-RMSD 损失函数 (带局部截断)。
    只计算预测坐标和真实坐标在局部邻域内的成对距离误差。
    """
    
    def __init__(self, reduction='mean', cutoff=15.0):
        super().__init__()
        self.reduction = reduction
        self.cutoff = cutoff  # 🚨 新增：截断距离，建议 10.0 ~ 15.0 埃
        
    def forward(
        self,
        pos_pred,
        pos_true,
        mask=None,
        batch_idx=None
    ):
        if batch_idx is None:
            batch_idx = torch.zeros(pos_pred.size(0), dtype=torch.long, device=pos_pred.device)
        
        # 转为密集矩阵 [B, N_max, 3]
        pos_pred_dense, batch_mask = to_dense_batch(pos_pred, batch_idx)
        pos_true_dense, _ = to_dense_batch(pos_true, batch_idx)
        
        # 计算安全的成对距离 [B, N_max, N_max]
        D_pred = torch.cdist(pos_pred_dense, pos_pred_dense, p=2.0)
        D_true = torch.cdist(pos_true_dense, pos_true_dense, p=2.0)
        
        mse = (D_pred - D_true) ** 2
        
        # 1. 有效节点掩码 (去除 padding 的虚拟节点)
        valid_2d = batch_mask.unsqueeze(1) * batch_mask.unsqueeze(2)
        
        # ================= 🚨 核心修复：引入局部距离截断 =================
        # 只惩罚真实距离在 cutoff 之内的原子对！释放全局结构的自由度。
        cutoff_mask = (D_true < self.cutoff).float()
        
        # 去除对角线（原子自己到自己的距离为0，不算作有效误差避免拉低 mean）
        eye_mask = 1.0 - torch.eye(D_true.size(1), device=D_true.device).unsqueeze(0)
        
        # 组合成最终的基础 mask
        base_mask = valid_2d * cutoff_mask * eye_mask
        # ===============================================================
        
        if mask is not None:
            mask_dense, _ = to_dense_batch(mask, batch_idx)
            mask_2d = mask_dense.unsqueeze(1) * mask_dense.unsqueeze(2)
            final_mask = mask_2d * base_mask
        else:
            final_mask = base_mask
        
        mse = mse * final_mask
        
        if self.reduction == 'mean':
            # 分母使用实际参与计算的有效 Pair 数量
            return mse.sum() / (final_mask.sum() + 1e-8)
        return mse.sum()


class KLLoss(nn.Module):
    """
    真实的 KL 散度计算（用于真实日志记录）。
    """
    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.reduction == 'batchmean':
            # 真实、未稀释的总 KL 散度
            return kl.sum(dim=-1).mean()
        elif self.reduction == 'mean':
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
        
    def get_beta(self):
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
    组合 D-RMSD 重建损失和带 Free Bits 的 KL 散度正则化。
    """
    
    def __init__(
        self,
        beta=0.1,
        recon_reduction='mean',
        kl_reduction='batchmean',
        free_bits=2.0  # 🚨 在这里引入 free_bits
    ):
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits
        self.drmsd_loss = DRMSDLoss(reduction=recon_reduction)
        self.kl_loss = KLLoss(reduction=kl_reduction)
        
    def forward(
        self,
        pos_pred,
        pos_true,
        mu,
        logvar,
        mask=None,
        batch_idx=None
    ):
        # 1. 计算重构误差
        recon_loss = self.drmsd_loss(pos_pred, pos_true, mask, batch_idx)
        
        # 2. 计算真实的 KL 散度（用于在 WandB 上透明监控！）
        raw_kl = self.kl_loss(mu, logvar)
        
        # 3. 🚨 核心魔法：计算用于反向传播的截断 KL (Hinge Loss)
        # 优化器只会看到这个 clamped_kl，所以低于 free_bits 时没有梯度
        clamped_kl = torch.clamp(raw_kl - self.free_bits, min=0.0)
        
        # 4. 组装总 Loss（给模型优化的真正目标）
        total_loss = recon_loss + self.beta * clamped_kl
        
        # 🚨 注意看返回值：我们返回 total_loss 给优化器，但返回 raw_kl 给 WandB！
        return total_loss, recon_loss, raw_kl


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

