"""
损失函数模块 (CMAE 架构)。
实现 InfoNCE 损失和基于掩码的 D-RMSD 重构损失，用于学习蛋白质互作流形。
"""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
import torch.distributed as dist

# ================= 1. 核心重构损失：Masked D-RMSD =================

class MaskedDRMSDLoss(nn.Module):
    """
    掩码 D-RMSD 损失函数 (带局部截断)。
    🚨 关键特性：只计算被掩码（被破坏）的原子与其它有效原子之间的相对距离误差。
    """
    
    def __init__(self, reduction='mean', cutoff=15.0):
        super().__init__()
        self.reduction = reduction
        self.cutoff = cutoff 
        
    def forward(
        self,
        pos_pred,
        pos_true,
        mask_v1,     # 👈 必须传入 forward 中生成的 mask_v1
        batch_idx=None
    ):
        if batch_idx is None:
            batch_idx = torch.zeros(pos_pred.size(0), dtype=torch.long, device=pos_pred.device)
        
        # 转为密集矩阵 [B, N_max, 3]
        pos_pred_dense, batch_mask = to_dense_batch(pos_pred, batch_idx)
        pos_true_dense, _ = to_dense_batch(pos_true, batch_idx)
        mask_v1_dense, _ = to_dense_batch(mask_v1.float(), batch_idx) # [B, N_max]
        
        # 计算成对距离 [B, N_max, N_max]
        D_pred = torch.cdist(pos_pred_dense, pos_pred_dense, p=2.0)
        D_true = torch.cdist(pos_true_dense, pos_true_dense, p=2.0)
        
        mse = (D_pred - D_true) ** 2
        
        # 1. 有效节点掩码 (去除 padding)
        valid_2d = batch_mask.unsqueeze(1) * batch_mask.unsqueeze(2)
        
        # 2. 局部距离截断掩码 (只关注 cutoff 内的物理互作)
        cutoff_mask = (D_true < self.cutoff).float()
        
        # 去除对角线 (自己到自己)
        eye_mask = 1.0 - torch.eye(D_true.size(1), device=D_true.device).unsqueeze(0)
        
        # 3. 🚨 核心掩码逻辑：至少有一个原子是被炸掉的 (mask_v1)
        # 如果 i 和 j 都没被炸掉，那模型就是作弊照抄，我们不奖励它。
        # 只有当 i 或 j 是被破坏的原子时，重构其距离才有意义。
        mask_i = mask_v1_dense.unsqueeze(2) # [B, N_max, 1]
        mask_j = mask_v1_dense.unsqueeze(1) # [B, 1, N_max]
        
        # mask_i + mask_j > 0 表示：这对原子中至少有一个属于被破坏的区域
        masked_region_mask = (mask_i + mask_j > 0.5).float()
        
        # 组合最终掩码
        final_mask = valid_2d * cutoff_mask * eye_mask * masked_region_mask
        
        mse = mse * final_mask
        
        if self.reduction == 'mean':
            # 分母使用实际参与计算的有效 Masked Pair 数量
            return mse.sum() / (final_mask.sum() + 1e-8)
        return mse.sum()

# ================= 2. 核心对比损失：InfoNCE =================

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        # 🚨 替换固定的 temperature，改为可学习的 logit_scale
        # 初始化为 ln(1/temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, z1, z2):
        # 跨 GPU 全局负样本收集 (All-Gather)
        if dist.is_initialized():
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            dist.all_gather(z1_list, z1)
            dist.all_gather(z2_list, z2)
            
            # 保留本地计算图的梯度
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            
            z1 = torch.cat(z1_list, dim=0)
            z2 = torch.cat(z2_list, dim=0)

        B = z1.size(0) 
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        
        # 🚨 限制最大 scale 为 100 (对应最低温度 0.01)，防止早期梯度爆炸
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        # 🚨 计算余弦相似度并乘以可学习的 scale (代替除以温度)
        logits = torch.matmul(z, z.T) * logit_scale
        
        # 屏蔽对角线 (自己和自己的相似度设为极小值)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(mask, -1e9)
        
        # 构建分类 Target
        targets = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device)
        ], dim=0)
        
        return F.cross_entropy(logits, targets)

# ================= 3. 组合引擎：CMAE Loss =================

class CMAELoss(nn.Module):
    """
    Contrastive Masked Autoencoder 联合损失。
    包含：拉伸流形的 InfoNCE + 物理保真的 Masked_DRMSD
    """
    def __init__(
        self,
        temperature=0.1,
        lambda_contrast=1.0,
        lambda_recon=0.5,
        cutoff=15.0
    ):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_recon = lambda_recon
        
        self.info_nce_loss = InfoNCELoss(temperature=temperature)
        self.drmsd_loss = MaskedDRMSDLoss(reduction='mean', cutoff=cutoff)
        
    # 🚨 新增：方便外部 (train_cmae.py) 直接读取当前温度进行 WandB 记录
    @property
    def logit_scale(self):
        return self.info_nce_loss.logit_scale
        
    def forward(
        self,
        z1,
        z2,
        pos_pred_v1,
        pos_true,
        mask_v1,
        batch_idx
    ):
        # 1. 计算 InfoNCE 对比损失
        contrast_loss = self.info_nce_loss(z1, z2)
        
        # 2. 计算仅在 Masked 区域的重构损失
        recon_loss = self.drmsd_loss(pos_pred_v1, pos_true, mask_v1, batch_idx)
        
        # 3. 组合总损失
        total_loss = self.lambda_contrast * contrast_loss + self.lambda_recon * recon_loss
        
        return total_loss, contrast_loss, recon_loss