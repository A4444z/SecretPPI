"""
损失函数模块 (CMAE 架构)。
实现 InfoNCE 损失和基于掩码的 D-RMSD 重构损失，用于学习蛋白质互作流形。
"""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

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
    """
    InfoNCE / NT-Xent 对比损失。
    将同一个复合物的两个视图拉近，将 Batch 内其他复合物推开。
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        参数:
            z1: [B, D] L2 归一化后的视图1表征
            z2: [B, D] L2 归一化后的视图2表征
        """
        B = z1.shape[0]
        # 拼接成 [2B, D] 的大张量
        z = torch.cat([z1, z2], dim=0)
        
        # 计算余弦相似度矩阵 [2B, 2B] (因为 z 已经归一化，点乘就是余弦相似度)
        sim = torch.matmul(z, z.T) / self.temperature

        # 构建正样本索引
        # 对于 z1[i]，正样本是 z2[i]，索引为 i + B
        # 对于 z2[i]，正样本是 z1[i]，索引为 i (因为 i 本身是 i+B 减去 B)
        positives = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z1.device)

        # 提取正样本的相似度 [2B, 1]
        pos_sim = sim[torch.arange(2*B), positives].unsqueeze(1)

        # 构建 Logits 掩码，去除自身相似度 (对角线)
        logits_mask = ~torch.eye(2*B, dtype=torch.bool, device=z.device)
        
        # 取出非自身的相似度 [2B, 2B - 1] 作为分母候选
        logits = sim[logits_mask].view(2*B, -1)

        # InfoNCE = -log( exp(pos) / sum(exp(all_except_self)) )
        # 为了数值稳定，通常用 log_softmax 或手动平移
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum(dim=1, keepdim=True)
        loss = - torch.log(torch.exp(pos_sim) / denom)
        
        return loss.mean()

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

# ================= 附录：清理冗余 =================
# 旧的 KLLoss, BetaScheduler, VAELoss, CoordinateDecoder 已被彻底删除。