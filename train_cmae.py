"""
GlueVAE/CMAE 分布式训练脚本 (DDP)。
使用 torchrun + DistributedDataParallel，支持多卡并行训练。
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DistributedSampler, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
import yaml
from tqdm import tqdm
from datetime import timedelta
import datetime

# ================= 🚨 新的 CMAE 导入 =================
from src.models.glue_cmae import GlueVAE  # 假设你在 glue_cmae.py 里类名还是叫 GlueVAE
from src.utils.loss_cmae import CMAELoss
# 移除了 VAELoss 和 BetaScheduler
from src.data.dataset import GlueVAEDataset
# ====================================================

def set_seed(seed=42):
    """设置随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载 YAML 配置文件。"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config, rank):
    """设置 wandb 日志（如果启用且 rank 仅 rank 0）。"""
    if rank == 0 and config['logging']['use_wandb']:
        try:
            import wandb
            wandb.init(
                project=config['logging']['project'],
                entity=config['logging']['entity'],
                config=config
            )
            return wandb
        except ImportError:
            print("Warning: wandb not installed, disabling logging")
            config['logging']['use_wandb'] = False
    return None


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,            # 👈 删除了 beta_scheduler
    epoch,
    config,
    rank,
    wandb_logger=None
    ):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_contrast_loss = 0.0  # 👈 替换了 kl_loss
    num_batches = 0
    
    log_interval = config['logging']['log_interval']
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        if rank == 0 and epoch == 0 and batch_idx == 0:
            print("\n" + "="*40)
            print("🚀 首批数据健康体检报告 (CMAE版)")
            print(f"  总原子数: {batch.pos.size(0)}")
            print(f"  [pos] 是否含 NaN/Inf: {torch.isnan(batch.pos).any().item() or torch.isinf(batch.pos).any().item()}")
            print(f"  [pos] 数值范围: 最小值 {batch.pos.min().item():.2f}, 最大值 {batch.pos.max().item():.2f}")
            print(f"  [is_ligand] 是否存在: {hasattr(batch, 'is_ligand')}")
            print("="*40 + "\n")
        
        # ================= 🚨 新的前向传播接口 =================
        z1, z2, pos_pred_v1, mask_v1, batch_entropy = model(
            z=batch.x,
            vector_features=batch.vector_features,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            pos=batch.pos,
            residue_index=batch.residue_index,
            is_ligand=batch.is_ligand,            # 👈 新增：受体配体标签
            mask_interface=batch.mask_interface,  
            batch_idx=batch.batch                 
        )
        
        if rank == 0 and epoch == 0 and batch_idx == 0:
            print("\n[DEBUG] forward finite check")
            print("  z1 finite:", torch.isfinite(z1).all().item())
            print("  z2 finite:", torch.isfinite(z2).all().item())
            print("  pos_pred_v1 finite:", torch.isfinite(pos_pred_v1).all().item())
            print(f"  mask_v1 sum (破坏的原子数): {mask_v1.sum().item()}")
        
        # ================= 🚨 新的损失计算接口 =================
        loss, contrast_loss, recon_loss = criterion(
            z1=z1,
            z2=z2,
            pos_pred_v1=pos_pred_v1,
            pos_true=batch.pos,
            mask_v1=mask_v1,
            batch_idx=batch.batch
        )

        # 🚨 核心逻辑：加入熵正则化惩罚 (减去熵，即鼓励注意力分散)
        # 这里的 0.01 是控制熵正则化强度的超参数
        # 使用一个新的变量名 step_loss，千万不要覆盖外层的 total_loss

        ent_weight = config['training'].get('entropy_weight', 0.01)
        step_loss = loss - ent_weight * batch_entropy
        
        if rank == 0 and epoch == 0 and batch_idx == 0:
            print("\n[DEBUG] loss finite check")
            print("  loss finite:", torch.isfinite(loss).item())
            print("  contrast_loss finite:", torch.isfinite(contrast_loss).item())
            print("  recon_loss finite:", torch.isfinite(recon_loss).item())
        
        optimizer.zero_grad()
        step_loss.backward()  # 👈 对 step_loss 反向传播
        
        # ================= 🚨 新增：高精度梯度探针 =================
        if rank == 0 and batch_idx % 50 == 0:
            proj_grad = model.module.projector.mlp[0].weight.grad
            attn_grad = model.module.attn_pooling.attn_mlp[0].weight.grad
            scale_grad = criterion.logit_scale.grad
            
            # 显式判断 None，并使用科学计数法 (.3e) 打印极其微小的梯度
            proj_str = "None" if proj_grad is None else f"{proj_grad.norm().item():.3e}"
            attn_str = "None" if attn_grad is None else f"{attn_grad.norm().item():.3e}"
            scale_str = "None" if scale_grad is None else f"{scale_grad.item():.3e}"
            
            print(f"\n[GRAD CHECK] Proj: {proj_str} | Attn: {attn_str} | Temp Scale: {scale_str}")
        # =========================================================================

        max_grad_norm = config['training']['max_grad_norm']
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # 累加器保持 float 相加
        total_loss += step_loss.item() 
        total_recon_loss += recon_loss.item()
        total_contrast_loss += contrast_loss.item()
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{step_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'contrast': f'{contrast_loss.item():.4f}'
            })
        
        if (batch_idx + 1) % log_interval == 0 and rank == 0 and wandb_logger is not None:

            current_temp = 1.0 / criterion.logit_scale.exp().item()

            wandb_logger.log({
                'train/loss': step_loss.item(),
                'train/recon_loss': recon_loss.item(),
                'train/contrast_loss': contrast_loss.item(),
                'train/entropy': batch_entropy.item(),  # 👈 新增：在 WandB 监控注意力熵！
                'train/temperature': current_temp,  # 👈 新增：监控当前温度！
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': batch_idx
            })
        # =========================================================
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'contrast_loss': total_contrast_loss / num_batches
    }

@torch.no_grad()
def validate(
    model,
    val_loader,
    criterion,
    device,
    epoch,
    config,
    rank,
    wandb_logger=None
):
    """验证模型，并计算 Top-1 检索准确率。"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_contrast_loss = 0.0
    
    # 新增：用于统计相似度和准确率
    all_top1_acc = []
    all_pos_sim = []
    all_neg_sim = []
    
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader
    
    for batch in pbar:
        batch = batch.to(device)
        
        # 前向传播
        graph_z1, graph_z2, pos_pred_v1, mask_v1, batch_entropy = model(
            z=batch.x,
            vector_features=batch.vector_features,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            pos=batch.pos,
            residue_index=batch.residue_index,
            is_ligand=batch.is_ligand,
            mask_interface=batch.mask_interface,
            batch_idx=batch.batch
        )
        
        # 计算基础 Loss
        loss, contrast_loss, recon_loss = criterion(
            z1=graph_z1,
            z2=graph_z2,
            pos_pred_v1=pos_pred_v1,
            pos_true=batch.pos,
            mask_v1=mask_v1,
            batch_idx=batch.batch
        )
        
        # ================= 🚨 新增：潜在相似度与检索测试逻辑 =================
        ent_weight = config['training'].get('entropy_weight', 0.01)
        val_total_loss = loss - ent_weight * batch_entropy
        
        # ================= 🚨 修复 2：验证集全局检索 (All-Gather) =================
        if dist.is_initialized():
            z1_list = [torch.zeros_like(graph_z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(graph_z2) for _ in range(dist.get_world_size())]
            dist.all_gather(z1_list, graph_z1)
            dist.all_gather(z2_list, graph_z2)
            
            # 把当前 rank 的放在最前面，确保对角线 targets 依然成立
            rank_idx = dist.get_rank()
            z1_list[rank_idx] = graph_z1
            z2_list[rank_idx] = graph_z2
            
            graph_z1_global = torch.cat(z1_list, dim=0)
            graph_z2_global = torch.cat(z2_list, dim=0)
        else:
            graph_z1_global = graph_z1
            graph_z2_global = graph_z2

        # 1. 计算余弦相似度矩阵 [B, B]
        # 因为 graph_z 已做 L2 归一化，点积即余弦相似度
        sim_matrix = torch.matmul(graph_z1_global, graph_z2_global.T) 
        
        # 2. 计算 Top-1 检索准确率 (这一行预测的是不是它自己)
        preds = sim_matrix.argmax(dim=-1)
        targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        top1_acc = (preds == targets).float().mean()
        
        # 3. 统计正负样本相似度
        pos_sim = torch.diagonal(sim_matrix).mean() # 对角线：正样本
        
        # 负样本：排除对角线后的平均值
        mask_neg = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
        neg_sim = sim_matrix[mask_neg].mean()
        
        # 汇总
        all_top1_acc.append(top1_acc.item())
        all_pos_sim.append(pos_sim.item())
        all_neg_sim.append(neg_sim.item())
        # ===================================================================
        
        # 🚨 修复 1 的收尾：累加的 loss 改为 val_total_loss
        total_loss += val_total_loss.item() 
        total_recon_loss += recon_loss.item()
        total_contrast_loss += contrast_loss.item()
        num_batches += 1
    
    # 计算全验证集的平均指标
    avg_loss = total_loss / num_batches
    avg_recon = total_recon_loss / num_batches
    avg_contrast = total_contrast_loss / num_batches
    avg_top1 = sum(all_top1_acc) / len(all_top1_acc)
    avg_pos_sim = sum(all_pos_sim) / len(all_pos_sim)
    avg_neg_sim = sum(all_neg_sim) / len(all_neg_sim)
    
    if rank == 0:
        print(f"\n[VAL REPORT] Top-1 Acc: {avg_top1:.4f} | Pos Sim: {avg_pos_sim:.4f} | Neg Sim: {avg_neg_sim:.4f}")
        
        if wandb_logger is not None:
            wandb_logger.log({
                'val/loss': avg_loss,
                'val/recon_loss': avg_recon,
                'val/contrast_loss': avg_contrast,
                'val/retrieval_top1': avg_top1,    # 👈 检索准确率
                'val/sim_positive': avg_pos_sim,  # 👈 正样本相似度
                'val/sim_negative': avg_neg_sim,  # 👈 负样本相似度
                'epoch': epoch
            })
    
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon,
        'contrast_loss': avg_contrast,
        'top1_acc': avg_top1
    }


import os
import torch
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP

def save_checkpoint(
    model,
    optimizer,
    criterion,
    epoch,
    step,
    save_dir,
    is_best=False
):
    """保存模型检查点（仅 rank 0）。"""
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state_dict,
        'criterion_state_dict': criterion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 🚨 获取当前进程号 (PID)
    pid = os.getpid()
    
    # 🚨 构造带有 进程号 和 epoch 的文件名
    # 格式为：checkpoint_进程号_epoch_n.pt
    filename = f"checkpoint_{pid}_epoch_{epoch}.pt"
    save_path = os.path.join(save_dir, filename)
    
    # 保存实体文件
    torch.save(checkpoint, save_path)
    
    # 顺手保存一个 `checkpoint_latest.pt` (加上进程号以免多开任务时互相覆盖)
    latest_path = os.path.join(save_dir, f'checkpoint_{pid}_latest.pt')
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = os.path.join(save_dir, f'checkpoint_{pid}_best.pt')
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description='GlueVAE CMAE DDP Training')
    parser.add_argument('--config', type=str, default='config_cmae.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--overfit_test', action='store_true', help='Enable overfit test mode')
    args = parser.parse_args()
    
    # DDP 初始化
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(hours=10)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"=== DDP Training Starting (CMAE) ===")
        print(f"World size: {world_size}")
        print(f"Using device: {device}")
    
    set_seed(args.seed + rank)
    
    config = load_config(args.config)
    
    wandb_logger = setup_wandb(config, rank)
    
    aug_config = config.get('augmentation', {})
    train_aug = aug_config.get('train', {})
    use_rotation = train_aug.get('random_rotation', True)
    
    if rank == 0:
        print("Loading dataset...")
    
    if args.overfit_test:
        if rank == 0:
            print("!!! RUNNING IN OVERFIT TEST MODE !!!")
        batch_size = config['data']['batch_size']
        full_dataset = GlueVAEDataset(
            root=config['data']['root_dir'],
            lmdb_path=config['data']['lmdb_path'],
            split='train',
            exclude_pdb_json=None,
            random_rotation=use_rotation,
            max_samples=batch_size
        )
        train_dataset = full_dataset
        val_dataset = full_dataset
        if rank == 0:
            print(f"Overfit test: Using {len(train_dataset)} samples for both train and val")
            
    else: 
        val_aug = aug_config.get('val', {})
        val_use_rotation = val_aug.get('random_rotation', False)
        
        current_max_samples = config['data'].get('max_samples', None)

        if rank == 0:
            print("Rank 0: 开始构建 PyG 元文件与 LMDB 缓存...")
            train_full_dataset = GlueVAEDataset(
                root=config['data']['root_dir'],
                lmdb_path=config['data']['lmdb_path'],
                split='train',
                exclude_pdb_json=config['data'].get('exclude_pdb_json'),
                random_rotation=use_rotation,
                max_samples=current_max_samples
            )
            val_full_dataset = GlueVAEDataset(
                root=config['data']['root_dir'],
                lmdb_path=config['data']['lmdb_path'],
                split='val',
                exclude_pdb_json=config['data'].get('exclude_pdb_json'),
                random_rotation=val_use_rotation,
                max_samples=current_max_samples
            )
            total_len = len(train_full_dataset)
            _ = len(val_full_dataset)
            print("Rank 0: 缓存构建完毕！")

        dist.barrier(device_ids=[local_rank])

        if rank != 0:
            import time
            print(f"Rank {rank}: 正在等待 NFS 同步 (10秒)...")
            time.sleep(10) 
            
            train_full_dataset = GlueVAEDataset(
                root=config['data']['root_dir'],
                lmdb_path=config['data']['lmdb_path'],
                split='train',
                exclude_pdb_json=config['data'].get('exclude_pdb_json'),
                random_rotation=use_rotation,
                max_samples=current_max_samples
            )
            val_full_dataset = GlueVAEDataset(
                root=config['data']['root_dir'],
                lmdb_path=config['data']['lmdb_path'],
                split='val',
                exclude_pdb_json=config['data'].get('exclude_pdb_json'),
                random_rotation=val_use_rotation,
                max_samples=current_max_samples
            )
            total_len = len(train_full_dataset)

        dist.barrier(device_ids=[local_rank])

        # 👇 -------- 把这几行粘贴进去 -------- 👇
        # 直接使用我们在上面配置好的专门的 train 和 val 数据集，杜绝索引错位！
        train_dataset = train_full_dataset
        val_dataset = val_full_dataset
        # 👆 -------- 粘贴结束 -------- 👆

        if rank == 0:
            print(f"Total dataset size: {total_len}")
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Val dataset size: {len(val_dataset)}")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=(config['data']['num_workers'] > 0)
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=(config['data']['num_workers'] > 0)
    )
    
    if rank == 0:
        print("Initializing model...")
    
    # 🚨 注意：这里不要传 latent_dim 了，Projector 里写死或者在 config 里加 proj_dim
    model = GlueVAE(
        hidden_dim=config['model']['hidden_dim'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        edge_dim=config['model']['edge_dim'],
        vocab_size=config['model']['vocab_size'],
        use_gradient_checkpointing=config['model']['use_gradient_checkpointing'],
        mask_noise=config['model'].get('mask_noise',0.5)
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
    
    # ================= 🚨 新增：实例化 CMAELoss =================
    # 从配置文件中读取超参数，如果没有就用默认值
    temp = config['training'].get('temperature', 0.1)
    l_contrast = config['training'].get('lambda_contrast', 1.0)
    l_recon = config['training'].get('lambda_recon', 0.5)
    recon_cutoff = config['training'].get('recon_cutoff', 15.0) # 👈 新增读取
    
    criterion = CMAELoss(
        temperature=temp,
        lambda_contrast=l_contrast,
        lambda_recon=l_recon,
        cutoff=recon_cutoff  # 建议使用截断以限制损失仅在局部
    ).to(device)
    # =========================================================
    
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'Adam':
        optimizer = optim.Adam(
        [
        {'params': model.parameters()},  # 第一组：模型的参数
        {'params': criterion.parameters()} # 第二组：损失函数的参数
        ],
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=tuple(optimizer_config['betas']),
            eps=optimizer_config['eps']
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")
    
    scheduler_config = config['training']['scheduler']
    if scheduler_config['type'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'],
            min_lr=scheduler_config['min_lr']
        )
    else:
        scheduler = None
    
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume is not None:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location=device)
        
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        # 🚨 新增：安全加载 criterion 的状态（为了兼容旧版没有保存该字段的 checkpoint）
        if 'criterion_state_dict' in checkpoint:
            criterion.load_state_dict(checkpoint['criterion_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['step']
        
        if rank == 0:
            print(f"✅ Resumed at epoch {start_epoch}, step {global_step}")
    
    if rank == 0:
        print("Starting training...")
    
    save_dir = config['logging']['save_dir']
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        train_sampler.set_epoch(epoch)
        
        # 🚨 注意去掉了 beta_scheduler 传参
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, config, rank, wandb_logger
        )
        
        if rank == 0:
            # 打印日志换成了 contrast_loss
            print(f"Epoch {epoch} - Train: loss={train_metrics['loss']:.4f}, "
                  f"recon={train_metrics['recon_loss']:.4f}, "
                  f"contrast={train_metrics['contrast_loss']:.4f}")
        
        val_interval_epochs = config['logging'].get('val_interval', 5)
        should_validate = (epoch + 1) % val_interval_epochs == 0 or (epoch == config['training']['num_epochs'] - 1)
        
        if should_validate:
            val_sampler.set_epoch(epoch)
            val_metrics = validate(
                model, val_loader, criterion, device, epoch, config, rank, wandb_logger
            )
            
            if rank == 0:
                print(f"Epoch {epoch} - Val: loss={val_metrics['loss']:.4f}, "
                      f"recon={val_metrics['recon_loss']:.4f}, "
                      f"contrast={val_metrics['contrast_loss']:.4f}")
                
                if scheduler is not None:
                    scheduler.step(val_metrics['loss'])
                
                is_best = val_metrics['loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['loss']
                    print(f"New best val loss: {best_val_loss:.4f}")
                
                global_step = (epoch + 1) * len(train_loader)
                save_checkpoint(model, optimizer, criterion, epoch, global_step, save_dir, is_best)
        else:
            save_interval_epochs = config['logging'].get('save_interval', 10)
            if rank == 0 and (epoch + 1) % save_interval_epochs == 0:
                global_step = (epoch + 1) * len(train_loader)
                save_checkpoint(model, optimizer, criterion, epoch, global_step, save_dir, is_best=False)
    
    if rank == 0:
        print("Training complete!")
        if wandb_logger is not None:
            wandb_logger.finish()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()