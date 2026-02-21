"""
GlueVAE 分布式训练脚本 (DDP)。
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

from src.models.glue_vae_solo import GlueVAE
from src.utils.loss_solo import VAELoss, BetaScheduler
from src.data.dataset import GlueVAEDataset
from datetime import timedelta
import datetime


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
    beta_scheduler,
    device,
    epoch,
    config,
    rank,
    wandb_logger=None
):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
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
            print("🚀 首批数据健康体检报告")
            print(f"  总原子数: {batch.pos.size(0)}")
            print(f"  [pos] 是否含 NaN/Inf: {torch.isnan(batch.pos).any().item() or torch.isinf(batch.pos).any().item()}")
            print(f"  [pos] 数值范围: 最小值 {batch.pos.min().item():.2f}, 最大值 {batch.pos.max().item():.2f}")
            print(f"  [x (原子序数)] 是否含 NaN: {torch.isnan(batch.x).any().item()}")
            print(f"  [edge_attr] 是否含 NaN: {torch.isnan(batch.edge_attr).any().item()}")
            print(f"  [vector_features] 是否含 NaN: {torch.isnan(batch.vector_features).any().item()}")
            print("="*40 + "\n")
        
        pos_pred, mu, logvar = model(
            z=batch.x,
            vector_features=batch.vector_features,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            pos=batch.pos,
            residue_index=batch.residue_index,
            mask_interface=batch.mask_interface,  # 👈 新增：把界面掩码传给模型
            batch_idx=batch.batch                 # 👈 新增：必须传 batch 索引
        )
        
        if rank == 0 and epoch == 0 and batch_idx == 0:
            print("\n[DEBUG] forward finite check")
            print("  pos_pred finite:", torch.isfinite(pos_pred).all().item())
            print("  mu finite:", torch.isfinite(mu).all().item())
            print("  logvar finite:", torch.isfinite(logvar).all().item())
            print(f"  logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
            print(f"  pos_pred range: [{pos_pred.min().item():.4f}, {pos_pred.max().item():.4f}]")
        
        beta = beta_scheduler.update()
        criterion.beta = beta
        
        loss, recon_loss, kl_loss = criterion(
            pos_pred=pos_pred,
            pos_true=batch.pos,
            mu=mu,
            logvar=logvar,
            mask=batch.mask_interface, batch_idx=batch.batch
        )
        
        if rank == 0 and epoch == 0 and batch_idx == 0:
            print("\n[DEBUG] loss finite check")
            print("  loss finite:", torch.isfinite(loss).item())
            print("  recon finite:", torch.isfinite(recon_loss).item())
            print("  kl finite:", torch.isfinite(kl_loss).item())
        
        optimizer.zero_grad()
        loss.backward()
        
        if rank == 0 and epoch == 0 and batch_idx == 0:
            bad_grad = False
            for n, p in model.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    print(f"[DEBUG] bad grad in: {n}")
                    bad_grad = True
                    break
            print("  grad finite:", not bad_grad)
        
        max_grad_norm = config['training']['max_grad_norm']
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'beta': f'{beta:.4f}'
            })
        
        if (batch_idx + 1) % log_interval == 0 and rank == 0 and wandb_logger is not None:
            wandb_logger.log({
                'train/loss': loss.item(),
                'train/recon_loss': recon_loss.item(),
                'train/kl_loss': kl_loss.item(),
                'train/beta': beta,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
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
    """验证模型。"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader
    
    for batch in pbar:
        batch = batch.to(device)
        
        pos_pred, mu, logvar = model(
            z=batch.x,
            vector_features=batch.vector_features,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            pos=batch.pos,
            residue_index=batch.residue_index,
            mask_interface=batch.mask_interface,  # 👈 新增：把界面掩码传给模型
            batch_idx=batch.batch                 # 👈 新增：必须传 batch 索引
        )
        
        loss, recon_loss, kl_loss = criterion(
            pos_pred=pos_pred,
            pos_true=batch.pos,
            mu=mu,
            logvar=logvar,
            mask=batch.mask_interface, batch_idx=batch.batch
        )
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon_loss / num_batches
    avg_kl = total_kl_loss / num_batches
    
    if rank == 0 and wandb_logger is not None:
        wandb_logger.log({
            'val/loss': avg_loss,
            'val/recon_loss': avg_recon,
            'val/kl_loss': avg_kl,
            'epoch': epoch
        })
    
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon,
        'kl_loss': avg_kl
    }


def save_checkpoint(
    model,
    optimizer,
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
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # ================= 🚨 新增：时间戳和动态文件名 =================
    # 获取当前时间，格式例如：20260220_153045
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构造带有时间戳和 epoch 的文件名
    filename = f"checkpoint_{timestamp}_epoch_{epoch}.pt"
    save_path = os.path.join(save_dir, filename)
    
    # 保存带有时间戳的实体文件
    torch.save(checkpoint, save_path)
    # ==============================================================
    
    # 顺手保存一个 `checkpoint_latest.pt`
    # 这样做的好处是：你下次写启动脚本时，--resume 参数永远可以直接指向这个 latest 文件，不用每次都去抄长长的带时间戳的名字
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = os.path.join(save_dir, f'checkpoint_best.pt')
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description='GlueVAE DDP Training')
    parser.add_argument('--config', type=str, default='config_solo.yaml', help='Path to config file')
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
        print(f"=== DDP Training Starting ===")
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
    
    # if rank != 0:# 🔴 注意！这里必须 barrier 等待 rank 0 加载完成，否则会存8个缓存文件
    #     dist.barrier()

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
            
    else:  # 🔴 注意！这个 else 必须和最外层的 if args.overfit_test: 对齐！
        val_aug = aug_config.get('val', {})
        val_use_rotation = val_aug.get('random_rotation', False)
        
        # 安全读取 max_samples 传给 Dataset
        current_max_samples = config['data'].get('max_samples', None)

        # ================= 🚨 终极防线：绝对串行化构建数据集 =================
        # 第一阶段：Rank 0 独占执行，其他卡在后面的 barrier 罚站
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
            # 强行触发所有的扫库、写缓存动作
            total_len = len(train_full_dataset)
            _ = len(val_full_dataset)
            print("Rank 0: 缓存构建完毕！")

        # 【路障 1】Rank 1-7 之前一直在这里等。现在 Rank 0 到了，所有人一起放行。
        # (采纳 Codex 建议，加入 device_ids 消除 NCCL 警告)
        dist.barrier(device_ids=[local_rank])

        # 第二阶段：Rank 1-7 安全读取
        if rank != 0:
            import time
            print(f"Rank {rank}: 正在等待 NFS 同步 (10秒)...")
            time.sleep(10) # 给超算网络一点时间同步文件
            
            # 此时 PyG 的 processed 文件和我们自己的 pkl 都稳稳地在硬盘上了
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

        # 【路障 2】Rank 0 刚才瞬间就到了这里，等 Rank 1-7 读完数据后，大家再一起往下走。
        dist.barrier(device_ids=[local_rank])
        # =======================================================================

        if total_len == 1:
            train_dataset = train_full_dataset
            val_dataset = val_full_dataset
            if rank == 0:
                print("Warning: Only one sample in dataset, using it for both train and validation")
        else:
            train_len = max(1, int(total_len * config['data']['train_split']))
            val_len = max(1, total_len - train_len)

            if train_len + val_len > total_len:
                train_len = total_len // 2
                val_len = total_len - train_len

            indices = torch.randperm(
                total_len,
                generator=torch.Generator().manual_seed(args.seed)
            ).tolist()

            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len + val_len]

            train_dataset = Subset(train_full_dataset, train_indices)
            val_dataset = Subset(val_full_dataset, val_indices)

        if rank == 0:
            print(f"Total dataset size: {total_len}")
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Val dataset size: {len(val_dataset)}")

        # 🔴 新增这行：强行触发 val_full_dataset 的 _load_keys()，
            # 让 Rank 0 顺手把 keys_cache_val.pkl 也给生成了！
            # _ = len(val_full_dataset)
    
    # if rank == 0:#第二次调取barrier，启动其他7个进程
    #     dist.barrier()

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
    
    model = GlueVAE(
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        edge_dim=config['model']['edge_dim'],
        vocab_size=config['model']['vocab_size'],
        use_gradient_checkpointing=config['model']['use_gradient_checkpointing']
    ).to(device)
    
    # ✅ 修改为：告诉 DDP，如果发现有参数没用到，不要报错，直接忽略它们！
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
    
    criterion = VAELoss(beta=config['training']['beta']['start'])
    
    beta_scheduler = BetaScheduler(
        beta_start=config['training']['beta']['start'],
        beta_end=config['training']['beta']['end'],
        warmup_steps=config['training']['beta']['warmup_steps'],
        schedule_type=config['training']['beta']['schedule_type']
    )
    
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
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
    
    # ================= 🚨 修复 DDP 断点加载 Bug =================
    # 去掉了 `and rank == 0`，保证 8 张卡同时同步加载权重！
    if args.resume is not None:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        
        # map_location=device 确保每张卡只把权重读到自己的专属显存里，不会爆内存
        checkpoint = torch.load(args.resume, map_location=device)
        
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['step']
        
        if rank == 0:
            print(f"✅ Resumed at epoch {start_epoch}, step {global_step}")
    # ==========================================================
    
    if rank == 0:
        print("Starting training...")
    
    save_dir = config['logging']['save_dir']
    val_interval = config['logging']['val_interval']
    save_interval = config['logging']['save_interval']
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        train_sampler.set_epoch(epoch)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            beta_scheduler, device, epoch, config, rank, wandb_logger
        )
        
        if rank == 0:
            print(f"Epoch {epoch} - Train: loss={train_metrics['loss']:.4f}, "
                  f"recon={train_metrics['recon_loss']:.4f}, "
                  f"kl={train_metrics['kl_loss']:.4f}")
        
        val_interval_epochs = config['logging'].get('val_interval', 5)
        # 🔴 新增逻辑：达到间隔，或者是最后一个 Epoch，强制进行验证！
        should_validate = (epoch + 1) % val_interval_epochs == 0 or (epoch == config['training']['num_epochs'] - 1)
        
        if should_validate:
            val_sampler.set_epoch(epoch)
            val_metrics = validate(
                model, val_loader, criterion, device, epoch, config, rank, wandb_logger
            )
            
            if rank == 0:
                print(f"Epoch {epoch} - Val: loss={val_metrics['loss']:.4f}, "
                      f"recon={val_metrics['recon_loss']:.4f}, "
                      f"kl={val_metrics['kl_loss']:.4f}")
                
                if scheduler is not None:
                    scheduler.step(val_metrics['loss'])
                
                is_best = val_metrics['loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['loss']
                    print(f"New best val loss: {best_val_loss:.4f}")
                
                global_step = (epoch + 1) * len(train_loader)
                save_checkpoint(model, optimizer, epoch, global_step, save_dir, is_best)
        else:
            save_interval_epochs = config['logging'].get('save_interval', 10)
            if rank == 0 and (epoch + 1) % save_interval_epochs == 0:
                global_step = (epoch + 1) * len(train_loader)
                save_checkpoint(model, optimizer, epoch, global_step, save_dir, is_best=False)
    
    if rank == 0:
        print("Training complete!")
        if wandb_logger is not None:
            wandb_logger.finish()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
