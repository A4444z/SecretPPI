"""
GlueVAE 训练脚本。
负责初始化模型、加载数据集、定义损失函数并执行训练循环。
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import yaml
from tqdm import tqdm

# 导入我们的模块
from src.models.glue_vae_solo import GlueVAE
from src.utils.loss_solo import VAELoss, BetaScheduler
from src.data.dataset import GlueVAEDataset


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


def setup_wandb(config):
    """设置 wandb 日志（如果启用）。"""
    if config['logging']['use_wandb']:
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
    wandb_logger=None
):
    """
    训练一个 epoch。
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    log_interval = config['logging']['log_interval']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # 前向传播
        pos_pred, mu, logvar = model(
            z=batch.x,
            vector_features=batch.vector_features,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            pos=batch.pos,
            residue_index=batch.residue_index
        )
        
        # 更新 beta
        beta = beta_scheduler.update()
        criterion.beta = beta
        
        # 计算损失
        loss, recon_loss, kl_loss = criterion(
            pos_pred=pos_pred,
            pos_true=batch.pos,
            mu=mu,
            logvar=logvar,
            mask=batch.mask_interface, batch_idx=batch.batch
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        max_grad_norm = config['training']['max_grad_norm']
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'beta': f'{beta:.4f}'
        })
        
        # 日志记录
        if (batch_idx + 1) % log_interval == 0 and wandb_logger is not None:
            wandb_logger.log({
                'train/loss': loss.item(),
                'train/recon_loss': recon_loss.item(),
                'train/kl_loss': kl_loss.item(),
                'train/beta': beta,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': batch_idx
            })
    
    # 返回平均损失
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
    wandb_logger=None
):
    """
    验证模型。
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        batch = batch.to(device)
        
        pos_pred, mu, logvar = model(
            z=batch.x,
            vector_features=batch.vector_features,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            pos=batch.pos,
            residue_index=batch.residue_index
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
    
    if wandb_logger is not None:
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
    """保存模型检查点。"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 保存最新检查点
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    # 保存最佳检查点
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description='GlueVAE Training')
    parser.add_argument('--config', type=str, default='config_solo.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置 wandb
    wandb_logger = setup_wandb(config)
    
    # 加载数据集
    print("Loading dataset...")
    full_dataset = GlueVAEDataset(
        root=config['data']['root_dir'],
        lmdb_path=config['data']['lmdb_path'],
        split='train',
        exclude_pdb_json=config['data'].get('exclude_pdb_json')
    )
    
    # 划分训练集和验证集
    total_len = len(full_dataset)
    
    if total_len == 1:
        # 如果只有一个样本，就让它既作为训练集又作为验证集
        train_dataset = full_dataset
        val_dataset = full_dataset
        print("Warning: Only one sample in dataset, using it for both train and validation")
    else:
        train_len = max(1, int(total_len * config['data']['train_split']))  # 确保训练集至少有1个样本
        val_len = max(1, total_len - train_len)  # 确保验证集至少有1个样本
        
        if train_len + val_len > total_len:  # 如果超过总数，调整为各一半
            train_len = total_len // 2
            val_len = total_len - train_len
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(args.seed)
        )
    
    print(f"Total dataset size: {total_len}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # 创建 DataLoader
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # 初始化模型
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
    
    # 计算模型参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # 初始化损失函数
    criterion = VAELoss(beta=config['training']['beta']['start'])
    
    # 初始化 beta 调度器
    beta_scheduler = BetaScheduler(
        beta_start=config['training']['beta']['start'],
        beta_end=config['training']['beta']['end'],
        warmup_steps=config['training']['beta']['warmup_steps'],
        schedule_type=config['training']['beta']['schedule_type']
    )
    
    # 初始化优化器
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
    
    # 初始化学习率调度器
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
    
    # 恢复检查点
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['step']
        print(f"Resumed at epoch {start_epoch}, step {global_step}")
    
    # 训练循环
    print("Starting training...")
    save_dir = config['logging']['save_dir']
    val_interval = config['logging']['val_interval']
    save_interval = config['logging']['save_interval']
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            beta_scheduler, device, epoch, config, wandb_logger
        )
        
        print(f"Epoch {epoch} - Train: loss={train_metrics['loss']:.4f}, "
              f"recon={train_metrics['recon_loss']:.4f}, "
              f"kl={train_metrics['kl_loss']:.4f}")
        
        # 验证
        # 对于小数据集，确保每个epoch都进行验证
        should_validate = (len(train_loader) == 1) or ((epoch + 1) % max(1, val_interval // len(train_loader) + 1) == 0)
        
        if should_validate:
            val_metrics = validate(
                model, val_loader, criterion, device, epoch, config, wandb_logger
            )
            
            print(f"Epoch {epoch} - Val: loss={val_metrics['loss']:.4f}, "
                  f"recon={val_metrics['recon_loss']:.4f}, "
                  f"kl={val_metrics['kl_loss']:.4f}")
            
            # 更新学习率调度器
            if scheduler is not None:
                scheduler.step(val_metrics['loss'])
            
            # 保存最佳模型
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                print(f"New best val loss: {best_val_loss:.4f}")
            
            # 保存检查点
            global_step = (epoch + 1) * len(train_loader)
            save_checkpoint(model, optimizer, epoch, global_step, save_dir, is_best)
        else:
            # 即使不验证，也考虑定期保存检查点
            if (epoch + 1) % 1 == 0:  # 每个epoch都保存检查点用于测试
                global_step = (epoch + 1) * len(train_loader)
                save_checkpoint(model, optimizer, epoch, global_step, save_dir, is_best=False)
    
    print("Training complete!")
    
    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    main()