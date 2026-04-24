import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.models.glue_cmae import GlueVAE
from src.data.dataset import GlueVAEDataset

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_cmae.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--chunk_size', type=int, default=1024, help='每次处理的查询样本数')
    parser.add_argument('--save_features', action='store_true', help='是否保存提取的特征以备后用')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 特征保存路径
    feat_path = f"{args.checkpoint}_features.pt"

    if os.path.exists(feat_path):
        print(f"♻️ 发现已存在的特征缓存: {feat_path}, 正在直接加载...")
        data = torch.load(feat_path)
        Z1_global = data['Z1']
        Z2_global = data['Z2']
    else:
        print("Loading validation set and extracting features...")
        split_ratio = {
            'train': config['data'].get('train_split', 0.9),
            'val': config['data'].get('val_split', 0.05),
            'test': config['data'].get('test_split', 0.05),
        }
        val_dataset = GlueVAEDataset(
            root=config['data']['root_dir'],
            lmdb_path=config['data']['lmdb_path'],
            split='val',
            exclude_pdb_json=config['data'].get('exclude_pdb_json'),
            random_rotation=False,
            max_samples=config['data'].get('max_samples', None),
            cutoff_radius=config['training'].get('recon_cutoff', 15.0),
            split_ratio=split_ratio
        )
        val_loader = PyGDataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        model = GlueVAE(
            hidden_dim=config['model']['hidden_dim'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            edge_dim=config['model']['edge_dim'],
            vocab_size=config['model']['vocab_size'],
            use_gradient_checkpointing=False
        ).to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()

        all_z1, all_z2 = [], []
        for batch in tqdm(val_loader, desc="Feature Extraction"):
            batch = batch.to(device)
            # 调用 forward，由于 model.eval()，会进入确定性分支
            graph_z1, graph_z2, _, _, _, _ = model(
                z=batch.x, vector_features=batch.vector_features,
                edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                pos=batch.pos, residue_index=batch.residue_index,
                is_ligand=batch.is_ligand, mask_interface=batch.mask_interface,
                batch_idx=batch.batch
            )
            all_z1.append(graph_z1.cpu())
            all_z2.append(graph_z2.cpu())

        Z1_global = torch.cat(all_z1, dim=0)
        Z2_global = torch.cat(all_z2, dim=0)

        if args.save_features:
            torch.save({'Z1': Z1_global, 'Z2': Z2_global}, feat_path)
            print(f"💾 特征已保存至: {feat_path}")

    N_total = Z1_global.size(0)
    print(f"\n📊 正在进行全局检索测试 (库大小: {N_total})...")
    
    # ================= 🚨 分块计算核心逻辑 (避免 1.4TB OOM) =================
    top1_correct = 0
    top5_correct = 0
    total_pos_sim = 0.0
    
    # 将候选库放在 GPU 上 (约 300MB，非常安全)
    Z2_global = Z2_global.to(device) 
    
    for i in tqdm(range(0, N_total, args.chunk_size), desc="Chunked Matching"):
        end = min(i + args.chunk_size, N_total)
        z1_chunk = Z1_global[i:end].to(device) # [Chunk, D]
        
        # 计算该块的相似度矩阵 [Chunk, N_total]
        sim_chunk = torch.matmul(z1_chunk, Z2_global.T) 
        
        # 1. 正样本相似度 (对角线元素)
        total_pos_sim += torch.diagonal(sim_chunk[:, i:end]).sum().item()
        
        # 2. Top-1 命中数
        preds_top1 = sim_chunk.argmax(dim=-1)
        targets = torch.arange(i, end, device=device)
        top1_correct += (preds_top1 == targets).sum().item()
        
        # 3. Top-5 命中数
        _, preds_top5 = sim_chunk.topk(5, dim=-1)
        top5_correct += (preds_top5 == targets.unsqueeze(1)).any(dim=-1).sum().item()

    avg_top1 = top1_correct / N_total
    avg_top5 = top5_correct / N_total
    avg_pos_sim = total_pos_sim / N_total

    print("\n" + "="*40)
    print("🏆 深度检索报告 (Memory Efficient)")
    print("="*40)
    print(f"总样本数: {N_total}")
    print(f"Top-1 准确率: {avg_top1 * 100:.2f}%")
    print(f"Top-5 准确率: {avg_top5 * 100:.2f}%")
    print(f"正样本平均相似度: {avg_pos_sim:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()