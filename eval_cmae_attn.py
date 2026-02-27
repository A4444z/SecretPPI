import os
import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader
import yaml


from src.models.glue_cmae import GlueVAE
from src.data.dataset import GlueVAEDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='CMAE Test Set Evaluation (Old Model)')
    parser.add_argument('--config', type=str, default='config_cmae.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='训练好的 .pt 权重路径')
    parser.add_argument('--test_lmdb', type=str, required=True, help='测试集 LMDB 路径')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='输出结果目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    config = load_config(args.config)

    # 1. 构建测试数据集 (关闭随机旋转)
    print("加载测试集...")
    test_dataset = GlueVAEDataset(
        root=config['data']['root_dir'],
        lmdb_path=args.test_lmdb,
        split='test',
        random_rotation=False # 评估时必须关闭随机旋转
    )
    
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False, # 测试集不需要打乱，方便对应 PDB ID
        num_workers=4,
        drop_last=False
    )

    # 2. 初始化旧版模型并加载权重
    print("加载旧版 CMAE (scatter_mean) 模型...")
    # 🚨 注意：去掉了 mask_noise，因为旧版 __init__ 不接受这个参数
    model = GlueVAE(
        hidden_dim=config['model']['hidden_dim'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        edge_dim=config['model']['edge_dim'],
        vocab_size=config['model']['vocab_size'],
        use_gradient_checkpointing=False
    ).to(device)

    # 兼容 DDP 权重加载 (剥离 'module.' 前缀)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 3. 收集数据的容器
    all_z1 = []
    all_z2 = []
    all_pdb_ids = []
    
    total_rmsd = 0.0
    valid_rmsd_batches = 0

    # 4. 前向传播收集特征
    print("🧠 开始特征提取与物理重构评估...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            
            
            graph_z1, graph_z2, pos_pred_v1, mask_v1, batch_entropy, attn_guidance_loss = model(
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
                        
            # 收集对比向量 (确保在 CPU 上保存)
            all_z1.append(graph_z1.cpu())
            all_z2.append(graph_z2.cpu())
            if hasattr(batch, 'pdb_id'):
                all_pdb_ids.extend(batch.pdb_id)
                
            # 计算物理坐标 RMSD (只计算被 Mask 的区域)
            if mask_v1.sum() > 0:
                pos_true = batch.pos[mask_v1]
                pos_pred = pos_pred_v1[mask_v1]
                # 计算均方根误差 (Å)
                rmsd = torch.sqrt(F.mse_loss(pos_pred, pos_true)).item()
                total_rmsd += rmsd
                valid_rmsd_batches += 1

    # 拼接全集特征
    all_z1 = torch.cat(all_z1, dim=0) # [N, proj_dim]
    all_z2 = torch.cat(all_z2, dim=0) # [N, proj_dim]
    
    avg_mask_rmsd = total_rmsd / max(1, valid_rmsd_batches)

    # 5. 计算全局检索指标 (Global Retrieval)
    print("🔍 计算全局对比学习检索指标...")
    N = all_z1.size(0)
    
    # 计算全集余弦相似度矩阵 [N, N]
    sim_matrix = torch.matmul(all_z1, all_z2.T)
    
    # 正确答案是对角线上的元素
    targets = torch.arange(N)
    
    # 对相似度矩阵按行降序排列，获取排名索引
    sorted_indices = sim_matrix.argsort(dim=-1, descending=True)
    
    # 找到 Target 在排序后的队伍中排第几 (排名从 1 开始)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    
    # 计算量化指标
    mrr = (1.0 / ranks.float()).mean().item()
    top1 = (ranks == 1).float().mean().item()
    top5 = (ranks <= 5).float().mean().item()
    top10 = (ranks <= 10).float().mean().item()
    
    # 计算正负样本平均相似度
    pos_sim = torch.diagonal(sim_matrix).mean().item()
    mask_neg = ~torch.eye(N, dtype=torch.bool)
    neg_sim = sim_matrix[mask_neg].mean().item()

    # 6. 生成报告并保存
    report = {
        "dataset_size": N,
        "retrieval_metrics": {
            "MRR (平均倒数排名)": round(mrr, 4),
            "Top-1 Acc": round(top1, 4),
            "Top-5 Acc": round(top5, 4),
            "Top-10 Acc": round(top10, 4)
        },
        "manifold_metrics": {
            "Positive_Similarity (正样本)": round(pos_sim, 4),
            "Negative_Similarity (负样本)": round(neg_sim, 4),
            "Margin (间距)": round(pos_sim - neg_sim, 4)
        },
        "physics_metrics": {
            "Masked_Region_Coordinate_RMSD (Å)": round(avg_mask_rmsd, 4)
        }
    }

    # 打印到终端
    print("\n" + "="*40)
    print("📊 CMAE 测试集评估报告 (Old Model)")
    print("="*40)
    print(json.dumps(report, indent=4, ensure_ascii=False))
    print("="*40)

    # 导出 JSON 报告
    json_path = os.path.join(args.output_dir, 'cmae_eval_attn_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # 导出特征数据
    feature_path = os.path.join(args.output_dir, 'cmae_test_attn_features.pt')
    torch.save({
        'z1_receptor': all_z1,
        'z2_ligand': all_z2,
        'pdb_ids': all_pdb_ids,
        'sim_matrix': sim_matrix
    }, feature_path)
    
    print(f"✅ 评估完成！\n报告已保存至: {json_path}\n特征已保存至: {feature_path}")

if __name__ == "__main__":
    main()