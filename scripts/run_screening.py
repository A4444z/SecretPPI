import os
import sys

# 获取当前脚本所在目录的上一级（即项目根目录），并加入系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import glob
import yaml
import torch
from torch_geometric.data import Batch

# 从刚才的底层文件导入我们需要的类和函数
# 假设 inference_screening.py 放在和这个脚本同级的目录下
from inference_screening import VirtualScreener, parse_pdb_to_pyg, save_patch_with_attention_to_pdb

def main():
    print("==================================================")
    print("🚀 CMAE 高通量虚拟筛选流水线启动")
    print("==================================================")

    # ================= 1. 实验参数配置 =================
    CONFIG_PATH = 'config_cmae.yaml'
    MODEL_WEIGHTS = "checkpoints/checkpoint_1159610_epoch_39.pt"  # 你的模型权重
    
    # 靶点 (Key) 配置
    TARGET_PDB_PATH = "database/test_input_pdbs/1a30_protein.pdb"
    TARGET_RESIDUES = [0, 1, 2, 3, 4]  # 你感兴趣的靶点残基索引
    PATCH_RADIUS = 15.0
    
    # 候选库 (Locks) 配置
    CANDIDATE_DIR = "database/test_input_pdbs"  # 候选蛋白所在的文件夹
    NUM_SAMPLED_PATCHES = 20  # 每个候选蛋白表面自动采样的斑块数量
    
    # 输出配置
    OUTPUT_DIR = "screening_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ===================================================

    # 2. 初始化筛选器
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    screener = VirtualScreener(model_path=MODEL_WEIGHTS, config=config)
    
    # 3. 提取靶点 (Target) 流形特征
    print(f"\n🔑 正在解析目标蛋白 A (靶点): {TARGET_PDB_PATH}")
    protein_A_full = parse_pdb_to_pyg(TARGET_PDB_PATH)
    target_patch = screener.extract_patch_manual(
        full_protein_data=protein_A_full, 
        center_residue_indices=TARGET_RESIDUES, 
        radius=PATCH_RADIUS
    )
    z_target, _ = screener.get_latent_representation(target_patch)
    print("✅ 靶点流形特征提取完毕，准备进入高通量扫描！\n")

    # 4. 遍历候选库进行极速匹配
    candidate_files = glob.glob(os.path.join(CANDIDATE_DIR, "*_protein.pdb"))
    candidate_files = [f for f in candidate_files if os.path.basename(f) != os.path.basename(TARGET_PDB_PATH)]
    print(f"📂 发现 {len(candidate_files)} 个候选蛋白，开始一对多筛选...\n")
    
    screening_results = []

    for idx, cand_path in enumerate(candidate_files):
        pdb_name = os.path.basename(cand_path)
        
        try:
            cand_full = parse_pdb_to_pyg(cand_path)
            candidate_patches = screener.extract_patches_auto(cand_full, num_patches=NUM_SAMPLED_PATCHES)
            
            if not candidate_patches:
                continue
                
            batch_candidates = Batch.from_data_list(candidate_patches)
            z_candidates, attn_weights = screener.get_latent_representation(batch_candidates)
            
            # 计算余弦相似度并找最高分
            similarities = torch.matmul(z_candidates, z_target.T).squeeze(-1)
            best_patch_idx = torch.argmax(similarities).item()
            best_score = similarities[best_patch_idx].item()
            
            screening_results.append({
                'pdb_name': pdb_name,
                'best_score': best_score,
                'best_patch_data': candidate_patches[best_patch_idx],
                'best_attn': attn_weights[batch_candidates.batch == best_patch_idx].mean(dim=-1).cpu().numpy()
            })
            
        except Exception as e:
            print(f"❌ 解析 {pdb_name} 失败: {e}")

    # 5. 生成排行榜并输出可视化结果
    screening_results.sort(key=lambda x: x['best_score'], reverse=True)

    print("\n" + "="*50)
    print("🏆 虚拟筛选最终排行榜 (Top 10)")
    print("="*50)
    
    for rank, res in enumerate(screening_results[:10]):
        print(f"Rank {rank+1}: {res['pdb_name']:<20} | 匹配度 (Score): {res['best_score']:.4f}")
    print("="*50)
    
    # 将 Top-3 的候选靶点保存为带 Attention 的 PDB 文件
    if len(screening_results) > 0:
        print("\n💾 正在导出 Top-3 匹配靶点的热力图 PDB...")
        for i, res in enumerate(screening_results[:3]):
            out_name = os.path.join(OUTPUT_DIR, f"Top{i+1}_{res['pdb_name']}")
            save_patch_with_attention_to_pdb(
                res['best_patch_data'], 
                res['best_attn'], 
                out_path=out_name
            )
        print(f"✅ 导出完毕！请使用 PyMOL 查看 {OUTPUT_DIR} 目录下的结果。")

if __name__ == "__main__":
    main()