import os
import sys

# 获取当前脚本所在目录的上一级（即项目根目录），并加入系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import re
import glob
import yaml
import torch
from torch_geometric.data import Batch

# 从底层文件导入
from inference_screening import VirtualScreener, parse_pdb_to_pyg, save_patch_with_attention_to_pdb

def main():
    print("==================================================")
    print("🚀 CMAE 双盲高通量虚拟筛选 (All-to-All Scanning)")
    print("==================================================")

    # ================= 1. 实验参数配置 =================
    CONFIG_PATH = '/home/fit/liulei/WORK/SecretPPI/config_cmae.yaml'
    MODEL_WEIGHTS = "/home/fit/liulei/WORK/SecretPPI/checkpoints/checkpoint_1159610_epoch_39.pt"  
    
    # 靶点 A 配置 (无需指定残基了)
    TARGET_PDB_PATH = "/home/fit/liulei/WORK/SecretPPI/database/AFDB_human/AF-Q92560-F1-model_v6.pdb"
    NUM_TARGET_PATCHES = 20 # 蛋白 A 表面采样的斑块数
    
    # 候选库 B 配置
    CANDIDATE_DIR = "/home/fit/liulei/WORK/SecretPPI/database/AFDB_human"  
    NUM_SAMPLED_PATCHES = 20 # 每个蛋白 B 表面采样的斑块数

    # 🚨 新增：调试/验证阶段控制蛋白 B 数目的开关
    # 设为具体的数字 (如 50) 则只测 50 个；设为 None 则火力全开，扫整个库！
    MAX_CANDIDATES = None
    
    OUTPUT_DIR = "/home/fit/liulei/WORK/SecretPPI/screening_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ===================================================

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    screener = VirtualScreener(model_path=MODEL_WEIGHTS, config=config)
    
    # 3. 🎯 【改动点】对目标蛋白 A 进行自动全景采样
    print(f"\n🔑 正在解析并自动采样目标蛋白 A: {TARGET_PDB_PATH}")
    protein_A_full = parse_pdb_to_pyg(TARGET_PDB_PATH)
    
    # 使用自动提取代替手动提取
    target_patches = screener.extract_patches_auto(protein_A_full, num_patches=NUM_TARGET_PATCHES)
    batch_targets = Batch.from_data_list(target_patches)
    z_targets, attn_weights_A = screener.get_latent_representation(batch_targets) # [N_A, 128]
    print(f"✅ 蛋白 A 表面共提取了 {len(target_patches)} 个候选斑块！\n")

    # =====================================================================
    # 4. 准备实时 CSV 记录器 & 遍历候选库 B
    # =====================================================================
    candidate_files = glob.glob(os.path.join(CANDIDATE_DIR, "*.pdb"))
    candidate_files = [f for f in candidate_files if os.path.basename(f) != os.path.basename(TARGET_PDB_PATH)]
    
    if MAX_CANDIDATES is not None and MAX_CANDIDATES < len(candidate_files):
        import random
        print(f"⚠️ [调试模式] 原本有 {len(candidate_files)} 个候选蛋白。")
        random.seed(42) 
        random.shuffle(candidate_files)
        candidate_files = candidate_files[:MAX_CANDIDATES]
        print(f"⚠️ [调试模式] 已开启截断，本次只筛选 {MAX_CANDIDATES} 个蛋白 B！")
    else:
        print(f"📂 发现 {len(candidate_files)} 个候选蛋白，火力全开...")
    
    # 🎯 新增：创建 CSV 文件并写入表头
    # 🎯 修改：增加 UniProt_ID 列
    csv_filename = f"All_Scores_{os.path.basename(TARGET_PDB_PATH).replace('.pdb', '')}.csv"
    csv_filepath = os.path.join(OUTPUT_DIR, csv_filename)
    with open(csv_filepath, 'w', encoding='utf-8') as f:
        f.write("Candidate_PDB,UniProt_ID,Match_Score\n")
    print(f"📄 将实时记录所有打分到 CSV 表格: {csv_filepath}")
    
    print(f"开始矩阵交叉匹配...\n")
    
    screening_results = []
    MAX_KEEP = 100 # 内存中只保留排名前 100 的详情数据，防止爆内存

    for idx, cand_path in enumerate(candidate_files):
        pdb_name = os.path.basename(cand_path)
        
        try:
            cand_full = parse_pdb_to_pyg(cand_path)
            candidate_patches = screener.extract_patches_auto(cand_full, num_patches=NUM_SAMPLED_PATCHES)
            
            if not candidate_patches:
                continue
                
            batch_candidates = Batch.from_data_list(candidate_patches)
            z_candidates, attn_weights_B = screener.get_latent_representation(batch_candidates) # [N_B, 128]
            
            # 🎯 计算相似度矩阵并提取最大分
            similarities = torch.matmul(z_candidates, z_targets.T)
            best_score = torch.max(similarities).item()
            flat_idx = torch.argmax(similarities)
            
            best_idx_B = (flat_idx // similarities.shape[1]).item()
            best_idx_A = (flat_idx % similarities.shape[1]).item()
            
            # 🎯 使用正则表达式精准提取 UniProt ID
            # 从 "AF-Q3LI81-F1-model_v6.pdb" 中提取出 "Q3LI81"
            
            match = re.search(r'AF-(.+?)-F\d+', pdb_name)
            uniprot_id = match.group(1) if match else "UNKNOWN"
            
            # 🎯 提高精度：使用 .6f 保留 6 位小数
            with open(csv_filepath, 'a', encoding='utf-8') as f:
                f.write(f"{pdb_name},{uniprot_id},{best_score:.6f}\n")

            # 将详细结果录入内存排行榜（加上 uniprot_id）
            screening_results.append({
                'pdb_name': pdb_name,
                'uniprot_id': uniprot_id, # 👈 新增这一行
                'best_score': best_score,
                'best_patch_B': candidate_patches[best_idx_B],
                'best_attn_B': attn_weights_B[batch_candidates.batch == best_idx_B].mean(dim=-1).cpu().numpy(),
                'best_patch_A': target_patches[best_idx_A],
                'best_attn_A': attn_weights_A[batch_targets.batch == best_idx_A].mean(dim=-1).cpu().numpy()
            })
            
            # 🛡️ 内存护航：每扫描满 50 个，清理一次内存中排名靠后的庞大图对象
            if len(screening_results) > 200:
                screening_results.sort(key=lambda x: x['best_score'], reverse=True)
                screening_results = screening_results[:MAX_KEEP]
                
        except Exception as e:
            # 遇到解析错误，也在 CSV 中记录下来，方便事后排查
            with open(csv_filepath, 'a', encoding='utf-8') as f:
                f.write(f"{pdb_name},ERROR\n")
            print(f"❌ 解析 {pdb_name} 失败: {e}")

    # 5. 排行榜
    screening_results.sort(key=lambda x: x['best_score'], reverse=True)

    print("\n" + "="*60)
    print("🏆 双盲虚拟筛选最终排行榜 (Top 10)")
    print("="*60)
    for rank, res in enumerate(screening_results[:10]):
        # 👈 这里使用 res['uniprot_id'] 替代冗长的 pdb_name，并显示 6 位小数
        print(f"Rank {rank+1:02d}: {res['uniprot_id']:<15} | 最高匹配度: {res['best_score']:.6f}")
    print("="*60)
    
    # 6. 【高阶】同时导出蛋白 A 和蛋白 B 互相看对眼的两个口袋！
    if len(screening_results) > 0:
        print(f"\n💾 正在导出 Top-1 契合配对的热力图到 {OUTPUT_DIR} ...")
        top1 = screening_results[0]
        
        # 👈 文件名也换成干净的 UniProt ID
        save_patch_with_attention_to_pdb(
            top1['best_patch_A'], 
            top1['best_attn_A'], 
            out_path=os.path.join(OUTPUT_DIR, f"Top1_ProteinA_Pocket_for_{top1['uniprot_id']}.pdb")
        )
        save_patch_with_attention_to_pdb(
            top1['best_patch_B'], 
            top1['best_attn_B'], 
            out_path=os.path.join(OUTPUT_DIR, f"Top1_ProteinB_Pocket_from_{top1['uniprot_id']}.pdb")
        )
        print("✅ 导出完毕！模型不仅找出了蛋白 B 的靶点，还指出了它是用蛋白 A 的哪个部位结合的！")


if __name__ == "__main__":
    main()