import os
import sys
import glob
import yaml
import pandas as pd

# 获取当前脚本所在目录的上一级（即项目根目录），并加入系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入底层评价器
from scripts.eval_binder import BinderEvaluator

def main():
    print("==================================================")
    print("🚀 CMAE 零样本复合物批量打分流水线 (Binder Evaluation)")
    print("==================================================")

    # ================= 1. 实验参数配置 =================
    CONFIG_PATH = '/home/fit/liulei/WORK/SecretPPI/config_cmae.yaml'
    MODEL_WEIGHTS = "/home/fit/liulei/WORK/SecretPPI/checkpoints/checkpoint_1159610_epoch_39.pt"  
    
    # 复合物存放的文件夹 (假设里面都是对接好的 PDB 文件)
    COMPLEX_DIR = "/home/fit/liulei/WORK/SecretPPI/database/7dha/7dha_D_disulfide_renamed_relaxed"  
    
    # 🚨 链 ID 配置：这里假设你的批量复合物中，受体都是 A 链，配体/Binder 都是 B 链
    # 如果你的生成管道（如 AlphaFold3 或 RFdiffusion）出来的链 ID 不同，请在这里修改
    TARGET_CHAINS = ['A']
    BINDER_CHAINS = ['B']
    
    # 评价参数
    INTERFACE_CUTOFF = 8.0  # 判定为界面的距离阈值 (Å)
    NOISE_SCALE = 1.0       # 破坏界面的高斯噪声强度 (1.0Å 是一个严苛的物理考验)
    
    OUTPUT_DIR = "/home/fit/liulei/WORK/SecretPPI/evaluation_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ===================================================

    # 2. 初始化打分器
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    evaluator = BinderEvaluator(model_path=MODEL_WEIGHTS, config=config)

    # 3. 准备候选复合物文件与 CSV 记录器
    complex_files = glob.glob(os.path.join(COMPLEX_DIR, "*.pdb"))
    
    if len(complex_files) == 0:
        print(f"❌ 错误：在 {COMPLEX_DIR} 下没有找到任何 PDB 文件！")
        return

    print(f"📂 发现 {len(complex_files)} 个复合物结构，开始批量打分...")
    
    csv_filepath = os.path.join(OUTPUT_DIR, "Binder_Evaluation_Scores.csv")
    with open(csv_filepath, 'w', encoding='utf-8') as f:
        f.write("Complex_PDB,Num_Interface_Atoms,Reconstruction_RMSD\n")
    print(f"📄 实时打分将记录至: {csv_filepath}\n")

    results_list = []

    # 4. 遍历打分
    for idx, pdb_path in enumerate(complex_files):
        pdb_name = os.path.basename(pdb_path)
        
        try:
            # 解析复合物（严格区分受体和配体）
            complex_data = evaluator.parse_complex(
                pdb_path=pdb_path, 
                target_chains=TARGET_CHAINS, 
                binder_chains=BINDER_CHAINS
            )
            
            # 进行打分
            res, msg = evaluator.evaluate_binder(
                complex_data, 
                interface_cutoff=INTERFACE_CUTOFF, 
                noise_scale=NOISE_SCALE
            )
            
            if res is None:
                print(f"⚠️ 跳过 {pdb_name}: {msg}")
                with open(csv_filepath, 'a', encoding='utf-8') as f:
                    f.write(f"{pdb_name},0,ERROR_NO_INTERFACE\n")
                continue
                
            rmsd = res['rmsd']
            num_atoms = res['num_interface_atoms']
            
            # 实时写入 CSV
            with open(csv_filepath, 'a', encoding='utf-8') as f:
                f.write(f"{pdb_name},{num_atoms},{rmsd:.4f}\n")
                
            results_list.append({
                'pdb_name': pdb_name,
                'rmsd': rmsd,
                'num_atoms': num_atoms
            })
            
            # 打印进度
            if (idx + 1) % 10 == 0 or (idx + 1) == len(complex_files):
                print(f"⏳ 已评估 {idx+1}/{len(complex_files)} 个复合物...")
                
        except Exception as e:
            print(f"❌ 解析或评估 {pdb_name} 失败: {e}")
            with open(csv_filepath, 'a', encoding='utf-8') as f:
                f.write(f"{pdb_name},ERROR,{e}\n")

    # 5. 生成最终排行榜 (🚨 注意：RMSD 越小越好！)
    if len(results_list) > 0:
        results_list.sort(key=lambda x: x['rmsd'], reverse=False) # 升序排列
        
        print("\n" + "="*60)
        print("🏆 Binder 结合能力物理评价排行榜 (Top 10)")
        print("   (基于 CMAE 界面重构误差，RMSD 越小说明能量契合度越高)")
        print("="*60)
        for rank, res in enumerate(results_list[:10]):
            print(f"Rank {rank+1:02d}: {res['pdb_name']:<25} | 界面原子数: {res['num_atoms']:<4} | 重构 RMSD: {res['rmsd']:.4f} Å")
        print("="*60)
        print(f"📊 完整打分 CSV 已保存至: {csv_filepath}")

if __name__ == "__main__":
    main()