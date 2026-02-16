#!/usr/bin/env python
"""
测试原子级VAE架构，确保每个原子都有独立的特征
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
from src.data.dataset import GlueVAEDataset
from src.models.glue_vae_atom_level import GlueVAEAtomLevel

def test_atom_level_vae():
    print("=== 测试原子级VAE架构 ===\n")
    
    # 1. 加载测试数据
    print("正在加载测试数据...")
    dataset = GlueVAEDataset(
        root="test",
        split='train',
        lmdb_path="test/test_lmdb",
        max_atoms=5000  # 设大一点，不触发patch
    )
    
    if len(dataset) == 0:
        print("❌ 数据集为空！")
        return False
    
    sample = dataset[0]
    print(f"✅ 数据加载成功！")
    print(f"  原子数: {sample.num_nodes}")
    print(f"  边数: {sample.num_edges}")
    print()
    
    # 2. 创建原子级VAE模型
    print("正在创建原子级VAE模型...")
    model = GlueVAEAtomLevel(
        hidden_dim=64,  # 小一点，快速测试
        latent_dim=16,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    print(f"✅ 模型创建成功！")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 3. 前向传播测试
    print("正在测试前向传播...")
    model.eval()
    with torch.no_grad():
        pos_pred, mu, logvar = model(
            z=sample.x,
            vector_features=sample.vector_features,
            edge_index=sample.edge_index,
            edge_attr=sample.edge_attr,
            pos=sample.pos
        )
    
    print(f"✅ 前向传播成功！")
    print(f"  pos_pred 形状: {pos_pred.shape}")
    print(f"  mu 形状: {mu.shape}")
    print(f"  logvar 形状: {logvar.shape}")
    print()
    
    # 4. 关键测试：检查每个原子的mu是否不同！
    print("=== 关键验证：每个原子的潜在表示是否独立？ ===")
    
    # 检查同一残基的原子
    residue_indices = sample.residue_index.unique()
    
    print(f"\n找到 {len(residue_indices)} 个残基")
    
    # 选择第一个有多个原子的残基
    target_res_idx = None
    for res_idx in residue_indices:
        mask = (sample.residue_index == res_idx)
        if mask.sum() > 1:
            target_res_idx = res_idx
            break
    
    if target_res_idx is not None:
        print(f"\n选择残基 {target_res_idx.item()} (有 {mask.sum()} 个原子)")
        
        # 获取该残基中所有原子的mu
        mask = (sample.residue_index == target_res_idx)
        mu_residue = mu[mask]
        
        print(f"该残基中 {mu_residue.size(0)} 个原子的mu：")
        for i in range(mu_residue.size(0)):
            print(f"  原子{i}: {mu_residue[i, :5]}...")
        
        # 检查这些mu是否都不同
        all_same = True
        for i in range(1, mu_residue.size(0)):
            if not torch.allclose(mu_residue[0], mu_residue[i], atol=1e-6):
                all_same = False
                break
        
        print(f"\n✅ 同一残基的原子mu是否都不同? {not all_same}")
        
        if not all_same:
            print("   太好了！每个原子都有独立的潜在表示！")
        else:
            print("   ❌ 不好！同一残基的原子mu完全一样！")
    
    # 5. 总体检查：所有原子的mu是否都不同？
    print(f"\n=== 总体检查 ===")
    print(f"总原子数: {mu.size(0)}")
    
    # 计算两两之间的差异
    unique_mu = []
    for i in range(mu.size(0)):
        is_unique = True
        for j in range(i):
            if torch.allclose(mu[i], mu[j], atol=1e-6):
                is_unique = False
                break
        if is_unique:
            unique_mu.append(i)
    
    print(f"唯一mu的原子数: {len(unique_mu)}")
    print(f"所有原子mu都唯一? {len(unique_mu) == mu.size(0)}")
    
    if len(unique_mu) == mu.size(0):
        print("\n✅ 完美！每个原子都有完全独立的潜在表示！")
        return True
    else:
        print(f"\n⚠️ 有 {mu.size(0) - len(unique_mu)} 个原子的mu与其他原子重复")
        return False

if __name__ == "__main__":
    success = test_atom_level_vae()
    if success:
        print("\n🎉 原子级VAE测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 原子级VAE测试失败！")
        sys.exit(1)
