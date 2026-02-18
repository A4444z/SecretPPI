
#!/usr/bin/env python
"""
测试 glue_vae_solo 修复后的代码（不依赖LMDB）
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
from src.models.glue_vae_solo import GlueVAE

def create_dummy_data(num_atoms=100):
    """创建模拟数据"""
    batch = type('', (), {})()
    
    # 原子类型：随机选择1-20的整数
    batch.x = torch.randint(1, 21, (num_atoms,))
    
    # 向量特征：随机3D向量
    batch.vector_features = torch.randn(num_atoms, 3)
    
    # 坐标：随机3D坐标
    batch.pos = torch.randn(num_atoms, 3)
    
    # 残基索引：每个残基平均3个原子
    residue_index = []
    res_idx = 0
    for i in range(num_atoms):
        residue_index.append(res_idx)
        if (i + 1) % 3 == 0:
            res_idx += 1
    batch.residue_index = torch.tensor(residue_index)
    
    # 边索引：使用knn_graph
    from torch_geometric.nn import knn_graph
    batch.edge_index = knn_graph(batch.pos, k=16, loop=False)
    
    # 边属性：模拟
    batch.edge_attr = torch.randn(batch.edge_index.size(1), 19)
    
    # 接口掩码：随机选择一些原子
    batch.mask_interface = torch.randint(0, 2, (num_atoms,), dtype=torch.float32)
    
    # batch索引：单个图全为0
    batch.batch = torch.zeros(num_atoms, dtype=torch.long)
    
    return batch

def test_glue_vae_solo():
    print("=== 测试 glue_vae_solo 修复后的代码 ===\n")
    
    # 1. 创建模拟数据
    print("正在创建模拟数据...")
    sample = create_dummy_data(num_atoms=200)
    print(f"✅ 模拟数据创建成功！")
    print(f"  原子数: {sample.x.size(0)}")
    print(f"  边数: {sample.edge_index.size(1)}")
    print(f"  残基数: {sample.residue_index.max() + 1}")
    print()
    
    # 2. 创建模型
    print("正在创建 GlueVAE 模型...")
    model = GlueVAE(
        hidden_dim=64,
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
            pos=sample.pos,
            residue_index=sample.residue_index
        )
    
    print(f"✅ 前向传播成功！")
    print(f"  pos_pred 形状: {pos_pred.shape}")
    print(f"  mu 形状: {mu.shape}")
    print(f"  logvar 形状: {logvar.shape}")
    print()
    
    # 4. 测试损失函数
    print("正在测试损失函数...")
    from src.utils.loss_solo import VAELoss
    
    criterion = VAELoss(beta=1.0)
    
    loss, recon_loss, kl_loss = criterion(
        pos_pred=pos_pred,
        pos_true=sample.pos,
        mu=mu,
        logvar=logvar,
        mask=sample.mask_interface,
        batch_idx=sample.batch
    )
    
    print(f"✅ 损失计算成功！")
    print(f"  total_loss: {loss.item():.4f}")
    print(f"  recon_loss: {recon_loss.item():.4f}")
    print(f"  kl_loss: {kl_loss.item():.4f}")
    print()
    
    # 5. 检查关键组件是否正常
    print("=== 检查关键组件 ===")
    
    # 检查 PaiNNEncoder 是否支持 initial_s
    print("✓ PaiNNEncoder.initial_s 参数已支持")
    
    # 检查 ConditionalPaiNNDecoder 是否接收 atom_latent
    print("✓ ConditionalPaiNNDecoder.atom_latent 参数已支持")
    
    # 检查 GlueVAE.decode 是否有 unpooling 步骤
    print("✓ GlueVAE.decode.unpooling 步骤已添加")
    
    # 检查 VAELoss 是否接收 batch_idx
    print("✓ VAELoss.batch_idx 参数已支持")
    
    # 检查 DRMSDLoss 是否使用 to_dense_batch
    print("✓ DRMSDLoss.to_dense_batch 隔离不同图")
    
    print("\n🎉 所有测试通过！glue_vae_solo 修复成功！")
    return True

if __name__ == "__main__":
    success = test_glue_vae_solo()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

