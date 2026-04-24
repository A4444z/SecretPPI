# SecretPPI — Contrastive Masked Autoencoder for Protein-Protein Interactions

SecretPPI learns SE(3)-equivariant representations of protein-protein interfaces using a **Contrastive Masked Autoencoder (GlueCMAE)**. It jointly optimizes an InfoNCE contrastive objective and a masked coordinate-reconstruction objective to produce embeddings that capture both global interaction specificity and local structural geometry.

---

## Architecture

```
PDB Interface (receptor + ligand atoms)
          |
    [PaiNN Encoder]  — SE(3)-equivariant message passing (scalar s, vector v)
          |
  [Attention Pooling]  — multi-head atom→graph aggregation with entropy reg.
          |
      [Projector]  — MLP projection head for contrastive space
          |
    ┌─────┴─────┐
   z1 (view 1)  z2 (view 2)   ← two corrupted views via "blast masking"
    └─────┬─────┘
          |
   [InfoNCE Loss]  +  [Masked D-RMSD Loss]
```

**Two-view masking**: Each training sample generates two views by "blasting" a 10 Å-radius hole in (1) the receptor surface and (2) the ligand surface, severing cross-chain edges to force independent surface representation. The encoder learns to produce matching embeddings for the two views while a `ConditionalPaiNNDecoder` reconstructs the destroyed atoms' coordinates.

**Key design choices**:
- **PaiNN backbone** (`src/models/layers_solo.py`): coupled scalar `s ∈ ℝ^(N×d)` and vector `v ∈ ℝ^(N×d×3)` features; strict SE(3) equivariance
- **Multi-head attention pooling** (`GlueVAE.attn_pooling`): replaces scatter_mean; entropy regularization prevents attention collapse
- **Masked D-RMSD**: distance-based coordinate loss on masked atoms; SE(3)-invariant by construction
- **InfoNCE with GatherLayer**: cross-GPU gradient-preserving all-gather for in-batch negatives across DDP workers
- **Deterministic hash split**: PDB IDs are MD5-hashed to assign train/val/test splits, preventing leakage

---

## Repository Structure

```
SecretPPI/
├── src/
│   ├── data/
│   │   ├── dataset.py               # GlueVAEDataset — LMDB loading, patch sampling, graph building
│   │   └── extract_interface.py     # PDB → interface LMDB pipeline
│   ├── models/
│   │   ├── glue_cmae.py             # GlueVAE — full CMAE model
│   │   └── layers_solo.py           # PaiNN layers (Message, Update, Block, Encoder)
│   └── utils/
│       ├── geometry.py              # RBF encoding, radius graph utilities
│       └── loss_cmae.py             # CMAELoss, InfoNCELoss, MaskedDRMSDLoss, GatherLayer
├── scripts/
│   ├── build_lmdb_resumable.py      # Build interface LMDB with resume support
│   ├── inference_screening.py       # Screening utilities (use run_screening.py as entry point)
│   ├── run_screening.py             # Virtual screening entry point
│   ├── run_screening_n2n.py         # N-to-N screening
│   ├── eval_binder.py               # Binder evaluation
│   ├── run_eval_binder.py           # Binder eval runner
│   └── watch_and_run.py             # Filesystem watcher for batch jobs
├── train_cmae.py                    # DDP training script (torchrun, 8 GPUs)
├── eval_cmae_attn.py                # Test set evaluation — retrieval + RMSD
├── eval_retrieval.py                # Memory-efficient chunked retrieval metrics
├── config_cmae.yaml                 # Main training config
├── config_overfit.yaml              # Single-batch overfit sanity check
├── train_cmae_submt.sh              # SLURM submission script
└── README.md
```

---

## Quick Start

### 1. Environment

```bash
conda create -n secret python=3.10
conda activate secret
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.4
conda install -y -c pyg pytorch-geometric
conda install -y biotite mdanalysis
pip install lmdb tqdm pyyaml wandb torch-scatter
```

### 2. Build LMDB dataset

```bash
python scripts/build_lmdb_resumable.py \
    --pdb_dir database/3DComplex/raw \
    --lmdb_path database/3DComplex/processed_lmdb \
    --num_workers 8 \
    --progress_file processed_progress.txt
```

### 3. Train (single node, 8 GPUs)

```bash
export WANDB_API_KEY="your_key_here"
sbatch train_cmae_submt.sh
# or locally:
torchrun --nproc_per_node=8 train_cmae.py --config config_cmae.yaml
```

### 4. Sanity check (overfit on one batch)

```bash
torchrun --nproc_per_node=1 train_cmae.py --config config_overfit.yaml --overfit_test
```

### 5. Evaluate

```bash
python eval_cmae_attn.py \
    --config config_cmae.yaml \
    --checkpoint checkpoints/checkpoint_best.pt \
    --test_lmdb database/3DComplex/processed_lmdb \
    --output_dir eval_results/
```

---

## Configuration (`config_cmae.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `model.hidden_dim` | 128 | PaiNN hidden dimension |
| `model.num_encoder_layers` | 6 | PaiNN encoder depth |
| `model.num_decoder_layers` | 4 | Conditional decoder depth |
| `training.temperature` | 0.2 | InfoNCE temperature (learnable) |
| `training.lambda_contrast` | 1.0 | InfoNCE loss weight |
| `training.lambda_recon` | 1.0 | Masked D-RMSD weight |
| `training.entropy_weight` | 0.001 | Attention entropy regularization |
| `training.use_amp` | true | Mixed precision (AMP) |
| `data.train_split` | 0.9 | Train fraction (hash-based) |
| `data.val_split` | 0.05 | Validation fraction |
| `data.test_split` | 0.05 | Test fraction |

---

## Evaluation Metrics

- **Top-1 / Top-5 / Top-10 Retrieval Accuracy**: given receptor embedding z1, retrieve matching ligand embedding z2 from the test gallery
- **MRR** (Mean Reciprocal Rank): average 1/rank of the correct match
- **Positive/Negative Similarity Margin**: cosine similarity gap between matched and unmatched pairs
- **Masked Coordinate RMSD (Å)**: RMSD of reconstructed atoms in the masked (destroyed) region

---

## References

- PaiNN: Schütt et al. (2021) — https://arxiv.org/abs/2102.03150
- InfoNCE / MoCo: He et al. (2020) — https://arxiv.org/abs/1911.05722
- MAE: He et al. (2022) — https://arxiv.org/abs/2111.06377

---

## 中文说明

### 项目简介

SecretPPI（GlueCMAE）是一个专为**蛋白质-蛋白质相互作用(PPI)**设计的表示学习框架，采用**对比掩码自编码器**架构。模型同时优化两个目标：
1. **InfoNCE对比损失**：让受体和配体的嵌入向量相互靠近（同一复合物），远离不匹配的对（不同复合物）
2. **掩码坐标重建损失（Masked D-RMSD）**：重建被"破坏"区域的原子坐标，强迫模型学习局部几何结构

### 数据流

```
PDB界面数据 (受体 + 配体原子)
          ↓
    [PaiNN编码器]  — SE(3)等变图神经网络
          ↓
  [多头注意力池化]  — 原子级→图级聚合
          ↓
      [投影头]  — MLP投影到对比学习空间
          ↓
    ┌─────┴─────┐
   z1 (视角1)  z2 (视角2)   ← 通过"掩码破坏"生成两个视角
    └─────┬─────┘
          ↓
   [InfoNCE损失] + [掩码D-RMSD损失]
```

### 关键特性

- **SE(3)等变性**：旋转输入坐标 → 输出特征相应旋转，满足物理对称性
- **双视角掩码**：在受体表面和配体表面分别"炸掉"10Å半径的区域，切断跨链边，强迫独立的表面表示学习
- **哈希分组**：按PDB ID的MD5哈希值确定性地划分训练/验证/测试集，避免数据泄露
- **混合精度训练（AMP）**：在`config_cmae.yaml`中设置`use_amp: true`启用，约节省30-40%显存

### 训练集划分

数据集按PDB ID的MD5哈希值划分，确保：
- 同一蛋白质复合物的所有数据只出现在一个分组
- 训练集/验证集/测试集之间无重叠
- 划分是确定性的（相同PDB ID始终划分到相同分组）

### 常用命令

```bash
# 构建LMDB数据集
python scripts/build_lmdb_resumable.py --pdb_dir <PDB目录> --lmdb_path <输出路径>

# 8卡DDP训练
export WANDB_API_KEY="your_key"
torchrun --nproc_per_node=8 train_cmae.py --config config_cmae.yaml

# 从检查点恢复训练
torchrun --nproc_per_node=8 train_cmae.py --config config_cmae.yaml --resume checkpoints/checkpoint_latest.pt

# 评估
python eval_cmae_attn.py --checkpoint checkpoints/checkpoint_best.pt --test_lmdb <LMDB路径>
```
