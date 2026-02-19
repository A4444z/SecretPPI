#!/bin/bash
#SBATCH --job-name=GlueVAE_DDP
#SBATCH --partition=a01
#SBATCH --output=stdout.ddp.%j
#SBATCH --error=stderr.ddp.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32

# ===================== 环境配置 =====================
module load soft/anaconda3/config
source activate
conda activate /home/fit/liulei/WORK/miniconda3/envs/secret
export WANDB_API_KEY="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"
export PYTHONUNBUFFERED=1

# ===================== 项目配置 =====================
PROJECT_DIR="/home/fit/liulei/WORK/SecretPPI"
cd $PROJECT_DIR

# ===================== 环境检查 =====================
echo "========================================"
echo "当前工作目录: $(pwd)"
echo "Python 路径: $(which python)"
echo "Python 版本: $(python --version)"
echo "Conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "检查 CUDA..."
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count())"
echo "========================================"

# ===================== 启动 DDP 训练 =====================
echo "开始 GlueVAE 8卡 DDP 训练"
echo "配置文件: config_solo.yaml"
echo "========================================"

# 使用 torchrun 启动 (推荐方式)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_solo_ddp.py \
    --config config_solo.yaml

echo "========================================"
echo "训练结束！"
echo "========================================"
