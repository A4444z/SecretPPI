#!/bin/bash
#
# SLURM submission script for GlueVAE Overfit Test
#
#SBATCH -J GlueVAE_Overfit_Test
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o stdout.overfit_test.%j
#SBATCH -e stderr.overfit_test.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# ===================== 环境配置 - 与 train_solo_submit.sh 保持一致 =====================
module load soft/anaconda3/config
source activate
conda activate /home/fit/liulei/WORK/miniconda3/envs/secret
export WANDB_API_KEY="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"
export PYTHONUNBUFFERED=1
# ===================== 项目配置 =====================
PROJECT_DIR="/home/fit/liulei/WORK/SecretPPI"
cd $PROJECT_DIR

echo "当前工作目录: $(pwd)"
echo "Python 路径: $(which python)"
echo "Python 版本: $(python --version)"
echo "Conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "检查 CUDA..."
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count())"

# ===================== 启动过拟合测试 =====================
echo "========================================"
echo "开始 GlueVAE 过拟合测试"
echo "配置文件: config_overfit.yaml"
echo "========================================"

python train_solo.py --config config_overfit.yaml --overfit_test

echo "========================================"
echo "训练结束！"
echo "========================================"
