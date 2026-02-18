#!/bin/bash
#
# SLURM submission script for GlueVAE Full Training
#
#SBATCH -J GlueVAE_Full_Train
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o stdout.full_train.%j
#SBATCH -e stderr.full_train.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

# 环境设置
export PATH=/apps/gpu/cuda/v12.6.1/bin:$PATH
export LD_LIBRARY_PATH=/apps/gpu/cuda/v12.6.1/lib64:$LD_LIBRARY_PATH
module load soft/anaconda3/config
source activate
conda activate secret

# 设置工作目录
cd /home/fit/liulei/WORK/SecretPPI

# 打印环境信息
echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python: $(which python)"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "========================"
echo

# 运行完整训练
python train_solo.py \
    --config config_solo.yaml

