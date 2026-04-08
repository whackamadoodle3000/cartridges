#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -p ice-gpu
#SBATCH -o train_%j.out
#SBATCH -J rank-train

module load anaconda3
module load cuda/12.6.1
conda activate ~/scratch/envs/cartridges

export CARTRIDGES_DIR=~/cartridges
export CARTRIDGES_OUTPUT_DIR=~/scratch/cartridges_output
export HF_HOME=~/scratch/hf_cache
export WANDB_MODE=offline

PARQUET=$(find $CARTRIDGES_OUTPUT_DIR -name "dataset.parquet" | sort | tail -1)
echo "Using parquet: $PARQUET"

cd $CARTRIDGES_DIR

for ratio in 1.0 0.1 0.05 0.02; do
    echo "=== Training ratio=$ratio ==="
    python experiments/rank_analysis/train_configs.py --ratio $ratio --parquet $PARQUET
done
