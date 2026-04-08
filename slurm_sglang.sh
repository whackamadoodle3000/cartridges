#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -p ice-gpu
#SBATCH -o sglang_server_%j.out
#SBATCH -J sglang-server

module load anaconda3
module load cuda/12.6.1
conda activate ~/scratch/envs/cartridges

export HF_HOME=~/scratch/hf_cache

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.85
