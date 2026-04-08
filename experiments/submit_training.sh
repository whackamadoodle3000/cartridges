#!/bin/bash
PARQUET=$(find $HOME/scratch/cartridges_output -name "dataset.parquet" | sort | tail -1)
echo "Using parquet: $PARQUET"

for ratio in 1.0 0.1 0.05 0.02; do
    jobscript=$(mktemp /tmp/train_ratio_XXXX.sh)
    cat > $jobscript << INNEREOF
#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -o train_${ratio}_%j.out
#SBATCH -J train-${ratio}

module load anaconda3
module load cuda/12.6.1
conda activate $HOME/scratch/envs/cartridges

export CARTRIDGES_DIR=$HOME/cartridges
export CARTRIDGES_OUTPUT_DIR=$HOME/scratch/cartridges_output
export HF_HOME=$HOME/scratch/hf_cache
export WANDB_MODE=offline

cd \$CARTRIDGES_DIR
git pull --quiet
echo "=== Training ratio=${ratio} ==="
python experiments/rank_analysis/train_configs.py --ratio ${ratio} --parquet ${PARQUET}
INNEREOF
    jobid=$(sbatch $jobscript | awk '{print $4}')
    echo "Submitted ratio=${ratio} → job ${jobid}"
    rm $jobscript
done
