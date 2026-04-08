# Running Rank Analysis on PACE (ICE Cluster)

## Pipeline Overview

The experiment has 5 sequential phases:

1. **SGLang server** — Serve Qwen3-4B locally so self-study synthesis can call it.
2. **Synthesis** — Use the running model to generate 256 synthetic Q&A conversations about
   the document. The model asks and answers its own questions about chunks of the document.
   A single set of conversations is used for everything (training, selection, and rank
   measurement). *Cancel the SGLang job after this finishes.*
3. **Training** — For each compression ratio, run Cartridges gradient descent. The KV
   cache starts from a causal forward pass over the document (first N tokens at ratio N),
   then is optimized via gradient descent on the conversations to produce good model outputs
   when answering questions about the document. Output: `cache_last.pt` per ratio.
4. **Collection** — For all 8 conditions (init_full, trained_full, snapkv/trained × 3
   ratios), run a forward pass over the conversations using each condition's KV cache.
   Capture Q and K_conv via the activation hook, compute Y (the cartridge's real attention
   contribution), and save V and Y tensors.
5. **Plotting** — Load all 8 `.pt` files and produce bar charts, heatmaps, and summary
   plots of effective rank.

## SSH Login

```bash
ssh <gtusername>@login-ice.pace.gatech.edu
```

## Environment Setup

### 1. Load modules and create the conda environment

PACE ICE has `cuda/12.6.1`, `cuda/12.9.1`, and `cuda/13.0.1`. Use **`cuda/12.6.1`** — it has
an exact PyTorch wheel match (`cu126`) and is the most stable choice on A100s.

Create the environment in scratch, not home — the default `~/.conda/envs/` location is on a
home directory with a small quota (~25 GB). PyTorch + SGLang + model weights + checkpoints
easily exceed that. Scratch on PACE ICE is at `~/scratch`.

```bash
module load anaconda3
module load cuda/12.6.1

# Create env in scratch to avoid home quota limits
# Replace <gtusername> with your GT username
conda create --prefix ~/scratch/envs/cartridges python=3.12 -y
conda activate ~/scratch/envs/cartridges
```

To avoid typing the full path every time, add this to `~/.bashrc`:
```bash
conda config --append envs_dirs ~/scratch/envs
# After this you can use: conda activate cartridges
```

### 2. Install PyTorch 2.11 with CUDA 12.6

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> **Other available CUDA options and why we skip them:**
> - `cuda/12.9.1` — no exact cu129 PyTorch wheel; would require cu128 wheels relying on
>   backward compatibility. Fine, but not a clean match.
> - `cuda/13.0.1` — PyTorch 2.11 supports it (`cu130`), but 13.0 is very new and less
>   battle-tested on A100 hardware. Use if you have a specific reason.
> - `cuda/11.x` — not supported by PyTorch 2.11. Do not use.

### 3. Install cartridges and SGLang

```bash
git clone <your-cartridges-repo-url> ~/cartridges
cd ~/cartridges
pip install -e .

# SGLang for serving Qwen3-4B locally (synthesis phase only)
pip install "sglang[all]"
```

### 4. Verify GPU and CUDA

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.__version__)"
# Should print: True  12.6  2.11.x
```


## Environment Variables

Add to your `~/.bashrc` (replace `<gtusername>` with your GT username):

```bash
export CARTRIDGES_DIR=~/cartridges
export CARTRIDGES_OUTPUT_DIR=~/scratch/cartridges_output
export HF_HOME=~/scratch/hf_cache
export WANDB_MODE=offline
export SGLANG_URL=http://localhost:30000/v1
```

## Directory Setup

```bash
mkdir -p $CARTRIDGES_OUTPUT_DIR
mkdir -p $HF_HOME
mkdir -p ~/cartridges/data/rank_analysis
mkdir -p ~/cartridges/plots
```

## Copy the Document to PACE

The document (`qasper_e_line_203_context.txt`) lives in the repo root. Copy it to PACE
along with the rest of the repo — no separate download needed.

```bash
# Verify it's present after cloning/copying
ls ~/cartridges/qasper_e_line_203_context.txt
```

### Switching to a Different Document

The document path is hardcoded in two files. Update both before running any phase:

**1. `synthesize_config.py`** — controls what the model generates Q&A conversations about:
```python
DOC_PATH = os.path.join(CARTRIDGES_DIR, "qasper_e_line_203_context.txt")
# Change to e.g.:
DOC_PATH = os.path.join(CARTRIDGES_DIR, "my_other_doc.txt")
```

**2. `train_configs.py`** — controls `KVFromText`, the initial document forward pass:
```python
DEFAULT_DOC = os.path.join(CARTRIDGES_DIR, "qasper_e_line_203_context.txt")
# Change to e.g.:
DEFAULT_DOC = os.path.join(CARTRIDGES_DIR, "my_other_doc.txt")
```

Then re-run all 4 phases from scratch (synthesis → training → collection → plotting) — the
trained cartridges and the self-study parquet are document-specific and cannot be reused.

Token length check: Qwen3-4B's native context is 32,768 tokens. Verify your document fits:
```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4b')
text = open('qasper_e_line_203_context.txt').read()
n = len(tok.encode(text))
print(f'{n} tokens (limit: ~16k recommended, hard limit: 32768)')
"
```
If it exceeds ~28k tokens, either trim the file or update the `int(ratio * 30_000)`
approximation in `train_configs.py` to match the actual token count.

## Phase 0: Launch SGLang Server

The SGLang server needs to be running before synthesis. Submit this as a
separate job and wait for it to be ready.

`slurm_sglang.sh`:

```bash
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
```

```bash
sbatch slurm_sglang.sh
# Wait for "Server is ready" in the output log before proceeding
```

You can check status with:

```bash
# Find the job
squeue -u $USER
# Check the log
tail -f sglang_server_<jobid>.out
```

## Phase 1: Self-Study Synthesis

Run against the SGLang server from Phase 0. If the SGLang server is on a
compute node, you need to set `SGLANG_URL` to point to that node.

`slurm_synthesize.sh`:

```bash
#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -p ice-gpu
#SBATCH -o synth_%j.out
#SBATCH -J rank-synth

module load anaconda3
module load cuda/12.6.1
conda activate ~/scratch/envs/cartridges

export CARTRIDGES_DIR=~/cartridges
export CARTRIDGES_OUTPUT_DIR=~/scratch/cartridges_output
export HF_HOME=~/scratch/hf_cache
export WANDB_MODE=offline

# Point to the SGLang server node (adjust if needed)
export SGLANG_URL=http://localhost:30000/v1

cd $CARTRIDGES_DIR

python experiments/rank_analysis/synthesize_config.py
```

```bash
sbatch slurm_synthesize.sh
```

The parquet will be saved under a timestamped + UUID path like:
- `$CARTRIDGES_OUTPUT_DIR/<timestamp>-synthesize_config/<uuid>/artifact/dataset.parquet`

Find the exact path after it finishes:
```bash
find ~/scratch/cartridges_output -name "dataset.parquet" | sort | tail -1
```

Copy that path — you'll need it for `--parquet` in training and collection.

This single set of 256 conversations is used for everything: Cartridges training,
SnapKV token selection, and rank measurement.

## Phase 2: Train Cartridges (4 Ratios)

**Important: cancel the SGLang server job before submitting training.** Both SGLang and the
training job load Qwen3-4B onto the same GPU. They cannot run simultaneously on one A100.

```bash
# Find and cancel the SGLang job first
squeue -u $USER
scancel <sglang_job_id>
```

Submit one job per ratio — smaller jobs queue faster and failures don't block other ratios.

`submit_training.sh` (run this on the login/interactive node):

```bash
#!/bin/bash
# Run with: bash submit_training.sh

PARQUET=$(find $HOME/scratch/cartridges_output -name "dataset.parquet" | sort | tail -1)
echo "Using parquet: $PARQUET"

for ratio in 1.0 0.1 0.05 0.02; do
    jobscript=$(mktemp /tmp/train_ratio_XXXX.sh)
    cat > $jobscript << INNEREOF
#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -o $HOME/cartridges/train_${ratio}_%j.out
#SBATCH -J train-${ratio}

module load anaconda3
module load cuda/12.6.1
conda activate $HOME/scratch/envs/cartridges

export CARTRIDGES_DIR=$HOME/cartridges
export CARTRIDGES_OUTPUT_DIR=$HOME/scratch/cartridges_output
export HF_HOME=$HOME/scratch/hf_cache
export WANDB_MODE=offline

cd \$CARTRIDGES_DIR
echo "=== Training ratio=${ratio} ==="
python experiments/rank_analysis/train_configs.py --ratio ${ratio} --parquet ${PARQUET}
INNEREOF

    jobid=$(sbatch $jobscript | awk '{print $4}')
    echo "Submitted ratio=${ratio} → job ${jobid}"
    rm $jobscript
done
```

```bash
bash submit_training.sh
```

> **Note:** Uses `$HOME` instead of `~` in the SBATCH `-o` line — `~` is not expanded by
> SLURM and creates a literal `~/` subdirectory inside your home dir.

Checkpoints will be saved at:
- `$CARTRIDGES_OUTPUT_DIR/rank_analysis_ratio_<r>/cache_last.pt`

## Phase 3: Collect KV Data (8 Conditions)

`slurm_collect.sh`:

```bash
#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -p ice-gpu
#SBATCH -o collect_%j.out
#SBATCH -J rank-collect

module load anaconda3
module load cuda/12.6.1
conda activate ~/scratch/envs/cartridges

export CARTRIDGES_DIR=~/cartridges
export CARTRIDGES_OUTPUT_DIR=~/scratch/cartridges_output
export HF_HOME=~/scratch/hf_cache
export WANDB_MODE=offline

cd $CARTRIDGES_DIR

DOC=qasper_e_line_203_context.txt
# Find the parquet generated by synthesize_config.py (timestamped path)
PARQUET=$(find $CARTRIDGES_OUTPUT_DIR -name "dataset.parquet" | sort | tail -1)
echo "Using parquet: $PARQUET"
OUT=data/rank_analysis

# 1) init_full
python experiments/rank_analysis/collect_kv_data.py \
    --condition init_full --document $DOC \
    --parquet $PARQUET --output-dir $OUT

# 2) trained_full (ratio 1.0)
python experiments/rank_analysis/collect_kv_data.py \
    --condition trained --ratio 1.0 \
    --trained-checkpoint $CARTRIDGES_OUTPUT_DIR/rank_analysis_ratio_1.0/cache_last.pt \
    --document $DOC --parquet $PARQUET --output-dir $OUT

# 3-8) snapkv and trained for each ratio
for ratio in 0.1 0.05 0.02; do
    echo "=== SnapKV ratio=$ratio ==="
    python experiments/rank_analysis/collect_kv_data.py \
        --condition snapkv --ratio $ratio \
        --document $DOC --parquet $PARQUET --output-dir $OUT

    echo "=== Trained ratio=$ratio ==="
    python experiments/rank_analysis/collect_kv_data.py \
        --condition trained --ratio $ratio \
        --trained-checkpoint $CARTRIDGES_OUTPUT_DIR/rank_analysis_ratio_$ratio/cache_last.pt \
        --document $DOC --parquet $PARQUET --output-dir $OUT
done
```

```bash
sbatch slurm_collect.sh
```

Output files: `data/rank_analysis/{condition}.pt` (13 files total).

## Phase 4: Generate Plots (CPU Only)

`slurm_plot.sh`:

```bash
#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -p ice-cpu
#SBATCH -o plot_%j.out
#SBATCH -J rank-plot

module load anaconda3
conda activate ~/scratch/envs/cartridges

cd ~/cartridges

python experiments/rank_analysis/plot_rank.py \
    --input-dir data/rank_analysis/ \
    --output-dir plots/
```

```bash
sbatch slurm_plot.sh
```

Output plots:
- `plots/per_head_bars.pdf` — 288 per-head bar charts
- `plots/heatmaps.pdf` — 8 heatmaps (V and Y side by side)
- `plots/summary.pdf` — mean erank vs condition
- `plots/interesting_heads.pdf` — top interesting heads analysis

## GPU Recommendation

| Phase       | GPU         | System RAM | Wall Time |
|-------------|-------------|------------|-----------|
| SGLang      | A100 (any)  | 32 GB      | 8 hours   |
| Synthesis   | A100 (any)  | 32 GB      | 4 hours   |
| Training    | A100 (any)  | 64 GB      | 12 hours  |
| Collection  | A100 (any)  | 64 GB      | 4 hours   |
| Plotting    | None (CPU)  | 16 GB      | 1 hour    |

Peak GPU memory estimates:
- **Training (ratio 1.0, worst case):** ~27 GB (model 8 GB + KV 4.4 GB + gradients 4.4 GB + overhead 10 GB)
- **Collection:** ~12 GB GPU, ~18 GB CPU RAM for hook tensors
- **SGLang serving:** ~10 GB

An A100 40 GB is sufficient for all phases. An A100 80 GB provides extra headroom.

## Troubleshooting

**SGLang server not reachable from another node:**
If synthesis or collection runs on a different node than the SGLang server,
you need to set `SGLANG_URL` to the server node's hostname:

```bash
export SGLANG_URL=http://<sglang-node-hostname>:30000/v1
```

You can find the hostname from the SGLang job output or with `scontrol show job <jobid>`.

**Out of memory during training:**
Reduce `global_batch_size` in `train_configs.py` (e.g., from 32 to 16).

**Out of memory during collection:**
The hook captures Q and K_conv to CPU RAM. If CPU RAM is insufficient,
reduce the number of eval conversations by truncating `eval_self_study.parquet`
or lowering `max_tokens` in `tokenize_eval_conversations()`.
