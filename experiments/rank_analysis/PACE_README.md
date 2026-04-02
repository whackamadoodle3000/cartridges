# Running Rank Analysis on PACE (ICE Cluster)

## Pipeline Overview

The experiment has 5 sequential phases:

1. **SGLang server** — Serve Qwen3-4B locally so self-study synthesis can call it.
2. **Synthesis** — Use the running model to generate synthetic Q&A conversations about
   Frankenstein (train split: 256, eval split: 64). The model asks and answers its own
   questions about chunks of the document. *Cancel the SGLang job after this finishes.*
3. **Training** — For each compression ratio, run Cartridges gradient descent. The KV
   cache starts from a causal forward pass over the document (first N tokens at ratio N),
   then is optimized via gradient descent on the 256 train-split conversations to produce
   good model outputs when answering questions about the document. Output: `cache_last.pt`
   per ratio.
4. **Collection** — For all 13 conditions (init_full, trained_full, snapkv/trained ×5),
   run a forward pass over the 64 eval-split conversations using each condition's KV cache.
   Capture Q and K_conv via the activation hook, compute Y (the cartridge's real attention
   contribution), and save V and Y tensors.
5. **Plotting** — Load all 13 `.pt` files and produce bar charts, heatmaps, and summary
   plots of effective rank.

## SSH Login

```bash
ssh <gtusername>@login-ice.pace.gatech.edu
```

## Environment Setup

```bash
# Create conda env
module load anaconda3
conda create -n cartridges python=3.12 -y
conda activate cartridges

# Clone and install
git clone <your-cartridges-repo-url> ~/cartridges
cd ~/cartridges
pip install -e .

# SGLang for local model serving
pip install "sglang[all]"

# Matplotlib for plotting (no display needed)
pip install matplotlib
```

## Environment Variables

Add to your `~/.bashrc` or set in each SLURM script:

```bash
export CARTRIDGES_DIR=~/cartridges
export CARTRIDGES_OUTPUT_DIR=~/cartridges_output
export WANDB_MODE=offline
export SGLANG_URL=http://localhost:30000/v1
```

## Directory Setup

```bash
mkdir -p $CARTRIDGES_OUTPUT_DIR
mkdir -p ~/cartridges/experiments/rank_analysis/data
mkdir -p ~/cartridges/data/rank_analysis
mkdir -p ~/cartridges/plots
```

## Download the Document

```bash
cd ~/cartridges/experiments/rank_analysis/data
wget -O frankenstein.txt https://www.gutenberg.org/cache/epub/84/pg84.txt
```

### Switching to a Different Document

The document path is referenced in two places. Update both before running any phase:

**1. `synthesize_config.py`** — controls what the model generates Q&A conversations about:
```python
DOC_PATH = os.path.join(CARTRIDGES_DIR, "experiments/rank_analysis/data/frankenstein.txt")
# Change to e.g.:
DOC_PATH = os.path.join(CARTRIDGES_DIR, "experiments/rank_analysis/data/longhealthtext.txt")
```

**2. `train_configs.py`** — controls `KVFromText`, the initial document forward pass:
```python
DEFAULT_DOC = os.path.join(CARTRIDGES_DIR, "experiments/rank_analysis/data/frankenstein.txt")
# Change to e.g.:
DEFAULT_DOC = os.path.join(CARTRIDGES_DIR, "experiments/rank_analysis/data/longhealthtext.txt")
```

Then re-run all 4 phases from scratch (synthesis → training → collection → plotting) — the
trained cartridges and eval parquets are document-specific and cannot be reused.

Token length check: Qwen3-4B's `max_position_embeddings` is 40960. Verify your document
fits by running:
```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4b')
text = open('experiments/rank_analysis/data/longhealthtext.txt').read()
n = len(tok.encode(text))
print(f'{n} tokens (limit: 40960)')
"
```
If it exceeds ~35k tokens, either trim the file or the `KVFromText` max_tokens computation
in `train_configs.py` (currently uses `int(ratio * 30_000)` as an approximation).

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
conda activate cartridges

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
conda activate cartridges

export CARTRIDGES_DIR=~/cartridges
export CARTRIDGES_OUTPUT_DIR=~/cartridges_output
export WANDB_MODE=offline

# Point to the SGLang server node (adjust if needed)
export SGLANG_URL=http://localhost:30000/v1

cd $CARTRIDGES_DIR

# Train set: 256 conversations
python experiments/rank_analysis/synthesize_config.py --split train

# Eval set: 64 conversations
python experiments/rank_analysis/synthesize_config.py --split eval
```

```bash
sbatch slurm_synthesize.sh
```

The parquet files will be saved at:
- `$CARTRIDGES_OUTPUT_DIR/rank_analysis_synth_train/artifact/dataset.parquet`
- `$CARTRIDGES_OUTPUT_DIR/rank_analysis_synth_eval/artifact/dataset.parquet`

## Phase 2: Train Cartridges (6 Ratios)

**Important: cancel the SGLang server job before submitting training.** Both SGLang and the
training job load Qwen3-4B onto the same GPU. They cannot run simultaneously on one A100.

```bash
# Find and cancel the SGLang job first
squeue -u $USER
scancel <sglang_job_id>
```

Each ratio trains a separate cartridge. They can run sequentially in one job.

`slurm_train.sh`:

```bash
#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -p ice-gpu
#SBATCH -o train_%j.out
#SBATCH -J rank-train

module load anaconda3
conda activate cartridges

export CARTRIDGES_DIR=~/cartridges
export CARTRIDGES_OUTPUT_DIR=~/cartridges_output
export WANDB_MODE=offline

cd $CARTRIDGES_DIR

for ratio in 1.0 0.5 0.2 0.1 0.05 0.02; do
    echo "=== Training ratio=$ratio ==="
    python experiments/rank_analysis/train_configs.py --ratio $ratio
done
```

```bash
sbatch slurm_train.sh
```

Checkpoints will be saved at:
- `$CARTRIDGES_OUTPUT_DIR/rank_analysis_ratio_<r>/cache_last.pt`

## Phase 3: Collect KV Data (13 Conditions)

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
conda activate cartridges

export CARTRIDGES_DIR=~/cartridges
export CARTRIDGES_OUTPUT_DIR=~/cartridges_output
export WANDB_MODE=offline

cd $CARTRIDGES_DIR

DOC=experiments/rank_analysis/data/frankenstein.txt
EVAL=$CARTRIDGES_OUTPUT_DIR/rank_analysis_synth_eval/artifact/dataset.parquet
OUT=data/rank_analysis

# 1) init_full
python experiments/rank_analysis/collect_kv_data.py \
    --condition init_full --document $DOC \
    --eval-parquet $EVAL --output-dir $OUT

# 2) trained_full (ratio 1.0)
python experiments/rank_analysis/collect_kv_data.py \
    --condition trained --ratio 1.0 \
    --trained-checkpoint $CARTRIDGES_OUTPUT_DIR/rank_analysis_ratio_1.0/cache_last.pt \
    --document $DOC --eval-parquet $EVAL --output-dir $OUT

# 3-12) snapkv and trained for each ratio
for ratio in 0.5 0.2 0.1 0.05 0.02; do
    echo "=== SnapKV ratio=$ratio ==="
    python experiments/rank_analysis/collect_kv_data.py \
        --condition snapkv --ratio $ratio \
        --document $DOC --eval-parquet $EVAL --output-dir $OUT

    echo "=== Trained ratio=$ratio ==="
    python experiments/rank_analysis/collect_kv_data.py \
        --condition trained --ratio $ratio \
        --trained-checkpoint $CARTRIDGES_OUTPUT_DIR/rank_analysis_ratio_$ratio/cache_last.pt \
        --document $DOC --eval-parquet $EVAL --output-dir $OUT
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
conda activate cartridges

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
- `plots/heatmaps.pdf` — 13 heatmaps (V and Y side by side)
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
