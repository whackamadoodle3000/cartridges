"""
Run KV data collection (collect_kv_data.py) on Modal for all conditions.

Conditions:
  - init_full                (full document, no training)
  - trained_full             (trained on full document, ratio 1.0)
  - trained_0.1 / 0.05 / 0.02
  - snapkv_0.1  / 0.05 / 0.02

USAGE
-----
modal run infra/modal_collect.py \
    --parquet outputs/.../dataset.parquet \
    --checkpoints-dir ./cartridges_checkpoints
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal

MINUTES = 60
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")

# ---------------------------------------------------------------------------
# Image (same as training)
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/whackamadoodle3000/cartridges /root/cartridges",
        "cd /root/cartridges && uv pip install --system -e .",
        "pip install torch --index-url https://download.pytorch.org/whl/cu126",
        "pip install 'numpy<2'",
        "pip install pyyaml",
    )
    .run_commands("cd /root/cartridges && git pull", force_build=True)
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
data_vol     = modal.Volume.from_name("cartridges-data",    create_if_missing=True)
output_vol   = modal.Volume.from_name("cartridges-outputs", create_if_missing=True)
results_vol  = modal.Volume.from_name("cartridges-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache",  create_if_missing=True)

app = modal.App("cartridges-collect")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:1",
    timeout=2 * MINUTES * 60,
    volumes={
        "/data":                    data_vol,
        "/outputs":                 output_vol,
        "/results":                 results_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def collect(condition: str, ratio: Optional[float], checkpoint_volume_path: Optional[str]):
    import subprocess, sys, os

    doc_path = "/root/cartridges/qasper_e_line_203_context.txt"
    parquet  = "/data/dataset.parquet"
    out_dir  = "/results"

    cmd = [
        sys.executable,
        "/root/cartridges/experiments/rank_analysis/collect_kv_data.py",
        "--condition", condition,
        "--document", doc_path,
        "--parquet", parquet,
        "--output-dir", out_dir,
    ]
    if ratio is not None:
        cmd += ["--ratio", str(ratio)]
    if checkpoint_volume_path is not None:
        cmd += ["--trained-checkpoint", checkpoint_volume_path]

    env = {
        **os.environ,
        "CARTRIDGES_DIR": "/root/cartridges",
        "CARTRIDGES_OUTPUT_DIR": "/outputs",
        "WANDB_MODE": "offline",
        "HF_HOME": "/root/.cache/huggingface",
    }

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"collect_kv_data.py failed with exit code {result.returncode}")
    results_vol.commit()
    print("Done.", flush=True)


@app.local_entrypoint()
def main(
    parquet: str,
    checkpoints_dir: str = "./cartridges_checkpoints",
):
    """
    parquet:          local path to dataset.parquet
    checkpoints_dir:  local dir with downloaded checkpoints (from modal_train.py)
                      Layout: <checkpoints_dir>/<run-dir>/<uuid>/cache-step*.pt
    """
    local_parquet = Path(parquet)
    assert local_parquet.exists(), f"Parquet not found: {local_parquet}"

    # Upload parquet
    print(f"Uploading parquet to Modal volume...")
    with data_vol.batch_upload(force=True) as batch:
        batch.put_file(str(local_parquet), local_parquet.name)
    print("Parquet uploaded.")

    # Find local checkpoints and upload them to the data volume
    ckpt_dir = Path(checkpoints_dir)
    ratio_to_ckpt = {}
    for run_dir in sorted(ckpt_dir.glob("*train_configs")):
        for config_yaml in run_dir.glob("*/config.yaml"):
            import yaml
            cfg = yaml.safe_load(config_yaml.read_text())
            name = cfg.get("name", "")
            for ratio in [1.0, 0.1, 0.05, 0.02]:
                if f"ratio_{ratio}" in name:
                    # Find the checkpoint file
                    uuid_dir = config_yaml.parent
                    ckpts = sorted(uuid_dir.glob("cache-step*.pt")) + sorted(uuid_dir.glob("cache_last.pt"))
                    if ckpts:
                        ratio_to_ckpt[ratio] = ckpts[-1]  # take latest step

    print(f"\nFound checkpoints: { {r: str(p.name) for r, p in ratio_to_ckpt.items()} }")

    # Upload checkpoints to Modal data volume
    print("Uploading checkpoints to Modal volume...")
    with data_vol.batch_upload(force=True) as batch:
        for ratio, local_ckpt in ratio_to_ckpt.items():
            remote_name = f"cache_ratio_{ratio}.pt"
            batch.put_file(str(local_ckpt), remote_name)
            print(f"  {ratio} → /data/{remote_name}")
    print("Checkpoints uploaded.")

    # Build job list
    jobs = []
    jobs.append(("init_full", None, None))                    # full doc, no training

    for ratio in [1.0, 0.1, 0.05, 0.02]:
        if ratio in ratio_to_ckpt:
            jobs.append(("trained", ratio, f"/data/cache_ratio_{ratio}.pt"))
        else:
            print(f"WARNING: no checkpoint for ratio={ratio}, skipping")

    for ratio in [0.1, 0.05, 0.02]:
        jobs.append(("snapkv", ratio, None))

    print(f"\nRunning {len(jobs)} collection jobs in parallel...")
    for _ in collect.starmap(jobs):
        pass

    print("\nAll collection jobs done!")
    print("Download results with:")
    print("  modal volume get cartridges-results / ./rank_analysis_results/")
