"""
Run Cartridges rank-analysis training on Modal.

Trains one cartridge per compression ratio. All ratios run in parallel
on separate H100 containers.

USAGE
-----
# Upload parquet and run all 4 ratios:
modal run infra/modal_train.py --parquet outputs/.../dataset.parquet

# Single ratio:
modal run infra/modal_train.py --parquet outputs/.../dataset.parquet --ratio 0.1
"""
import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GPU_TYPE   = os.environ.get("GPU_TYPE", "H100")
RATIOS     = [1.0, 0.1, 0.05, 0.02]
MINUTES    = 60

# ---------------------------------------------------------------------------
# Image: CUDA + cartridges installed from current repo
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/whackamadoodle3000/cartridges /root/cartridges",
        "cd /root/cartridges && uv pip install --system -e .",
        # Install GPU-enabled PyTorch explicitly
        "pip install torch --index-url https://download.pytorch.org/whl/cu126",
        "pip install 'numpy<2'",
    )
)

# ---------------------------------------------------------------------------
# Volumes: parquet input + checkpoint output
# ---------------------------------------------------------------------------
data_vol     = modal.Volume.from_name("cartridges-data",    create_if_missing=True)
output_vol   = modal.Volume.from_name("cartridges-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache",  create_if_missing=True)

app = modal.App("cartridges-train")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:1",
    timeout=4 * MINUTES * 60,   # 4 hours
    volumes={
        "/data":                  data_vol,
        "/outputs":               output_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def train(ratio: float, parquet_remote: str):
    import subprocess, os, sys

    env = {
        **os.environ,
        "CARTRIDGES_DIR":        "/root/cartridges",
        "CARTRIDGES_OUTPUT_DIR": "/outputs",
        "WANDB_MODE":            "offline",
    }

    cmd = [
        sys.executable,
        "/root/cartridges/experiments/rank_analysis/train_configs.py",
        "--ratio", str(ratio),
        "--parquet", parquet_remote,
    ]
    print(f"[ratio={ratio}] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, check=True)
    output_vol.commit()
    print(f"[ratio={ratio}] Done. Exit code: {result.returncode}", flush=True)


@app.local_entrypoint()
def main(
    parquet: str,
    ratio: Optional[float] = None,
):
    """
    parquet: local path to dataset.parquet
    ratio:   optional single ratio to train; if omitted trains all 4
    """
    # Upload parquet to Modal Volume
    local_path = Path(parquet)
    assert local_path.exists(), f"Parquet not found: {local_path}"

    remote_path = f"/data/{local_path.name}"
    print(f"Uploading {local_path} → Modal volume at {remote_path} ...")
    with data_vol.batch_upload() as batch:
        batch.put_file(str(local_path), remote_path)
    print("Upload complete.")

    ratios_to_run = [ratio] if ratio is not None else RATIOS
    print(f"Submitting training for ratios: {ratios_to_run}")

    # Run all ratios in parallel
    for result in train.starmap([(r, remote_path) for r in ratios_to_run]):
        pass

    print("All training runs complete. Checkpoints saved to Modal volume 'cartridges-outputs'.")
    print("Download checkpoints with:")
    print("  modal volume get cartridges-outputs / ./cartridges_checkpoints/")
