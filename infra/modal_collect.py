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
    )
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

    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    results_vol.commit()
    print("Done.", flush=True)


@app.local_entrypoint()
def main(
    parquet: str,
    checkpoints_dir: str = "./cartridges_checkpoints",
):
    """
    parquet:          local path to dataset.parquet
    checkpoints_dir:  local directory with downloaded training checkpoints,
                      expected layout:
                        <checkpoints_dir>/
                          2026-...-train_configs/<uuid>/artifact/cache_last.pt  (ratio 1.0)
                          ...
                      OR pass the Modal volume paths directly if you haven't downloaded yet.
    """
    # Upload parquet (idempotent — re-uploading is fine)
    local_parquet = Path(parquet)
    assert local_parquet.exists(), f"Parquet not found: {local_parquet}"
    print(f"Uploading parquet to Modal volume...")
    with data_vol.batch_upload() as batch:
        batch.put_file(str(local_parquet), local_parquet.name)
    print("Parquet uploaded.")

    ratios = [1.0, 0.1, 0.05, 0.02]

    # Find checkpoints in the output volume
    # Training saves to /outputs/<run-dir>/<uuid>/artifact/cache_last.pt
    # We look for checkpoint files matching each ratio by run name
    def find_checkpoint(ratio: float) -> Optional[str]:
        """Return the volume path to cache_last.pt for a given ratio."""
        # Try to find it by listing the output volume
        try:
            entries = list(output_vol.listdir("/"))
            for entry in entries:
                if f"ratio_{ratio}" in entry.path or f"ratio-{ratio}" in entry.path:
                    # Walk into it to find cache_last.pt
                    for sub in output_vol.listdir(f"/{entry.path}"):
                        for artifact in output_vol.listdir(f"/{entry.path}/{sub.path}/artifact"):
                            if "cache_last" in artifact.path or "cache-step" in artifact.path:
                                return f"/outputs/{entry.path}/{sub.path}/artifact/{artifact.path}"
        except Exception as e:
            print(f"Warning: could not find checkpoint for ratio {ratio}: {e}")
        return None

    # Build job list
    jobs = []

    # Baseline: full document, no training
    jobs.append(("init_full", None, None))

    # Trained conditions
    for ratio in ratios:
        ckpt = find_checkpoint(ratio)
        if ckpt is None:
            print(f"WARNING: no checkpoint found for ratio={ratio}, skipping trained condition")
        else:
            print(f"Found checkpoint for ratio={ratio}: {ckpt}")
            condition = "trained" if ratio < 1.0 else "trained"
            jobs.append((condition, ratio, ckpt))

    # SnapKV conditions (no checkpoint needed)
    for ratio in [0.1, 0.05, 0.02]:
        jobs.append(("snapkv", ratio, None))

    print(f"\nRunning {len(jobs)} collection jobs in parallel...")
    for _ in collect.starmap(jobs):
        pass

    print("\nAll collection jobs done!")
    print("Download results with:")
    print("  modal volume get cartridges-results / ./rank_analysis_results/")
