"""
Run rank analysis plotting on Modal (CPU only) and save plots to a volume.

USAGE
-----
modal run infra/modal_plot.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "matplotlib", "numpy")
    .run_commands(
        "git clone https://github.com/whackamadoodle3000/cartridges /root/cartridges",
    )
    .run_commands("cd /root/cartridges && git pull", force_build=True)
)

results_vol = modal.Volume.from_name("cartridges-results", create_if_missing=False)
plots_vol   = modal.Volume.from_name("cartridges-plots",   create_if_missing=True)

app = modal.App("cartridges-plot")


@app.function(
    image=image,
    cpu=4,
    memory=32768,  # 32 GB RAM for loading all .pt files
    timeout=30 * 60,
    volumes={
        "/results": results_vol,
        "/plots":   plots_vol,
    },
)
def plot():
    import subprocess, sys
    cmd = [
        sys.executable,
        "/root/cartridges/experiments/rank_analysis/plot_rank.py",
        "--input-dir",  "/results",
        "--output-dir", "/plots",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    plots_vol.commit()
    print("Plots saved to Modal volume 'cartridges-plots'.", flush=True)


@app.local_entrypoint()
def main():
    plot.remote()
    print("\nDone! Download plots with:")
    print("  modal volume get cartridges-plots / ./rank_analysis_plots/")
