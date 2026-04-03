"""
Plot effective rank analysis results.

Loads .pt files produced by collect_kv_data.py and generates:
  1. Per-head bar charts (multi-page PDF)
  2. Heatmaps per condition (PDF)
  3. Summary line plot (PDF)
  4. Interesting heads callout (PDF + printed table)

Usage:
    python plot_rank.py --input-dir data/rank_analysis/ --output-dir plots/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch


N_LAYERS = 36
N_KV_HEADS = 8

CONDITION_ORDER = [
    "init_full",
    "trained_full",
    "snapkv_0.1", "trained_0.1",
    "snapkv_0.05", "trained_0.05",
    "snapkv_0.02", "trained_0.02",
]

CONDITION_COLORS = {
    "init_full": "#1f77b4",
    "trained_full": "#ff7f0e",
    "snapkv_0.1": "#e377c2", "trained_0.1": "#7f7f7f",
    "snapkv_0.05": "#bcbd22", "trained_0.05": "#17becf",
    "snapkv_0.02": "#aec7e8", "trained_0.02": "#ffbb78",
}


def effective_rank(mat: torch.Tensor) -> float:
    """Raw effective rank, range [1, min(rows, cols)]. No normalization."""
    s = torch.linalg.svdvals(mat.float())
    s = s[s > 1e-10]
    if s.numel() == 0:
        return 1.0
    p = s / s.sum()
    return (-(p * p.log()).sum()).exp().item()


def load_all_conditions(input_dir: str) -> dict:
    """Load all .pt files from input_dir. Returns {condition_name: data_dict}."""
    data = {}
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(input_dir, fname)
        d = torch.load(path, map_location="cpu", weights_only=False)
        cond = d["condition"]
        data[cond] = d
        print(f"Loaded {cond}: T_cart={d['metadata']['T_cart']}, "
              f"n_conv_tokens={d['metadata']['n_conv_tokens']}")
    return data


def compute_all_ranks(data: dict) -> dict:
    """Compute erank_V and erank_Y for every (condition, layer, head).

    Returns {condition: {"erank_V": np.array(N_LAYERS, N_KV_HEADS),
                          "erank_Y": np.array(N_LAYERS, N_KV_HEADS)}}
    """
    ranks = {}
    for cond, d in data.items():
        erank_V = np.zeros((N_LAYERS, N_KV_HEADS))
        erank_Y = np.zeros((N_LAYERS, N_KV_HEADS))

        for l in range(N_LAYERS):
            for h in range(N_KV_HEADS):
                V_h = d["values"][l][h]          # (T_cart, head_dim)
                Y_h = d["attention_outputs"][l][h]  # (n_conv_tokens, head_dim)
                erank_V[l, h] = effective_rank(V_h)
                erank_Y[l, h] = effective_rank(Y_h)

        ranks[cond] = {"erank_V": erank_V, "erank_Y": erank_Y}
        print(f"{cond}: mean erank_V={erank_V.mean():.2f}, mean erank_Y={erank_Y.mean():.2f}")

    return ranks


def plot_per_head_bars(ranks: dict, output_dir: str):
    """Plot 1: one figure per (layer, head) with bars for each condition."""
    path = os.path.join(output_dir, "per_head_bars.pdf")
    conds = [c for c in CONDITION_ORDER if c in ranks]
    n_conds = len(conds)

    with PdfPages(path) as pdf:
        for l in range(N_LAYERS):
            for h in range(N_KV_HEADS):
                fig, axes = plt.subplots(1, 2, figsize=(max(14, n_conds * 0.8), 5))

                for ax_idx, (metric_key, metric_label) in enumerate([
                    ("erank_V", "erank(V)"),
                    ("erank_Y", "erank(Y)"),
                ]):
                    ax = axes[ax_idx]
                    vals = [ranks[c][metric_key][l, h] for c in conds]
                    colors = [CONDITION_COLORS.get(c, "#333333") for c in conds]
                    bars = ax.bar(range(n_conds), vals, color=colors)
                    ax.set_xticks(range(n_conds))
                    ax.set_xticklabels(conds, rotation=60, ha="right", fontsize=7)
                    ax.set_ylabel(metric_label)
                    ax.set_title(f"{metric_label}")

                    for bar, val in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{val:.1f}", ha="center", va="bottom", fontsize=6)

                fig.suptitle(f"Layer {l}, KV Head {h}", fontsize=13, fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved {N_LAYERS * N_KV_HEADS} per-head bar charts to {path}")


def plot_heatmaps(ranks: dict, output_dir: str):
    """Plot 2: one page per condition, V and Y heatmaps side by side."""
    path = os.path.join(output_dir, "heatmaps.pdf")
    conds = [c for c in CONDITION_ORDER if c in ranks]

    with PdfPages(path) as pdf:
        for cond in conds:
            fig, axes = plt.subplots(1, 2, figsize=(18, 4))

            for ax_idx, (metric_key, metric_label) in enumerate([
                ("erank_V", "erank(V)"),
                ("erank_Y", "erank(Y)"),
            ]):
                ax = axes[ax_idx]
                mat = ranks[cond][metric_key]  # (N_LAYERS, N_KV_HEADS)
                im = ax.imshow(mat.T, aspect="auto", cmap="viridis",
                               origin="lower")
                ax.set_xlabel("Layer")
                ax.set_ylabel("KV Head")
                ax.set_title(f"{metric_label}")
                ax.set_xticks(range(0, N_LAYERS, 4))
                ax.set_yticks(range(N_KV_HEADS))
                plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)

            fig.suptitle(f"Condition: {cond}", fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved {len(conds)} heatmaps to {path}")


def plot_summary(ranks: dict, output_dir: str):
    """Plot 3: mean +/- std erank across all heads, one point per condition."""
    path = os.path.join(output_dir, "summary.pdf")
    conds = [c for c in CONDITION_ORDER if c in ranks]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(conds))

    for metric_key, metric_label, color in [
        ("erank_V", "erank(V)", "#1f77b4"),
        ("erank_Y", "erank(Y)", "#d62728"),
    ]:
        means = [ranks[c][metric_key].mean() for c in conds]
        stds = [ranks[c][metric_key].std() for c in conds]

        ax.plot(x, means, "o-", label=metric_label, color=color)
        ax.fill_between(x, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Effective Rank")
    ax.set_title("Mean Effective Rank Across All 288 (Layer, Head) Pairs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved summary plot to {path}")


def plot_interesting_heads(ranks: dict, output_dir: str):
    """Plot 4: top interesting heads analysis + printed table."""
    path = os.path.join(output_dir, "interesting_heads.pdf")
    conds = [c for c in CONDITION_ORDER if c in ranks]

    trained_conds = [c for c in conds if c.startswith("trained_")]
    snapkv_conds = [c for c in conds if c.startswith("snapkv_")]

    head_ids = [(l, h) for l in range(N_LAYERS) for h in range(N_KV_HEADS)]

    print("\n=== Interesting Heads Analysis ===\n")

    # 1) Lowest mean erank_V across trained conditions
    if trained_conds:
        mean_trained_V = np.mean(
            [ranks[c]["erank_V"] for c in trained_conds], axis=0
        )  # (N_LAYERS, N_KV_HEADS)
        flat_idx = np.argsort(mean_trained_V.ravel())[:5]
        print("Top 5 lowest mean erank_V across trained conditions (most compressible):")
        table1 = []
        for idx in flat_idx:
            l, h = head_ids[idx]
            val = mean_trained_V[l, h]
            table1.append((l, h, val))
            print(f"  Layer {l}, Head {h}: mean erank_V = {val:.2f}")
    else:
        table1 = []

    # 2) Largest snapkv - trained gap for erank_V
    ratios = ["0.5", "0.2", "0.1", "0.05", "0.02"]
    gap_accumulator = np.zeros((N_LAYERS, N_KV_HEADS))
    gap_count = 0
    for r in ratios:
        sk = f"snapkv_{r}"
        tr = f"trained_{r}"
        if sk in ranks and tr in ranks:
            gap_accumulator += ranks[sk]["erank_V"] - ranks[tr]["erank_V"]
            gap_count += 1

    if gap_count > 0:
        gap_avg = gap_accumulator / gap_count
        flat_idx = np.argsort(-gap_avg.ravel())[:5]
        print("\nTop 5 largest erank_V(snapkv) - erank_V(trained) gap (training reduces rank most):")
        table2 = []
        for idx in flat_idx:
            l, h = head_ids[idx]
            val = gap_avg[l, h]
            table2.append((l, h, val))
            print(f"  Layer {l}, Head {h}: avg gap = {val:.2f}")
    else:
        table2 = []

    # 3) Where erank_Y diverges most from erank_V
    divergence = np.zeros((N_LAYERS, N_KV_HEADS))
    div_count = 0
    for c in conds:
        divergence += np.abs(ranks[c]["erank_Y"] - ranks[c]["erank_V"])
        div_count += 1

    if div_count > 0:
        divergence_avg = divergence / div_count
        flat_idx = np.argsort(-divergence_avg.ravel())[:5]
        print("\nTop 5 where erank_Y diverges most from erank_V:")
        table3 = []
        for idx in flat_idx:
            l, h = head_ids[idx]
            val = divergence_avg[l, h]
            table3.append((l, h, val))
            print(f"  Layer {l}, Head {h}: avg |erank_Y - erank_V| = {val:.2f}")
    else:
        table3 = []

    # Plot the three tables
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    titles = [
        "Most compressible (lowest mean erank_V, trained)",
        "Training most reduces rank (snapkv - trained gap)",
        "Largest |erank_Y - erank_V| divergence",
    ]
    tables = [table1, table2, table3]
    col_labels_list = [
        ["Layer", "Head", "Mean erank_V"],
        ["Layer", "Head", "Avg Gap"],
        ["Layer", "Head", "Avg |Y-V| Divergence"],
    ]

    for ax, title, table, col_labels in zip(axes, titles, tables, col_labels_list):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        if table:
            cell_text = [[str(l), str(h), f"{v:.2f}"] for l, h, v in table]
            t = ax.table(cellText=cell_text, colLabels=col_labels,
                         loc="center", cellLoc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(9)
            t.scale(1, 1.5)
        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                    fontsize=10, color="gray", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved interesting heads to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing .pt files from collect_kv_data.py")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading condition data...")
    data = load_all_conditions(args.input_dir)

    print("Computing effective ranks...")
    ranks = compute_all_ranks(data)

    print("\nGenerating plots...")
    plot_per_head_bars(ranks, args.output_dir)
    plot_heatmaps(ranks, args.output_dir)
    plot_summary(ranks, args.output_dir)
    plot_interesting_heads(ranks, args.output_dir)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
