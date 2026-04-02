"""
Collect KV cache data and attention outputs for rank analysis.

For each condition, saves V_cart and the cartridge's real attention contribution Y
per layer and KV head.

Usage:
    python collect_kv_data.py --condition init_full --document data/frankenstein.txt \
        --eval-parquet data/eval_self_study.parquet --output-dir data/rank_analysis/

    python collect_kv_data.py --condition trained --ratio 0.5 \
        --trained-checkpoint $OUT/rank_analysis_ratio_0.5/cache_last.pt \
        --document data/frankenstein.txt --eval-parquet data/eval_self_study.parquet \
        --output-dir data/rank_analysis/

    python collect_kv_data.py --condition snapkv --ratio 0.5 \
        --document data/frankenstein.txt --eval-parquet data/eval_self_study.parquet \
        --output-dir data/rank_analysis/
"""
import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.text import KVFromText
from cartridges.initialization.tokenization_utils import MODEL_TO_SYSTEM_PROMPT_TOKENIZER
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.models.qwen.modeling_qwen3 import Qwen3Attention


MODEL_NAME = "Qwen/Qwen3-4b"
N_LAYERS = 36
N_KV_HEADS = 8
N_Q_HEADS = 32
HEAD_DIM = 128
GQA_RATIO = N_Q_HEADS // N_KV_HEADS  # 4


def load_model(device: str = "cuda"):
    """Load Qwen3-4B and tokenizer."""
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    attn_config = AttnConfig(n_layers=N_LAYERS, n_heads=N_KV_HEADS, head_dim=HEAD_DIM)
    return model, tokenizer, attn_config


def build_init_cache(model, tokenizer, attn_config, doc_path: str, max_tokens=None):
    """Run a causal forward pass over the document and return the resulting cache."""
    initializer = KVFromText(KVFromText.Config(
        text_source=doc_path,
        max_tokens=max_tokens,
    ))
    cache = initializer.initialize_kv_cache(tokenizer, model, attn_config)
    return cache


def extract_kv(cache: TrainableCache):
    """Extract keys and values as plain tensors from a TrainableCache.

    Returns lists of length n_layers, each tensor shape (n_kv_heads, T_cart, head_dim).
    """
    keys = []
    values = []
    for l in range(len(cache.trainable_keys)):
        k = cache.trainable_keys[l].detach().squeeze(0)  # (n_kv_heads, T, head_dim)
        v = cache.trainable_values[l].detach().squeeze(0)
        keys.append(k)
        values.append(v)
    return keys, values


def run_forward_with_hooks(model, cache, input_ids, device="cuda"):
    """Run a single forward pass with activation capture enabled.

    Returns activation records grouped by layer index.
    """
    cache = cache.to(device)
    cache.clear()

    Qwen3Attention._activation_store = []
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            seq_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=device)
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=device)
            model(
                input_ids=input_ids.to(device),
                seq_ids=seq_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                mode="generate",
            )

    records = Qwen3Attention._activation_store
    Qwen3Attention._activation_store = None

    by_layer = defaultdict(list)
    for rec in records:
        by_layer[rec["layer_idx"]].append(rec)

    Q_per_layer = {}
    K_conv_per_layer = {}
    for layer_idx, recs in by_layer.items():
        Q_per_layer[layer_idx] = torch.cat(
            [r["query_states"] for r in recs], dim=2
        )  # (1, n_q_heads, total_tokens, head_dim)
        K_conv_per_layer[layer_idx] = torch.cat(
            [r["key_states"] for r in recs], dim=2
        )  # (1, n_kv_heads, total_tokens, head_dim)

    return Q_per_layer, K_conv_per_layer


def compute_attention_outputs(keys, values, Q_per_layer, K_conv_per_layer):
    """Compute the cartridge's real attention contribution Y per layer/head.

    Exactly replicates the model's causal attention:
      - Every conv token attends to ALL T_cart cartridge tokens (no causal mask).
      - Conv token i attends only to conv tokens 0..i (lower-triangular causal mask).

    Y[i] = softmax_causal(Q_i @ [K_cart; K_conv[:i+1]].T / sqrt(d))[:T_cart] @ V_cart

    Computed vectorised with a boolean mask — no per-token loop needed.

    Returns a list of length n_layers, each tensor shape (n_kv_heads, n_conv_tokens, head_dim).
    """
    Y_all = []
    scale = 1.0 / math.sqrt(HEAD_DIM)

    for l in range(N_LAYERS):
        K_cart_l = keys[l].float()       # (n_kv_heads, T_cart, head_dim)
        V_cart_l = values[l].float()      # (n_kv_heads, T_cart, head_dim)
        Q_l = Q_per_layer[l][0].float()   # (n_q_heads, n_conv, head_dim)
        K_conv_l = K_conv_per_layer[l][0].float()  # (n_kv_heads, n_conv, head_dim)

        T_cart = K_cart_l.shape[1]
        n_conv = K_conv_l.shape[1]

        # Causal mask for the conv-to-conv block: query i can only see key j <= i.
        # Shape (n_conv, n_conv); True = keep, False = mask out.
        causal_mask = torch.ones(n_conv, n_conv, dtype=torch.bool).tril()

        Y_heads = []

        for h in range(N_KV_HEADS):
            K_cart_h = K_cart_l[h]   # (T_cart, head_dim)
            V_cart_h = V_cart_l[h]   # (T_cart, head_dim)
            K_conv_h = K_conv_l[h]   # (n_conv, head_dim)
            K_all = torch.cat([K_cart_h, K_conv_h], dim=0)  # (T_cart + n_conv, head_dim)

            q_start = h * GQA_RATIO
            Q_h = Q_l[q_start:q_start + GQA_RATIO]  # (4, n_conv, head_dim)

            scores = torch.einsum("gnd,kd->gnk", Q_h, K_all) * scale
            # (4, n_conv, T_cart + n_conv)

            # Mask the conv-to-conv block: future conv keys get -inf before softmax.
            scores_conv = scores[:, :, T_cart:].masked_fill(
                ~causal_mask.unsqueeze(0), float("-inf")
            )  # (4, n_conv, n_conv)
            scores = torch.cat([scores[:, :, :T_cart], scores_conv], dim=-1)

            attn = F.softmax(scores, dim=-1)[:, :, :T_cart]  # (4, n_conv, T_cart)
            attn_avg = attn.mean(dim=0)   # (n_conv, T_cart)
            Y_h = attn_avg @ V_cart_h     # (n_conv, head_dim)
            Y_heads.append(Y_h)

        Y_all.append(torch.stack(Y_heads, dim=0).to(torch.float32))

    return Y_all


def build_snapkv_cache(
    model, tokenizer, attn_config, doc_path: str, ratio: float, device="cuda"
):
    """Build a SnapKV-compressed cache by selecting top tokens per head."""
    full_cache = build_init_cache(model, tokenizer, attn_config, doc_path, max_tokens=None)
    full_keys, full_values = extract_kv(full_cache)
    T_full = full_keys[0].shape[1]
    T_cart = int(ratio * T_full)

    content = Path(doc_path).read_text()
    tokenize_fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tokenizer.name_or_path.lower()]
    input_ids = tokenize_fn(tokenizer=tokenizer, content=content, max_tokens=None).squeeze(0)

    obs_window = 64
    obs_cache = TrainableCache(config=attn_config)
    Qwen3Attention._activation_store = []
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ids = input_ids.to(device)
            seq_ids = torch.zeros(ids.shape[0], dtype=torch.long, device=device)
            pos_ids = torch.arange(ids.shape[0], dtype=torch.long, device=device)
            model(
                input_ids=ids, seq_ids=seq_ids, position_ids=pos_ids,
                past_key_values=obs_cache, use_cache=True, mode="generate",
            )

    records = Qwen3Attention._activation_store
    Qwen3Attention._activation_store = None

    by_layer = defaultdict(list)
    for rec in records:
        by_layer[rec["layer_idx"]].append(rec)

    Q_obs_per_layer = {}
    for layer_idx, recs in by_layer.items():
        Q_all = torch.cat([r["query_states"] for r in recs], dim=2)
        Q_obs_per_layer[layer_idx] = Q_all[:, :, -obs_window:, :]  # (1, n_q_heads, 64, head_dim)

    snap_keys_list = []
    snap_values_list = []

    for l in range(N_LAYERS):
        K_full_l = full_keys[l].float()   # (n_kv_heads, T_full, head_dim)
        V_full_l = full_values[l].float()
        Q_obs_l = Q_obs_per_layer[l][0].float()  # (n_q_heads, 64, head_dim)

        scale = 1.0 / math.sqrt(HEAD_DIM)
        snap_k_heads = []
        snap_v_heads = []

        for h in range(N_KV_HEADS):
            K_h = K_full_l[h]  # (T_full, head_dim)
            q_start = h * GQA_RATIO
            Q_obs_h = Q_obs_l[q_start:q_start + GQA_RATIO]  # (4, 64, head_dim)

            scores = torch.einsum("gnd,td->gnt", Q_obs_h, K_h) * scale
            attn = F.softmax(scores, dim=-1)  # (4, 64, T_full)
            vote = attn.mean(dim=0).sum(dim=0)  # (T_full,)

            vote_pooled = F.avg_pool1d(
                vote.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2, stride=1
            ).squeeze()

            _, indices = torch.topk(vote_pooled, T_cart)
            indices_sorted, _ = torch.sort(indices)

            snap_k_heads.append(K_full_l[h][indices_sorted])
            snap_v_heads.append(V_full_l[h][indices_sorted])

        snap_keys_list.append(
            torch.stack(snap_k_heads, dim=0).unsqueeze(0).to(torch.bfloat16)
        )  # (1, n_kv_heads, T_cart, head_dim)
        snap_values_list.append(
            torch.stack(snap_v_heads, dim=0).unsqueeze(0).to(torch.bfloat16)
        )

    snap_cache = TrainableCache(
        config=attn_config,
        init_keys=snap_keys_list,
        init_values=snap_values_list,
    )
    return snap_cache


def tokenize_eval_conversations(eval_parquet: str, tokenizer) -> torch.Tensor:
    """Load and tokenize eval self-study conversations into a flat token sequence.

    Skips 'system' messages because they contain a document chunk that was used
    during synthesis but should NOT be present during rank analysis — the cartridge
    already provides the document context, exactly as in real inference.
    """
    import json

    df = pd.read_parquet(eval_parquet)

    all_ids = []
    for _, row in df.iterrows():
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        # Keep only user+assistant turns (no system prompt with document chunk).
        # During real inference the cartridge IS the document; there is no system prompt.
        qa_messages = [m for m in messages if m.get("role") != "system"]
        if not qa_messages:
            continue

        ids = tokenizer.apply_chat_template(
            qa_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        all_ids.extend(ids)

    # Cap to stay within Qwen3-4B's 32,768-token context limit.
    # With a ~16k-token document, init_full uses ~16k cartridge positions, leaving ~16k
    # for conv tokens. Use 8k to be safe and keep GPU memory reasonable.
    max_tokens = 8_000
    if len(all_ids) > max_tokens:
        all_ids = all_ids[:max_tokens]

    return torch.tensor(all_ids, dtype=torch.long)


def save_result(
    condition: str,
    values: list,
    attention_outputs: list,
    metadata: dict,
    output_dir: str,
):
    """Save collected data to a .pt file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{condition}.pt"
    path = os.path.join(output_dir, filename)

    values_cpu = [v.cpu() for v in values]
    attn_cpu = [y.cpu() for y in attention_outputs]

    torch.save({
        "condition": condition,
        "values": values_cpu,
        "attention_outputs": attn_cpu,
        "metadata": metadata,
    }, path)

    total_size = sum(v.numel() * v.element_size() for v in values_cpu)
    total_size += sum(y.numel() * y.element_size() for y in attn_cpu)
    print(f"Saved {path} ({total_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Collect KV data for rank analysis")
    parser.add_argument("--condition", required=True,
                        choices=["init_full", "snapkv", "trained"])
    parser.add_argument("--ratio", type=float, default=None,
                        help="Compression ratio (required for snapkv/trained)")
    parser.add_argument("--document", required=True, help="Path to document text file")
    parser.add_argument("--trained-checkpoint", type=str, default=None,
                        help="Path to cache_last.pt (required for trained condition)")
    parser.add_argument("--eval-parquet", required=True,
                        help="Path to eval self-study parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.condition in ("snapkv", "trained") and args.ratio is None:
        parser.error("--ratio is required for snapkv and trained conditions")
    if args.condition == "trained" and args.trained_checkpoint is None:
        parser.error("--trained-checkpoint is required for trained condition")

    condition_name = args.condition
    if args.ratio is not None:
        if args.condition == "trained" and args.ratio >= 1.0:
            condition_name = "trained_full"
        elif args.condition != "init_full":
            condition_name = f"{args.condition}_{args.ratio}"

    print(f"=== Collecting data for condition: {condition_name} ===")

    print("Loading model...")
    model, tokenizer, attn_config = load_model(args.device)

    print("Building cache...")
    if args.condition == "init_full":
        cache = build_init_cache(model, tokenizer, attn_config, args.document, max_tokens=None)
    elif args.condition == "snapkv":
        cache = build_snapkv_cache(
            model, tokenizer, attn_config, args.document, args.ratio, args.device
        )
    elif args.condition == "trained":
        cache = TrainableCache.from_pretrained(args.trained_checkpoint, device=args.device)
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    keys, values = extract_kv(cache)
    T_cart = keys[0].shape[1]

    content = Path(args.document).read_text()
    tokenize_fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tokenizer.name_or_path.lower()]
    T_full_ids = tokenize_fn(tokenizer=tokenizer, content=content, max_tokens=None).squeeze(0)
    T_full = T_full_ids.shape[0]

    print(f"T_cart={T_cart}, T_full={T_full}")

    print("Tokenizing eval conversations...")
    conv_ids = tokenize_eval_conversations(args.eval_parquet, tokenizer)
    n_conv_tokens = conv_ids.shape[0]
    print(f"Eval conversation tokens: {n_conv_tokens}")

    print("Running forward pass with hooks...")
    Q_per_layer, K_conv_per_layer = run_forward_with_hooks(
        model, cache, conv_ids, args.device
    )

    print("Computing attention outputs Y...")
    Y_all = compute_attention_outputs(keys, values, Q_per_layer, K_conv_per_layer)

    metadata = {
        "ratio": args.ratio if args.ratio is not None else 1.0,
        "T_cart": T_cart,
        "T_full": T_full,
        "n_conv_tokens": n_conv_tokens,
    }

    print("Saving results...")
    values_to_save = [v.float() for v in values]
    save_result(condition_name, values_to_save, Y_all, metadata, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
