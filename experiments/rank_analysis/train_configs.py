"""
Training configs for Cartridges rank analysis experiment.

Trains cartridges at various compression ratios on a given document
using self-study synthesized data. Run with:

    python train_configs.py --ratio 0.5

Or train all ratios sequentially:

    for r in 1.0 0.1 0.05 0.02; do python train_configs.py --ratio $r; done
"""
import argparse
import os

import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset


CARTRIDGES_DIR = os.environ["CARTRIDGES_DIR"]
OUTPUT_DIR = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")

DEFAULT_DOC = os.path.join(CARTRIDGES_DIR, "qasper_e_line_203_context.txt")
DEFAULT_PARQUET = os.path.join(OUTPUT_DIR, "rank_analysis_synth/artifact/dataset.parquet")


def make_train_config(
    ratio: float,
    doc_path: str = DEFAULT_DOC,
    parquet: str = DEFAULT_PARQUET,
) -> TrainConfig:
    max_tokens = None if ratio >= 1.0 else int(ratio * 30_000)

    if ratio >= 1.0:
        lr = 1e-3
    elif ratio >= 0.1:
        lr = 5e-3
    else:
        lr = 2e-2

    return TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="Qwen/Qwen3-4b",
            model_cls=FlexQwen3ForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(
            text_source=doc_path,
            max_tokens=max_tokens,
        ),

        lr=lr,
        epochs=1,
        global_batch_size=32,

        dataset=TrainDataset.Config(
            data_sources=[
                DataSource(path=parquet, type="local"),
            ],
            top_k_logits=20,
            packed_seq_length=1024,
            packing_mode="truncate",
        ),

        loss_eval_every_n_steps=16,
        loss_evals=[
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=parquet, type="local"),
                    packed_seq_length=1024,
                ),
                name_for_wandb="rank_analysis_loss",
            ),
        ],

        distributed_backend="gloo",
        save_every_n_steps=9999,
        save_after_training=True,
        name=f"rank_analysis_ratio_{ratio}",
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True,
                        help="Compression ratio (1.0 = full, 0.5 = 50%%, etc.)")
    parser.add_argument("--doc", type=str, default=DEFAULT_DOC)
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    args = parser.parse_args()

    # Clear sys.argv so pydrantic.main doesn't try to re-parse our flags
    sys.argv = sys.argv[:1]

    config = make_train_config(
        ratio=args.ratio,
        doc_path=args.doc,
        parquet=args.parquet,
    )
    pydrantic.main(config)
