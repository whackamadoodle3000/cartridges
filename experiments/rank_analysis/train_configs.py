"""
Training configs for Cartridges rank analysis experiment.

Trains cartridges at various compression ratios on a given document
using self-study synthesized data. Run with:

    python train_configs.py --ratio 0.5

Or train all ratios sequentially:

    for r in 1.0 0.5 0.2 0.1 0.05 0.02; do python train_configs.py --ratio $r; done
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

DEFAULT_DOC = os.path.join(CARTRIDGES_DIR, "experiments/rank_analysis/data/frankenstein.txt")
DEFAULT_TRAIN_PARQUET = os.path.join(OUTPUT_DIR, "rank_analysis_synth_train/artifact/dataset.parquet")
DEFAULT_EVAL_PARQUET = os.path.join(OUTPUT_DIR, "rank_analysis_synth_eval/artifact/dataset.parquet")


def make_train_config(
    ratio: float,
    doc_path: str = DEFAULT_DOC,
    train_parquet: str = DEFAULT_TRAIN_PARQUET,
    eval_parquet: str = DEFAULT_EVAL_PARQUET,
) -> TrainConfig:
    max_tokens = None if ratio >= 1.0 else int(ratio * 30_000)

    return TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="Qwen/Qwen3-4b",
            model_cls=FlexQwen3ForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(
            text_source=doc_path,
            max_tokens=max_tokens,
        ),

        lr=2e-2,
        epochs=1,
        global_batch_size=32,

        dataset=TrainDataset.Config(
            data_sources=[
                DataSource(path=train_parquet, type="local"),
            ],
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ),

        loss_eval_every_n_steps=16,
        loss_evals=[
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=eval_parquet, type="local"),
                    packed_seq_length=2048,
                ),
                name_for_wandb="rank_analysis_eval",
            ),
        ],

        distributed_backend="gloo",
        save_every_n_steps=9999,
        save_after_training=True,
        name=f"rank_analysis_ratio_{ratio}",
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True,
                        help="Compression ratio (1.0 = full, 0.5 = 50%%, etc.)")
    parser.add_argument("--doc", type=str, default=DEFAULT_DOC)
    parser.add_argument("--train-parquet", type=str, default=DEFAULT_TRAIN_PARQUET)
    parser.add_argument("--eval-parquet", type=str, default=DEFAULT_EVAL_PARQUET)
    args = parser.parse_args()

    config = make_train_config(
        ratio=args.ratio,
        doc_path=args.doc,
        train_parquet=args.train_parquet,
        eval_parquet=args.eval_parquet,
    )
    pydrantic.main(config)
