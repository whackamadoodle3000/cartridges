"""
Self-study synthesis for rank analysis experiment.

Generates 8192 synthetic Q&A conversations about the document using Qwen3-4B
served via Tokasaurus (Modal or local). A single set of conversations is used
for everything: Cartridges training, SnapKV token selection, and rank measurement.

Usage:
    export TOKASAURUS_URL="https://your-modal-deployment.modal.run"
    python synthesize_config.py
"""
import os

import pydrantic

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer


CARTRIDGES_DIR = os.environ["CARTRIDGES_DIR"]
OUTPUT_DIR = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")
TOKASAURUS_URL = os.environ.get("TOKASAURUS_URL", "http://localhost:8080")

DOC_PATH = os.path.join(CARTRIDGES_DIR, "qasper_e_line_203_context.txt")


client = TokasaurusClient.Config(
    model_name="Qwen/Qwen3-4b",
    url=TOKASAURUS_URL,
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        tools=[],
        resources=[
            TextFileResource.Config(
                path=DOC_PATH,
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                chunker=TokenChunker.Config(
                    tokenizer=client.model_name,
                    min_tokens_per_chunk=512,
                    max_tokens_per_chunk=1024,
                ),
            ),
        ],
    ),
    num_samples=8192,
    batch_size=1,
    max_num_batches_in_parallel=80,
    name="rank_analysis_synth",
    output_dir=OUTPUT_DIR,
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
)


if __name__ == "__main__":
    pydrantic.main([config])
