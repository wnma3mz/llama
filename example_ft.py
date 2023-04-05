# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import (
    ModelArgs,
    Transformer,
    TransformerBak,
    Tokenizer,
    LLaMA,
    PromptEmbedding,
    PromptTuningConfig,
)
from dataclasses import dataclass, field, asdict

@dataclass
class FTParams:
    # Fine-Tuning Model Params
    num_virtual_tokens: int = field(default=32)
    num_transformer_submodules: int = field(default=1)
    peft_type: str = field(default="PROMPT_TUNING")
    task_type: str = field(default="CAUSAL_LM")

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    # local_rank, world_size = 1, 1
    print(local_rank, world_size)

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tuning_ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    tuning_checkpoints = sorted(Path(tuning_ckpt_dir).glob("*.pth"))
    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # model = TransformerBak(model_args)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)


    kwargs = asdict(FTParams())
    kwargs["token_dim"] = params["dim"]
    peft_config = PromptTuningConfig(**kwargs)
    # word_embeddings is not used
    prompt_encoder = PromptEmbedding(peft_config, word_embeddings=model.tok_embeddings, tokenizer=tokenizer, init_prompt="")
    # prompt_encoder.load_state_dict(torch.load(tuning_checkpoints[local_rank], map_location="cpu"), strict=False)


    generator = LLaMA(model, tokenizer, prompt_encoder)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tuning_ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tuning_ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for prompt, result in zip(prompts, results):
        print(prompt, result.text)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
