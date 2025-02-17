# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple, List
import os
import sys
import torch
import time
import json
import torch.nn as nn
from pathlib import Path

from datasets import DataCollatorForSupervisedDataset, SupervisedTokenDataset
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from torch.utils.data import DataLoader

from llama import ModelArgs, TransformerTrain, Tokenizer, LLaMAFT
from dataclasses import dataclass, field, asdict

import warnings

warnings.filterwarnings("ignore")

# CUDA out of memory.


@dataclass
class Params:
    # Fine-Tuning Params
    lr: float = field(default=5e-5)
    num_epochs: int = field(default=1)
    batch_size: int = field(default=1)

    # File Params
    ckpt_dir: str = field(default="./ckpts/7B_fs4")
    tuning_ckpt_dir: str = field(default="./ckpts/7B_fsft4")
    dataset_fname: str = field(default="./datasets/alpaca_data_token.pkl")  # Just Test
    tokenizer_path: str = field(default="./ckpts/tokenizer.model")

    # Model Params
    max_seq_len: int = field(default=512)
    max_batch_size: int = field(default=1)


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    print(f"local_rank={local_rank}, world_size={world_size}")

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
):
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = checkpoints[local_rank]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = TransformerTrain(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    # Fine-Tuning The Total Model
    model_ft = LLaMAFT(model)
    print(f"Load Model Cost Time: {time.time() - start_time:.2f} s")
    return model_ft, tokenizer


def load_dataloader(fname, tokenizer):
    s1 = time.time()
    train_dataset = SupervisedTokenDataset(fname)
    print(f"Load Dataset Cost Time: {time.time() - s1:.2f} s")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
        batch_size=Params.batch_size,
        pin_memory=False,
    )
    return train_dataloader


def train_func(model_ft, optimizer, train_dataloader, local_rank):
    model_ft.train()
    for ep in range(Params.num_epochs):
        for batch in train_dataloader:
            output = model_ft(local_rank, **batch)
            loss = output.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Just Save PromptEncoder
        torch.save(
            model_ft.prompt_encoder.state_dict(),
            os.path.join(
                Params.tuning_ckpt_dir,
                f"ft_consolidated.{str(local_rank).zfill(2)}.pth",
            ),
        )


def main():
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    model_ft, tokenizer = load_model(
        Params.ckpt_dir,
        Params.tokenizer_path,
        local_rank,
        world_size,
        Params.max_seq_len,
        Params.max_batch_size,
    )

    train_dataloader = load_dataloader(Params.dataset_fname, tokenizer)

    # for name, params in model_ft.decoder.named_parameters():
    #     params.requires_grad = False

    base_opt = torch.optim.AdamW
    optimizer = base_opt(
        filter(lambda p: p.requires_grad, model_ft.parameters()), lr=Params.lr
    )
    train_func(model_ft, optimizer, train_dataloader, local_rank)


if __name__ == "__main__":
    if not os.path.isdir(Params.tuning_ckpt_dir):
        os.mkdir(Params.tuning_ckpt_dir)
    main()
