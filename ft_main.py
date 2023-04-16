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
import math
from datasets import (
    DataCollatorForSupervisedDataset,
    SupervisedTokenDataset,
    SupervisedDataset,
)
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from torch.utils.data import DataLoader

from llama import (
    ModelArgs,
    TransformerTrain,
    Tokenizer,
    LLaMAFT,
    PromptEmbedding,
    PromptTuningConfig,
)
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import numpy as np


@dataclass
class Params:
    # Fine-Tuning Params
    lr: float = field(default=3e-4)
    num_epochs: int = field(default=3)
    batch_size: int = field(default=2)

    # File Params
    ckpt_dir: str = field(default="./ckpts/7B_fs4")
    tuning_ckpt_dir: str = field(default="./ckpts/7B_ft4")
    dataset_fname: str = field(default="./datasets/alpaca_data")  # Just Test
    tokenizer_path: str = field(default="./ckpts/tokenizer.model")

    init_prompt: str = field(
        default="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    )

    # Model Params
    max_seq_len: int = field(default=512)
    max_batch_size: int = field(default=1)


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
    params,
) -> Tuple[LLaMAFT, Tokenizer]:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = checkpoints[local_rank]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    vir_tokens = FTParams.num_virtual_tokens * FTParams.num_transformer_submodules

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len + vir_tokens, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = TransformerTrain(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    kwargs = asdict(FTParams())
    kwargs["token_dim"] = params["dim"]
    if Params.init_prompt:
        init_token_ids = tokenizer.encode(Params.init_prompt, bos=True, eos=False)
        num_text_tokens = len(init_token_ids)
        if num_text_tokens > vir_tokens:
            init_token_ids = init_token_ids[:vir_tokens]
        elif num_text_tokens < vir_tokens:
            num_reps = math.ceil(vir_tokens / num_text_tokens)
            init_token_ids = init_token_ids * num_reps
        init_token_ids = init_token_ids[:vir_tokens]
        word_embedding_weights = model.tok_embeddings(
            torch.LongTensor(init_token_ids).to(local_rank)
        )
    else:
        word_embedding_weights = None

    peft_config = PromptTuningConfig(**kwargs)
    # Init From Model Word Embedding
    prompt_encoder = PromptEmbedding(peft_config, word_embedding_weights)
    model_ft = LLaMAFT(model, prompt_encoder)
    print(f"Load Model Cost Time: {time.time() - start_time:.2f} s")
    return model_ft, tokenizer


def load_dataloader(fname: str, tokenizer: Tokenizer):
    s1 = time.time()

    if os.path.isfile(fname + ".pkl"):
        # Read from cache file
        train_dataset = SupervisedTokenDataset(fname + ".pkl")
    else:
        # First Load to pickle the Dataset
        train_dataset = SupervisedDataset(fname + ".json", tokenizer)
        import pickle

        with open(fname.split(".json")[0] + "_token.pkl", "wb") as f:
            pickle.dump(
                {"input_ids": train_dataset.input_ids, "labels": train_dataset.labels},
                f,
            )
    print(f"Load Dataset Cost Time: {time.time() - s1:.2f} s")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSupervisedDataset(),
        batch_size=Params.batch_size,
        pin_memory=False,
    )
    return train_dataloader


def train_func(model_ft, optimizer, train_dataloader, local_rank):
    model_ft.to(local_rank)
    model_ft.train()
    for ep in range(Params.num_epochs):
        loss_lst = []
        with tqdm(train_dataloader, ncols=80, postfix="loss: *.****") as t:
            for batch in t:
                output = model_ft(local_rank, **batch)
                loss = output.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_lst.append(loss.data.item())
                t.postfix = "loss: {:.4f}".format(np.mean(loss_lst))
                # t.postfix = "loss: {:.4f}".format(loss_lst[-1])
        # Just Save PromptEncoder
        train_epoch_loss = np.sum(loss_lst) / len(train_dataloader)
        train_ppl = np.exp(train_epoch_loss)
        print(f"{ep}: {train_ppl} {train_epoch_loss}")
        if local_rank == 0:
            torch.save(
                model_ft.prompt_encoder.state_dict(),
                os.path.join(Params.tuning_ckpt_dir, f"prefix.{ep}.pth"),
            )
            np.save(f"loss.{ep}.npy", loss_lst)
            # np.load(f'loss.{ep}.npy')


def main():
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    with open(Path(Params.ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_ft, tokenizer = load_model(
        Params.ckpt_dir,
        Params.tokenizer_path,
        local_rank,
        world_size,
        Params.max_seq_len,
        Params.max_batch_size,
        params,
    )

    train_dataloader = load_dataloader(Params.dataset_fname, tokenizer)

    for name, params in model_ft.decoder.named_parameters():
        params.requires_grad = False

    base_opt = torch.optim.AdamW
    optimizer = base_opt(
        filter(lambda p: p.requires_grad, model_ft.parameters()), lr=Params.lr
    )
    train_func(model_ft, optimizer, train_dataloader, local_rank)


if __name__ == "__main__":
    if not os.path.isdir(Params.tuning_ckpt_dir):
        os.mkdir(Params.tuning_ckpt_dir)
    main()
