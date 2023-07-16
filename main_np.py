# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Any, Tuple, Optional
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from llama import (
    TransformerTorch,
    TransformerBlockNumpy,
    TransformerNumpy,
    AttentionNumpy,
    FeedForwardNumpy,
    RMSNormNumpy,
    LinearNumpy,
    ModelArgs,
    EmbeddingNumpy,
    Tokenizer,
    LLaMA,
)
import numpy as np
from dataclasses import dataclass, field, asdict


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def decode(tokenizer, tokens, max_gen_len):
    decoded, token = [], []
    for i, t in enumerate(tokens.tolist()):
        # cut to max gen len
        t = t[: len(tokens[i]) + max_gen_len]
        # just decode generate

        # cut to eos tok if any
        try:
            t = t[: t.index(tokenizer.eos_id)]
        except ValueError:
            pass
        # cut to -1 if any
        try:
            t = t[: t.index(-1)]
        except ValueError:
            pass

        decoded.append(tokenizer.decode(t))
        token.append(t)
    return decoded, token


def generate(tokenizer, model, x, max_gen_len=256, temperature=0.8, top_p=0.95):
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False)]
    tokens = torch.full((1, max_gen_len), tokenizer.pad_id).long()
    # For Prompt to Set Inputs Ids
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()
    input_text_mask = tokens != tokenizer.pad_id
    start_pos = min(len(x) for x in prompt_tokens)
    prev_pos = 0

    for cur_pos in range(start_pos, max_gen_len):
        logits = model(tokens[:, prev_pos:cur_pos], prev_pos)

        # Get The Next Token
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            # probs = torch.where(torch.isnan(probs), torch.full_like(probs, 1e-10), probs) # Replace Nan
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        # Update. only replace token if prompt has already been generated
        # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        # 如果当前token是空值，则替换为生成结果
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
    return tokens
    # decoded, token = decode(tokens, prompt_tokens, max_gen_len)
    # return decoded


if __name__ == "__main__":
    max_seq_len = 1024
    max_batch_size = 1
    params = {
        "n_layers": 1,
        # "dim": 4096,
        # "n_heads": 32
    }
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer_path = "/mnt/c/Users/lu/Downloads/llama_cpu/models/tokenizer.model"
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = TransformerTorch(model_args)

    # output_np = LinearNumpy(model.output.weight.detach().numpy())
    freqs_cis_np = model.freqs_cis.numpy()
    # torch.save(model.state_dict(), f"b{model_args.n_layers}.pkl")
    # tokens = generate(tokenizer, model, "Hello?")
    layers = []
    for i in range(model_args.n_layers):
        layers.append(TransformerBlockNumpy(
            AttentionNumpy(model.layers[i].attention),
            FeedForwardNumpy(model.layers[i].feed_forward),
            RMSNormNumpy(
                model.layers[i].attention_norm.weight.detach().numpy(),
                model.params.norm_eps,
            ),
            RMSNormNumpy(
                model.layers[i].ffn_norm.weight.detach().numpy(), model.params.norm_eps
            ),
        ))

    model_numpy = TransformerNumpy(
        EmbeddingNumpy(model.tok_embeddings.weight.detach().numpy()),
        layers,
        RMSNormNumpy(model.norm.weight.detach().numpy(), model.params.norm_eps),
        LinearNumpy(model.output.weight.detach().numpy()),
        freqs_cis_np,
    )

    # Torch计算
    tokens_lst = tokenizer.encode("Hello?", bos=True, eos=False)
    tokens = torch.tensor(tokens_lst).unsqueeze(0)
    tokens_arr = np.array(tokens)
    print(tokens_arr)

    prev_pos, cur_pos = 0, len(tokens)
    with torch.no_grad():
        output = model(tokens[:, prev_pos:cur_pos], prev_pos)

    # Numpy计算
    output_calc = model_numpy(tokens_arr[:, prev_pos:cur_pos], prev_pos)
    print(output)
    print("="*20)
    print(output_calc) # 存在浮点误差，在Attention计算qkv时
