# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List, Optional

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from dataclasses import dataclass
import torch.nn as nn
from llama.prompt_tuning import PromptEmbedding


def banned_token(logits):
    # for token_id in [2, 518, 29961]:
    # print(logits.shape)
    # for token_id in [1649, 7652, 14365, 22359, 27097]:
    #     logits[:, token_id] = 0
    return logits


@dataclass
class Output:
    text: str
    token: List[int]


class LLaMA:
    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        prompt_encoder: Optional[PromptEmbedding] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_encoder = prompt_encoder
        if self.prompt_encoder:
            self.peft_config = self.prompt_encoder.config
            self.prompt_tokens = torch.arange(
                self.peft_config.num_virtual_tokens
                * self.peft_config.num_transformer_submodules
            ).long()

    def encode(self, prompts):
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        return prompt_tokens

    def decode(self, tokens, prompt_tokens, max_gen_len):
        decoded, token = [], []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            # t = t[: len(prompt_tokens[i]) + max_gen_len]

            # just decode generate
            t = t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len]

            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            # cut to -1 if any
            try:
                t = t[: t.index(-1)]
            except ValueError:
                pass

            decoded.append(self.tokenizer.decode(t))
            token.append(t)
        return decoded, token

    def get_prompt(self, bsz: int):
        prompts = self.prompt_encoder.embedding.weight.repeat(bsz, 1, 1)
        return prompts
    
    def generate(
        self,
        prompts: List[str] = None,
        prompt_tokens: List[List[int]] = None,
        max_gen_len: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[Output]:
        params = self.model.params
        bsz = len(prompts)
        if prompts:
            assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
            prompt_tokens = self.encode(prompts)
        elif prompt_tokens is None:
            raise Exception("Please input prompts or prompt_tokens.")

        ft_prompts = None
        if self.prompt_encoder:
            ft_prompts = self.get_prompt(bsz)

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0

        last_token = -torch.ones(bsz)
        for cur_pos in range(start_pos, total_len):
            logits = self.model.generate(tokens[:, prev_pos:cur_pos], prev_pos, ft_prompts)

            logits = banned_token(logits)  # New Add

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            # Break Token Id
            if next_token[0] == 13 and last_token[0] == 13:
                break
            last_token = next_token

        decoded, token = self.decode(tokens, prompt_tokens, max_gen_len)
        return [
            Output(text=text, token=token_id) for text, token_id in zip(decoded, token)
        ]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
