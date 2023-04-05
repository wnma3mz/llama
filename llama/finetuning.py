# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List, Optional

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
import torch.nn as nn
from llama.prompt_tuning import PromptEmbedding
from dataclasses import dataclass, field

from .utils import _extend_attention_mask


@dataclass
class ModelOutput:
    loss: field(default=None)
    logits: field(default=None)


class LLaMAFT(nn.Module):
    def __init__(self, decoder: Transformer, prompt_encoder: PromptEmbedding):
        super().__init__()

        self.decoder = decoder
        self.prompt_encoder = prompt_encoder
        self.peft_config = self.prompt_encoder.config

        self.num_tokens = self.peft_config.num_virtual_tokens
        self.prompt_tokens = torch.arange(
            self.num_tokens * self.peft_config.num_transformer_submodules
        ).long()

    def get_prompt(self, bsz: int, local_rank):
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(bsz, -1).to(local_rank)
        prompts = self.prompt_encoder(prompt_tokens)
        return prompts

    def forward(
        self,
        local_rank,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        if input_ids is not None:
            input_ids = input_ids.to(local_rank)
            input_embeds = self.decoder.tok_embeddings(input_ids)
        elif input_embeds is not None:
            input_embeds = input_embeds.to(local_rank)
        else:
            raise SyntaxError("Please input input_ids or input_embeds.")

        bsz, seqlen_o, dim = input_embeds.shape

        # Concat Prompt and Embedding
        prompts = self.get_prompt(bsz, local_rank)
        prompts = prompts.to(input_embeds.dtype).to(local_rank)
        input_embeds = torch.cat((prompts, input_embeds), dim=1)

        bsz, seqlen, dim = input_embeds.shape

        # Build Attention Mask
        attention_mask = None
        if attention_mask is None:
            attention_mask = torch.ones((bsz, seqlen), dtype=torch.bool)
        else:
            attention_mask = _extend_attention_mask(attention_mask, self.num_tokens)

        logits = self.decoder(input_embeds=input_embeds, attention_mask=attention_mask)

        if labels is not None:
            # Concat Prompt and Labels
            prefix_labels = torch.full((bsz, self.num_tokens), -100)
            labels = torch.cat((prefix_labels, labels), dim=1)

            # Calculate Loss
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.decoder.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fn(shift_logits, shift_labels)
            print(loss)
            return ModelOutput(loss=loss, logits=logits)
        return ModelOutput(loss=None, logits=logits)