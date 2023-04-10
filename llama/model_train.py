# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from .utils import (
    apply_rotary_pos_emb,
    precompute_cos_sin,
    _make_causal_mask,
    _expand_mask,
)


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos,
        sin,
        mask: Optional[torch.Tensor],
        start_pos: int = 0,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        keys = xk[:, start_pos : start_pos + seqlen]
        values = xv[:, start_pos : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos,
        sin,
        mask: Optional[torch.Tensor],
        start_pos: int = 0,
    ):
        h = x + self.attention(self.attention_norm(x), cos, sin, mask, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        # dim = 4096, n_heads = 32, max_seq_len = 512
        self.cos_cached, self.sin_cached = precompute_cos_sin(
            self.params.max_seq_len,
            self.params.dim // self.params.n_heads,
            device=self.tok_embeddings.weight.device,
        )

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, start_pos: int, ft_prompts=None):
        # TODO: Now Generate Bad Text
        _bsz, seqlen_o = tokens.shape
        h = self.tok_embeddings(tokens)
        if ft_prompts is not None:
            ft_prompts = ft_prompts.to(h.dtype).to(h.device)
            h = torch.cat((ft_prompts, h), dim=1)
        _bsz, seqlen, dim = h.shape

        mask = None
        if seqlen_o > 1:
            mask = torch.ones((_bsz, seqlen), dtype=torch.bool, device=h.device)
            past_key_values_length = 0
            combined_attention_mask = _make_causal_mask(
                (_bsz, seqlen),
                h.dtype,
                device=h.device,
                past_key_values_length=past_key_values_length,
            )

            mask = _expand_mask(mask, h.dtype, seqlen) + combined_attention_mask

        cos = self.cos_cached[:, start_pos : start_pos + seqlen].to(h.dtype)
        sin = self.sin_cached[:, start_pos : start_pos + seqlen].to(h.dtype)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,  # In Train
    ):
        if input_embeds is None:
            _bsz, seqlen = input_ids.shape
            h = self.tok_embeddings(input_ids)
        else:
            # For Prompt Tuning
            _bsz, seqlen, dim = input_embeds.shape
            h = input_embeds

        if attention_mask is None:
            mask = None
        else:
            attention_mask = attention_mask.to(h.device)
            past_key_values_length = 0  # TODO
            combined_attention_mask = _make_causal_mask(
                (_bsz, seqlen),
                h.dtype,
                device=h.device,
                past_key_values_length=past_key_values_length,
            )
            mask = (
                _expand_mask(attention_mask, h.dtype, seqlen) + combined_attention_mask
            )

        cos = self.cos_cached[:, :seqlen].to(h.dtype)
        sin = self.sin_cached[:, :seqlen].to(h.dtype)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        h = self.norm(h)

        logits = self.output(h).float()
        return logits
