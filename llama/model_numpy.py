from typing import Any, Tuple, Optional, List

import torch
import numpy as np


def silu(x):
    return x / (1 + np.exp(-x))


def softmax(x, dim):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / np.sum(e_x, axis=dim, keepdims=True)


def view_as_complex(x):
    return x[..., 0] + 1j * x[..., 1]    

def view_as_real(x):
    bsz, a, b, len_ = x.shape
    x = np.concatenate((np.expand_dims(np.real(x), -1), np.expand_dims(np.imag(x), -1)), axis=-1)
    return x.reshape(bsz, a, b, len_*2)

def apply_rotary_emb(xq: np.ndarray, xk: np.ndarray, freqs_cis: np.ndarray):
    xq_ = view_as_complex(xq.astype(np.float32).reshape(*xq.shape[:-1], -1, 2))
    xk_ = view_as_complex(xk.astype(np.float32).reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = view_as_real(xq_ * freqs_cis)
    xk_out = view_as_real(xk_ * freqs_cis)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(*shape)


class EmbeddingNumpy:
    def __init__(self, weights) -> None:
        self.weights = weights

    def __call__(self, x):
        return self.weights[x]


class LinearNumpy:
    def __init__(self, weights, bias=None) -> None:
        self.weights, self.bias = weights, bias

    def __call__(self, x) -> Any:
        return (
            np.dot(x, self.weights.T) + self.bias
            if self.bias
            else np.dot(x, self.weights.T)
        )


class RMSNormNumpy:
    def __init__(self, weights, eps: float = 1e-6):
        self.eps = eps
        self.weight = weights

    def _norm(self, x):
        return x * np.reciprocal(
            np.sqrt(np.square(x).mean(-1, keepdims=True) + self.eps)
        )

    def __call__(self, x):
        return self._norm(x).astype(np.float32) * self.weight

def contiguous(arr):
    if isinstance(arr, np.ndarray):
        return arr.copy()
    else:
        return np.ascontiguousarray(arr)
    
class AttentionNumpy:
    def __init__(self, attention: torch.nn.Module):
        self.wq = LinearNumpy(attention.wq.weight.detach().numpy())
        self.wk = LinearNumpy(attention.wk.weight.detach().numpy())
        self.wv = LinearNumpy(attention.wv.weight.detach().numpy())
        self.wo = LinearNumpy(attention.wo.weight.detach().numpy())

        max_batch_size, max_seq_len = 1, 1024
        self.cache_k = np.zeros((max_batch_size, max_seq_len, attention.n_local_heads, attention.head_dim))
        self.cache_v = np.zeros((max_batch_size, max_seq_len, attention.n_local_heads, attention.head_dim))
        self.n_local_heads = attention.n_local_heads
        self.head_dim = attention.head_dim

    def __call__(
        self,
        x: np.ndarray,
        start_pos: int,
        freqs_cis: np.ndarray,
        mask: Optional[np.ndarray],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape((bsz, seqlen, self.n_local_heads, self.head_dim))
        xk = xk.reshape((bsz, seqlen, self.n_local_heads, self.head_dim))
        xv = xv.reshape((bsz, seqlen, self.n_local_heads, self.head_dim))

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose((0, 2, 1, 3))
        keys = keys.transpose((0, 2, 1, 3))
        values = values.transpose((0, 2, 1, 3))

        scores = np.matmul(xq, keys.transpose((0, 1, 3, 2))) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = softmax(scores.astype(np.float32), dim=-1).astype(np.float32)
        output = np.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        # output = contiguous(output.transpose((0, 2, 1, 3))).reshape(bsz, seqlen, -1)
        output = output.transpose((0, 2, 1, 3)).reshape(bsz, seqlen, -1)

        return self.wo(output)


class FeedForwardNumpy:
    def __init__(self, mlp: torch.nn.Module):
        self.w1 = LinearNumpy(mlp.w1.weight.detach().numpy())
        self.w2 = LinearNumpy(mlp.w2.weight.detach().numpy())
        self.w3 = LinearNumpy(mlp.w3.weight.detach().numpy())

    def __call__(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class TransformerBlockNumpy:
    def __init__(
        self,
        attention: AttentionNumpy,
        feed_forward: FeedForwardNumpy,
        attention_norm: RMSNormNumpy,
        ffn_norm: RMSNormNumpy,
    ):
        self.attention = attention
        self.feed_forward = feed_forward
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm

    def __call__(
        self,
        x: np.ndarray,
        start_pos: int,
        freqs_cis: np.ndarray,
        mask: Optional[np.ndarray],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TransformerNumpy:
    def __init__(
        self,
        tok_embeddings: EmbeddingNumpy,
        layers: List[TransformerBlockNumpy],
        norm: RMSNormNumpy,
        output: LinearNumpy,
        freqs_cis: np.ndarray,
    ) -> None:
        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.freqs_cis = freqs_cis

    def __call__(self, tokens: np.ndarray, start_pos: int):
        h = self.tok_embeddings(tokens)

        _bsz, seqlen = tokens.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        print(freqs_cis.shape)
        mask = None
        if seqlen > 1:
            mask = np.full((1, 1, seqlen, seqlen), float("-inf"))
            mask = np.triu(mask.squeeze(), k=start_pos + 1)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output
