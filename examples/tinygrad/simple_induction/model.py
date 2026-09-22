"""
Minimal attention-only transformer for mechanistic interpretability demos.

1 attention layer, no MLP, no LayerNorm. Every forward pass fills
`model.acts` with named activation Tensors:

    acts["embed"]        (B, T, d_model)
    acts["q"|"k"|"v"]    (B, n_heads, T, d_head)
    acts["attn_scores"]  (B, n_heads, T, T)   pre-softmax, masked
    acts["attn_pattern"] (B, n_heads, T, T)   post-softmax, in [0,1]
    acts["attn_out"]     (B, T, d_model)
    acts["resid_post"]   (B, T, d_model)
    acts["logits"]       (B, T, vocab)
"""

import math

from tinygrad import Tensor, nn


class Transformer:
    def __init__(
        self, vocab: int = 64, seq_len: int = 64, d_model: int = 64, n_heads: int = 4
    ):
        assert d_model % n_heads == 0
        self.vocab, self.seq_len = vocab, seq_len
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.d_model = d_model

        self.embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.unembed = nn.Linear(d_model, vocab, bias=False)

        # causal mask: -inf above the diagonal
        self.mask = Tensor.full((seq_len, seq_len), float("-inf")).triu(1)

        self.acts: dict[str, Tensor] = {}

    def parameters(self):
        return nn.state.get_parameters(self)

    def _split(self, x: Tensor) -> Tensor:
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def __call__(self, tokens: Tensor, head_mask=None) -> Tensor:
        B, T = tokens.shape
        a = self.acts
        a.clear()

        x = self.embed(tokens) + self.pos_embed(Tensor.arange(T))
        a["embed"] = x

        q = self._split(self.w_q(x))
        a["q"] = q
        k = self._split(self.w_k(x))
        a["k"] = k
        v = self._split(self.w_v(x))
        a["v"] = v

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores + self.mask[:T, :T]
        a["attn_scores"] = scores

        pattern = scores.softmax(-1)
        a["attn_pattern"] = pattern  # (B, H, T, T) — the money plot

        z = pattern @ v  # (B, H, T, d_head)
        if head_mask is not None:
            z = z * head_mask.reshape(1, self.n_heads, 1, 1)
        attn_out = self.w_o(z.transpose(1, 2).reshape(B, T, self.d_model))
        a["attn_out"] = attn_out

        resid = x + attn_out
        a["resid_post"] = resid

        logits = self.unembed(resid)
        a["logits"] = logits
        return logits
