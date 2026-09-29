"""Tiny single-head NumPy attention with no-PE / absolute sinusoidal / RoPE."""
from __future__ import annotations

from typing import Literal

import numpy as np

from rope import apply_rope, rope_cos_sin, sinusoidal_pe

PEKind = Literal["none", "absolute", "rope"]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class TinyAttention:
    """Single-head attention: X -> Q,K,V linear, scaled-dot-product, out proj.

    Positional encoding modes
    -------------------------
    none     : no position signal
    absolute : add sinusoidal PE to token embeddings before QKV
    rope     : apply RoPE to Q and K after the Q/K projections
    """

    def __init__(
        self,
        d_model: int,
        *,
        pe: PEKind = "none",
        rope_base: float = 10000.0,
        rng: np.random.Generator | None = None,
        scale: float = 0.1,
    ) -> None:
        if d_model < 2 or d_model % 2 != 0:
            raise ValueError("d_model must be even and >= 2")
        self.d_model = d_model
        self.pe = pe
        self.rope_base = rope_base
        rng = rng or np.random.default_rng()
        self.W_q = rng.normal(0, scale, size=(d_model, d_model))
        self.W_k = rng.normal(0, scale, size=(d_model, d_model))
        self.W_v = rng.normal(0, scale, size=(d_model, d_model))
        self.W_o = rng.normal(0, scale, size=(d_model, d_model))
        self.b_o = np.zeros(d_model)

    def parameters(self) -> list[np.ndarray]:
        return [self.W_q, self.W_k, self.W_v, self.W_o, self.b_o]

    def _prepare_x(self, x: np.ndarray) -> np.ndarray:
        """x: (B, T, D) — optionally add absolute PE."""
        if self.pe != "absolute":
            return x
        pe = sinusoidal_pe(x.shape[1], self.d_model, base=self.rope_base)
        return x + pe[None, :, :]

    def qkv_logits(
        self,
        x: np.ndarray,
        *,
        q_positions: np.ndarray | None = None,
        k_positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return Q, K, V and attention logits (B, T, T) before softmax."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x[None, ...]
        x = self._prepare_x(x)
        B, T, D = x.shape
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        if self.pe == "rope":
            if q_positions is None:
                cos_q, sin_q = rope_cos_sin(T, D, base=self.rope_base)
            else:
                cos_q, sin_q = None, None
            if k_positions is None:
                cos_k, sin_k = cos_q, sin_q
            else:
                cos_k, sin_k = None, None
            Q = apply_rope(Q, cos_q, sin_q, positions=q_positions, base=self.rope_base)
            K = apply_rope(K, cos_k, sin_k, positions=k_positions, base=self.rope_base)
        scale = 1.0 / np.sqrt(D)
        logits = (Q @ K.transpose(0, 2, 1)) * scale
        return Q, K, V, logits

    def forward(
        self,
        x: np.ndarray,
        *,
        q_positions: np.ndarray | None = None,
        k_positions: np.ndarray | None = None,
        return_weights: bool = False,
    ):
        Q, K, V, logits = self.qkv_logits(
            x, q_positions=q_positions, k_positions=k_positions
        )
        weights = softmax(logits, axis=-1)
        ctx = weights @ V
        out = ctx @ self.W_o + self.b_o
        if return_weights:
            return out, weights, logits
        return out


def clone_attention(src: TinyAttention, pe: PEKind) -> TinyAttention:
    """Clone weights into a new TinyAttention with a different PE mode."""
    dst = TinyAttention(src.d_model, pe=pe, rope_base=src.rope_base, scale=0.0)
    dst.W_q = src.W_q.copy()
    dst.W_k = src.W_k.copy()
    dst.W_v = src.W_v.copy()
    dst.W_o = src.W_o.copy()
    dst.b_o = src.b_o.copy()
    return dst
