"""Rotary Position Embeddings (RoPE) from scratch in NumPy.

Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).

For even dimension d and base theta (default 10000), each pair of dims (2i, 2i+1) is
rotated by angle m * theta_i where theta_i = base^(-2i/d) and m is the absolute position.

Key property: the attention score <RoPE(q, m), RoPE(k, n)> depends only on the
relative offset (m - n) (and the content vectors q, k), not on absolute positions.
"""
from __future__ import annotations

import numpy as np


def rope_freqs(dim: int, base: float = 10000.0) -> np.ndarray:
    """Inverse frequencies theta_i = base^(-2i/d) for i in 0 .. dim/2 - 1.

    Parameters
    ----------
    dim : even embedding dimension
    base : RoPE base (default 10000, same as Transformer PE)

    Returns
    -------
    freqs : (dim/2,) float64
    """
    if dim < 2 or dim % 2 != 0:
        raise ValueError(f"dim must be even and >= 2, got {dim}")
    if base <= 0:
        raise ValueError("base must be positive")
    i = np.arange(0, dim, 2, dtype=np.float64)
    return base ** (-i / dim)


def rope_cos_sin(
    seq_len: int, dim: int, base: float = 10000.0, offset: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build cos/sin tables for positions [offset, offset+seq_len).

    Returns
    -------
    cos, sin : each (seq_len, dim) -- values duplicated across each pair so
               elementwise multiply with x works after interleaving.
    """
    freqs = rope_freqs(dim, base)  # (d/2,)
    positions = np.arange(offset, offset + seq_len, dtype=np.float64)[:, None]  # (T, 1)
    angles = positions * freqs[None, :]  # (T, d/2)
    # Interleave: [th0, th0, th1, th1, ...] so shape is (T, d)
    cos = np.repeat(np.cos(angles), 2, axis=-1)
    sin = np.repeat(np.sin(angles), 2, axis=-1)
    return cos, sin


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Map (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...).

    Used in the compact RoPE form: RoPE(x) = x * cos + rotate_half(x) * sin.
    """
    x = np.asarray(x, dtype=np.float64)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = -x_odd
    out[..., 1::2] = x_even
    return out


def apply_rope(
    x: np.ndarray,
    cos: np.ndarray | None = None,
    sin: np.ndarray | None = None,
    *,
    positions: np.ndarray | None = None,
    base: float = 10000.0,
) -> np.ndarray:
    """Apply RoPE to the last dimension of x.

    Parameters
    ----------
    x : (..., T, D) with D even
    cos, sin : optional (T, D) tables from rope_cos_sin; if omitted, built from
               ``positions`` (shape (T,)) or arange(T).
    positions : optional absolute positions for each of the T tokens
    base : RoPE base when building tables

    Returns
    -------
    x_rot : same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {x.shape[-1]}")
    T, D = x.shape[-2], x.shape[-1]

    if cos is None or sin is None:
        if positions is None:
            cos, sin = rope_cos_sin(T, D, base=base)
        else:
            positions = np.asarray(positions, dtype=np.float64).reshape(-1)
            if positions.shape[0] != T:
                raise ValueError(f"positions length {positions.shape[0]} != T={T}")
            freqs = rope_freqs(D, base)
            angles = positions[:, None] * freqs[None, :]
            cos = np.repeat(np.cos(angles), 2, axis=-1)
            sin = np.repeat(np.sin(angles), 2, axis=-1)

    # Broadcast cos/sin over leading dims
    while cos.ndim < x.ndim:
        cos = cos[None, ...]
        sin = sin[None, ...]
    return x * cos + rotate_half(x) * sin


def apply_rope_pair(
    x: np.ndarray, position: int | float, base: float = 10000.0
) -> np.ndarray:
    """Apply RoPE to a single vector (D,) or batch (..., D) at one position."""
    x = np.asarray(x, dtype=np.float64)
    D = x.shape[-1]
    freqs = rope_freqs(D, base)
    angle = float(position) * freqs  # (d/2,)
    cos = np.repeat(np.cos(angle), 2)
    sin = np.repeat(np.sin(angle), 2)
    return x * cos + rotate_half(x) * sin


def rope_dot(q: np.ndarray, k: np.ndarray, m: int, n: int, base: float = 10000.0) -> float:
    """Scalar <RoPE(q, m), RoPE(k, n)> -- used to demo relative dependence."""
    q_r = apply_rope_pair(q, m, base)
    k_r = apply_rope_pair(k, n, base)
    return float(np.dot(q_r, k_r))


def relative_rope_kernel(
    q: np.ndarray, k: np.ndarray, offset: int, base: float = 10000.0
) -> float:
    """Equivalent relative form: rotate q by +offset relative to k at 0.

    For RoPE, <RoPE(q,m), RoPE(k,n)> = <RoPE(q, m-n), k> when k is unrotated
    (or equivalently rotate q by (m-n) and leave k fixed).
    """
    q_rel = apply_rope_pair(q, offset, base)
    return float(np.dot(q_rel, k))


def check_relative_property(
    dim: int = 32,
    base: float = 10000.0,
    n_trials: int = 32,
    seed: int = 42,
    atol: float = 1e-9,
) -> dict:
    """Verify <RoPE(q,m), RoPE(k,n)> == <RoPE(q, m-n), k> and shift invariance of dots."""
    rng = np.random.default_rng(seed)
    max_abs_err = 0.0
    max_shift_err = 0.0
    for _ in range(n_trials):
        q = rng.normal(size=dim)
        k = rng.normal(size=dim)
        m = int(rng.integers(0, 64))
        n = int(rng.integers(0, 64))
        abs_dot = rope_dot(q, k, m, n, base)
        rel_dot = relative_rope_kernel(q, k, m - n, base)
        max_abs_err = max(max_abs_err, abs(abs_dot - rel_dot))

        # Shift both positions by delta -- absolute dots should be unchanged
        delta = int(rng.integers(1, 20))
        shifted = rope_dot(q, k, m + delta, n + delta, base)
        max_shift_err = max(max_shift_err, abs(abs_dot - shifted))

    return {
        "n_trials": n_trials,
        "dim": dim,
        "base": base,
        "max_abs_vs_rel_err": float(max_abs_err),
        "max_shift_invariance_err": float(max_shift_err),
        "pass": bool(max_abs_err < atol and max_shift_err < atol),
        "atol": atol,
    }


def sinusoidal_pe(seq_len: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Absolute sinusoidal PE (Vaswani et al.) for comparison -- shape (T, D)."""
    pe = np.zeros((seq_len, dim), dtype=np.float64)
    if seq_len == 0:
        return pe
    pos = np.arange(seq_len, dtype=np.float64)[:, None]
    i = np.arange(0, dim, 2, dtype=np.float64)
    div = base ** (i / dim)
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div[: pe[:, 1::2].shape[1]])
    return pe
