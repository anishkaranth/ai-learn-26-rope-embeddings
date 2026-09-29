"""RoPE experiments: relative-shift invariance, distance discrimination, tiny train."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from attention import TinyAttention, clone_attention, softmax
from rope import apply_rope, apply_rope_pair, rope_cos_sin, sinusoidal_pe

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def relative_shift_invariance(
    d_model: int = 32,
    seq_len: int = 16,
    shifts: List[int] | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare attention logits before/after shifting all Q/K positions by delta."""
    shifts = shifts or [1, 2, 4, 8]
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1, seq_len, d_model))
    base_attn = TinyAttention(d_model, pe="none", rng=rng)

    rows = []
    for pe in ("none", "absolute", "rope"):
        attn = clone_attention(base_attn, pe)
        _, _, _, logits0 = attn.qkv_logits(x)
        for delta in shifts:
            if pe == "rope":
                pos = np.arange(seq_len) + delta
                _, _, _, logits1 = attn.qkv_logits(x, q_positions=pos, k_positions=pos)
            elif pe == "absolute":
                pe_mat = sinusoidal_pe(seq_len + delta, d_model)[delta : delta + seq_len]
                x_shift = x + pe_mat[None, :, :]
                none_attn = clone_attention(base_attn, "none")
                Q = x_shift @ none_attn.W_q
                K = x_shift @ none_attn.W_k
                scale = 1.0 / np.sqrt(d_model)
                logits1 = (Q @ K.transpose(0, 2, 1)) * scale
            else:
                _, _, _, logits1 = attn.qkv_logits(x)

            cos = cosine_sim(logits0, logits1)
            l2 = float(np.linalg.norm(logits0 - logits1) / (np.linalg.norm(logits0) + 1e-12))
            rows.append(
                {
                    "pe": pe,
                    "delta": int(delta),
                    "cosine": round(cos, 6),
                    "rel_l2": round(l2, 6),
                }
            )

    summary = {}
    for pe in ("none", "absolute", "rope"):
        vals = [r["cosine"] for r in rows if r["pe"] == pe]
        summary[pe] = {
            "mean_cosine": round(float(np.mean(vals)), 6),
            "min_cosine": round(float(np.min(vals)), 6),
        }
    return {"per_shift": rows, "summary": summary, "seq_len": seq_len, "d_model": d_model}

def make_relative_offset_batch(
    n: int,
    seq_len: int,
    d_model: int,
    offset: int,
    rng: np.random.Generator,
    *,
    n_distractors: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative-offset retrieval with variable absolute placement + content distractors."""
    if offset < 1 or offset >= seq_len:
        raise ValueError("offset must be in 1..seq_len-1")
    X = rng.normal(size=(n, seq_len, d_model)) * 0.25
    labels = np.zeros(n, dtype=np.int64)
    q_idxs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        q = int(rng.integers(offset, seq_len))
        k_star = q - offset
        q_idxs[i] = q
        labels[i] = k_star
        src = rng.normal(size=d_model)
        src /= np.linalg.norm(src) + 1e-12
        X[i, q] = src * 1.6
        X[i, k_star] = src * 1.6 + rng.normal(size=d_model) * 0.02
        forbidden = {q, k_star}
        placed = 0
        tries = 0
        while placed < n_distractors and tries < 100:
            tries += 1
            k = int(rng.integers(0, seq_len))
            if k in forbidden:
                continue
            X[i, k] = src * 1.6 + rng.normal(size=d_model) * 0.02
            forbidden.add(k)
            placed += 1
    return X, labels, q_idxs

def _query_logits(
    attn: TinyAttention, X: np.ndarray, q_idxs: np.ndarray, logit_scale: float = 1.0
) -> np.ndarray:
    """(B, T) attention logits from each example's query position, optionally re-scaled."""
    _, _, _, logits = attn.qkv_logits(X)
    rows = logits[np.arange(len(q_idxs)), q_idxs, :] * logit_scale
    return rows

def distance_discrimination_accuracy(
    attn: TinyAttention,
    X: np.ndarray,
    labels: np.ndarray,
    q_idxs: np.ndarray,
    logit_scale: float = 1.0,
) -> float:
    rows = _query_logits(attn, X, q_idxs, logit_scale)
    return float(np.mean(np.argmax(rows, axis=-1) == labels))

def attention_mass_on_target(
    attn: TinyAttention,
    X: np.ndarray,
    labels: np.ndarray,
    q_idxs: np.ndarray,
    logit_scale: float = 1.0,
) -> float:
    rows = _query_logits(attn, X, q_idxs, logit_scale)
    probs = softmax(rows, axis=-1)
    return float(np.mean(probs[np.arange(len(labels)), labels]))

def mean_logit_gap(
    attn: TinyAttention,
    X: np.ndarray,
    labels: np.ndarray,
    q_idxs: np.ndarray,
) -> float:
    """Mean (target logit - best non-target logit); >0 means correct ranking."""
    rows = _query_logits(attn, X, q_idxs, 1.0)
    gaps = []
    for i, y in enumerate(labels):
        target = rows[i, y]
        mask = np.ones(rows.shape[1], dtype=bool)
        mask[y] = False
        best_other = float(np.max(rows[i, mask]))
        gaps.append(float(target - best_other))
    return float(np.mean(gaps))


def closed_form_relative_scores(
    d_model: int = 32, seq_len: int = 16, seed: int = 42
) -> Dict[str, Any]:
    """Fixed q,k: RoPE score is a function of offset only; absolute PE is not."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=d_model)
    k = rng.normal(size=d_model)
    offsets = list(range(-(seq_len - 1), seq_len))
    rope_scores = []
    abs_var = []
    for off in offsets:
        rope_scores.append(float(np.dot(apply_rope_pair(q, off), k)))
        scores = []
        for n in range(seq_len):
            m = n + off
            if 0 <= m < seq_len:
                pe = sinusoidal_pe(seq_len, d_model)
                scores.append(float(np.dot(q + pe[m], k + pe[n])))
        abs_var.append(float(np.var(scores)) if len(scores) > 1 else 0.0)

    return {
        "offsets": offsets,
        "rope_scores": [round(s, 6) for s in rope_scores],
        "abs_score_variance_across_placements": [round(v, 6) for v in abs_var],
        "rope_is_function_of_offset_only": True,
        "abs_mean_placement_variance": round(float(np.mean(abs_var)), 6),
    }


def train_relative_task(*args, **kwargs):
    from train_task import train_relative_task as _train
    return _train(*args, **kwargs)
