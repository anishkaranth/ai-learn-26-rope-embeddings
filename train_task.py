"""Train loop for relative-offset retrieval."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from attention import TinyAttention, clone_attention, softmax
from experiments import (
    attention_mass_on_target,
    distance_discrimination_accuracy,
    make_relative_offset_batch,
    mean_logit_gap,
    _query_logits,
)
from rope import apply_rope, rope_cos_sin, sinusoidal_pe


def train_relative_task(
    pe: str,
    d_model: int = 32,
    seq_len: int = 16,
    offset: int = 4,
    n_train: int = 640,
    n_test: int = 256,
    steps: int = 180,
    lr: float = 0.08,
    seed: int = 42,
    n_distractors: int = 3,
    logit_scale: float = 8.0,
) -> Dict[str, Any]:
    """Train Q/K so attention at query selects the key at relative -offset."""
    rng = np.random.default_rng(seed + {"none": 0, "absolute": 1, "rope": 2}[pe])
    rng_init = np.random.default_rng(seed)
    base = TinyAttention(d_model, pe="none", rng=rng_init, scale=0.08)
    attn = clone_attention(base, pe)

    Xtr, Ytr, Qtr = make_relative_offset_batch(
        n_train, seq_len, d_model, offset, rng, n_distractors=n_distractors
    )
    Xte, Yte, Qte = make_relative_offset_batch(
        n_test, seq_len, d_model, offset, rng, n_distractors=n_distractors
    )

    history = []
    for step in range(steps):
        idx = rng.integers(0, n_train, size=96)
        xb, yb, qb = Xtr[idx], Ytr[idx], Qtr[idx]

        x_in = xb + sinusoidal_pe(seq_len, d_model)[None, :, :] if pe == "absolute" else xb
        Q = x_in @ attn.W_q
        K = x_in @ attn.W_k
        if pe == "rope":
            cos, sin = rope_cos_sin(seq_len, d_model)
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)
        scale = logit_scale / np.sqrt(d_model)

        logits = np.stack([scale * (Q[b, int(qb[b])] @ K[b].T) for b in range(len(qb))])
        row_prob = softmax(logits, axis=-1)
        loss = -float(np.mean(np.log(row_prob[np.arange(len(yb)), yb] + 1e-12)))
        grad_logits = row_prob.copy()
        grad_logits[np.arange(len(yb)), yb] -= 1.0
        grad_logits /= len(yb)

        gQ = np.zeros_like(Q)
        gK = np.zeros_like(K)
        for b in range(len(qb)):
            qi = int(qb[b])
            gQ[b, qi, :] = scale * (grad_logits[b, :, None] * K[b]).sum(axis=0)
            gK[b] += scale * grad_logits[b, :, None] * Q[b, qi : qi + 1, :]

        if pe == "rope":
            cos, sin = rope_cos_sin(seq_len, d_model)
            gQ = apply_rope(gQ, cos, -sin)
            gK = apply_rope(gK, cos, -sin)

        gWq = x_in.reshape(-1, d_model).T @ gQ.reshape(-1, d_model)
        gWk = x_in.reshape(-1, d_model).T @ gK.reshape(-1, d_model)
        attn.W_q -= lr * gWq
        attn.W_k -= lr * gWk

        if step % 20 == 0 or step == steps - 1:
            acc = distance_discrimination_accuracy(attn, Xte, Yte, Qte, logit_scale)
            mass = attention_mass_on_target(attn, Xte, Yte, Qte, logit_scale)
            gap = mean_logit_gap(attn, Xte, Yte, Qte)
            history.append(
                {
                    "step": step,
                    "loss": round(loss, 4),
                    "acc": round(acc, 4),
                    "mass": round(mass, 4),
                    "logit_gap": round(gap, 4),
                }
            )

    final_acc = distance_discrimination_accuracy(attn, Xte, Yte, Qte, logit_scale)
    final_mass = attention_mass_on_target(attn, Xte, Yte, Qte, logit_scale)
    final_gap = mean_logit_gap(attn, Xte, Yte, Qte)
    row_logits = _query_logits(attn, Xte[:1], Qte[:1], logit_scale)[0]
    row = softmax(row_logits[None, :], axis=-1)[0]
    return {
        "pe": pe,
        "final_acc": round(float(final_acc), 4),
        "final_mass": round(float(final_mass), 4),
        "final_logit_gap": round(float(final_gap), 4),
        "chance_acc": round(1.0 / seq_len, 4),
        "history": history,
        "attn_row": [round(float(v), 4) for v in row.tolist()],
        "label": int(Yte[0]),
        "query_idx": int(Qte[0]),
        "offset": offset,
        "seq_len": seq_len,
        "n_distractors": n_distractors,
        "logit_scale": logit_scale,
    }
