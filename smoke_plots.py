"""Matplotlib SVG plots + RESULTS.md writer for RoPE smoke."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update(
    {
        "svg.hashsalt": "ai-learn-26",
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)
COLS = ["#e76f51", "#e9c46a", "#2a9d8f", "#126782", "#8338ec"]
PE_ORDER = ["none", "absolute", "rope"]
PE_LABELS = {"none": "no PE", "absolute": "absolute sin", "rope": "RoPE"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names: List[str] = []

    from rope import rope_freqs

    d = m["config"]["d_model"]
    freqs = rope_freqs(d, m["config"]["rope_base"])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(np.arange(len(freqs)), freqs, "o-", color=COLS[3], ms=4)
    ax.set_xlabel("pair index i")
    ax.set_ylabel(r"$\\theta_i = base^{-2i/d}$")
    ax.set_yscale("log")
    ax.set_title(f"RoPE frequencies (d={d}, base={m['config']['rope_base']})")
    ax.grid(True, alpha=0.3, which="both")
    names.append(_save(fig, out / "rope_frequencies.svg"))

    inv = m["relative_invariance"]["per_shift"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for pe, c in zip(PE_ORDER, COLS):
        xs = [r["delta"] for r in inv if r["pe"] == pe]
        ys = [r["cosine"] for r in inv if r["pe"] == pe]
        ax.plot(xs, ys, "o-", color=c, label=PE_LABELS[pe])
    ax.set_xlabel("position shift delta")
    ax.set_ylabel("cosine(logits, logits after shift)")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Relative-shift invariance of attention logits")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    names.append(_save(fig, out / "relative_invariance.svg"))

    train = {r["pe"]: r for r in m["train_results"]}
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    xs = np.arange(len(PE_ORDER))
    accs = [train[p]["final_acc"] for p in PE_ORDER]
    masses = [train[p]["final_mass"] for p in PE_ORDER]
    w = 0.35
    ax.bar(xs - w / 2, accs, w, color=COLS[2], label="top-1 acc")
    ax.bar(xs + w / 2, masses, w, color=COLS[0], label="attn mass on target")
    ax.axhline(train["none"]["chance_acc"], color="#555", ls="--", lw=1, label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels([PE_LABELS[p] for p in PE_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Relative-offset retrieval (test)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    names.append(_save(fig, out / "accuracy_comparison.svg"))

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), sharey=True)
    for ax, pe in zip(axes, PE_ORDER):
        row = np.array(train[pe]["attn_row"])
        ax.bar(np.arange(len(row)), row, color=COLS[2] if pe == "rope" else COLS[1] if pe == "absolute" else "#aaa")
        ax.axvline(train[pe]["label"], color=COLS[0], ls="--", lw=1.2, label="target")
        ax.axvline(train[pe]["query_idx"], color=COLS[3], ls=":", lw=1.2, label="query")
        ax.set_title(PE_LABELS[pe])
        ax.set_xlabel("key index")
        ax.set_ylim(0, max(0.5, float(row.max()) * 1.15))
    axes[0].set_ylabel("attn weight")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Sample attention from query (sharpened softmax)", fontsize=11, y=1.02)
    fig.tight_layout()
    names.append(_save(fig, out / "attention_rows.svg"))

    cf = m["closed_form"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(cf["offsets"], cf["rope_scores"], "-", color=COLS[4], label="RoPE <R(q,delta), k>")
    ax.set_xlabel("relative offset delta = m - n")
    ax.set_ylabel("dot product")
    ax.set_title("RoPE attention score depends only on relative offset")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "rope_score_vs_offset.svg"))

    return names


def write_results_md(out: Path, m: Dict[str, Any], plot_names: List[str]) -> None:
    inv = m["relative_invariance"]["summary"]
    prop = m["relative_property"]
    lines = [
        "# Results: ai-learn-26-rope-embeddings",
        "",
        f"Real output of `python run_smoke.py` (seed {m['seed']}, CPU, {m['wall_time_s']} s wall time).",
        "",
        "## Setup",
        f"- d_model={m['config']['d_model']}, seq_len={m['config']['seq_len']}, "
        f"RoPE base={m['config']['rope_base']}, relative offset={m['config']['offset']}.",
        f"- Relative-offset retrieval: {m['config']['n_distractors']} content-matched distractors; "
        "query/source placement varies so absolute slot memorization fails.",
        f"- Tiny single-head NumPy attention; train Q/K for {m['config']['steps']} steps, "
        f"logit_scale={m['config']['logit_scale']}.",
        "",
        "## Relative property check",
        f"- max |<RoPE(q,m),RoPE(k,n)> - <RoPE(q,m-n),k>| = **{prop['max_abs_vs_rel_err']:.2e}**",
        f"- max shift-invariance error = **{prop['max_shift_invariance_err']:.2e}** -> pass={prop['pass']}",
        "",
        "## Relative-shift invariance (attention logits cosine after +delta)",
        "| PE | mean cosine | min cosine |",
        "|---|---|---|",
    ]
    for pe in PE_ORDER:
        lines.append(
            f"| {PE_LABELS[pe]} | {inv[pe]['mean_cosine']} | {inv[pe]['min_cosine']} |"
        )
    lines += [
        "",
        "## Relative-offset retrieval (test)",
        "| PE | top-1 acc | attn mass | logit gap |",
        "|---|---|---|---|",
    ]
    for r in m["train_results"]:
        lines.append(
            f"| {PE_LABELS[r['pe']]} | **{r['final_acc']}** | {r['final_mass']} | {r['final_logit_gap']} |"
        )
    lines += [
        "",
        "## Headline",
        f"- RoPE relative invariance cosine: **{inv['rope']['mean_cosine']}** "
        f"(absolute sin: {inv['absolute']['mean_cosine']}).",
        f"- Task accuracy: RoPE **{m['headline']['rope_acc']}** > "
        f"absolute **{m['headline']['absolute_acc']}** > "
        f"no-PE **{m['headline']['none_acc']}** (chance {m['headline']['chance_acc']}).",
        f"- Closed-form: absolute PE score variance across placements of the same offset "
        f"= **{m['closed_form']['abs_mean_placement_variance']}** (RoPE variance = 0 by construction).",
        "",
        "## Plots",
    ]
    for name in plot_names:
        lines.append(f"- ![{name}]({name})")
    lines += [
        "",
        "## Notes",
        "- Softmax uses an educational logit_scale so peaks are visible; rankings use the same scores.",
        "- Toy single-head attention, not a full Transformer -- the RoPE math is the real one.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
