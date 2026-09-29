#!/usr/bin/env python3
"""RoPE smoke: relative property, shift invariance, relative-offset train -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from experiments import (
    closed_form_relative_scores,
    relative_shift_invariance,
    train_relative_task,
)
from rope import check_relative_property
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {
    "d_model": 32,
    "seq_len": 16,
    "rope_base": 10000.0,
    "offset": 4,
    "n_distractors": 3,
    "n_train": 640,
    "n_test": 256,
    "steps": 180,
    "lr": 0.08,
    "logit_scale": 8.0,
    "shifts": [1, 2, 4, 8],
}


def _compact(js: str) -> str:
    js = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)
    return re.sub(r"\{\n([^{}\[\]]*?)\n\s*\}", lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}", js)


def r4(x: Any) -> float:
    return round(float(x), 4)


def main() -> None:
    t0 = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    prop = check_relative_property(dim=CFG["d_model"], base=CFG["rope_base"], seed=SEED)
    inv = relative_shift_invariance(
        d_model=CFG["d_model"],
        seq_len=CFG["seq_len"],
        shifts=CFG["shifts"],
        seed=SEED,
    )
    cf = closed_form_relative_scores(
        d_model=CFG["d_model"], seq_len=CFG["seq_len"], seed=SEED
    )

    train_results = []
    for pe in ("none", "absolute", "rope"):
        train_results.append(
            train_relative_task(
                pe,
                d_model=CFG["d_model"],
                seq_len=CFG["seq_len"],
                offset=CFG["offset"],
                n_train=CFG["n_train"],
                n_test=CFG["n_test"],
                steps=CFG["steps"],
                lr=CFG["lr"],
                seed=SEED,
                n_distractors=CFG["n_distractors"],
                logit_scale=CFG["logit_scale"],
            )
        )

    by_pe = {r["pe"]: r for r in train_results}
    wall = round(time.perf_counter() - t0, 2)
    headline = {
        "rope_invariance_cosine": inv["summary"]["rope"]["mean_cosine"],
        "absolute_invariance_cosine": inv["summary"]["absolute"]["mean_cosine"],
        "none_invariance_cosine": inv["summary"]["none"]["mean_cosine"],
        "rope_acc": by_pe["rope"]["final_acc"],
        "absolute_acc": by_pe["absolute"]["final_acc"],
        "none_acc": by_pe["none"]["final_acc"],
        "rope_mass": by_pe["rope"]["final_mass"],
        "absolute_mass": by_pe["absolute"]["final_mass"],
        "none_mass": by_pe["none"]["final_mass"],
        "rope_logit_gap": by_pe["rope"]["final_logit_gap"],
        "chance_acc": by_pe["none"]["chance_acc"],
        "relative_property_pass": prop["pass"],
        "abs_placement_variance": cf["abs_mean_placement_variance"],
    }

    metrics: Dict[str, Any] = {
        "project": "ai-learn-26-rope-embeddings",
        "seed": SEED,
        "config": CFG,
        "relative_property": {
            "max_abs_vs_rel_err": prop["max_abs_vs_rel_err"],
            "max_shift_invariance_err": prop["max_shift_invariance_err"],
            "pass": prop["pass"],
            "n_trials": prop["n_trials"],
        },
        "relative_invariance": inv,
        "closed_form": {
            "offsets": cf["offsets"],
            "rope_scores": cf["rope_scores"],
            "abs_mean_placement_variance": cf["abs_mean_placement_variance"],
            "rope_is_function_of_offset_only": True,
            "abs_score_variance_across_placements": cf["abs_score_variance_across_placements"],
        },
        "train_results": [
            {
                "pe": r["pe"],
                "final_acc": r["final_acc"],
                "final_mass": r["final_mass"],
                "final_logit_gap": r["final_logit_gap"],
                "chance_acc": r["chance_acc"],
                "history": r["history"],
                "attn_row": r["attn_row"],
                "label": r["label"],
                "query_idx": r["query_idx"],
                "offset": r["offset"],
                "seq_len": r["seq_len"],
                "n_distractors": r["n_distractors"],
                "logit_scale": r["logit_scale"],
            }
            for r in train_results
        ],
        "headline": headline,
        "wall_time_s": wall,
    }

    plot_names = make_plots(RESULTS, metrics)
    write_results_md(RESULTS, metrics, plot_names)

    (RESULTS / "metrics.json").write_text(
        _compact(json.dumps(metrics, indent=1)) + "\n", encoding="utf-8"
    )
    shot = {
        "project": "ai-learn-26-rope-embeddings",
        "seed": SEED,
        "config": {
            "d_model": CFG["d_model"],
            "seq_len": CFG["seq_len"],
            "offset": CFG["offset"],
            "rope_base": CFG["rope_base"],
            "steps": CFG["steps"],
            "logit_scale": CFG["logit_scale"],
        },
        "headline": headline,
        "wall_time_s": wall,
    }
    (RESULTS / "JSON.shot").write_text(
        _compact(json.dumps(shot, indent=1)) + "\n", encoding="utf-8"
    )

    print("=== ai-learn-26-rope-embeddings smoke ===")
    print(f"seed={SEED} wall_time_s={wall}")
    print(f"relative_property_pass={prop['pass']}")
    print("invariance:", inv["summary"])
    for r in train_results:
        print(
            f"  {r['pe']:8s} acc={r['final_acc']} mass={r['final_mass']} gap={r['final_logit_gap']}"
        )
    print("plots:", ", ".join(plot_names))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
