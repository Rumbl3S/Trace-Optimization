"""
benchmark_comparison.py — trace_use vs common baselines, with publication-quality charts.

Strategies compared (all on the same 152 labeled component records):
  1. No Verification        — take first answer, never check
  2. Random Gating          — verify a random X% of components
  3. Task-Text Routing      — gate on kNN over task/question embedding only
  4. Answer-Only Routing    — gate on kNN over final-answer text only
  5. trace_use              — gate on kNN over full reasoning trace (our approach)
  6. Always Verify          — verify every component (upper bound)

All predictions are leave-one-out (no data leakage).
Embeddings: OpenAI text-embedding-3-small via concurrent batch calls.
"""
from __future__ import annotations

import json
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents import _load_env   # loads .env before any client is created
_load_env()
from forecast import knn_predict, auc as roc_auc

RESULTS = Path(__file__).parent / "results"
OUT_DIR = RESULTS / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_HIGHLIGHT = "#2E7D6B"   # trace_use — teal
C_SECOND    = "#5BA4CF"   # second best
C_NEUTRAL   = "#B0BEC5"   # baselines
C_WORST     = "#CFD8DC"
C_ALWAYS    = "#546E7A"
BG          = "#FAFAFA"
GRID        = "#E8ECEF"

# ── load data ─────────────────────────────────────────────────────────────────
def load_records():
    path = RESULTS / "component_records.json"
    with open(path) as f:
        recs = json.load(f)
    return recs

# ── embedding helpers ─────────────────────────────────────────────────────────
def _extract_final_answer(text: str) -> str:
    m = re.search(r"answer\s*:\s*(.+)", text, re.I)
    return (m.group(1) if m else text[-300:]).strip()

def embed_batch(texts: list[str], client) -> np.ndarray:
    texts = [t[:8000] for t in texts]
    r = client.embeddings.create(model="text-embedding-3-small", input=texts)
    v = np.array([d.embedding for d in r.data], dtype="float32")
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def embed_all_parallel(text_lists: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Embed multiple corpora in parallel threads (one thread per corpus)."""
    from openai import OpenAI
    client = OpenAI()
    results = {}
    BATCH = 256

    def embed_corpus(name, texts):
        out = []
        for i in range(0, len(texts), BATCH):
            out.append(embed_batch(texts[i:i+BATCH], client))
        return name, np.vstack(out)

    with ThreadPoolExecutor(max_workers=len(text_lists)) as ex:
        futures = {ex.submit(embed_corpus, name, texts): name
                   for name, texts in text_lists.items()}
        for fut in as_completed(futures):
            name, vecs = fut.result()
            results[name] = vecs
            print(f"  embedded '{name}': {vecs.shape}")
    return results

# ── LOO kNN prediction -> failure score ──────────────────────────────────────
def loo_fail_scores(vecs: np.ndarray, labels: list[int], k: int = 10) -> list[float]:
    """Leave-one-out kNN -> P(success) for each point -> convert to P(fail)."""
    preds_success = knn_predict(vecs.tolist(), labels, k=k)
    return [1.0 - p for p in preds_success]

# ── budget curve: failures caught vs fraction of budget spent ─────────────────
def budget_curve(fail_scores: list[float], labels: list[int],
                 budgets: list[float]) -> list[float]:
    """For each budget fraction, what fraction of failures do we catch
    by intervening on the top-budget% highest-scoring components?"""
    n = len(labels)
    total_failures = sum(1 for l in labels if l == 0)
    if total_failures == 0:
        return [0.0] * len(budgets)
    order = sorted(range(n), key=lambda i: fail_scores[i], reverse=True)
    results = []
    for b in budgets:
        k = max(1, int(round(b * n)))
        top_k = order[:k]
        caught = sum(1 for i in top_k if labels[i] == 0)
        results.append(caught / total_failures)
    return results

def random_curve(budgets: list[float]) -> list[float]:
    return [b for b in budgets]   # random catches budget% of failures in expectation

# ── chart 1: AUC leaderboard bar chart (like the reference image) ────────────
def plot_auc_leaderboard(strategy_aucs: dict, datasets: dict, save_path: Path):
    items = sorted(
        [(k, v) for k, v in strategy_aucs.items()
         if k not in ("No Verification", "Always Verify")],
        key=lambda x: x[1], reverse=True
    )
    names, aucs = zip(*items)
    max_auc = max(aucs)

    col_map = {
        "trace_use":         C_HIGHLIGHT,
        "Answer-Only":       "#5BA4CF",
        "Task-Text Routing": "#7986CB",
        "Random Gating":     C_WORST,
    }

    def draw_bars(ax, ns, vs, title):
        colors = [col_map.get(n, C_NEUTRAL) for n in ns]
        bars = ax.bar(range(len(ns)), vs, color=colors,
                      width=0.55, zorder=3, edgecolor="white", linewidth=1.0)
        ax.set_facecolor(BG)
        ax.set_ylim(0.45, max(vs) + 0.12)
        ax.axhline(0.5, color="#EF5350", linewidth=1.2, linestyle="--", zorder=2)
        ax.text(len(ns) - 0.5, 0.502, "chance", fontsize=7.5,
                color="#EF5350", va="bottom", ha="right")
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(ns, rotation=30, ha="right", fontsize=9, color="#333")
        ax.set_ylabel("ROC-AUC", fontsize=9, color="#555")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#1A1A2E", pad=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.grid(axis="y", color=GRID, zorder=0, linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#DDD")
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", color="#CCC", labelsize=8.5)
        for bar, val in zip(bars, vs):
            is_top = (val == max(vs))
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=9 if is_top else 8,
                    fontweight="bold" if is_top else "normal",
                    color=C_HIGHLIGHT if is_top else "#666")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=BG)
    fig.suptitle("Failure Prediction AUC — trace_use vs Common Approaches",
                 fontsize=13, fontweight="bold", color="#1A1A2E", y=1.02)

    draw_bars(axes[0], names, aucs, f"All Tasks  (n=152)")

    for ax, src in zip(axes[1:], ("fanout", "musique")):
        data = datasets[src]
        n    = data["_n"]
        sub  = {k: data[k] for k in [nm for nm in names if nm in data]}
        sub_items = sorted(sub.items(), key=lambda x: x[1], reverse=True)
        sn, sv = zip(*sub_items)
        draw_bars(ax, sn, sv, f"{'FanOutQA' if src=='fanout' else 'MuSiQue'}  (n={n})")

    legend = [
        mpatches.Patch(color=C_HIGHLIGHT, label="trace_use (ours)"),
        mpatches.Patch(color="#5BA4CF",   label="Answer-Only routing"),
        mpatches.Patch(color="#7986CB",   label="Task-text routing"),
        mpatches.Patch(color=C_WORST,     label="Random (no signal)"),
        plt.Line2D([0], [0], color="#EF5350", linestyle="--", linewidth=1.2,
                   label="Chance (AUC = 0.50)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.1))

    fig.subplots_adjust(wspace=0.35, bottom=0.22)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  saved: {save_path}")

# ── chart 2: budget efficiency curve ─────────────────────────────────────────
def plot_budget_curve(strategy_curves: dict, budgets: list[float], save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle("Verification Budget Efficiency — Failures Caught vs Budget Spent",
                 fontsize=13, fontweight="bold", y=1.02, color="#1A1A2E")

    styles = {
        "No Verification":        ("#CFD8DC",   "--", 1.4),
        "Random Gating":          ("#78909C",   "-.", 1.8),
        "Task-Text Routing":      ("#7986CB",   ":",  2.0),
        "Answer-Only":            ("#5BA4CF",   "--", 2.0),
        "trace_use":              (C_HIGHLIGHT, "-",  3.0),
        "Always Verify":          (C_ALWAYS,    ":",  1.4),
        "trace_use (fanout)":     (C_HIGHLIGHT, "-",  2.5),
        "trace_use (musique)":    ("#26A69A",   "--", 2.5),
    }

    for ax_idx, (ax, (src_label, curves)) in enumerate(
            zip(axes, strategy_curves.items())):
        rand_key = "Random Gating"
        rand_ys  = [b * 100 for b in budgets]

        for name, ys in curves.items():
            if name not in styles:
                continue
            col, ls, lw = styles[name]
            label = "trace_use (ours)" if name == "trace_use" else name
            ax.plot([b * 100 for b in budgets], [y * 100 for y in ys],
                    color=col, linestyle=ls, linewidth=lw, label=label, zorder=3)

        # shade trace_use advantage over random
        for tu_key in ("trace_use", "trace_use (fanout)", "trace_use (musique)"):
            if tu_key in curves:
                tu_ys = [y * 100 for y in curves[tu_key]]
                ax.fill_between([b * 100 for b in budgets], rand_ys, tu_ys,
                                where=[t > r for t, r in zip(tu_ys, rand_ys)],
                                alpha=0.10, color=C_HIGHLIGHT)

        ax.set_facecolor(BG)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#DDD")
        ax.set_xlabel("Verify Budget (% of components)", fontsize=10, color="#444")
        ax.set_ylabel("Failures Caught (%)", fontsize=10, color="#444")
        ax.set_title(src_label, fontsize=11, fontweight="bold", color="#1A1A2E", pad=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.tick_params(colors="#666", labelsize=8.5)
        ax.legend(loc="upper left", fontsize=8, frameon=True,
                  framealpha=0.92, edgecolor="#DDD", labelspacing=0.3)

    fig.subplots_adjust(wspace=0.3, bottom=0.12)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  saved: {save_path}")

# ── chart 3: multiplier bars at fixed budgets ─────────────────────────────────
def plot_multiplier_bars(strategy_curves: dict, budgets_to_show: list[float],
                         save_path: Path):
    """How many times MORE failures does trace_use catch vs random, at fixed budgets?"""
    fig, axes = plt.subplots(1, len(budgets_to_show), figsize=(14, 5), facecolor=BG)
    fig.suptitle("Catch Rate vs Random Baseline — Each Bar = Failures Caught at That Budget",
                 fontsize=12, fontweight="bold", y=1.02, color="#1A1A2E")

    all_strategies = ["No Verification", "Random Gating",
                      "Task-Text Routing", "Answer-Only", "trace_use", "Always Verify"]
    col_map = {
        "No Verification":   C_WORST,
        "Random Gating":     "#90A4AE",
        "Task-Text Routing": C_SECOND,
        "Answer-Only":       "#7986CB",
        "trace_use":         C_HIGHLIGHT,
        "Always Verify":     C_ALWAYS,
    }

    # use "All Tasks" curves
    all_curves = list(strategy_curves.values())[0]
    budget_list = list(np.linspace(0, 1, 201))

    def catch_at(name, b):
        if name not in all_curves:
            return 0.0
        idx = min(range(len(budget_list)), key=lambda i: abs(budget_list[i] - b))
        return all_curves[name][idx] * 100

    for ax, b in zip(axes, budgets_to_show):
        vals = [(s, catch_at(s, b)) for s in all_strategies]
        vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
        names, ys = zip(*vals_sorted)
        cols = [col_map.get(n, C_NEUTRAL) for n in names]

        bars = ax.bar(range(len(names)), ys, color=cols,
                      width=0.62, zorder=3, edgecolor="white", linewidth=0.7)
        ax.set_facecolor(BG)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", color=GRID, zorder=0, linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#DDD")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.5, color="#333")
        ax.set_ylabel("Failures Caught (%)", fontsize=9, color="#444")
        ax.set_title(f"Budget = {int(b*100)}%", fontsize=11,
                     fontweight="bold", color="#1A1A2E")
        ax.tick_params(axis="x", length=0)

        for bar, val in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if val == max(ys) else "normal",
                    color=C_HIGHLIGHT if val == max(ys) else "#555")

    legend = [mpatches.Patch(color=c, label=n)
              for n, c in col_map.items()]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"  saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading component records...")
    recs = load_records()
    labels = [r["label"] for r in recs]
    srcs   = [r["src"]   for r in recs]

    print(f"  {len(recs)} components  |  failures={sum(1 for l in labels if l==0)}  "
          f"successes={sum(1 for l in labels if l==1)}")

    # build text corpora
    traces       = [r["trace"] for r in recs]
    task_texts   = [r["task"] + " " + r.get("subq", "") for r in recs]
    answer_texts = [_extract_final_answer(r["trace"]) for r in recs]

    print("\nEmbedding all corpora in parallel...")
    vecs = embed_all_parallel({
        "trace":       traces,
        "task_text":   task_texts,
        "answer_only": answer_texts,
    })

    # ── LOO predictions for each strategy ────────────────────────────────────
    print("\nRunning LOO kNN predictions (k=10)...")
    K = 10
    fail_scores = {
        "No Verification":   [0.0] * len(labels),          # never flags anything
        "Random Gating":     list(np.random.default_rng(42).uniform(0, 1, len(labels))),
        "Task-Text Routing": loo_fail_scores(vecs["task_text"],   labels, K),
        "Answer-Only":       loo_fail_scores(vecs["answer_only"], labels, K),
        "trace_use":         loo_fail_scores(vecs["trace"],       labels, K),
        "Always Verify":     [1.0] * len(labels),           # always flags everything
    }

    # ── AUCs — roc_auc expects higher score = more likely label=1 (success)
    #          so pass success_score = 1 - fail_score ──────────────────────────
    strategy_aucs = {}
    for name, scores in fail_scores.items():
        if name in ("No Verification", "Always Verify"):
            strategy_aucs[name] = 0.5
        else:
            success_scores = [1.0 - s for s in scores]
            strategy_aucs[name] = roc_auc(labels, success_scores)
        print(f"  {name:<22s}  AUC={strategy_aucs[name]:.3f}")

    # per-dataset AUCs
    datasets = {}
    for src in ("fanout", "musique"):
        idx = [i for i, s in enumerate(srcs) if s == src]
        sub_labels = [labels[i] for i in idx]
        datasets[src] = {"_n": len(idx)}
        for name, scores in fail_scores.items():
            sub_scores = [scores[i] for i in idx]
            if name in ("No Verification", "Always Verify"):
                datasets[src][name] = 0.5
            else:
                datasets[src][name] = roc_auc(sub_labels, [1.0 - s for s in sub_scores])

    # ── budget curves ─────────────────────────────────────────────────────────
    budgets = list(np.linspace(0, 1, 201))
    print("\nComputing budget curves...")

    def curves_for_subset(idx_subset):
        sub_labels = [labels[i] for i in idx_subset]
        return {
            name: budget_curve([scores[i] for i in idx_subset], sub_labels, budgets)
            for name, scores in fail_scores.items()
        }

    all_idx     = list(range(len(labels)))
    fanout_idx  = [i for i, s in enumerate(srcs) if s == "fanout"]
    musique_idx = [i for i, s in enumerate(srcs) if s == "musique"]

    strategy_curves = {
        "All Tasks":    curves_for_subset(all_idx),
        "FanOutQA":     curves_for_subset(fanout_idx),
        "MuSiQue":      curves_for_subset(musique_idx),
    }

    # ── print key numbers ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  KEY NUMBERS")
    print("=" * 60)
    for budget_pct in [0.10, 0.20, 0.30, 0.50]:
        b_idx = min(range(len(budgets)), key=lambda i: abs(budgets[i] - budget_pct))
        rand  = strategy_curves["All Tasks"]["Random Gating"][b_idx] * 100
        tu    = strategy_curves["All Tasks"]["trace_use"][b_idx] * 100
        mult  = tu / rand if rand > 0 else float("inf")
        print(f"  Budget {int(budget_pct*100):3d}%:  Random={rand:.0f}%  "
              f"trace_use={tu:.0f}%  ({mult:.2f}x random)")
    print("=" * 60)

    # ── generate charts ───────────────────────────────────────────────────────
    print("\nGenerating charts...")

    # fix dataset dict for chart 1 (remove _n before plotting)
    ds_for_chart = {}
    for src, d in datasets.items():
        n = d["_n"]
        ds_for_chart[src] = {k: v for k, v in d.items() if k != "_n"}
        ds_for_chart[src]["_n"] = n

    plot_auc_leaderboard(
        strategy_aucs,
        ds_for_chart,
        OUT_DIR / "1_auc_leaderboard.png",
    )
    plot_budget_curve(
        {
            "All Tasks": strategy_curves["All Tasks"],
            "FanOutQA vs MuSiQue": {
                "trace_use (fanout)":  strategy_curves["FanOutQA"]["trace_use"],
                "trace_use (musique)": strategy_curves["MuSiQue"]["trace_use"],
                "Random Gating":       strategy_curves["All Tasks"]["Random Gating"],
                "No Verification":     strategy_curves["All Tasks"]["No Verification"],
                "Always Verify":       strategy_curves["All Tasks"]["Always Verify"],
                "Answer-Only":         strategy_curves["All Tasks"]["Answer-Only"],
                "Task-Text Routing":   strategy_curves["All Tasks"]["Task-Text Routing"],
            },
        },
        budgets,
        OUT_DIR / "2_budget_efficiency.png",
    )
    plot_multiplier_bars(
        {"All Tasks": strategy_curves["All Tasks"]},
        [0.10, 0.20, 0.30, 0.50],
        OUT_DIR / "3_catch_rate_by_budget.png",
    )

    print(f"\nAll charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
