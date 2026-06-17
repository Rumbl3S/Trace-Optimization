"""
learning_diagnostic.py — answers "are we actually learning anything?"

Three questions:
  1. AUC vs store size   — does kNN quality improve as more traces are stored?
  2. P(fail) distribution — do real failures score higher than real passes?
  3. Calibration          — when we predict P(fail)=0.7, do ~70% actually fail?

Uses the 152 labeled component traces from component_records.json.
Embeds once, then simulates streaming (add one trace, predict the rest).
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from agents import _build_openai, _load_env
from forecast import knn_predict_cross, cosine

_load_env()

OUT  = Path("eval/results/charts")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("eval/results/component_embeddings.npy")
LABEL_CACHE = Path("eval/results/component_labels.npy")

BG   = "#FAFAFA"
GRID = "#E8ECEF"
C_PASS = "#4CAF50"
C_FAIL = "#EF5350"
C_MAIN = "#2E7D6B"

# ── load records ──────────────────────────────────────────────────────────────
recs   = json.loads(Path("eval/results/component_records.json").read_text())
traces = [r["trace"] for r in recs]
labels = [r["label"] for r in recs]
n      = len(recs)
print(f"Loaded {n} records: {sum(labels)} pass, {n-sum(labels)} fail")

# ── embed (cache so we only pay once) ────────────────────────────────────────
if CACHE.exists() and LABEL_CACHE.exists():
    print("Loading cached embeddings...")
    vecs   = np.load(str(CACHE))
    labels = np.load(str(LABEL_CACHE)).tolist()
else:
    print("Embedding traces with OpenAI (this is a one-time cost)...")
    embedder = _build_openai()
    from concurrent.futures import ThreadPoolExecutor
    BATCH = 20
    batches = [traces[i:i+BATCH] for i in range(0, n, BATCH)]
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(embedder, b) for b in batches]
        for i, f in enumerate(futs):
            results.extend(f.result())
            print(f"  batch {i+1}/{len(batches)} done")
    vecs = np.array(results, dtype="float32")
    np.save(str(CACHE), vecs)
    np.save(str(LABEL_CACHE), np.array(labels))
    print(f"Saved embeddings: {vecs.shape}")

labels = [int(x) for x in labels]

# ── 1. AUC vs store size (streaming simulation) ───────────────────────────────
# Strategy: use records 0..i as store, predict on i+1..n.
# Track AUC as store grows from 10 → 152.
print("\nSimulating streaming: AUC vs store size...")

MIN_STORE = 5   # need at least some of each class
store_sizes  = []
auc_scores   = []
random_auc   = []

shuffle_idx = np.random.RandomState(42).permutation(n)
sv = vecs[shuffle_idx]
sl = [labels[i] for i in shuffle_idx]

for store_end in range(MIN_STORE, n - 10, 3):
    store_v = sv[:store_end].tolist()
    store_l = sl[:store_end]

    # skip if store doesn't have both classes
    if len(set(store_l)) < 2:
        continue

    test_v  = sv[store_end:].tolist()
    test_l  = sl[store_end:]

    k = min(10, store_end - 1)
    success_scores = knn_predict_cross(store_v, store_l, test_v, k=k)

    if len(set(test_l)) < 2:
        continue

    # label=1 means pass; success_scores are P(pass); AUC = discriminability pass vs fail
    auc = roc_auc_score(test_l, success_scores)
    store_sizes.append(store_end)
    auc_scores.append(auc)
    random_auc.append(0.5)

print(f"  Final AUC (full store {n}): {auc_scores[-1]:.3f}")
print(f"  AUC at store=20: {auc_scores[min(5, len(auc_scores)-1)]:.3f}")
print(f"  AUC at store=50: {next((a for s,a in zip(store_sizes,auc_scores) if s>=50), None):.3f}")

# ── 2. P(fail) distribution for actual pass vs fail ───────────────────────────
print("\nComputing P(fail) distribution (LOO cross-validation)...")
k = 10
store_v_all = vecs.tolist()
store_l_all = labels

# LOO: for each record, predict using all other records
loo_pred_fail = []
for i in range(n):
    leave_v = [store_v_all[j] for j in range(n) if j != i]
    leave_l = [store_l_all[j] for j in range(n) if j != i]
    if len(set(leave_l)) < 2:
        loo_pred_fail.append(0.5)
        continue
    succ = knn_predict_cross(leave_v, leave_l, [store_v_all[i]], k=k)[0]
    loo_pred_fail.append(1.0 - succ)   # P(fail) = 1 - P(pass)

loo_pred = np.array(loo_pred_fail)     # P(fail): higher = more likely failure
loo_true = np.array(labels)            # 1=pass, 0=fail

fail_preds = loo_pred[loo_true == 0]   # P(fail) scores for actual failures
pass_preds = loo_pred[loo_true == 1]   # P(fail) scores for actual passes

# roc_auc_score(y_true, scores) where label=1 is positive; use P(pass) = 1-P(fail)
overall_auc = roc_auc_score(loo_true, 1.0 - loo_pred)
print(f"  LOO AUC: {overall_auc:.3f}")
print(f"  Mean P(fail) | actual fail: {fail_preds.mean():.3f}")
print(f"  Mean P(fail) | actual pass: {pass_preds.mean():.3f}")
print(f"  Separation (delta): {fail_preds.mean() - pass_preds.mean():.3f}")

# ── 3. Calibration ────────────────────────────────────────────────────────────
print("\nCalibration...")
cal_bins     = np.arange(0, 1.1, 0.1)
bin_midpoint = []
bin_actual   = []
bin_count    = []
for lo, hi in zip(cal_bins[:-1], cal_bins[1:]):
    mask = (loo_pred >= lo) & (loo_pred < hi)
    if mask.sum() > 0:
        # actual failure rate in this predicted-P(fail) bin
        actual_fail_rate = (loo_true[mask] == 0).mean()
        bin_midpoint.append((lo + hi) / 2)
        bin_actual.append(actual_fail_rate)
        bin_count.append(mask.sum())

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=BG)
fig.suptitle("trace_use — Is the Forecaster Actually Learning?\n"
             f"152 labeled component traces, kNN k=10, LOO AUC = {overall_auc:.3f}",
             fontsize=12, fontweight="bold", color="#1A1A2E", y=1.02)

for ax in axes:
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#DDD")
    ax.tick_params(colors="#666", labelsize=9)

# --- subplot 1: AUC vs store size ---
ax = axes[0]
ax.plot(store_sizes, auc_scores, color=C_MAIN, linewidth=2.0, zorder=4, label="trace_use kNN")
ax.axhline(0.5, color="#90A4AE", linewidth=1.2, linestyle="--", label="random chance")
ax.fill_between(store_sizes, 0.5, auc_scores, alpha=0.12, color=C_MAIN)

# annotate milestones
for milestone, label in [(10, "10 traces"), (50, "50 traces"), (100, "100 traces")]:
    idx = next((i for i, s in enumerate(store_sizes) if s >= milestone), None)
    if idx:
        ax.annotate(f"  {auc_scores[idx]:.2f}", (store_sizes[idx], auc_scores[idx]),
                    fontsize=8, color=C_MAIN, va="center")
        ax.scatter([store_sizes[idx]], [auc_scores[idx]], s=50, color=C_MAIN, zorder=5)

ax.set_xlabel("Store size (traces added)", fontsize=10, color="#444", labelpad=6)
ax.set_ylabel("AUC (test set)", fontsize=10, color="#444", labelpad=6)
ax.set_title("Does it improve with more data?", fontsize=10.5, fontweight="bold", color="#1A1A2E")
ax.set_ylim(0.40, 0.90)
ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="#DDD")
ax.grid(axis="both", color=GRID, linewidth=0.8)
ax.spines[["top","right","left"]].set_visible(False)

# --- subplot 2: P(fail) distribution ---
ax = axes[1]
bins = np.linspace(0, 1, 15)

ax.hist(pass_preds, bins=bins, color=C_PASS, alpha=0.7, label=f"Actual PASS (n={len(pass_preds)})", zorder=3)
ax.hist(fail_preds, bins=bins, color=C_FAIL, alpha=0.7, label=f"Actual FAIL (n={len(fail_preds)})", zorder=3)
ax.axvline(0.5, color="#555", linewidth=1.5, linestyle="--", alpha=0.7)
ax.text(0.51, ax.get_ylim()[1]*0.98 if ax.get_ylim()[1] > 0 else 10,
        "threshold=0.5", fontsize=8, color="#555", va="top")

# add mean lines
ax.axvline(pass_preds.mean(), color=C_PASS, linewidth=1.5, linestyle=":")
ax.axvline(fail_preds.mean(), color=C_FAIL, linewidth=1.5, linestyle=":")
ax.text(pass_preds.mean() - 0.02, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 8,
        f"μ={pass_preds.mean():.2f}", fontsize=7.5, color=C_PASS, ha="right")
ax.text(fail_preds.mean() + 0.02, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 8,
        f"μ={fail_preds.mean():.2f}", fontsize=7.5, color=C_FAIL, ha="left")

ax.set_xlabel("Predicted P(fail)", fontsize=10, color="#444", labelpad=6)
ax.set_ylabel("Count", fontsize=10, color="#444", labelpad=6)
ax.set_title("Are failure scores higher for real failures?", fontsize=10.5,
             fontweight="bold", color="#1A1A2E")
ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="#DDD")
ax.grid(axis="both", color=GRID, linewidth=0.8)
ax.spines[["top","right","left"]].set_visible(False)

# --- subplot 3: calibration ---
ax = axes[2]
if bin_midpoint:
    scatter = ax.scatter(bin_midpoint, bin_actual,
                         s=[max(20, c*3) for c in bin_count],
                         c=bin_actual, cmap="RdYlGn_r", vmin=0, vmax=1,
                         zorder=4, edgecolors="white", linewidth=0.8)
    ax.plot(bin_midpoint, bin_actual, color="#555", linewidth=1.2,
            linestyle="--", alpha=0.5, zorder=3)

    # ideal calibration line
    ax.plot([0, 1], [0, 1], color=C_MAIN, linewidth=1.5, linestyle="-",
            alpha=0.6, label="perfect calibration", zorder=2)

    for xm, ya, c in zip(bin_midpoint, bin_actual, bin_count):
        ax.annotate(f"n={c}", (xm, ya), xytext=(0, 8), textcoords="offset points",
                    fontsize=7, color="#888", ha="center")

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.10)
ax.set_xlabel("Predicted P(fail)", fontsize=10, color="#444", labelpad=6)
ax.set_ylabel("Actual failure rate", fontsize=10, color="#444", labelpad=6)
ax.set_title("When we predict P(fail)=X, do X% actually fail?", fontsize=10.5,
             fontweight="bold", color="#1A1A2E")
ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="#DDD")
ax.grid(axis="both", color=GRID, linewidth=0.8)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(colors="#666", labelsize=9)

plt.tight_layout(pad=1.5)
p = OUT / "7_learning_diagnostic.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\nSaved: {p}")

# ── text summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DIAGNOSTIC SUMMARY")
print("=" * 55)
print(f"LOO AUC on 152 traces:   {overall_auc:.3f}  (0.5 = chance, 1.0 = perfect)")
print(f"Mean P(fail) | actual fail:  {fail_preds.mean():.3f}")
print(f"Mean P(fail) | actual pass:  {pass_preds.mean():.3f}")
print(f"Score separation (delta):    {fail_preds.mean()-pass_preds.mean():.3f}")
print()
print("At threshold 0.50:")
predicted_fail = (loo_pred >= 0.5)
tp = ((predicted_fail) & (loo_true == 0)).sum()
fp = ((predicted_fail) & (loo_true == 1)).sum()
fn = ((~predicted_fail) & (loo_true == 0)).sum()
tn = ((~predicted_fail) & (loo_true == 1)).sum()
print(f"  True positives (correct fail flags): {tp}")
print(f"  False positives (wasted retries):    {fp}")
print(f"  False negatives (missed failures):   {fn}")
print(f"  True negatives (correct pass):       {tn}")
prec = tp/(tp+fp) if (tp+fp)>0 else 0
rec  = tp/(tp+fn) if (tp+fn)>0 else 0
print(f"  Precision: {prec:.2f}   Recall: {rec:.2f}")
print()
print("Verdict:")
if overall_auc > 0.70:
    print(f"  YES — traces carry real signal (AUC {overall_auc:.2f} >> 0.5).")
    print(f"  Real failures score +{fail_preds.mean()-pass_preds.mean():.2f} higher P(fail) on average.")
    print(f"  The forecaster IS learning from trajectories.")
elif overall_auc > 0.60:
    print(f"  PARTIAL — some signal (AUC {overall_auc:.2f}), but noisy with small store.")
else:
    print(f"  WEAK — AUC {overall_auc:.2f}, traces may not be informative enough.")
print()
print("Store size needed for reliable predictions:")
reliable_idx = next((i for i, a in enumerate(auc_scores) if a >= 0.70), None)
if reliable_idx is not None:
    print(f"  AUC crosses 0.70 at store_size = {store_sizes[reliable_idx]}")
    print(f"  Below that, forecaster is unreliable.")
