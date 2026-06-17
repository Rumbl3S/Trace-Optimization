"""Generate charts and tables from large_run.json results."""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS = Path(__file__).parent / "results"
data = json.loads((RESULTS / "large_run.json").read_text())

# ── 1. Derive per-component metrics ──────────────────────────────────────────
# label=1 → pass, label=0 → fail (verifier perspective)
# retried=True → forecaster triggered intervention
records = data  # flat list

# ── 2. Aggregate by (domain, task) ───────────────────────────────────────────
tasks = defaultdict(lambda: {
    "domain": "", "task": "", "n": 0,
    "n_pass": 0, "n_fail": 0, "n_retry": 0,
    "caught": 0, "missed": 0, "false_alarm": 0,
    "p_fail_vals": [], "labels": [],
})

for r in records:
    key = r["task"][:70]
    t = tasks[key]
    t["domain"]  = r["domain"]
    t["task"]    = r["task"][:70]
    t["n"]      += 1
    t["labels"].append(r["label"])
    if r["label"] == 1:
        t["n_pass"] += 1
    else:
        t["n_fail"] += 1
    if r["retried"]:
        t["n_retry"] += 1
        if r["label"] == 0:
            t["caught"] += 1       # true positive
        else:
            t["false_alarm"] += 1  # false positive
    else:
        if r["label"] == 0:
            t["missed"] += 1       # false negative
    if r["p_fail"] is not None:
        t["p_fail_vals"].append(r["p_fail"])

tasks = dict(tasks)

# ── 3. Aggregate by domain ────────────────────────────────────────────────────
domains = defaultdict(lambda: {"n":0,"n_pass":0,"n_fail":0,"n_retry":0,
                               "caught":0,"missed":0,"false_alarm":0,
                               "tasks":0})
for t in tasks.values():
    d = domains[t["domain"]]
    for k in ("n","n_pass","n_fail","n_retry","caught","missed","false_alarm"):
        d[k] += t[k]
    d["tasks"] += 1

# ── 4. Session-level AUC (exclude null p_fail = cold-start components) ────────
from sklearn.metrics import roc_auc_score

valid = [(r["label"], r["p_fail"]) for r in records if r["p_fail"] is not None]
auc_labels  = [v[0] for v in valid]
auc_pfail   = [v[1] for v in valid]
# label=1=pass; p_fail predicts failure → use 1-p_fail for P(pass)
auc_score = float("nan")
if len(set(auc_labels)) == 2:
    auc_score = roc_auc_score(auc_labels, [1.0 - p for p in auc_pfail])

# ── 5. Print summary table ────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'LARGE RUN — COMPONENT-LEVEL RESULTS':^90}")
print("="*90)
print(f"\n{'Task (truncated)':<48} {'Domain':<11} {'Comp':>4} {'Pass':>4} {'Fail':>4} "
      f"{'Retry':>5} {'Caught':>6} {'Missed':>6} {'FP':>4}")
print("-"*90)

domain_order = ["Algorithm","Debugging","Math","Systems","Research","Coding"]
for dom in domain_order:
    for key, t in sorted(tasks.items(), key=lambda x: x[0]):
        if t["domain"] != dom:
            continue
        print(f"{t['task']:<48} {t['domain']:<11} {t['n']:>4} {t['n_pass']:>4} "
              f"{t['n_fail']:>4} {t['n_retry']:>5} {t['caught']:>6} {t['missed']:>6} "
              f"{t['false_alarm']:>4}")
    # domain subtotal
    d = domains[dom]
    recall = d["caught"] / d["n_fail"] if d["n_fail"] else float("nan")
    prec   = d["caught"] / d["n_retry"] if d["n_retry"] else float("nan")
    print(f"  {'── '+dom+' SUBTOTAL':<46} {'':<11} {d['n']:>4} {d['n_pass']:>4} "
          f"{d['n_fail']:>4} {d['n_retry']:>5} {d['caught']:>6} {d['missed']:>6} "
          f"{d['false_alarm']:>4}   recall={recall:.0%}  prec={prec:.0%}")
    print()

# Overall
total = {k: sum(domains[d][k] for d in domains) for k in ("n","n_pass","n_fail","n_retry","caught","missed","false_alarm")}
overall_recall = total["caught"] / total["n_fail"] if total["n_fail"] else float("nan")
overall_prec   = total["caught"] / total["n_retry"] if total["n_retry"] else float("nan")
print("="*90)
print(f"{'OVERALL':<48} {'':<11} {total['n']:>4} {total['n_pass']:>4} "
      f"{total['n_fail']:>4} {total['n_retry']:>5} {total['caught']:>6} {total['missed']:>6} "
      f"{total['false_alarm']:>4}   recall={overall_recall:.0%}  prec={overall_prec:.0%}")
print(f"\nSession AUC  (P(pass) vs actual label, n={len(valid)} components with p_fail): {auc_score:.3f}")
print("="*90)

# ── 6. P(fail) distribution by outcome ───────────────────────────────────────
pass_pf = [r["p_fail"] for r in records if r["p_fail"] is not None and r["label"]==1]
fail_pf = [r["p_fail"] for r in records if r["p_fail"] is not None and r["label"]==0]
print(f"\nP(fail) | actual pass : mean={np.mean(pass_pf):.3f}  median={np.median(pass_pf):.3f}")
print(f"P(fail) | actual fail : mean={np.mean(fail_pf):.3f}  median={np.median(fail_pf):.3f}")
print(f"Separation gap        : {np.mean(fail_pf)-np.mean(pass_pf):.3f}")

# ── 7. Charts ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

COLORS = {"Algorithm":"#4C72B0","Debugging":"#DD8452","Math":"#55A868",
          "Systems":"#C44E52","Research":"#8172B2","Coding":"#937860"}

# ── Chart A: Domain accuracy bar chart ────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
dom_names  = domain_order
dom_acc    = [domains[d]["n_pass"] / domains[d]["n"] for d in dom_names]
dom_colors = [COLORS[d] for d in dom_names]
bars = ax_a.bar(dom_names, dom_acc, color=dom_colors, edgecolor="white", linewidth=0.8)
ax_a.set_ylim(0, 1.12)
ax_a.set_ylabel("Component pass rate")
ax_a.set_title("A — Pass Rate by Domain", fontweight="bold")
ax_a.axhline(total["n_pass"]/total["n"], ls="--", color="gray", lw=1.2, label="Overall")
ax_a.legend(fontsize=8)
ax_a.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, dom_acc):
    ax_a.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.0%}",
              ha="center", va="bottom", fontsize=8, fontweight="bold")

# ── Chart B: P(fail) scatter — actual outcome ──────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
jitter_pass = np.random.default_rng(42).uniform(-0.07, 0.07, len(pass_pf))
jitter_fail = np.random.default_rng(42).uniform(-0.07, 0.07, len(fail_pf))
ax_b.scatter(np.array(pass_pf)+jitter_pass, np.ones(len(pass_pf)), alpha=0.25,
             s=14, color="#55A868", label=f"Pass (n={len(pass_pf)})")
ax_b.scatter(np.array(fail_pf)+jitter_fail, np.zeros(len(fail_pf)), alpha=0.50,
             s=20, color="#C44E52", label=f"Fail (n={len(fail_pf)})")
ax_b.axvline(0.35, ls="--", color="navy", lw=1.2, label="θ=0.35")
ax_b.set_xlim(-0.05, 1.05)
ax_b.set_yticks([0, 1])
ax_b.set_yticklabels(["Fail", "Pass"])
ax_b.set_xlabel("P(fail) forecast")
ax_b.set_title("B — P(fail) vs Actual Outcome", fontweight="bold")
ax_b.legend(fontsize=8)

# ── Chart C: Caught vs missed failures by domain ──────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
caught_vals = [domains[d]["caught"] for d in dom_names]
missed_vals = [domains[d]["missed"] for d in dom_names]
x = np.arange(len(dom_names))
w = 0.38
b1 = ax_c.bar(x - w/2, caught_vals, w, label="Caught (TP)", color="#55A868", edgecolor="white")
b2 = ax_c.bar(x + w/2, missed_vals, w, label="Missed (FN)", color="#C44E52", edgecolor="white")
ax_c.set_xticks(x)
ax_c.set_xticklabels(dom_names, rotation=30)
ax_c.set_ylabel("# failures")
ax_c.set_title("C — Caught vs Missed Failures", fontweight="bold")
ax_c.legend(fontsize=8)
for bar in list(b1)+list(b2):
    h = bar.get_height()
    if h > 0:
        ax_c.text(bar.get_x()+bar.get_width()/2, h+0.08, str(int(h)),
                  ha="center", va="bottom", fontsize=8)

# ── Chart D: P(fail) histogram by outcome ────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
bins = np.linspace(0, 1, 14)
ax_d.hist(pass_pf, bins=bins, alpha=0.6, color="#55A868", label="Pass", density=True)
ax_d.hist(fail_pf, bins=bins, alpha=0.6, color="#C44E52", label="Fail", density=True)
ax_d.axvline(0.35, ls="--", color="navy", lw=1.2, label="θ=0.35")
ax_d.set_xlabel("P(fail)")
ax_d.set_ylabel("Density")
ax_d.set_title("D — P(fail) Distribution by Outcome", fontweight="bold")
ax_d.legend(fontsize=8)

# ── Chart E: Per-task intervention count vs fail count ────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
task_list = [t for t in tasks.values()]
retry_counts = [t["n_retry"] for t in task_list]
fail_counts  = [t["n_fail"]  for t in task_list]
dom_c = [COLORS[t["domain"]] for t in task_list]
ax_e.scatter(fail_counts, retry_counts, c=dom_c, s=55, alpha=0.8, edgecolors="white", lw=0.5)
max_v = max(max(retry_counts), max(fail_counts)) + 0.5
ax_e.plot([0, max_v], [0, max_v], "k--", lw=0.8, alpha=0.4, label="retry = fail")
ax_e.set_xlabel("Actual failures per task")
ax_e.set_ylabel("Interventions (retries) per task")
ax_e.set_title("E — Retries vs Failures (per task)", fontweight="bold")
for d, c in COLORS.items():
    ax_e.scatter([], [], color=c, s=40, label=d)
ax_e.legend(fontsize=7, ncol=2)

# ── Chart F: Recall & Precision by domain ─────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
dom_recall = []
dom_prec   = []
for d in dom_names:
    dd = domains[d]
    dom_recall.append(dd["caught"]/dd["n_fail"] if dd["n_fail"] else 0.0)
    dom_prec.append(dd["caught"]/dd["n_retry"] if dd["n_retry"] else 0.0)

x = np.arange(len(dom_names))
ax_f.bar(x - 0.2, dom_recall, 0.38, label="Recall", color="#4C72B0", edgecolor="white")
ax_f.bar(x + 0.2, dom_prec,   0.38, label="Precision", color="#DD8452", edgecolor="white")
ax_f.set_xticks(x)
ax_f.set_xticklabels(dom_names, rotation=30)
ax_f.set_ylim(0, 1.2)
ax_f.set_ylabel("Score")
ax_f.set_title("F — Recall & Precision by Domain", fontweight="bold")
ax_f.legend(fontsize=8)
ax_f.axhline(overall_recall, ls="--", color="#4C72B0", lw=0.8, alpha=0.6)
ax_f.axhline(overall_prec,   ls="--", color="#DD8452",  lw=0.8, alpha=0.6)
for i, (r, p) in enumerate(zip(dom_recall, dom_prec)):
    ax_f.text(i-0.2, r+0.04, f"{r:.0%}", ha="center", fontsize=7)
    ax_f.text(i+0.2, p+0.04, f"{p:.0%}", ha="center", fontsize=7)

fig.suptitle(
    f"Large-Scale Integration Run — 20 Tasks × 6 Domains  |  "
    f"AUC={auc_score:.3f}  recall={overall_recall:.0%}  prec={overall_prec:.0%}",
    fontsize=12, fontweight="bold", y=0.98,
)

out = RESULTS / "8_large_run.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved → {out}")
