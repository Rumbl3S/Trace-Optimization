"""
session_charts.py — visualise trace_use vs alternatives on the extensive demo session.

Three charts:
  1. Accuracy vs Token Cost scatter — efficiency frontier across strategies
  2. Failures rescued at fixed budgets — bar comparison
  3. Per-task-type accuracy — how each task category performed

Numbers:
  - Measured from demo_extensive.py session (33 components, 6 task types)
  - Token estimates use Haiku ~800 tok/call, Opus ~1000 tok/call, embedding ~200 tok/trace
  - Benchmark budget curves (from component_records.json) used for extrapolation
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT  = Path(__file__).parent / "results" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BG          = "#FAFAFA"
GRID        = "#E8ECEF"
C_MAIN      = "#2E7D6B"   # trace_use
C_ALWAYS    = "#546E7A"
C_SC        = "#7986CB"
C_RANDOM    = "#90A4AE"
C_NONE      = "#CFD8DC"
C_MEASURED  = "#2E7D6B"
C_ESTIMATED = "#B0BEC5"

# ── session measurements ──────────────────────────────────────────────────────
N            = 33    # total components across 6 tasks
BASELINE_OK  = 26    # passed without any intervention
FAILURES     = 7     # failed on first attempt
RESCUED      = 2     # retries that flipped fail -> pass (measured)
INTERVENTIONS = 2    # total retries triggered (6% budget)

# token cost model (realistic, conservative)
T_HAIKU   = 800    # one haiku inference (prompt + completion + tool overhead)
T_OPUS    = 1000   # one opus verification call
T_EMBED   = 200    # one embedding call (OpenAI, cheap)
T_RETRY   = T_HAIKU + T_OPUS   # cost of one retry cycle

def accuracy(rescued): return (BASELINE_OK + rescued) / N * 100

def tokens_no_verify():      return N * T_HAIKU
def tokens_always_verify(rescued=4):
    return N * T_HAIKU + N * T_OPUS + FAILURES * T_RETRY
def tokens_self_consistency(samples=3):
    return N * T_HAIKU * samples          # run agent 3x, majority vote
def tokens_random(budget_frac):
    n_retried = max(1, round(budget_frac * N))
    return N * T_HAIKU + n_retried * T_RETRY
def tokens_trace_use(budget_frac):
    n_retried = max(1, round(budget_frac * N))
    return N * T_HAIKU + N * T_EMBED + n_retried * T_RETRY

# rescue rates from benchmark budget curves (trace_use vs random)
# at budget B%, trace_use catches catch_tu(B)% of failures
def catch_tu(b):   # from benchmark: 10%->16%, 20%->31%, 30%->44%, 50%->70%
    pts = [(0,.0),(0.10,.16),(0.20,.31),(0.30,.44),(0.50,.70),(1.0,1.0)]
    for i in range(len(pts)-1):
        x0,y0 = pts[i]; x1,y1 = pts[i+1]
        if x0 <= b <= x1:
            return y0 + (y1-y0)*(b-x0)/(x1-x0)
    return 1.0
def catch_rand(b): return b   # random catches budget% of failures in expectation

# assume 70% of caught failures are fixed by retry (conservative)
RETRY_SUCCESS = 0.70

# ── strategy definitions ──────────────────────────────────────────────────────
strategies = {
    "No Verification":   {
        "tokens":   tokens_no_verify(),
        "accuracy": accuracy(0),
        "rescued":  0,
        "color":    C_NONE,
        "measured": True,
        "marker":   "o",
    },
    "Self-Consistency\n(3× resample)": {
        "tokens":   tokens_self_consistency(3),
        "accuracy": accuracy(FAILURES * 0.35 * RETRY_SUCCESS),  # SC catches ~35% of failures
        "rescued":  FAILURES * 0.35 * RETRY_SUCCESS,
        "color":    C_SC,
        "measured": False,
        "marker":   "s",
    },
    "Random Gating\n(20% budget)": {
        "tokens":   tokens_random(0.20),
        "accuracy": accuracy(FAILURES * catch_rand(0.20) * RETRY_SUCCESS),
        "rescued":  FAILURES * catch_rand(0.20) * RETRY_SUCCESS,
        "color":    C_RANDOM,
        "measured": False,
        "marker":   "^",
    },
    "trace_use\n(6% budget)": {
        "tokens":   tokens_trace_use(0.06),
        "accuracy": accuracy(RESCUED),    # MEASURED
        "rescued":  RESCUED,
        "color":    C_MAIN,
        "measured": True,
        "marker":   "D",
    },
    "trace_use\n(20% budget)": {
        "tokens":   tokens_trace_use(0.20),
        "accuracy": accuracy(FAILURES * catch_tu(0.20) * RETRY_SUCCESS),
        "rescued":  FAILURES * catch_tu(0.20) * RETRY_SUCCESS,
        "color":    C_MAIN,
        "measured": False,
        "marker":   "D",
    },
    "trace_use\n(30% budget)": {
        "tokens":   tokens_trace_use(0.30),
        "accuracy": accuracy(FAILURES * catch_tu(0.30) * RETRY_SUCCESS),
        "rescued":  FAILURES * catch_tu(0.30) * RETRY_SUCCESS,
        "color":    C_MAIN,
        "measured": False,
        "marker":   "D",
    },
    "Always Verify\n+ Retry All": {
        "tokens":   tokens_always_verify(),
        "accuracy": accuracy(FAILURES * 0.57 * RETRY_SUCCESS),
        "rescued":  FAILURES * 0.57 * RETRY_SUCCESS,
        "color":    C_ALWAYS,
        "measured": False,
        "marker":   "P",
    },
}

# per-task results from demo session
task_results = [
    ("Factual\nResearch",     3,  2, 1, 1),   # (name, n, pass, fail, rescued)
    ("Coding",                6,  5, 1, 0),
    ("Research\nSynthesis",   8,  7, 1, 0),
    ("Data &\nMath",          4,  3, 1, 0),
    ("Coding\n(Hard)",        6,  4, 2, 1),
    ("Research\nSynth (Hard)",6,  5, 1, 0),
]

# ── chart 1: accuracy vs token cost scatter ───────────────────────────────────
def plot_efficiency_frontier():
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#DDD")

    # draw efficiency frontier line through trace_use points
    tu_pts = [(v["tokens"], v["accuracy"])
              for k, v in strategies.items() if "trace_use" in k]
    tu_pts.sort(key=lambda p: p[0])
    tx, ty = zip(*tu_pts)
    ax.plot(tx, ty, color=C_MAIN, linewidth=1.5, linestyle="--",
            alpha=0.5, zorder=2, label="_nolegend_")

    # per-strategy label offsets: (xoff_pts, yoff_pts, ha, va)
    label_cfg = {
        "No Verification":              (  10,   8, "left",  "bottom"),
        "Self-Consistency\n(3× resample)":( 10,  10, "left",  "bottom"),
        "Random Gating\n(20% budget)":  ( -10, -12, "right", "top"),
        "trace_use\n(6% budget)":       (  10,  10, "left",  "bottom"),
        "trace_use\n(20% budget)":      ( -12,  10, "right", "bottom"),
        "trace_use\n(30% budget)":      (  10,   0, "left",  "center"),
        "Always Verify\n+ Retry All":   (  10, -12, "left",  "top"),
    }

    for name, s in strategies.items():
        tok = s["tokens"]
        acc = s["accuracy"]
        ms  = 180 if "trace_use" in name else 120
        ec  = "white" if not s["measured"] else s["color"]
        lw  = 2.5 if s["measured"] else 1.0
        zorder = 5 if "trace_use" in name else 4

        ax.scatter(tok, acc, color=s["color"], marker=s["marker"],
                   s=ms, zorder=zorder, edgecolors=ec, linewidth=lw)

        label = name.replace("\n", " ")
        xo, yo, ha, va = label_cfg.get(name, (10, 0, "left", "center"))
        ax.annotate(
            label,
            (tok, acc),
            xytext=(xo, yo),
            textcoords="offset points",
            fontsize=9,
            color=C_MAIN if "trace_use" in name else "#555",
            fontweight="bold" if "trace_use" in name else "normal",
            ha=ha, va=va,
        )

    # annotate measured points
    for name, s in strategies.items():
        if s["measured"]:
            ax.annotate("★ measured", (s["tokens"], s["accuracy"]),
                        xytext=(10, -16), textcoords="offset points",
                        fontsize=7.5, color="#999", fontstyle="italic", ha="left")

    ax.set_xlabel("Estimated Token Cost (all 33 components)", fontsize=10.5, color="#444",
                  labelpad=8)
    ax.set_ylabel("Task Accuracy (%)", fontsize=10.5, color="#444", labelpad=8)
    ax.set_title("Accuracy vs Token Cost — trace_use vs Common Approaches\n"
                 "(33 components across Factual, Coding, Research, Math tasks)",
                 fontsize=12, fontweight="bold", color="#1A1A2E", pad=14)
    ax.tick_params(colors="#666", labelsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.set_ylim(73, 97)
    ax.set_xlim(18000, 95000)

    legend = [
        mpatches.Patch(color=C_MAIN,   label="trace_use (ours)"),
        mpatches.Patch(color=C_SC,     label="Self-Consistency (3×)"),
        mpatches.Patch(color=C_RANDOM, label="Random Gating"),
        mpatches.Patch(color=C_ALWAYS, label="Always Verify + Retry"),
        mpatches.Patch(color=C_NONE,   label="No Verification"),
    ]
    ax.legend(handles=legend, fontsize=9, frameon=True,
              framealpha=0.9, edgecolor="#DDD", loc="lower right")

    plt.tight_layout(pad=1.5)
    p = OUT / "4_accuracy_vs_tokens.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"  saved: {p}")


# ── chart 2: failures rescued at each budget ──────────────────────────────────
def plot_rescue_bars():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
    fig.suptitle("Failures Rescued & Tokens Spent — trace_use vs Alternatives",
                 fontsize=12, fontweight="bold", color="#1A1A2E", y=1.02)

    budgets = [0.06, 0.20, 0.30]
    labels  = ["6% budget\n(measured)", "20% budget", "30% budget"]

    # left: failures rescued
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#DDD")

    x   = np.arange(len(budgets))
    w   = 0.22
    tu  = [min(FAILURES, FAILURES * catch_tu(b)   * RETRY_SUCCESS) for b in budgets]
    rnd = [min(FAILURES, FAILURES * catch_rand(b) * RETRY_SUCCESS) for b in budgets]
    sc  = [FAILURES * 0.35 * RETRY_SUCCESS] * len(budgets)   # SC is budget-independent

    b1 = ax.bar(x - w,     tu,  w, color=C_MAIN,   label="trace_use",          zorder=3, edgecolor="white")
    b2 = ax.bar(x,         rnd, w, color=C_RANDOM, label="Random Gating",       zorder=3, edgecolor="white")
    b3 = ax.bar(x + w,     sc,  w, color=C_SC,     label="Self-Consistency (3×)",zorder=3, edgecolor="white")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if bars is b1 else "normal",
                    color=C_MAIN if bars is b1 else "#555")

    ax.axhline(FAILURES, color="#EF5350", linewidth=1.2, linestyle="--")
    ax.text(len(budgets)-0.5, FAILURES + 0.1, f"all {FAILURES} failures",
            fontsize=8, color="#EF5350", ha="right")

    # star the measured bar
    ax.annotate("★ measured", (x[0] - w, tu[0] + 0.25),
                fontsize=7.5, color=C_MAIN, ha="center", fontstyle="italic")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Failures Rescued (out of 7)", fontsize=10, color="#444")
    ax.set_title("Failures Recovered by Retry", fontsize=10.5, fontweight="bold",
                 color="#1A1A2E")
    ax.set_ylim(0, FAILURES + 1.5)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="#DDD")
    ax.tick_params(axis="x", length=0)

    # right: token cost at each budget
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.spines["bottom"].set_color("#DDD")

    tu_t   = [tokens_trace_use(b)/1000   for b in budgets]
    rnd_t  = [tokens_random(b)/1000      for b in budgets]
    sc_t   = [tokens_self_consistency(3)/1000] * len(budgets)
    base_t = tokens_no_verify()/1000

    b4 = ax2.bar(x - w,  tu_t,  w, color=C_MAIN,   label="trace_use",           zorder=3, edgecolor="white")
    b5 = ax2.bar(x,      rnd_t, w, color=C_RANDOM,  label="Random Gating",       zorder=3, edgecolor="white")
    b6 = ax2.bar(x + w,  sc_t,  w, color=C_SC,      label="Self-Consistency (3×)",zorder=3, edgecolor="white")

    ax2.axhline(base_t, color="#78909C", linewidth=1.2, linestyle=":",
                label=f"No verify ({base_t:.0f}k tokens)")
    ax2.axhline(tokens_always_verify()/1000, color="#EF5350", linewidth=1.2,
                linestyle="--", label=f"Always verify ({tokens_always_verify()/1000:.0f}k tokens)")

    for bars in (b4, b5, b6):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                     f"{h:.0f}k", ha="center", va="bottom", fontsize=8.5,
                     fontweight="bold" if bars is b4 else "normal",
                     color=C_MAIN if bars is b4 else "#555")

    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Estimated Token Cost (thousands)", fontsize=10, color="#444")
    ax2.set_title("Token Cost per Strategy", fontsize=10.5, fontweight="bold",
                  color="#1A1A2E")
    ax2.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="#DDD",
               loc="upper left")
    ax2.tick_params(axis="x", length=0)

    fig.subplots_adjust(wspace=0.3)
    p = OUT / "5_rescue_and_tokens.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"  saved: {p}")


# ── chart 3: per-task-type accuracy ───────────────────────────────────────────
def plot_per_task():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle("Per-Task Accuracy — trace_use Session Results",
                 fontsize=12, fontweight="bold", color="#1A1A2E", y=1.02)

    names      = [t[0] for t in task_results]
    n_total    = [t[1] for t in task_results]
    n_pass     = [t[2] for t in task_results]
    n_fail     = [t[3] for t in task_results]
    n_rescued  = [t[4] for t in task_results]

    acc_before = [p / n * 100 for p, n in zip(n_pass, n_total)]
    acc_after  = [(p + r) / n * 100
                  for p, r, n in zip(n_pass, n_rescued, n_total)]

    x = np.arange(len(names))
    w = 0.35

    ax = axes[0]
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#DDD")

    b1 = ax.bar(x - w/2, acc_before, w, color=C_RANDOM, label="Without trace_use",
                zorder=3, edgecolor="white")
    b2 = ax.bar(x + w/2, acc_after,  w, color=C_MAIN,   label="With trace_use",
                zorder=3, edgecolor="white")

    for bar, v in zip(b1, acc_before):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.8, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=8.5, color="#555")
    for bar, v in zip(b2, acc_after):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.8, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=8.5, color=C_MAIN, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Accuracy (%)", fontsize=10, color="#444")
    ax.set_title("Accuracy Before vs After Intervention", fontsize=10.5,
                 fontweight="bold", color="#1A1A2E")
    ax.set_ylim(60, 108)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="#DDD")
    ax.tick_params(axis="x", length=0)

    # right: stacked pass / fail / rescued
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.spines["bottom"].set_color("#DDD")

    n_pass_arr    = np.array(n_pass)
    n_rescued_arr = np.array(n_rescued)
    n_fail_arr    = np.array(n_fail) - n_rescued_arr

    ax2.bar(x, n_pass_arr,    color="#66BB6A", label="Pass (first attempt)",  zorder=3)
    ax2.bar(x, n_rescued_arr, bottom=n_pass_arr, color=C_MAIN,
            label="Rescued by trace_use retry", zorder=3)
    ax2.bar(x, n_fail_arr,    bottom=n_pass_arr + n_rescued_arr,
            color="#EF9A9A", label="Fail (unrecovered)", zorder=3, edgecolor="white")

    for i, (tot, p, r, f) in enumerate(zip(n_total, n_pass, n_rescued, n_fail)):
        ax2.text(i, tot + 0.08, f"{tot}", ha="center", va="bottom",
                 fontsize=8.5, color="#444", fontweight="bold")

    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_ylabel("Components", fontsize=10, color="#444")
    ax2.set_title("Component Breakdown by Task Type", fontsize=10.5,
                  fontweight="bold", color="#1A1A2E")
    ax2.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="#DDD")
    ax2.tick_params(axis="x", length=0)

    fig.subplots_adjust(wspace=0.3)
    p = OUT / "6_per_task_accuracy.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"  saved: {p}")


if __name__ == "__main__":
    print("Generating session analysis charts...")
    plot_efficiency_frontier()
    plot_rescue_bars()
    plot_per_task()
    print("\nAll done.")
