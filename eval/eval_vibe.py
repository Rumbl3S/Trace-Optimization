"""eval/eval_vibe.py — Large-scale vibecoding brain evaluation.

12-task expression parser build, one feature per task.
Tasks 1-4 are designed to PASS (basic ops); tasks 5-12 contain hidden edge-case
tests that haiku consistently fails (** right-assoc, unary-minus precedence, etc.).

Produces:
  eval/results/vibe_timeline.png   — p_fail over the session, fires marked
  eval/results/vibe_embedding.png  — PCA of trace embeddings (pass vs fail)
  eval/results/vibe_roc.png        — ROC curve + AUC / Spearman
  eval/results/vibe_run.json       — raw results ledger

Run:
    python eval/eval_vibe.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agents import _build_openai
from brain import BrainAgent

# ── constants ─────────────────────────────────────────────────────────────────
THRESHOLD = 0.25
K         = 5
MODEL     = "claude-haiku-4-5-20251001"
OUT       = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

NO_EVAL = (
    "Do NOT use Python's eval(), compile(), or exec(). "
    "Use standalone functions only — no classes. "
    "The entry point must be a top-level function, not a method."
)

_SYS = (
    "You are a Python coding assistant. "
    "Output a SINGLE ```python ... ``` code block with the complete implementation. "
    "Embed any reasoning as Python comments inside the block. "
    "Do not write BNF grammars or explanatory prose outside the code block."
)

COT = (
    "\n\nWork through your approach step by step as Python comments. "
    "Then provide the complete implementation in a ```python code block."
)

# ── LLM call ──────────────────────────────────────────────────────────────────

def _call(prompt: str) -> tuple[str, int]:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=_SYS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    return text, msg.usage.input_tokens + msg.usage.output_tokens


# ── code extraction ────────────────────────────────────────────────────────────

def _extract_code(text: str) -> str | None:
    closed   = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    unclosed = re.findall(r"```(?:python|py)[^\n]*\n(.*?)$", text, re.DOTALL)
    blocks   = closed + unclosed
    if not blocks:
        return None

    def _ok(b: str) -> bool:
        try:
            compile(b, "<c>", "exec")
            return True
        except Exception:
            return False

    for b in blocks:
        b = b.strip()
        if ("def " in b or "class " in b) and _ok(b):
            return b
    for b in blocks:
        b = b.strip()
        if _ok(b):
            return b
    return None


def _run(code: str, check_fn: Callable) -> tuple[bool, str]:
    ns: dict = {}
    try:
        exec(compile(code, "<vibe>", "exec"), ns)
    except Exception as e:
        return False, f"Compile error: {e}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return bool(check_fn(ns)), ""
    except Exception as e:
        return False, f"Test error: {e}"


# ── check functions ────────────────────────────────────────────────────────────

def _ev(ns) -> Callable | None:
    return ns.get("evaluate")


def check_basic(ns) -> bool:
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("2 + 3") - 5.0) < 1e-9 and
            abs(ev("10 - 4") - 6.0) < 1e-9 and
            abs(ev("3 * 4") - 12.0) < 1e-9 and
            abs(ev("10 / 2") - 5.0) < 1e-9 and
            abs(ev("2 + 3 * 4") - 14.0) < 1e-9
        )
    except Exception:
        return False


def check_parens(ns) -> bool:
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("(2 + 3) * 4") - 20.0) < 1e-9 and
            abs(ev("10 - 3 - 2") - 5.0) < 1e-9 and      # left-assoc
            abs(ev("12 / 4 / 3") - 1.0) < 1e-9 and      # left-assoc
            abs(ev("2 * (3 + 4) * 5") - 70.0) < 1e-9
        )
    except Exception:
        return False


def check_power_basic(ns) -> bool:
    """Task 3: ** exists, only trivial cases (no right-assoc trap yet)."""
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("2 ** 3") - 8.0) < 1e-9 and
            abs(ev("3 ** 2") - 9.0) < 1e-9 and
            abs(ev("2 + 3 ** 2") - 11.0) < 1e-9 and
            abs(ev("(2 + 1) ** 3") - 27.0) < 1e-9
        )
    except Exception:
        return False


def check_unary_plus_float(ns) -> bool:
    """Task 4: unary + and float literals."""
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("+3") - 3.0) < 1e-9 and
            abs(ev("1.5 + 2.5") - 4.0) < 1e-9 and
            abs(ev("3.14 * 2") - 6.28) < 1e-9 and
            abs(ev("+(2 + 3)") - 5.0) < 1e-9
        )
    except Exception:
        return False


def check_power_right_assoc(ns) -> bool:
    """Task 5 HIDDEN: ** must be right-associative. LLMs almost always fail."""
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("2 ** 3 ** 2") - 512.0) < 1e-9 and   # 2**(3**2)=512 NOT 64
            abs(ev("2 ** 2 ** 3") - 256.0) < 1e-9 and   # 2**(2**3)=256 NOT 64
            abs(ev("(2 ** 3) ** 2") - 64.0) < 1e-9 and  # parens override
            abs(ev("4 ** 3 ** 2") - 262144.0) < 1e-9    # 4**(3**2)=4**9
        )
    except Exception:
        return False


def check_unary_minus(ns) -> bool:
    """Task 6 HIDDEN: unary minus has LOWER precedence than **."""
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("-3") - (-3.0)) < 1e-9 and
            abs(ev("--3") - 3.0) < 1e-9 and
            abs(ev("-(2 + 3)") - (-5.0)) < 1e-9 and
            abs(ev("-2 ** 2") - (-4.0)) < 1e-9 and       # -(2**2)=-4 NOT (-2)**2=4
            abs(ev("2 ** -1") - 0.5) < 1e-9 and           # right side can be negative
            abs(ev("-2 ** 2 + 1") - (-3.0)) < 1e-9
        )
    except Exception:
        return False


def check_comparisons(ns) -> bool:
    """Task 7 HIDDEN: comparisons + prior edge cases."""
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("3 > 2") - 1.0) < 1e-9 and
            abs(ev("2 > 3") - 0.0) < 1e-9 and
            abs(ev("2 + 3 > 4") - 1.0) < 1e-9 and
            abs(ev("-2 ** 2 < 0") - 1.0) < 1e-9 and      # -4 < 0
            abs(ev("10 - 3 - 2 == 5") - 1.0) < 1e-9 and  # left-assoc
            abs(ev("2 ** 3 ** 2 == 512") - 1.0) < 1e-9   # right-assoc
        )
    except Exception:
        return False


def check_variables(ns) -> bool:
    """Task 8 HIDDEN: variables with prior edge cases."""
    ev = _ev(ns)
    if not ev: return False
    try:
        assert abs(ev("x + 1", {"x": 3}) - 4.0) < 1e-9
        assert abs(ev("-x ** 2", {"x": 3}) - (-9.0)) < 1e-9    # -(3**2)=-9
        assert abs(ev("x ** y ** z", {"x": 2, "y": 3, "z": 2}) - 512.0) < 1e-9
        assert abs(ev("10 - a - b == 5", {"a": 3, "b": 2}) - 1.0) < 1e-9
        try:
            ev("undef", {})
            return False
        except (NameError, KeyError):
            return True
    except Exception:
        return False


def check_modulo(ns) -> bool:
    """Task 9 HIDDEN: modulo + right-assoc."""
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("10 % 3") - 1.0) < 1e-9 and
            abs(ev("10 % 3 + 1") - 2.0) < 1e-9 and
            abs(ev("2 ** 3 % 3") - 2.0) < 1e-9 and       # (2**3)%3 = 8%3 = 2
            abs(ev("-7 % 3") - 2.0) < 1e-9                # Python semantics
        )
    except Exception:
        return False


def check_floor_div(ns) -> bool:
    """Task 10 HIDDEN: floor division + right-assoc."""
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("7 // 2") - 3.0) < 1e-9 and
            abs(ev("2 ** 3 // 3") - 2.0) < 1e-9 and      # 8//3=2
            abs(ev("-7 // 2") - (-4.0)) < 1e-9 and        # floor division
            abs(ev("10 // 3 // 1") - 3.0) < 1e-9          # left-assoc
        )
    except Exception:
        return False


def check_builtins(ns) -> bool:
    """Task 11 HIDDEN: abs() and round() callable inside expression."""
    ev = _ev(ns)
    if not ev: return False
    try:
        return (
            abs(ev("abs(-3)") - 3.0) < 1e-9 and
            abs(ev("abs(-2 ** 2)") - 4.0) < 1e-9 and     # abs(-(2**2))=abs(-4)=4
            abs(ev("round(3.7)") - 4.0) < 1e-9 and
            abs(ev("abs(-3) + round(1.4)") - 4.0) < 1e-9
        )
    except Exception:
        return False


def check_integration(ns) -> bool:
    """Task 12 HIDDEN: all features combined."""
    ev = _ev(ns)
    if not ev: return False
    try:
        # right-assoc + unary minus + variables
        assert abs(ev("abs(-x) ** y ** z", {"x": 2, "y": 3, "z": 2}) - 512.0) < 1e-9
        # comparisons + variables
        assert abs(ev("x > y", {"x": 5, "y": 3}) - 1.0) < 1e-9
        # modulo + floor div
        assert abs(ev("2 ** 3 % 3") - 2.0) < 1e-9
        assert abs(ev("7 // 2") - 3.0) < 1e-9
        # negative unary
        assert abs(ev("-2 ** 2") - (-4.0)) < 1e-9
        return True
    except Exception:
        return False


# ── task definitions ──────────────────────────────────────────────────────────

TASKS = [
    {
        "name": "evaluate: +,-,*,/",
        "prompt": (
            "Write a Python function `evaluate(expr: str) -> float` that parses and "
            "evaluates arithmetic expressions with +, -, *, /. "
            "Respect operator precedence (* and / before + and -). "
            f"{NO_EVAL}"
        ),
        "check": check_basic,
    },
    {
        "name": "evaluate: parentheses + left-assoc",
        "prompt": (
            "Write a Python function `evaluate(expr: str) -> float` that handles "
            "+, -, *, / with parentheses and correct left-to-right associativity. "
            "Example: 10-3-2=5 (not 9), 12/4/3=1 (not 9). "
            f"{NO_EVAL}"
        ),
        "check": check_parens,
    },
    {
        "name": "evaluate: add ** power",
        "prompt": (
            "Write a Python function `evaluate(expr: str) -> float` that supports "
            "+, -, *, /, ** (power). ** has higher precedence than *, /, +, -. "
            f"{NO_EVAL}"
        ),
        "check": check_power_basic,
    },
    {
        "name": "evaluate: unary + and floats",
        "prompt": (
            "Write a Python function `evaluate(expr: str) -> float` that supports "
            "+, -, *, /, ** plus unary + operator and floating-point number literals. "
            "Example: +3=3, 1.5+2.5=4. "
            f"{NO_EVAL}"
        ),
        "check": check_unary_plus_float,
    },
    {
        "name": "** right-associative [HIDDEN]",
        "prompt": (
            "Extend your `evaluate` function so that ** is right-associative "
            "(as in Python and mathematics). "
            f"{NO_EVAL}"
        ),
        "check": check_power_right_assoc,
    },
    {
        "name": "unary minus: -2**2=-4 [HIDDEN]",
        "prompt": (
            "Extend your `evaluate` function to support unary minus. "
            "Examples: -3=-3, -(2+3)=-5. "
            f"{NO_EVAL}"
        ),
        "check": check_unary_minus,
    },
    {
        "name": "comparison ops [HIDDEN]",
        "prompt": (
            "Extend `evaluate` to support comparison operators: < > <= >= == != "
            "Return 1.0 for True and 0.0 for False. "
            "Comparisons have lower precedence than arithmetic. "
            f"{NO_EVAL}"
        ),
        "check": check_comparisons,
    },
    {
        "name": "variables dict [HIDDEN]",
        "prompt": (
            "Extend `evaluate` to accept `evaluate(expr, variables=None) -> float`. "
            "Variable names match [a-zA-Z_][a-zA-Z0-9_]*. Raise NameError for undefined. "
            f"{NO_EVAL}"
        ),
        "check": check_variables,
    },
    {
        "name": "modulo % [HIDDEN]",
        "prompt": (
            "Extend `evaluate` to support the modulo operator % with the same "
            "precedence as * and /. Use Python floor-division semantics. "
            f"{NO_EVAL}"
        ),
        "check": check_modulo,
    },
    {
        "name": "floor division // [HIDDEN]",
        "prompt": (
            "Extend `evaluate` to support floor division // with the same "
            "precedence as * and /. Use Python semantics (-7//2=-4). "
            f"{NO_EVAL}"
        ),
        "check": check_floor_div,
    },
    {
        "name": "abs() and round() [HIDDEN]",
        "prompt": (
            "Extend `evaluate` so that abs() and round() can be called as functions "
            "inside expressions, e.g. abs(-3)=3, round(3.7)=4. "
            f"{NO_EVAL}"
        ),
        "check": check_builtins,
    },
    {
        "name": "full integration [HIDDEN]",
        "prompt": (
            "Write a complete `evaluate(expr, variables=None) -> float` that supports "
            "all features: +,-,*,/,**,%,// with correct precedence and associativity, "
            "unary + and -, parentheses, comparisons, variable lookup, abs(), round(). "
            f"{NO_EVAL}"
        ),
        "check": check_integration,
    },
]


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    idx:    int
    name:   str
    label:  int
    p_fail: float | None
    fired:  bool
    detail: str
    vec:    np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


# ── diagnostics ───────────────────────────────────────────────────────────────

def _diagnose(code: str) -> str:
    ns: dict = {}
    try:
        exec(compile(code, "<v>", "exec"), ns)
    except Exception:
        return "exec error"
    ev = ns.get("evaluate")
    if not ev:
        return "evaluate() not defined"
    probes = [
        ("2**3**2",  512.0, "** right-assoc: 2**(3**2)=512"),
        ("-2**2",    -4.0,  "unary<**: -(2**2)=-4"),
        ("10-3-2",   5.0,   "subtraction left-assoc"),
        ("2**2**3",  256.0, "** right-assoc: 2**(2**3)=256"),
    ]
    hits = []
    for expr, expected, label in probes:
        try:
            got = ev(expr)
            if abs(float(got) - expected) > 1e-9:
                hits.append(f"{expr}={got}≠{expected} [{label}]")
        except Exception as exc:
            hits.append(f"{expr}→error({exc})")
    return "; ".join(hits[:2]) if hits else "unknown failure"


# ── main run ──────────────────────────────────────────────────────────────────

def run() -> list[TaskResult]:
    embedder = _build_openai()
    brain    = BrainAgent(embedder, k=K, threshold=THRESHOLD,
                          check_interval=9999, min_chars=999999)

    results: list[TaskResult] = []

    print(f"\n{'─'*72}")
    print(f"  VibeSession — 12-task expression parser  |  threshold={THRESHOLD}")
    print(f"  model={MODEL}")
    print(f"{'─'*72}\n")

    for i, task in enumerate(TASKS):
        n = i + 1
        print(f"Task {n:>2}/12: {task['name']}")

        # Pre-generation brain query
        p_fail, warning = brain.failure_store.query(task["prompt"])
        fired = warning is not None

        prompt = task["prompt"]
        if warning:
            prompt = (
                f"[BRAIN WARNING — P(fail)={p_fail:.0%}]\n"
                f"A previous similar task failed:\n{warning}\n\n"
                f"Task: {prompt}"
            )
            print(f"          ⚡ BRAIN fired  p_fail={p_fail:.2f}")
        elif p_fail is not None:
            print(f"          p_fail={p_fail:.2f}")

        # Generate
        text, tokens = _call(prompt + COT)

        # Verify
        code = _extract_code(text)
        if code is None:
            passed, detail = False, "No code block found"
        else:
            passed, detail = _run(code, task["check"])
            if not passed and not detail and code:
                detail = _diagnose(code)

        label = int(passed)
        brain.store(text, label)

        status = "PASS" if passed else "FAIL"
        print(f"          {status}  {detail[:60] if detail else ''}")

        # Grab the embedding just stored
        store  = brain.failure_store
        vec    = store._traces[-1].vec if store._traces else np.array([])

        results.append(TaskResult(
            idx=n, name=task["name"], label=label,
            p_fail=p_fail, fired=fired, detail=detail, vec=vec,
        ))

    return results


# ── graphs ────────────────────────────────────────────────────────────────────

def plot_timeline(results: list[TaskResult], threshold: float) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        pass

    xs      = [r.idx for r in results]
    p_vals  = [r.p_fail for r in results]
    labels  = [r.label for r in results]
    fired   = [r.fired for r in results]

    # Shaded danger zone
    ax.axhspan(threshold, 1.05, alpha=0.08, color="gold", zorder=0)
    ax.axhline(threshold, color="goldenrod", linestyle="--", linewidth=1.2,
               label=f"threshold = {threshold}", zorder=1)

    # Connect non-None p_fail points
    xs_known = [r.idx for r in results if r.p_fail is not None]
    pf_known = [r.p_fail for r in results if r.p_fail is not None]
    if xs_known:
        ax.plot(xs_known, pf_known, color="steelblue", linewidth=1.5,
                marker="o", markersize=5, zorder=2)

    # Per-task markers
    for r in results:
        pf = r.p_fail if r.p_fail is not None else -0.04
        c  = "green" if r.label == 1 else "red"
        mk = "o" if r.label == 1 else "X"
        ax.scatter(r.idx, pf, s=120, color=c, marker=mk, zorder=4,
                   edgecolors="white", linewidths=0.8)
        if r.fired:
            ax.scatter(r.idx, r.p_fail, s=260, marker="*",
                       color="gold", edgecolors="darkorange",
                       linewidths=0.8, zorder=5)
            ax.annotate("⚡fired", (r.idx, r.p_fail),
                        textcoords="offset points", xytext=(4, 8),
                        fontsize=8, color="darkorange")

    ax.set_xlim(0.5, len(results) + 0.5)
    ax.set_ylim(-0.08, 1.05)
    ax.set_xticks(range(1, len(results) + 1))
    ax.set_xticklabels([f"{r.idx}" for r in results])
    ax.set_xlabel("Task #", fontsize=11)
    ax.set_ylabel("p_fail (before generation)", fontsize=11)
    ax.set_title("Brain p_fail over vibecoding session", fontsize=13, fontweight="bold")

    pass_patch = mpatches.Patch(color="green", label="PASS")
    fail_patch = mpatches.Patch(color="red",   label="FAIL")
    fire_patch = mpatches.Patch(color="gold",  label="⚡ brain fired")
    ax.legend(handles=[pass_patch, fail_patch, fire_patch,
                        mpatches.Patch(color="goldenrod", alpha=0.4,
                                       label=f"threshold={threshold}")],
              loc="upper left", fontsize=9)

    plt.tight_layout()
    out = OUT / "vibe_timeline.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


def plot_embedding(results: list[TaskResult]) -> None:
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    vecs   = np.stack([r.vec for r in results])
    labels = [r.label for r in results]

    pca = PCA(n_components=2)
    xy  = pca.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(8, 7))
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        pass

    for i, r in enumerate(results):
        x, y  = xy[i]
        c     = "green" if r.label == 1 else "red"
        mk    = "o" if r.label == 1 else "X"
        sz    = 80 + i * 12
        ax.scatter(x, y, s=sz, color=c, marker=mk,
                   edgecolors="white", linewidths=0.7, zorder=3, alpha=0.85)
        ax.annotate(str(r.idx), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=9, color="white" if r.label == 0 else "black")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)", fontsize=10)
    ax.set_title(
        "Trace embedding space (PCA)\nPasses cluster separately from failures",
        fontsize=12, fontweight="bold",
    )

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="green", label="PASS"),
        mpatches.Patch(color="red",   label="FAIL"),
    ], fontsize=10)

    plt.tight_layout()
    out = OUT / "vibe_embedding.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_roc(results: list[TaskResult]) -> None:
    from sklearn.metrics import roc_curve, roc_auc_score
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt

    # Only tasks where p_fail was computed
    known = [r for r in results if r.p_fail is not None]
    if len(known) < 4:
        print("Not enough p_fail values for ROC curve.")
        return

    y_true  = [r.label for r in known]
    scores  = [1.0 - r.p_fail for r in known]   # higher = more likely pass

    try:
        auc_val = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
    except Exception as e:
        print(f"ROC skipped: {e}")
        return

    p_fails = [r.p_fail for r in known]
    errors  = [1 - r.label for r in known]
    sp_r    = spearmanr(p_fails, errors).statistic

    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        pass

    ax.plot(fpr, tpr, color="steelblue", linewidth=2.5,
            label=f"Brain (AUC={auc_val:.2f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance (AUC=0.50)")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        f"Brain Failure Forecaster\nAUC={auc_val:.2f}  Spearman={sp_r:.2f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = OUT / "vibe_roc.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    return auc_val, sp_r


# ── JSON ledger ───────────────────────────────────────────────────────────────

def save_json(results: list[TaskResult], auc: float | None, spearman: float | None) -> None:
    n_pass  = sum(r.label for r in results)
    n_fail  = len(results) - n_pass
    n_fires = sum(r.fired for r in results)

    data = {
        "n_tasks":  len(results),
        "n_pass":   n_pass,
        "n_fail":   n_fail,
        "n_fires":  n_fires,
        "auc":      round(auc, 4) if auc is not None else None,
        "spearman": round(spearman, 4) if spearman is not None else None,
        "tasks": [
            {
                "idx":    r.idx,
                "name":   r.name,
                "label":  r.label,
                "p_fail": round(r.p_fail, 4) if r.p_fail is not None else None,
                "fired":  r.fired,
                "detail": r.detail,
            }
            for r in results
        ],
    }
    out = OUT / "vibe_run.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"Saved {out}")
    return data


def print_table(results: list[TaskResult]) -> None:
    print(f"\n{'─'*72}")
    print(f"  {'#':>2}  {'Task':<34}  {'Result':6}  {'p_fail':7}  {'Fire':5}")
    print(f"{'─'*72}")
    for r in results:
        status = "PASS" if r.label else "FAIL"
        pf     = f"{r.p_fail:.2f}" if r.p_fail is not None else "  —  "
        fire   = "⚡" if r.fired else "  "
        print(f"  {r.idx:>2}  {r.name[:34]:<34}  {status:6}  {pf:7}  {fire}")
    n_pass  = sum(r.label for r in results)
    n_fires = sum(r.fired for r in results)
    print(f"{'─'*72}")
    print(f"  {n_pass}/{len(results)} passed  |  {n_fires} brain fires")
    print(f"{'─'*72}\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run()
    print_table(results)

    auc_val = sp_val = None
    try:
        plot_timeline(results, THRESHOLD)
        plot_embedding(results)
        roc_out = plot_roc(results)
        if roc_out:
            auc_val, sp_val = roc_out
    except ImportError as e:
        print(f"[graph skipped] {e}")

    data = save_json(results, auc_val, sp_val)
    print(f"\n  AUC={auc_val:.2f}  Spearman={sp_val:.2f}" if auc_val else "")
    print("Done.\n")
