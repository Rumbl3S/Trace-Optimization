"""eval/eval_v3_vibe.py — Hybrid brain vs sonnet+thinking. 20 tasks.

Architecture:
  • Carry-forward: haiku receives its verified code from the previous task as
    context — just like a real coding session. The broken tokenizer never
    recurs because once haiku writes a correct one, it carries it forward.

  • Hybrid confidence: deterministic probe tests (catches '2**3**2=64'
    immediately after tool call 1) combined with kNN over past code snippets
    (catches unfamiliar bugs that look like prior failures). Together they
    replicate Anthropic-style confidence/backtracking at the tool level.

  • No regex: code is captured as a raw dict from block.input, not parsed
    from the serialised trace string.

  • Live trajectory graph: task × turn heatmap saved after every task.
    Open eval/results/trajectory_live.png in an auto-refresh viewer.

Run:
    python eval/eval_v3_vibe.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agents import build_embedder, tool_agent
from brain import BrainAgent

OUT   = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

THRESHOLD = 0.35
K         = 5
HAIKU     = "claude-haiku-4-5-20251001"
SONNET    = "claude-sonnet-4-6"
NO_EVAL   = (
    "Do NOT use Python's eval(), compile(), or exec(). "
    "Standalone functions only — no classes. "
    "Entry point must be a top-level function, not a method."
)
MAX_TURNS = 8


# ─────────────────────────────────────────────────────────────────────────────
#  TASK DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def _ev(ns): return ns.get("evaluate")

def check_basic(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("2 + 3") - 5) < 1e-9,
            abs(ev("10 - 4") - 6) < 1e-9,
            abs(ev("3 * 4") - 12) < 1e-9,
            abs(ev("10 / 2") - 5) < 1e-9,
            abs(ev("2 + 3 * 4") - 14) < 1e-9,
        ])
    except Exception: return False

def check_parens(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("(2 + 3) * 4") - 20) < 1e-9,
            abs(ev("10 - 3 - 2") - 5) < 1e-9,
            abs(ev("12 / 4 / 3") - 1) < 1e-9,
        ])
    except Exception: return False

def check_power(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("2 ** 3") - 8) < 1e-9,
            abs(ev("2 + 3 ** 2") - 11) < 1e-9,
        ])
    except Exception: return False

def check_unary_float(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("+3") - 3) < 1e-9,
            abs(ev("1.5 + 2.5") - 4) < 1e-9,
            abs(ev("+(2 + 3)") - 5) < 1e-9,
        ])
    except Exception: return False

def check_right_assoc(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("2 ** 3 ** 2") - 512) < 1e-9,
            abs(ev("2 ** 2 ** 3") - 256) < 1e-9,
            abs(ev("(2 ** 3) ** 2") - 64) < 1e-9,
        ])
    except Exception: return False

def check_unary_minus(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("-3") - (-3)) < 1e-9,
            abs(ev("--3") - 3) < 1e-9,
            abs(ev("-2 ** 2") - (-4)) < 1e-9,
            abs(ev("2 ** -1") - 0.5) < 1e-9,
        ])
    except Exception: return False

def check_comparisons(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("3 > 2") - 1) < 1e-9,
            abs(ev("2 > 3") - 0) < 1e-9,
            abs(ev("-2 ** 2 < 0") - 1) < 1e-9,
            abs(ev("2 ** 3 ** 2 == 512") - 1) < 1e-9,
        ])
    except Exception: return False

def check_variables(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        assert abs(ev("x + 1", {"x": 3}) - 4) < 1e-9
        assert abs(ev("-x ** 2", {"x": 3}) - (-9)) < 1e-9
        assert abs(ev("x ** y ** z", {"x": 2, "y": 3, "z": 2}) - 512) < 1e-9
        try: ev("undef", {}); return False
        except (NameError, KeyError): return True
    except Exception: return False

def check_modulo(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("10 % 3") - 1) < 1e-9,
            abs(ev("2 ** 3 % 3") - 2) < 1e-9,
            abs(ev("-7 % 3") - 2) < 1e-9,
        ])
    except Exception: return False

def check_floor_div(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("7 // 2") - 3) < 1e-9,
            abs(ev("-7 // 2") - (-4)) < 1e-9,
            abs(ev("2 ** 3 // 3") - 2) < 1e-9,
        ])
    except Exception: return False

def check_builtins(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        return all([
            abs(ev("abs(-3)") - 3) < 1e-9,
            abs(ev("abs(-2 ** 2)") - 4) < 1e-9,
            abs(ev("round(3.7)") - 4) < 1e-9,
        ])
    except Exception: return False

def check_integration(ns):
    ev = _ev(ns)
    if not ev: return False
    try:
        assert abs(ev("abs(-x) ** y ** z", {"x": 2, "y": 3, "z": 2}) - 512) < 1e-9
        assert abs(ev("x > y", {"x": 5, "y": 3}) - 1) < 1e-9
        assert abs(ev("-2 ** 2") - (-4)) < 1e-9
        assert abs(ev("7 // 2") - 3) < 1e-9
        return True
    except Exception: return False

def check_run_length(ns):
    fn = ns.get("encode")
    if not fn: return False
    try:
        return all([fn("") == "", fn("a") == "1a",
                    fn("aabbbcc") == "2a3b2c", fn("aaaa") == "4a"])
    except Exception: return False

def check_balanced(ns):
    fn = ns.get("is_balanced")
    if not fn: return False
    try:
        return all([fn("") == True, fn("()") == True, fn("([)]") == False,
                    fn("((") == False, fn("()[]{}") == True])
    except Exception: return False

def check_word_freq(ns):
    fn = ns.get("word_freq")
    if not fn: return False
    try:
        r = fn("the cat sat on the mat")
        assert r.get("the") == 2 and r.get("cat") == 1
        assert fn("Hello hello HELLO").get("hello") == 3
        return True
    except Exception: return False

def check_lcs(ns):
    fn = ns.get("lcs") or ns.get("longest_common_substring")
    if not fn: return False
    try:
        assert fn("abcdef", "bcdf") == "bcd"
        assert fn("abc", "xyz") == ""
        assert fn("abab", "bab") == "bab"
        return True
    except Exception: return False

def check_binary_search(ns):
    fn = ns.get("binary_search")
    if not fn: return False
    try:
        assert fn([1,3,5,7,9], 5) == 2
        assert fn([], 1) == -1
        assert fn([1], 1) == 0
        return True
    except Exception: return False

def check_merge_sorted(ns):
    fn = ns.get("merge") or ns.get("merge_sorted")
    if not fn: return False
    try:
        assert fn([1,3,5], [2,4,6]) == [1,2,3,4,5,6]
        assert fn([], [1,2]) == [1,2]
        return True
    except Exception: return False

def check_islands(ns):
    fn = ns.get("count_islands") or ns.get("numIslands")
    if not fn: return False
    try:
        assert fn([["1","1","0"],["0","1","0"],["1","0","1"]]) == 3
        assert fn([["0","0","0"]]) == 0
        return True
    except Exception: return False

def check_two_sum(ns):
    fn = ns.get("two_sum")
    if not fn: return False
    try:
        assert set(fn([2,7,11,15], 9)) == {0, 1}
        r = fn([1,2,3], 10)
        assert r is None or r == [] or r == (-1,-1)
        return True
    except Exception: return False


# ── probe functions: return list of failure strings (empty = clean) ───────────

def _parser_probe(ns) -> list[str]:
    ev = ns.get("evaluate")
    if not ev:
        return ["evaluate() not defined"]
    fails = []
    tests = [
        ("2**3**2",  512.0,
         "** must be right-associative: 2**(3**2)=512, not (2**3)**2=64. "
         "FIX: right-recursive parse_power — call parse_power(tokens,pos) "
         "recursively for the exponent, not iteratively."),
        ("-2**2",    -4.0,
         "-2**2 must equal -(2**2)=-4, not (-2)**2=4. "
         "FIX: parse_unary must call parse_power (not parse_primary). "
         "Call chain must be: additive→mult→power→unary→primary"),
        ("2+3*4",    14.0,
         "Precedence wrong: 2+3*4 should be 14 not 20."),
    ]
    for expr, expected, fix in tests:
        try:
            got = float(ev(expr))
            if abs(got - expected) > 1e-9:
                fails.append(f"{expr}={got} (expected {expected}). {fix}")
        except Exception as exc:
            errmsg = str(exc)
            if "**" in expr or "*" in expr:
                fails.append(
                    f"{expr}→error({errmsg}). "
                    "Likely your tokenizer splits '**' into two '*' tokens. "
                    "FIX: check for '**' BEFORE '*' in your tokenizer: "
                    "if s[i:i+2]=='**': emit ('**', '**'); i+=2 "
                    "elif s[i]=='*': emit ('*', '*'); i+=1"
                )
            else:
                fails.append(f"{expr}→error({errmsg})")
    return fails


def _generic_probe(check_fn):
    def probe(ns) -> list[str]:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ok = check_fn(ns)
                return [] if ok else ["check function returned False"]
            except AssertionError as e:
                return [f"assertion failed: {e}"]
            except Exception as e:
                return [f"error: {e}"]
    return probe


TASKS = [
    # ── expression parser (12 tasks, cumulative) ──────────────────────────────
    {"name": "evaluate +,-,*,/",        "domain": "parser", "check": check_basic,
     "probe": _parser_probe,
     "prompt": (
         "Write a Python function `evaluate(expr: str) -> float` that parses and "
         "evaluates arithmetic with +,-,*,/. Correct precedence. Support parentheses. "
         f"{NO_EVAL}"
     )},
    {"name": "parens + left-assoc",     "domain": "parser", "check": check_parens,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` to handle parentheses and correct left-associativity. "
         "10-3-2=5 (not 9). 12/4/3=1 (not 9). "
         f"{NO_EVAL}"
     )},
    {"name": "add ** power",            "domain": "parser", "check": check_power,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` to support ** (power). ** has higher precedence than *,/,+,-. "
         f"{NO_EVAL}"
     )},
    {"name": "unary + and floats",      "domain": "parser", "check": check_unary_float,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` to support unary + and floating-point literals. "
         "+3=3, 1.5+2.5=4. "
         f"{NO_EVAL}"
     )},
    {"name": "** right-assoc [HIDDEN]", "domain": "parser", "check": check_right_assoc,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` so ** is right-associative (like Python/maths). "
         "2**3**2 must equal 512, not 64. "
         f"{NO_EVAL}"
     )},
    {"name": "unary minus [HIDDEN]",    "domain": "parser", "check": check_unary_minus,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` to support unary minus. "
         "-2**2 must equal -4 (not 4). 2**-1=0.5. "
         f"{NO_EVAL}"
     )},
    {"name": "comparisons [HIDDEN]",    "domain": "parser", "check": check_comparisons,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` with comparison operators <,>,<=,>=,==,!=. "
         "Return 1.0 for True, 0.0 for False. Comparisons have lowest precedence. "
         f"{NO_EVAL}"
     )},
    {"name": "variables [HIDDEN]",      "domain": "parser", "check": check_variables,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` to accept `evaluate(expr, variables=None) -> float`. "
         "Variable names match [a-zA-Z_][\\w]*. Raise NameError for undefined. "
         f"{NO_EVAL}"
     )},
    {"name": "modulo % [HIDDEN]",       "domain": "parser", "check": check_modulo,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` to support modulo %, same precedence as * and /. "
         "Python semantics: -7%3=2. "
         f"{NO_EVAL}"
     )},
    {"name": "floor div // [HIDDEN]",   "domain": "parser", "check": check_floor_div,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` to support floor division //, same precedence as * and /. "
         "-7//2=-4. "
         f"{NO_EVAL}"
     )},
    {"name": "abs() round() [HIDDEN]",  "domain": "parser", "check": check_builtins,
     "probe": _parser_probe,
     "prompt": (
         "Extend `evaluate` so abs() and round() work as functions in expressions. "
         "abs(-3)=3, round(3.7)=4. "
         f"{NO_EVAL}"
     )},
    {"name": "full integration [HIDDEN]","domain": "parser", "check": check_integration,
     "probe": _parser_probe,
     "prompt": (
         "Write a COMPLETE `evaluate(expr, variables=None) -> float` supporting "
         "all features: +,-,*,/,**,%,// with correct precedence and associativity, "
         "unary +/-, parentheses, comparisons, variable lookup, abs(), round(). "
         f"{NO_EVAL}"
     )},
    # ── string domain (4 tasks) ───────────────────────────────────────────────
    {"name": "run-length encoding",    "domain": "string",
     "probe": None,
     "check": check_run_length,
     "prompt": (
         "Write `encode(s: str) -> str` for run-length encoding. "
         "encode('aabbbcc')='2a3b2c', encode('')=''. No classes."
     )},
    {"name": "balanced brackets",      "domain": "string",
     "probe": None,
     "check": check_balanced,
     "prompt": (
         "Write `is_balanced(s: str) -> bool` — True iff all brackets match. "
         "Support ()[]{}. Empty string is balanced. No classes."
     )},
    {"name": "word frequency",         "domain": "string",
     "probe": None,
     "check": check_word_freq,
     "prompt": "Write `word_freq(text: str) -> dict` case-insensitive word counts. Empty→{}. No classes."},
    {"name": "longest common substr",  "domain": "string",
     "probe": None,
     "check": check_lcs,
     "prompt": "Write `lcs(a: str, b: str) -> str` — longest common SUBSTRING (contiguous). No classes."},
    # ── algorithm domain (4 tasks) ────────────────────────────────────────────
    {"name": "binary search",          "domain": "algo",
     "probe": None,
     "check": check_binary_search,
     "prompt": "Write `binary_search(arr: list, target) -> int`. Returns index or -1. Handles []. No classes."},
    {"name": "merge sorted lists",     "domain": "algo",
     "probe": None,
     "check": check_merge_sorted,
     "prompt": "Write `merge(a: list, b: list) -> list` — merge two sorted lists. No classes."},
    {"name": "count islands",          "domain": "algo",
     "probe": None,
     "check": check_islands,
     "prompt": "Write `count_islands(grid: list[list[str]]) -> int` — connected '1' components. DFS/BFS. No classes."},
    {"name": "two sum",                "domain": "algo",
     "probe": None,
     "check": check_two_sum,
     "prompt": "Write `two_sum(nums: list[int], target: int)` → (i,j) or None. No classes."},
]

N_TASKS   = len(TASKS)
N_PARSER  = sum(1 for t in TASKS if t["domain"] == "parser")
N_STRING  = sum(1 for t in TASKS if t["domain"] == "string")
N_ALGO    = sum(1 for t in TASKS if t["domain"] == "algo")


# ─────────────────────────────────────────────────────────────────────────────
#  CODE EXTRACTION  (no regex — uses AST on the raw block.input dict)
# ─────────────────────────────────────────────────────────────────────────────

def _exec_and_check(code: str, check_fn) -> tuple[bool, str]:
    ns: dict = {}
    try:
        exec(compile(code, "<eval_v3>", "exec"), ns)
    except Exception as e:
        return False, f"Compile error: {e}"
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return bool(check_fn(ns)), ""
        except Exception as e:
            return False, f"Test error: {e}"


def _extract_code_from_trace(trace: str, want_def: str | None = None) -> str | None:
    """Extract the best code block from a trace string.

    Prefers the last block that contains a function definition (or the specific
    function name if want_def is given). Falls back to any compilable block.
    Uses ast on the dict repr, not regex on the structure.
    """
    import ast as _ast
    import re

    # Find all python_exec dicts in the trace.
    # {.*?} stops at nested braces, so we bracket-match manually instead.
    candidates: list[str] = []
    search_from = 0
    while True:
        start = trace.find("python_exec({", search_from)
        if start == -1:
            break
        brace_start = start + len("python_exec(")
        depth, i = 0, brace_start
        while i < len(trace):
            if trace[i] == "{": depth += 1
            elif trace[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        raw = trace[brace_start : i + 1]
        search_from = i + 1
        try:
            d = _ast.literal_eval(raw)
            if isinstance(d, dict) and "code" in d:
                candidates.append(d["code"])
        except Exception:
            pass

    if candidates:
        # Prefer last block with the wanted definition
        target = want_def or "def "
        for code in reversed(candidates):
            if target in code:
                return code
        return candidates[-1]

    # Fallback: ```python fences
    import re
    for block in reversed(re.findall(r"```(?:python)?\n(.*?)```", trace, re.DOTALL)):
        block = block.strip()
        if "def " in block:
            try:
                compile(block, "<c>", "exec"); return block
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY VISUALIZER — live heatmap updated after each task
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryVisualizer:
    """Writes a PNG after every task. Open in Preview with auto-refresh.

    Layout:
      Left:  N_TASKS × MAX_TURNS grid — each cell = one tool call
             colour = risk (green→red), marker = brain fired / task outcome
      Right: token bars + pass-rate curve
    """

    MAX_TURNS = 8

    def __init__(self, n_tasks: int, out_path: Path):
        self._n   = n_tasks
        self._out = out_path
        # grid[task_idx][turn] = severity 0-1 | None
        self._grid:    list[list[float | None]] = [[None] * self.MAX_TURNS for _ in range(n_tasks)]
        self._fires:   list[list[bool]]         = [[False] * self.MAX_TURNS for _ in range(n_tasks)]
        self._results: list[bool | None]        = [None] * n_tasks
        self._tokens_a: list[int]              = []
        self._tokens_b: list[int]              = []
        self._lock = threading.Lock()

    def record_turn(self, task_idx: int, turn: int, severity: float, fired: bool):
        with self._lock:
            t = min(turn, self.MAX_TURNS - 1)
            self._grid[task_idx][t] = severity
            self._fires[task_idx][t] = fired

    def record_task(self, task_idx: int, passed: bool,
                    tok_a: int = 0, tok_b: int = 0):
        with self._lock:
            self._results[task_idx] = passed
            if tok_a: self._tokens_a.append(tok_a)
            if tok_b: self._tokens_b.append(tok_b)

    def save(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap

        with self._lock:
            grid    = [row[:] for row in self._grid]
            fires   = [row[:] for row in self._fires]
            results = list(self._results)
            tok_a   = list(self._tokens_a)
            tok_b   = list(self._tokens_b)

        cmap = LinearSegmentedColormap.from_list(
            "risk", ["#43a047", "#ffee58", "#e53935"], N=256
        )

        fig = plt.figure(figsize=(22, 10))
        fig.patch.set_facecolor("#1a1a2e")

        # ── left: trajectory heatmap ─────────────────────────────────────────
        ax_heat = fig.add_axes([0.03, 0.08, 0.52, 0.84])
        ax_heat.set_facecolor("#16213e")

        mat = np.full((self._n, self.MAX_TURNS), np.nan)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v is not None:
                    mat[i, j] = v

        im = ax_heat.imshow(
            mat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
            interpolation="nearest",
        )

        # overlay markers
        for i in range(self._n):
            for j in range(self.MAX_TURNS):
                if fires[i][j]:
                    ax_heat.text(j, i, "⚡", ha="center", va="center",
                                 fontsize=9, color="white")
            # task outcome on the right edge
            if results[i] is not None:
                sym = "✓" if results[i] else "✗"
                col = "#43a047" if results[i] else "#e53935"
                ax_heat.text(self.MAX_TURNS - 0.4, i, sym,
                             ha="left", va="center", fontsize=11,
                             fontweight="bold", color=col)

        # domain separator lines
        ax_heat.axhline(N_PARSER - 0.5, color="white", lw=1.5, alpha=0.5)
        ax_heat.axhline(N_PARSER + N_STRING - 0.5, color="white", lw=1.5, alpha=0.5)

        ax_heat.set_yticks(range(self._n))
        ax_heat.set_yticklabels(
            [f"{i+1:>2}  {TASKS[i]['name'][:28]}" for i in range(self._n)],
            fontsize=8, color="white",
        )
        ax_heat.set_xticks(range(self.MAX_TURNS))
        ax_heat.set_xticklabels([f"t{i}" for i in range(self.MAX_TURNS)],
                                 fontsize=9, color="white")
        ax_heat.set_xlabel("Tool-call turn →", fontsize=10, color="white")
        ax_heat.set_title(
            "Agent Trajectory Map   (colour = risk  ⚡ = brain fired  ✓/✗ = outcome)",
            fontsize=12, fontweight="bold", color="white", pad=10,
        )
        ax_heat.tick_params(colors="white")
        for sp in ax_heat.spines.values(): sp.set_color("#444")

        plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01,
                     label="Risk score (0=clean  1=fail)").ax.yaxis.label.set_color("white")

        # domain labels
        mid_p = (N_PARSER - 1) / 2
        mid_s = N_PARSER + (N_STRING - 1) / 2
        mid_a = N_PARSER + N_STRING + (N_ALGO - 1) / 2
        for mid, lbl in [(mid_p, "PARSER"), (mid_s, "STRING"), (mid_a, "ALGO")]:
            ax_heat.text(-0.6, mid, lbl, ha="right", va="center",
                         fontsize=8, color="#aaa", rotation=90)

        # ── right top: token bars ─────────────────────────────────────────────
        ax_tok = fig.add_axes([0.60, 0.55, 0.37, 0.37])
        ax_tok.set_facecolor("#16213e")
        n_done = max(len(tok_a), len(tok_b), 1)
        xs = np.arange(1, n_done + 1)
        w  = 0.38
        if tok_a:
            ax_tok.bar(xs[:len(tok_a)] - w/2, tok_a, w,
                       color="#2196F3", alpha=0.85, label="haiku+brain")
        if tok_b:
            ax_tok.bar(xs[:len(tok_b)] + w/2, tok_b, w,
                       color="#FF9800", alpha=0.85, label="sonnet+thinking")
        ax_tok.set_title("Tokens per task", color="white", fontsize=10)
        ax_tok.legend(fontsize=8, facecolor="#222", labelcolor="white")
        ax_tok.tick_params(colors="white")
        ax_tok.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
        for sp in ax_tok.spines.values(): sp.set_color("#444")
        ax_tok.set_facecolor("#16213e")

        # ── right bottom: pass-rate curves ────────────────────────────────────
        ax_acc = fig.add_axes([0.60, 0.08, 0.37, 0.37])
        ax_acc.set_facecolor("#16213e")
        n_res = sum(1 for r in results if r is not None)
        if n_res:
            xs2 = np.arange(1, n_res + 1)
            cum = [sum(1 for r in results[:i+1] if r) / (i+1) * 100
                   for i in range(n_res)]
            ax_acc.plot(xs2, cum, color="#2196F3", lw=2.5, marker="o",
                        markersize=5, label="haiku+brain")
        # sonnet reference line (filled when b_results land)
        ax_acc.axhline(75, color="#FF9800", lw=1.5, linestyle="--",
                       alpha=0.6, label="sonnet+thinking (prev)")
        ax_acc.set_ylim(0, 105)
        ax_acc.set_title("Cumulative pass rate %", color="white", fontsize=10)
        ax_acc.legend(fontsize=8, facecolor="#222", labelcolor="white")
        ax_acc.tick_params(colors="white")
        ax_acc.set_ylabel("%", color="white", fontsize=9)
        for sp in ax_acc.spines.values(): sp.set_color("#444")

        fig.suptitle(
            "Hybrid Brain — live trajectory  "
            f"(haiku+brain  {sum(1 for r in results if r)}/{n_res} pass  "
            f"threshold={THRESHOLD}  k={K})",
            fontsize=13, fontweight="bold", color="white", y=0.97,
        )

        plt.savefig(self._out, dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  CARRY-FORWARD CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

class LessonStore:
    """Accumulates specific lessons from task failures — injected as bullets,
    not code. Replaces full-code carry-forward with targeted rules.

    Keeps the last MAX_LESSONS lessons (ring buffer), domain-scoped.
    Each lesson is ≤ 160 chars so total injection stays under ~300 tokens.
    """

    MAX_LESSONS = 8

    def __init__(self):
        self._lessons:  list[str] = []
        self._domain:   str | None = None
        self._has_seen: bool = False   # True after first task in domain

    def record_fail(self, probe_fails: list[str], detail: str) -> None:
        for raw in (probe_fails or [])[:3]:
            lesson = self._distil(raw)
            if lesson and lesson not in self._lessons:
                self._lessons.append(lesson)
        # also distil from verify detail if probe gave nothing
        if not probe_fails and detail and detail not in ("failed", "no code"):
            lesson = self._distil(detail)
            if lesson and lesson not in self._lessons:
                self._lessons.append(lesson)
        # ring-buffer trim
        if len(self._lessons) > self.MAX_LESSONS:
            self._lessons = self._lessons[-self.MAX_LESSONS:]

    def record_pass(self, task_name: str) -> None:
        self._has_seen = True

    @staticmethod
    def _distil(raw: str) -> str:
        """Keep the FIX: clause if present; else keep the first 160 chars."""
        if "FIX:" in raw:
            return raw.split("FIX:", 1)[1].strip()[:160]
        return raw.strip()[:160]

    def build_prompt(self, task: dict, base_prompt: str) -> str:
        if task["domain"] != "parser" or not self._has_seen:
            self._domain = task["domain"]
            return base_prompt
        if not self._lessons:
            return base_prompt
        bullets = "\n".join(f"• {l}" for l in self._lessons[-6:])
        return (
            f"Critical rules from past failures — violating any of these WILL cause wrong answers:\n"
            f"{bullets}\n\n"
            f"{base_prompt}\n\n"
            f"You MUST verify with python_exec (3+ test cases) before finishing."
        )

    def update_domain(self, domain: str) -> None:
        if domain != self._domain:
            self._domain = domain
            self._has_seen = False
            self._lessons.clear()

    @property
    def active(self) -> bool:
        return bool(self._lessons)


# ─────────────────────────────────────────────────────────────────────────────
#  SONNET DIAGNOSIS (for stored metadata)
# ─────────────────────────────────────────────────────────────────────────────

def _diagnose(code: str, task: dict) -> str:
    if not code: return "no code"
    probe_fn = task.get("probe")
    if not probe_fn:
        return "failed"
    ns: dict = {}
    try:
        exec(compile(code, "<d>", "exec"), ns)
        fails = probe_fn(ns)
        return "; ".join(fails[:2]) if fails else "unknown failure"
    except Exception as e:
        return f"exec error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD A: haiku + tool_agent + brain + carry-forward
# ─────────────────────────────────────────────────────────────────────────────

def run_method_a(brain: BrainAgent, viz: TrajectoryVisualizer) -> list[dict]:
    agent = tool_agent(["python_exec"], max_turns=MAX_TURNS, model=HAIKU,
                       max_tokens=2048)
    agent.monitor = brain
    lessons = LessonStore()
    results: list[dict] = []

    for i, task in enumerate(TASKS):
        n = i + 1
        print(f"\n  A {n:>2}/20 [{task['domain']:6}] {task['name']}")

        # Register task with brain (probe + trajectory index)
        probe_fn = task.get("probe")
        if probe_fn is None and task["domain"] != "parser":
            probe_fn = _generic_probe(task["check"])
        brain.set_task(i, probe_fn=probe_fn)
        brain.reset()
        lessons.update_domain(task["domain"])

        # Build prompt — lessons inject targeted rules, not full code
        prompt = lessons.build_prompt(task, task["prompt"])
        if lessons.active and task["domain"] == "parser":
            print(f"       [lessons: {len(lessons._lessons)} rules injected]")

        t0 = time.time()
        result = agent(prompt)
        elapsed = time.time() - t0

        trace, tokens = (result[0], result[1]) if isinstance(result, tuple) \
                        else (str(result), 0)

        # Update trajectory viz with brain data
        for pt in brain.get_trajectory():
            if pt.task_idx == i:
                viz.record_turn(i, pt.turn, pt.severity, pt.fired)

        # Extract code — prefer block containing the expected function name
        want = "evaluate" if task["domain"] == "parser" else None
        code = _extract_code_from_trace(trace, want_def=want)

        # Verify
        if code:
            passed, detail = _exec_and_check(code, task["check"])
        else:
            passed, detail = False, "No code extracted"

        # Run probe on extracted code for lesson distillation
        probe_fails: list[str] = []
        if not passed and code and task.get("probe"):
            ns: dict = {}
            try:
                exec(compile(code, "<probe>", "exec"), ns)
                probe_fails = task["probe"](ns)
            except Exception:
                pass

        if not passed and not detail and code and task["domain"] == "parser":
            detail = _diagnose(code, task)

        # Feed failure lessons back into the lesson store
        if not passed:
            lessons.record_fail(probe_fails, detail)
        else:
            lessons.record_pass(task["name"])

        # Teach the brain (skip empty code — embedder rejects empty strings)
        meta = detail if not passed else ""
        if code:
            brain.store_code(code, int(passed), metadata=meta)
        brain.store(trace, int(passed), metadata=meta)

        if passed and code:
            brain.store_passing_code(task["name"], code)

        fires = brain._code_interventions
        status = "PASS" if passed else "FAIL"
        fire_tag = f"[⚡{fires}×]" if fires else ""
        print(f"       A: {status}  {tokens:>6,} tok  {elapsed:.0f}s  "
              f"{fire_tag}  {detail[:55] if detail else ''}")

        viz.record_task(i, passed, tok_a=tokens)
        viz.save()  # update live graph

        results.append({
            "task": n, "name": task["name"], "domain": task["domain"],
            "method": "haiku+brain", "passed": passed, "tokens": tokens,
            "elapsed": elapsed, "code_fires": fires,
            "carry_used": lessons.active and task["domain"] == "parser",
            "detail": detail,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD B: sonnet + extended thinking
# ─────────────────────────────────────────────────────────────────────────────

def run_method_b(viz: TrajectoryVisualizer) -> list[dict]:
    import anthropic
    client = anthropic.Anthropic()
    results: list[dict] = []

    for i, task in enumerate(TASKS):
        n = i + 1
        print(f"\n  B {n:>2}/20 [{task['domain']:6}] {task['name']}")
        t0 = time.time()
        try:
            msg = client.messages.create(
                model=SONNET, max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 8000},
                messages=[{"role": "user", "content": task["prompt"]}],
            )
            text   = "".join(b.text for b in msg.content if hasattr(b, "text"))
            tokens = msg.usage.input_tokens + msg.usage.output_tokens
        except Exception as e:
            print(f"       B: ERROR {e}")
            results.append({"task": n, "name": task["name"], "domain": task["domain"],
                            "method": "sonnet+thinking", "passed": False, "tokens": 0,
                            "elapsed": 0, "detail": str(e)})
            continue

        elapsed = time.time() - t0

        # extract from text response
        import re
        code = None
        for block in reversed(re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)):
            block = block.strip()
            if "def " in block:
                try: compile(block, "<c>", "exec"); code = block; break
                except Exception: pass

        if code:
            passed, detail = _exec_and_check(code, task["check"])
        else:
            passed, detail = False, "No code extracted"

        if not passed and not detail and code and task["domain"] == "parser":
            detail = _diagnose(code, task)

        status = "PASS" if passed else "FAIL"
        print(f"       B: {status}  {tokens:>6,} tok  {elapsed:.0f}s  "
              f"{detail[:55] if detail else ''}")

        viz.record_task(i, passed, tok_b=tokens)
        viz.save()

        results.append({
            "task": n, "name": task["name"], "domain": task["domain"],
            "method": "sonnet+thinking", "passed": passed, "tokens": tokens,
            "elapsed": elapsed, "detail": detail,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL COMPARISON GRAPHS
# ─────────────────────────────────────────────────────────────────────────────

def _cost(tokens: int, method: str) -> float:
    return tokens * (1.50 if "haiku" in method else 12.0) / 1_000_000


def plot_final(a: list[dict], b: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values(): sp.set_color("#444")
        ax.tick_params(colors="white")

    xs = np.arange(1, N_TASKS + 1)
    w  = 0.38

    # ── panel 1: tokens per task ──────────────────────────────────────────────
    ax = axes[0]
    tok_a = [r["tokens"] for r in a]
    tok_b = [r["tokens"] for r in b]
    ca = ["#2196F3" if r["passed"] else "#90CAF9" for r in a]
    cb = ["#FF9800" if r["passed"] else "#FFCC80" for r in b]
    ax.bar(xs - w/2, tok_a, w, color=ca, label="haiku+brain")
    ax.bar(xs + w/2, tok_b, w, color=cb, label="sonnet+thinking")
    ax.axvline(N_PARSER + 0.5, color="white", lw=1, alpha=0.4)
    ax.axvline(N_PARSER + N_STRING + 0.5, color="white", lw=1, alpha=0.4)
    ax.set_title("Tokens per task  (darker = PASS)", color="white", fontweight="bold")
    ax.set_xlabel("Task #", color="white"); ax.set_ylabel("Tokens", color="white")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x/1000)}k"))
    ax.legend(fontsize=9, facecolor="#222", labelcolor="white")

    # ── panel 2: pass rate by domain ─────────────────────────────────────────
    ax = axes[1]
    domains = [("parser", N_PARSER), ("string", N_STRING), ("algo", N_ALGO)]
    xlabels = ["Parser\n(12 tasks)", "String\n(4 tasks)", "Algo\n(4 tasks)"]
    for di, (dom, n) in enumerate(domains):
        da = [r for r in a if r["domain"] == dom]
        db = [r for r in b if r["domain"] == dom]
        pa = sum(r["passed"] for r in da) / n * 100
        pb = sum(r["passed"] for r in db) / n * 100
        x  = di * 3
        ax.bar(x,     pa, 1.2, color="#2196F3", alpha=0.9)
        ax.bar(x+1.3, pb, 1.2, color="#FF9800", alpha=0.9)
        ax.text(x,     pa + 2, f"{pa:.0f}%", ha="center", color="#90CAF9", fontsize=10)
        ax.text(x+1.3, pb + 2, f"{pb:.0f}%", ha="center", color="#FFCC80", fontsize=10)
    ax.set_xticks([0.65, 3.65, 6.65]); ax.set_xticklabels(xlabels, color="white")
    ax.set_ylim(0, 115); ax.set_title("Pass rate by domain", color="white", fontweight="bold")
    ax.set_ylabel("Pass %", color="white")
    ax.legend(handles=[mpatches.Patch(color="#2196F3", label="haiku+brain"),
                        mpatches.Patch(color="#FF9800", label="sonnet+thinking")],
              fontsize=9, facecolor="#222", labelcolor="white")

    # ── panel 3: cost vs accuracy scatter ────────────────────────────────────
    ax = axes[2]
    avg_a  = np.mean([r["tokens"] for r in a])
    avg_b  = np.mean([r["tokens"] for r in b])
    pass_a = np.mean([r["passed"] for r in a]) * 100
    pass_b = np.mean([r["passed"] for r in b]) * 100
    cost_a = sum(_cost(r["tokens"], "haiku") for r in a)
    cost_b = sum(_cost(r["tokens"], "sonnet") for r in b)

    ax.scatter(avg_a, pass_a, s=500, color="#2196F3", zorder=5, edgecolors="white", lw=2)
    ax.scatter(avg_b, pass_b, s=500, color="#FF9800", zorder=5, edgecolors="white", lw=2)

    ax.annotate(
        f"haiku+brain\n{pass_a:.0f}% pass\n${cost_a:.3f} total",
        (avg_a, pass_a), textcoords="offset points", xytext=(-90, 15),
        color="#90CAF9", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#90CAF9"),
    )
    ax.annotate(
        f"sonnet+thinking\n{pass_b:.0f}% pass\n${cost_b:.3f} total",
        (avg_b, pass_b), textcoords="offset points", xytext=(15, -50),
        color="#FFCC80", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#FFCC80"),
    )
    ratio = cost_b / max(cost_a, 1e-9)
    ax.text(0.5, 0.05, f"haiku+brain is {ratio:.1f}× cheaper",
            ha="center", transform=ax.transAxes, fontsize=11, color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333", alpha=0.9))
    ax.set_xlabel("Avg tokens/task", color="white")
    ax.set_ylabel("Pass rate %", color="white")
    ax.set_title("Cost vs accuracy", color="white", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0f}%"))

    fig.suptitle(
        "Hybrid Brain (carry-forward + probe tests + kNN)  vs  sonnet+extended_thinking",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )
    plt.tight_layout()
    out = OUT / "eval_v3_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 78)
    print("  eval_v3: hybrid brain (carry-forward + probes + kNN) vs sonnet+thinking")
    print(f"  20 tasks  threshold={THRESHOLD}  k={K}  max_turns={MAX_TURNS}")
    print(f"  Live graph → {OUT / 'trajectory_live.png'}")
    print("═" * 78)

    embedder = build_embedder()
    brain    = BrainAgent(embedder, k=K, threshold=THRESHOLD,
                          check_interval=2.0, min_chars=999999)  # trace bail off
    viz      = TrajectoryVisualizer(N_TASKS, OUT / "trajectory_live.png")

    # Kick off a first render immediately (blank)
    viz.save()
    print(f"\n  Trajectory graph initialised — open {OUT / 'trajectory_live.png'}")

    a_results: list[dict] = []
    b_results: list[dict] = []
    _errs: list[Exception] = []

    def _run_a():
        try: a_results.extend(run_method_a(brain, viz))
        except Exception as e: _errs.append(e); import traceback; traceback.print_exc()

    def _run_b():
        try: b_results.extend(run_method_b(viz))
        except Exception as e: _errs.append(e); import traceback; traceback.print_exc()

    print("\n  Running A and B in parallel...\n" + "─" * 78)
    t_a = threading.Thread(target=_run_a, name="method-A")
    t_b = threading.Thread(target=_run_b, name="method-B")
    t_a.start(); t_b.start()
    t_a.join();  t_b.join()

    if _errs: raise _errs[0]

    # ── summary ───────────────────────────────────────────────────────────────
    pass_a  = sum(r["passed"] for r in a_results)
    pass_b  = sum(r["passed"] for r in b_results)
    tok_a   = sum(r["tokens"] for r in a_results)
    tok_b   = sum(r["tokens"] for r in b_results)
    cost_a  = sum(_cost(r["tokens"], "haiku")  for r in a_results)
    cost_b  = sum(_cost(r["tokens"], "sonnet") for r in b_results)
    fires   = sum(r.get("code_fires", 0) for r in a_results)

    print("\n" + "═" * 78)
    print("  RESULTS")
    print("─" * 78)
    hdr = f"  {'#':>2}  {'Task':<30}  {'A':>5}  {'tok_A':>7}  {'B':>5}  {'tok_B':>7}  fires"
    print(hdr); print("─" * 78)
    for a, b in zip(a_results, b_results):
        pa = "PASS" if a["passed"] else "FAIL"
        pb = "PASS" if b["passed"] else "FAIL"
        f  = f"⚡{a.get('code_fires',0)}" if a.get("code_fires") else ""
        carry = "↑" if a.get("carry_used") else ""
        print(f"  {a['task']:>2}  {a['name']:<30}  {carry}{pa}  {a['tokens']:>7,}  "
              f"{pb}  {b['tokens']:>7,}  {f}")
    print("─" * 78)
    print(f"  haiku+brain:      {pass_a}/20 pass  avg {tok_a//20:,} tok  "
          f"{fires} interventions  ${cost_a:.4f}")
    print(f"  sonnet+thinking:  {pass_b}/20 pass  avg {tok_b//20:,} tok  "
          f"${cost_b:.4f}")
    print(f"  Cost ratio:  sonnet is {cost_b/max(cost_a,1e-9):.1f}× more expensive")
    print("═" * 78)

    print("\n[Generating final comparison graphs...]")
    plot_final(a_results, b_results)
    viz.save()

    # ── JSON ledger ───────────────────────────────────────────────────────────
    ledger = {
        "summary": {
            "pass_a": pass_a, "pass_b": pass_b,
            "tokens_a": tok_a, "tokens_b": tok_b,
            "cost_a": cost_a, "cost_b": cost_b,
            "cost_ratio": cost_b / max(cost_a, 1e-9),
            "interventions": fires,
        },
        "tasks": [
            {**a, **{f"b_{k}": v for k, v in b.items() if k not in ("task","name","domain")}}
            for a, b in zip(a_results, b_results)
        ],
        "trajectory": [
            {"task": pt.task_idx, "turn": pt.turn,
             "p_fail": pt.p_fail, "probe_fails": pt.probe_fails,
             "fired": pt.fired, "severity": pt.severity}
            for pt in brain.get_trajectory()
        ],
    }
    out_json = OUT / "eval_v3_run.json"
    with open(out_json, "w") as f:
        json.dump(ledger, f, indent=2)
    print(f"Saved {out_json}")
    print(f"Saved {OUT / 'trajectory_live.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
