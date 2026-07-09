#!/usr/bin/env python3
"""eval_learn.py — Cold-start motif learning benchmark.

Tests whether BrainAgent can learn from first-occurrence failures and then
fire predictive warnings before repeat failures in the same error family.

4 motif families × 8 tasks (1 discovery + 5 recurrence + 2 near-miss) = 32 tasks.
No seed_motifs() — pure cold start.

Key metrics:
  first_fail_rate           : discovery tasks that failed (expect high)
  motif_extraction_rate     : failed families where a motif was extracted
  repeat_prevention_rate    : recurrence tasks where brain fired pre-execution
  false_positive_rate       : near-miss correct tasks where brain fired (expect low)

Run:
    python eval/eval_learn.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from trace_use import build_embedder, tool_agent
from trace_use import BrainAgent

OUT = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

HAIKU     = "claude-haiku-4-5-20251001"
MAX_TURNS = 5


# ── code extraction + execution ───────────────────────────────────────────────

def _exec_ns(code: str) -> "dict | str":
    ns: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<lb>", "exec"), ns)
        return ns
    except Exception as e:
        return str(e)


def _extract_last_code(trace: str) -> str:
    """Extract the last python_exec code block from a tool_agent trace."""
    last_code = ""
    idx = 0
    while True:
        start = trace.find("[tool:python_exec(", idx)
        if start < 0:
            break
        brace = trace.find("{", start)
        if brace < 0:
            break
        depth = 0
        end = brace
        for i, ch in enumerate(trace[brace:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = brace + i + 1
                    break
        try:
            d = json.loads(trace[brace:end])
            code = d.get("code", "")
            if code:
                last_code = code
        except Exception:
            pass
        idx = end
    if not last_code:
        blocks = re.findall(r"```python\n(.*?)```", trace, re.DOTALL)
        if blocks:
            last_code = blocks[-1]
    return last_code


def _verify(trace: str, check_fn: Callable) -> tuple[bool, str]:
    code = _extract_last_code(trace)
    if not code:
        return False, "no python_exec call found in trace"
    ns = _exec_ns(code)
    if isinstance(ns, str):
        return False, f"exec error: {ns[:200]}"
    try:
        return check_fn(ns)
    except Exception as e:
        return False, f"verifier error: {e}"


TOL = 1e-4


def _close(a: float, b: float) -> bool:
    return abs(a - b) < TOL


def _make_check(fn_name: str, args_list: list, expected_list: list) -> Callable:
    """Build a verifier that calls fn_name with each set of args and checks output.

    Convention:
      tuple  → unpack as positional args:   fn(a, b, c)
      list   → pass as a single list arg:   fn([a, b, c])
      scalar → call directly:               fn(x)
    """
    def check(ns: dict) -> tuple[bool, str]:
        fn = ns.get(fn_name)
        if fn is None:
            return False, f"function '{fn_name}' not defined"
        for args, expected in zip(args_list, expected_list):
            try:
                if isinstance(args, tuple):
                    got = fn(*args)   # (a, b) → fn(a, b)
                elif isinstance(args, list):
                    got = fn(args)    # [a, b] → fn([a, b])
                else:
                    got = fn(args)    # scalar → fn(x)
            except Exception as e:
                return False, f"{fn_name}({args}) raised {type(e).__name__}: {e}"
            if isinstance(expected, int):
                if int(round(float(got))) != expected:
                    return False, f"{fn_name}({args}) = {got!r}, expected {expected}"
            else:
                if not _close(float(got), float(expected)):
                    return False, (
                        f"{fn_name}({args}) = {float(got):.6f}, "
                        f"expected {float(expected):.6f}"
                    )
        return True, ""
    return check


# ── Task definitions ──────────────────────────────────────────────────────────
#
# DESIGN PRINCIPLE:
#   Discovery tasks: NO expected value given, NO formula hints.
#     Haiku naturally uses the wrong approach; verifier catches it.
#   Recurrence tasks: explicit expected value — haiku can see the target,
#     but will still often use the same wrong approach from the same domain framing.
#   Near-miss tasks: correct approach is simpler (e.g. multiply, not sqrt).
#     Brain must NOT fire on these.

TASKS: list[dict] = []

# ════════════════════════════════════════════════════════════════════════════════
# Family A: geomean
# Error:  arithmetic mean of rates → sum(r)/n
# Correct: geometric mean → (∏(1+r))^(1/n) - 1
# Haiku naturally uses arithmetic mean when asked for "average" without specifying geometric.
# ════════════════════════════════════════════════════════════════════════════════

TASKS.append({
    "name": "A1_discovery", "family": "geomean", "kind": "discovery",
    "prompt": (
        "Write `avg_annual_return(returns: list) -> float`.\n"
        "returns is a list of yearly return rates (e.g. 0.10 = 10%).\n"
        "Return the average annual return an investor earned over the period.\n"
        "Call python_exec to implement and test the function with "
        "avg_annual_return([0.10, 0.50, -0.10])."
    ),
    # Geometric mean ≈ 0.14016. Arithmetic mean = 0.16667 → FAIL.
    "check": _make_check("avg_annual_return",
        [[0.10, 0.50, -0.10]],
        [(1.1 * 1.5 * 0.9) ** (1 / 3) - 1]),
})

TASKS.append({
    "name": "A2_recurrence", "family": "geomean", "kind": "recurrence",
    # Framed as "mean period rate" — haiku will use arithmetic mean (wrong)
    # Brain motif should fire and warn: use geometric mean, not arithmetic
    "prompt": (
        "Write `mean_period_rate(rates: list) -> float`.\n"
        "rates is a list of period-over-period growth rates.\n"
        "Return the mean rate per period.\n"
        "Test: mean_period_rate([0.05, 0.20, 0.05]) ≈ 0.09545.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("mean_period_rate",
        [[0.05, 0.20, 0.05]],
        [(1.05 * 1.20 * 1.05) ** (1 / 3) - 1]),
})

TASKS.append({
    "name": "A3_recurrence", "family": "geomean", "kind": "recurrence",
    # "Average annual return" → haiku uses arithmetic mean (wrong)
    "prompt": (
        "Write `avg_return(returns: list) -> float`.\n"
        "returns is a list of annual investment returns.\n"
        "Return the representative annual return over the period.\n"
        "Test: avg_return([0.10, 0.50, -0.10]) ≈ 0.14016.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("avg_return",
        [[0.10, 0.50, -0.10]],
        [(1.1 * 1.5 * 0.9) ** (1 / 3) - 1]),
})

TASKS.append({
    "name": "A4_recurrence", "family": "geomean", "kind": "recurrence",
    # "Average growth" for a sequence → haiku uses arithmetic mean (wrong)
    "prompt": (
        "Write `average_growth(values: list) -> float`.\n"
        "values is a list of annual growth rates.\n"
        "Return the average annual growth rate.\n"
        "Test: average_growth([0.20, -0.20, 0.30]) ≈ 0.07189.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("average_growth",
        [[0.20, -0.20, 0.30]],
        [(1.20 * 0.80 * 1.30) ** (1 / 3) - 1]),
})

TASKS.append({
    "name": "A5_recurrence", "family": "geomean", "kind": "recurrence",
    # "Annual return for a portfolio" → haiku uses arithmetic mean
    "prompt": (
        "Write `portfolio_annual_return(returns: list) -> float`.\n"
        "returns is a list of annual returns. Return the average annual return.\n"
        "Test: portfolio_annual_return([0.10, 0.50, -0.10]) ≈ 0.14016.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("portfolio_annual_return",
        [[0.10, 0.50, -0.10]],
        [(1.1 * 1.5 * 0.9) ** (1 / 3) - 1]),
})

TASKS.append({
    "name": "A6_recurrence", "family": "geomean", "kind": "recurrence",
    # Factors framed as "average factor" → haiku uses arithmetic mean
    "prompt": (
        "Write `avg_growth_factor(factors: list) -> float`.\n"
        "factors is a list of annual growth multipliers (e.g. 1.10 = 10% gain).\n"
        "Return the average annual growth factor.\n"
        "Test: avg_growth_factor([1.10, 1.50, 0.90]) ≈ 1.14016.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("avg_growth_factor",
        [[1.10, 1.50, 0.90]],
        [(1.10 * 1.50 * 0.90) ** (1 / 3)]),
})

# Near-miss: arithmetic mean IS correct here — brain must NOT fire
TASKS.append({
    "name": "A7_near_miss", "family": "geomean", "kind": "near_miss",
    "prompt": (
        "Write `mean_return(returns: list) -> float`.\n"
        "Return the arithmetic mean (simple average) of the returns.\n"
        "Test: mean_return([0.10, 0.50, -0.10]) ≈ 0.16667.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("mean_return",
        [[0.10, 0.50, -0.10]],
        [(0.10 + 0.50 - 0.10) / 3]),
})

TASKS.append({
    "name": "A8_near_miss", "family": "geomean", "kind": "near_miss",
    "prompt": (
        "Write `weighted_ret(returns: list, weights: list) -> float`.\n"
        "Return the weighted arithmetic mean return.\n"
        "Test: weighted_ret([0.05, 0.10, 0.15], [0.5, 0.3, 0.2]) ≈ 0.085.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("weighted_ret",
        [([0.05, 0.10, 0.15], [0.5, 0.3, 0.2])],
        [0.05 * 0.5 + 0.10 * 0.3 + 0.15 * 0.2]),
})

# ════════════════════════════════════════════════════════════════════════════════
# Family B: vol_annualize
# Error:  std * n  (linear scaling of volatility)
# Correct: std * sqrt(n)  (square-root-of-time rule)
# Haiku without explicit guidance tends to multiply linearly.
# ════════════════════════════════════════════════════════════════════════════════

TASKS.append({
    "name": "B1_discovery", "family": "vol_annualize", "kind": "discovery",
    "prompt": (
        "Write `scale_std(daily_std: float, n_days: int) -> float`.\n"
        "daily_std is the standard deviation of returns for one day.\n"
        "Return the standard deviation over n_days trading days.\n"
        "Call python_exec to implement and test with scale_std(0.01, 252)."
    ),
    # sqrt rule: 0.01*sqrt(252) ≈ 0.15875. Linear: 0.01*252 = 2.52 → FAIL.
    "check": _make_check("scale_std",
        [(0.01, 252)],
        [0.01 * (252 ** 0.5)]),
})

TASKS.append({
    "name": "B2_recurrence", "family": "vol_annualize", "kind": "recurrence",
    # "Total standard deviation over N months" — haiku might use linear
    "prompt": (
        "Write `monthly_std_to_annual(monthly_std: float) -> float`.\n"
        "monthly_std is the standard deviation of monthly returns.\n"
        "Return the standard deviation over 12 months (the annual std).\n"
        "Test: monthly_std_to_annual(0.03) ≈ 0.10392.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("monthly_std_to_annual",
        [(0.03,)],
        [0.03 * (12 ** 0.5)]),
})

TASKS.append({
    "name": "B3_recurrence", "family": "vol_annualize", "kind": "recurrence",
    "prompt": (
        "Write `weekly_std_to_annual(weekly_std: float) -> float`.\n"
        "weekly_std is the standard deviation of weekly returns.\n"
        "Return the annual standard deviation (52 weeks per year).\n"
        "Test: weekly_std_to_annual(0.02) ≈ 0.14422.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("weekly_std_to_annual",
        [(0.02,)],
        [0.02 * (52 ** 0.5)]),
})

TASKS.append({
    "name": "B4_recurrence", "family": "vol_annualize", "kind": "recurrence",
    "prompt": (
        "Write `period_std_scaled(std: float, from_n: int, to_n: int) -> float`.\n"
        "std is the standard deviation over from_n periods.\n"
        "Return the standard deviation for to_n periods.\n"
        "Test: period_std_scaled(0.01, 1, 252) ≈ 0.15875.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("period_std_scaled",
        [(0.01, 1, 252)],
        [0.01 * (252 ** 0.5)]),
})

TASKS.append({
    "name": "B5_recurrence", "family": "vol_annualize", "kind": "recurrence",
    "prompt": (
        "Write `std_over_horizon(daily_std: float, days: int) -> float`.\n"
        "daily_std is the standard deviation of daily returns.\n"
        "Return the standard deviation over a horizon of days.\n"
        "Test: std_over_horizon(0.01, 10) ≈ 0.03162.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("std_over_horizon",
        [(0.01, 10)],
        [0.01 * (10 ** 0.5)]),
})

TASKS.append({
    "name": "B6_recurrence", "family": "vol_annualize", "kind": "recurrence",
    "prompt": (
        "Write `quarterly_std_to_annual(q_std: float) -> float`.\n"
        "q_std is the quarterly return standard deviation.\n"
        "Return the annual standard deviation (4 quarters in a year).\n"
        "Test: quarterly_std_to_annual(0.05) ≈ 0.10.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("quarterly_std_to_annual",
        [(0.05,)],
        [0.05 * (4 ** 0.5)]),
})

# Near-miss B: linear scaling IS correct for means — brain must NOT fire
TASKS.append({
    "name": "B7_near_miss", "family": "vol_annualize", "kind": "near_miss",
    "prompt": (
        "Write `annualize_mean(daily_mean: float, trading_days: int = 252) -> float`.\n"
        "daily_mean is the arithmetic mean of daily returns.\n"
        "Return the annualized expected return (means scale linearly with time).\n"
        "Test: annualize_mean(0.0004, 252) ≈ 0.1008.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("annualize_mean",
        [(0.0004, 252)],
        [0.0004 * 252]),
})

TASKS.append({
    "name": "B8_near_miss", "family": "vol_annualize", "kind": "near_miss",
    "prompt": (
        "Write `total_variance(daily_var: float, days: int) -> float`.\n"
        "daily_var is the daily return variance (variance scales linearly).\n"
        "Return the total variance over days periods.\n"
        "Test: total_variance(0.0001, 252) ≈ 0.0252.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("total_variance",
        [(0.0001, 252)],
        [0.0001 * 252]),
})

# ════════════════════════════════════════════════════════════════════════════════
# Family C: compound_vs_simple
# Error:  rate * n  (simple multiplication for periodic→annual return)
# Correct: (1+rate)^n - 1  (compound interest)
# Haiku without explicit "compound" framing often uses simple multiplication.
# ════════════════════════════════════════════════════════════════════════════════

TASKS.append({
    "name": "C1_discovery", "family": "compound_vs_simple", "kind": "discovery",
    "prompt": (
        "Write `total_annual_return(monthly_r: float) -> float`.\n"
        "monthly_r is the return rate each month for 12 months.\n"
        "Return the total annual return.\n"
        "Call python_exec to implement and test with total_annual_return(0.01)."
    ),
    # Compound: (1.01)^12 - 1 ≈ 0.12683. Simple: 0.01*12 = 0.12 → FAIL if haiku adds.
    "check": _make_check("total_annual_return",
        [(0.01,)],
        [(1.01) ** 12 - 1]),
})

TASKS.append({
    "name": "C2_recurrence", "family": "compound_vs_simple", "kind": "recurrence",
    # "Annual return from daily" → haiku might multiply (wrong)
    "prompt": (
        "Write `daily_return_annualized(daily_r: float, n: int = 252) -> float`.\n"
        "daily_r is the return each trading day. Return the annual return.\n"
        "Test: daily_return_annualized(0.001) ≈ 0.28367.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("daily_return_annualized",
        [(0.001,)],
        [(1.001) ** 252 - 1]),
})

TASKS.append({
    "name": "C3_recurrence", "family": "compound_vs_simple", "kind": "recurrence",
    "prompt": (
        "Write `weekly_return_annualized(weekly_r: float) -> float`.\n"
        "weekly_r is the return each week. Return the total annual return.\n"
        "Test: weekly_return_annualized(0.005) ≈ 0.29664.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("weekly_return_annualized",
        [(0.005,)],
        [(1.005) ** 52 - 1]),
})

TASKS.append({
    "name": "C4_recurrence", "family": "compound_vs_simple", "kind": "recurrence",
    "prompt": (
        "Write `quarterly_return_annualized(q_r: float) -> float`.\n"
        "q_r is the return each quarter. Return the annual return.\n"
        "Test: quarterly_return_annualized(0.03) ≈ 0.12551.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("quarterly_return_annualized",
        [(0.03,)],
        [(1.03) ** 4 - 1]),
})

TASKS.append({
    "name": "C5_recurrence", "family": "compound_vs_simple", "kind": "recurrence",
    "prompt": (
        "Write `periodic_return_annualized(rate: float, n: int) -> float`.\n"
        "rate is the per-period return, n is periods per year.\n"
        "Return the total annual return.\n"
        "Test: periodic_return_annualized(0.02, 6) ≈ 0.12616.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("periodic_return_annualized",
        [(0.02, 6)],
        [(1.02) ** 6 - 1]),
})

TASKS.append({
    "name": "C6_recurrence", "family": "compound_vs_simple", "kind": "recurrence",
    "prompt": (
        "Write `ear(nominal: float, m: int) -> float`.\n"
        "nominal is the nominal annual rate, m is compounding frequency.\n"
        "Return the effective annual rate: (1 + nominal/m)^m - 1.\n"
        "Test: ear(0.12, 12) ≈ 0.12683.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("ear",
        [(0.12, 12)],
        [(1 + 0.12 / 12) ** 12 - 1]),
})

# Near-miss C: division (not exponentiation) is correct
TASKS.append({
    "name": "C7_near_miss", "family": "compound_vs_simple", "kind": "near_miss",
    "prompt": (
        "Write `apr_monthly(annual_rate: float) -> float`.\n"
        "annual_rate is a nominal APR. Return the monthly rate: annual_rate / 12.\n"
        "Test: apr_monthly(0.12) ≈ 0.01.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("apr_monthly",
        [(0.12,)],
        [0.12 / 12]),
})

TASKS.append({
    "name": "C8_near_miss", "family": "compound_vs_simple", "kind": "near_miss",
    "prompt": (
        "Write `annual_to_monthly(annual_r: float) -> float`.\n"
        "Return the monthly equivalent: (1+annual_r)^(1/12) - 1.\n"
        "Test: annual_to_monthly(0.12683) ≈ 0.01.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("annual_to_monthly",
        [(0.12683,)],
        [(1.12683) ** (1 / 12) - 1]),
})

# ════════════════════════════════════════════════════════════════════════════════
# Family D: log_base
# Error:  math.log(x) — natural log (base e)
# Correct: math.log10(x) — base-10 log
# When haiku is told to "use logarithms" for digit counting, it often uses natural log.
# ════════════════════════════════════════════════════════════════════════════════

TASKS.append({
    "name": "D1_discovery", "family": "log_base", "kind": "discovery",
    "prompt": (
        "Write `count_digits(n: int) -> int`.\n"
        "Return the number of decimal digits in positive integer n.\n"
        "Implement using math.log — do NOT use len(str(n)).\n"
        "Call python_exec to implement and test with "
        "count_digits(1), count_digits(9), count_digits(10), count_digits(100)."
    ),
    # log10-based: floor(log10(n))+1. Natural log gives wrong answer for n>2.
    # Haiku reading "use math.log" might write math.log(n) without base → FAIL.
    "check": _make_check("count_digits",
        [1, 9, 10, 100, 1000],
        [1,  1,  2,   3,    4]),
})

TASKS.append({
    "name": "D2_recurrence", "family": "log_base", "kind": "recurrence",
    "prompt": (
        "Write `order_of_magnitude(x: float) -> int`.\n"
        "Returns floor(log10(x)) for positive x.\n"
        "Test: order_of_magnitude(500) = 2, order_of_magnitude(1000) = 3.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("order_of_magnitude",
        [500, 1000, 1500, 100],
        [2,   3,    3,    2]),
})

TASKS.append({
    "name": "D3_recurrence", "family": "log_base", "kind": "recurrence",
    "prompt": (
        "Write `decibels(power_ratio: float) -> float`.\n"
        "dB = 10 * log10(power_ratio).\n"
        "Test: decibels(100.0) = 20.0, decibels(1000.0) = 30.0.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("decibels",
        [100.0, 1000.0],
        [20.0,  30.0]),
})

TASKS.append({
    "name": "D4_recurrence", "family": "log_base", "kind": "recurrence",
    "prompt": (
        "Write `ph_value(h_conc: float) -> float`.\n"
        "pH = -log10(h_conc).\n"
        "Test: ph_value(1e-7) = 7.0, ph_value(1e-3) = 3.0.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("ph_value",
        [1e-7, 1e-3],
        [7.0,  3.0]),
})

TASKS.append({
    "name": "D5_recurrence", "family": "log_base", "kind": "recurrence",
    "prompt": (
        "Write `richter(intensity_ratio: float) -> float`.\n"
        "M = log10(intensity_ratio).\n"
        "Test: richter(1000.0) = 3.0, richter(1e6) = 6.0.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("richter",
        [1000.0, 1e6],
        [3.0,    6.0]),
})

TASKS.append({
    "name": "D6_recurrence", "family": "log_base", "kind": "recurrence",
    "prompt": (
        "Write `bel(power_ratio: float) -> float`.\n"
        "Bel level = log10(power_ratio).\n"
        "Test: bel(100.0) = 2.0, bel(1000.0) = 3.0.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("bel",
        [100.0, 1000.0],
        [2.0,   3.0]),
})

# Near-miss D: natural log IS correct
TASKS.append({
    "name": "D7_near_miss", "family": "log_base", "kind": "near_miss",
    "prompt": (
        "Write `log_return(p0: float, p1: float) -> float`.\n"
        "Continuously compounded return = ln(p1/p0). Use math.log (natural log).\n"
        "Test: log_return(100.0, 110.0) ≈ 0.09531.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("log_return",
        [(100.0, 110.0)],
        [math.log(110 / 100)]),
})

TASKS.append({
    "name": "D8_near_miss", "family": "log_base", "kind": "near_miss",
    "prompt": (
        "Write `continuous_growth(p: float, r: float, t: float) -> float`.\n"
        "A = P * e^(r*t). Use math.exp.\n"
        "Test: continuous_growth(1000, 0.05, 2) ≈ 1105.17.\n"
        "Call python_exec to implement and test."
    ),
    "check": _make_check("continuous_growth",
        [(1000.0, 0.05, 2.0)],
        [1000.0 * math.exp(0.05 * 2.0)]),
})


# ── Build ordered task list ────────────────────────────────────────────────────
# Order: all 4 discovery → interleaved recurrences (4 families × 5 each) → near-misses

_fam_order = ["geomean", "vol_annualize", "compound_vs_simple", "log_base"]

_ordered: list[dict] = []

# 4 discovery tasks first
for fam in _fam_order:
    for t in TASKS:
        if t["family"] == fam and t["kind"] == "discovery":
            _ordered.append(t)

# Recurrences interleaved (max 5 per family × 4 families = 20)
_rec_by_fam: dict[str, list[dict]] = {}
for t in TASKS:
    if t["kind"] == "recurrence":
        _rec_by_fam.setdefault(t["family"], []).append(t)

max_rec = max(len(v) for v in _rec_by_fam.values())
for i in range(max_rec):
    for fam in _fam_order:
        tasks_for_fam = _rec_by_fam.get(fam, [])
        if i < len(tasks_for_fam):
            _ordered.append(tasks_for_fam[i])

# Near-miss tasks last
for fam in _fam_order:
    for t in TASKS:
        if t["family"] == fam and t["kind"] == "near_miss":
            _ordered.append(t)

TASKS = _ordered


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_benchmark() -> None:
    print("\n" + "═" * 72)
    print("  eval_learn — Cold-start motif learning benchmark")
    print(f"  {len(TASKS)} tasks | 4 families | 0 pre-seeded motifs")
    print("═" * 72)

    embedder = build_embedder()
    brain    = BrainAgent(embedder, threshold=0.45, k=5)
    # NO seed_motifs() — pure cold start

    agent = tool_agent(["python_exec"], max_turns=MAX_TURNS, model=HAIKU, max_tokens=4096)
    agent.monitor = brain

    results: list[dict] = []

    for i, task in enumerate(TASKS):
        n = i + 1
        print(f"\n[{n:02d}/{len(TASKS)}] {task['name']:<22} ({task['family']}, {task['kind']})")

        brain.set_task(i, task=task["prompt"][:300])
        brain.reset()

        t0 = time.time()
        try:
            trace, tokens = agent(task["prompt"])
        except Exception as e:
            print(f"  agent error: {e}")
            trace = f"[agent error: {e}]"
            tokens = 0

        elapsed = time.time() - t0

        # Capture fire state BEFORE store() — reset() on next task clears it
        fired     = brain.last_fire is not None
        fire_info = {k: v for k, v in brain.last_fire.items()} if fired else None

        passed, detail = _verify(trace, task["check"])

        # Store first-attempt trace + label; triggers motif extraction on failure
        brain.store(trace, int(passed), detail[:300] if not passed else "")

        status   = "PASS" if passed else "FAIL"
        fire_tag = " [FIRED]" if fired else ""
        detail_s = detail[:80] if detail else "ok"
        print(f"  → {status}{fire_tag} | {elapsed:.1f}s | {tokens} tok | {detail_s}")

        results.append({
            "idx":      i,
            "name":     task["name"],
            "family":   task["family"],
            "kind":     task["kind"],
            "passed":   passed,
            "fired":    fired,
            "detail":   detail[:300],
            "tokens":   tokens,
            "elapsed":  round(elapsed, 1),
            "fire_info": fire_info,
        })

    # ── Compute metrics ────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  RESULTS")
    print("═" * 72)

    by_kind: dict[str, list[dict]] = {"discovery": [], "recurrence": [], "near_miss": []}
    for r in results:
        by_kind[r["kind"]].append(r)

    discoveries = by_kind["discovery"]
    recurrences = by_kind["recurrence"]
    near_misses = by_kind["near_miss"]

    n_disc_fail     = sum(1 for r in discoveries if not r["passed"])
    n_rec_fired     = sum(1 for r in recurrences if r["fired"])
    n_nm_fp         = sum(1 for r in near_misses if r["fired"])
    families_failed = {r["family"] for r in discoveries if not r["passed"]}
    families_total  = {r["family"] for r in discoveries}

    first_fail_rate  = n_disc_fail / max(len(discoveries), 1)
    motif_extr_rate  = len(families_failed) / max(len(families_total), 1)
    rep_prev_rate    = n_rec_fired / max(len(recurrences), 1)
    fp_rate          = n_nm_fp / max(len(near_misses), 1)

    # Per-family breakdown
    print(f"\n{'Family':<26} {'Disc fail':<14} {'Rec fired':<14} {'NM false+':}")
    print("-" * 66)
    for fam in _fam_order:
        disc = [r for r in results if r["family"] == fam and r["kind"] == "discovery"]
        rec  = [r for r in results if r["family"] == fam and r["kind"] == "recurrence"]
        nm   = [r for r in results if r["family"] == fam and r["kind"] == "near_miss"]
        print(
            f"  {fam:<24} "
            f"{sum(1 for r in disc if not r['passed'])}/{len(disc):<12} "
            f"{sum(1 for r in rec if r['fired'])}/{len(rec):<12} "
            f"{sum(1 for r in nm if r['fired'])}/{len(nm)}"
        )

    print("\n  OVERALL")
    print(f"  first_fail_rate          : {first_fail_rate:.0%}  ({n_disc_fail}/{len(discoveries)} discovery tasks failed)")
    print(f"  motif_extraction_rate    : {motif_extr_rate:.0%}  ({len(families_failed)}/{len(families_total)} families have learnable failure)")
    print(f"  repeat_prevention_rate   : {rep_prev_rate:.0%}  ({n_rec_fired}/{len(recurrences)} recurrence tasks — brain fired)")
    print(f"  false_positive_rate      : {fp_rate:.0%}  ({n_nm_fp}/{len(near_misses)} near-miss tasks — brain fired incorrectly)")
    print(f"  total_tokens             : {sum(r['tokens'] for r in results):,}")
    print(f"  total_time               : {sum(r['elapsed'] for r in results):.0f}s")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": HAIKU,
        "cold_start": True,
        "n_tasks": len(TASKS),
        "metrics": {
            "first_fail_rate":        round(first_fail_rate, 4),
            "motif_extraction_rate":  round(motif_extr_rate, 4),
            "repeat_prevention_rate": round(rep_prev_rate, 4),
            "false_positive_rate":    round(fp_rate, 4),
            "total_tokens":           sum(r["tokens"] for r in results),
            "total_time_s":           round(sum(r["elapsed"] for r in results), 1),
        },
        "per_family": {},
        "tasks": results,
    }
    for fam in _fam_order:
        disc = [r for r in results if r["family"] == fam and r["kind"] == "discovery"]
        rec  = [r for r in results if r["family"] == fam and r["kind"] == "recurrence"]
        nm   = [r for r in results if r["family"] == fam and r["kind"] == "near_miss"]
        summary["per_family"][fam] = {
            "disc_fail":  sum(1 for r in disc if not r["passed"]),
            "disc_total": len(disc),
            "rec_fired":  sum(1 for r in rec if r["fired"]),
            "rec_total":  len(rec),
            "nm_fp":      sum(1 for r in nm if r["fired"]),
            "nm_total":   len(nm),
        }

    out_path = OUT / "eval_learn.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("═" * 72)


if __name__ == "__main__":
    run_benchmark()
