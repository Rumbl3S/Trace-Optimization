"""vibe_session.py — Brain-monitored vibe-coding: 30-task personal finance toolkit.

The brain fires MID-GENERATION when the tool-agent's trajectory enters a
region that historically preceded failures. It injects both:
  STOP: [the exact mistake seen in the nearest stored failure]
  FIX:  [the working approach from the nearest stored success]

No pre-seeding. The brain starts cold and learns as the session progresses.
By task ~5 it has enough stored runs to start firing meaningful interventions.

Tasks (sequential, difficulty increases, later phases share recurring failure patterns):
  Phase 1 — Data model:       Transaction, Store, running_balance, monthly_summary
  Phase 2 — Analytics:        savings_rate, expense_ratios, budget, date_filter
  Phase 3 — Forecasting:      moving_avg, cash_flow, compound_interest, loan
  Phase 4 — Multi-asset:      rebalancing, time_weighted_return, allocation  (tasks 13-14)
  Phase 4b— Analytics (HARD): weighted_vol, rolling_sharpe, sharpe, sortino, info_ratio
                               Shared trap: population std (n not n-1); compound rf
  Phase 5 — Return-based risk: drawdown_series, calmar, capm, tracking_error, cvar
                               Shared trap: compound price series from returns
  Phase 6 — Advanced (HARD):  parametric_var, omega, GBM Monte Carlo, risk_report
                               Combines all traps

Run:
    python vibe_session.py
"""
from __future__ import annotations

import contextlib
import datetime
import io
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from trace_use import build_embedder, tool_agent
from trace_use import BrainAgent
from eval.eval_hard import _extract_code as _extract_tool_code

console = Console()

HAIKU     = "claude-haiku-4-5-20251001"
MAX_TURNS = 12
NO_EVAL   = (
    "Do NOT use Python's eval(), compile(), or exec(). "
    "Write standalone functions only. No classes unless the task requires it."
)


# ══════════════════════════════════════════════════════════════════════════════
#  ABSTRACT FAILURE MOTIFS
#  Named failure patterns given to brain before task 1 via seed_motifs().
#  Each motif fires when the pattern regex matches code — never from domain
#  similarity alone. neg_pattern suppresses false positives when correct code
#  coincidentally matches the surface.
# ══════════════════════════════════════════════════════════════════════════════

_BRAIN_MOTIFS = [
    {
        "id": "additive_rate_conversion",
        "name": "Additive rate conversion",
        "description": "rf/252 is additive, not compound. Correct: (1+rf)**(1/252)-1",
        "surface_pattern": r"(rf|risk_free|rate)\s*[^(=\n]*?/\s*252",
        "neg_pattern": r"\(1\s*\+.*?\)\s*\*\*|\bexp\s*\(",
        "task_keywords": ["sharpe", "risk_free", "risk free", "rf", "rate"],
        "confidence": 0.88,
        "recommendation": "rf_daily = (1 + risk_free_annual)**(1/252) - 1",
    },
    {
        "id": "sample_std_for_population",
        "name": "Sample std instead of population std",
        "description": "ddof=1 or statistics.stdev divides by n-1; population needs ddof=0",
        "surface_pattern": r"ddof\s*=\s*1|statistics\.stdev\s*\(",
        "neg_pattern": r"ddof\s*=\s*0",
        "task_keywords": ["volatility", "vol", "std", "standard deviation", "variance"],
        "confidence": 0.92,
        "recommendation": "np.std(x, ddof=0)  or  sum((x-mu)**2)/n",
    },
    {
        "id": "missing_ito_correction",
        "name": "Missing Ito correction in GBM",
        "description": "GBM drift must be (mu - 0.5*sigma^2)/dt, not mu/dt",
        "surface_pattern": r"\bdrift\s*=\s*mu|\bmu_annual\s*/\s*252",
        "neg_pattern": r"0\.5\s*\*?\s*sigma|-\s*0\.5",
        "task_keywords": ["gbm", "monte carlo", "geometric brownian", "simulation", "stochastic"],
        "confidence": 0.85,
        "recommendation": "drift = (mu_annual - 0.5 * sigma_annual**2) / 252",
    },
    {
        "id": "sortino_wrong_denominator",
        "name": "Sortino downside dev divides by count of negatives",
        "description": "Should divide by len(ALL returns), not len(negative returns)",
        "surface_pattern": r"/\s*len\s*\(.*?(?:neg|below|downside|bad|loss)",
        "neg_pattern": "",
        "task_keywords": ["sortino", "downside", "downside deviation", "downside risk"],
        "confidence": 0.88,
        "recommendation": "sqrt(sum(min(r-mar,0)**2 for r in returns) / len(returns))",
    },
    {
        "id": "arithmetic_annualization",
        "name": "Arithmetic annualization instead of geometric",
        "description": "mean(daily_returns)*252 is wrong; use compound (1+total)**(252/n)-1",
        "surface_pattern": r"\bmean\s*\(.*?\)\s*\*\s*252|sum\s*\(.*?\)\s*/\s*\w+\s*\)\s*\*\s*252",
        "neg_pattern": r"\*\*\s*\(\s*252|\bpow\s*\(",
        "task_keywords": ["calmar", "annuali", "annual return", "cagr"],
        "confidence": 0.80,
        "recommendation": "(1 + total_return)**(252/n) - 1  where total_return = product(1+r)-1",
    },
    {
        "id": "two_tailed_var_zscore",
        "name": "Two-tailed z-score for one-tailed VaR",
        "description": "VaR at 95% needs one-tailed z=1.6449, not two-tailed z=1.96",
        "surface_pattern": r"\b1\.96\b|\b2\.576\b",
        "neg_pattern": r"\b1\.6449\b|\b1\.645\b|\b2\.326\b",
        "task_keywords": ["var", "value at risk", "parametric", "confidence"],
        "confidence": 0.85,
        "recommendation": "z = {0.95: 1.6449, 0.99: 2.3263}[confidence]",
    },
    {
        "id": "tracking_error_wrong_annualization",
        "name": "Tracking error scaled by 252 not sqrt(252)",
        "description": (
            "Tracking error is a volatility measure — it annualizes by multiplying std by "
            "sqrt(252), not by 252. Multiplying by 252 annualizes a mean/return, not a std."
        ),
        "surface_pattern": r"(?:np\.)?std\s*\([^)]*\)\s*\*\s*252\b",
        "neg_pattern": r"sqrt\s*\(\s*252\s*\)|252\s*\*\*\s*0\.5|\*\s*\(\s*252\s*\)",
        "task_keywords": ["information ratio", "tracking error", "active return", "te", "ir"],
        "confidence": 0.86,
        "recommendation": "tracking_error = np.std(active_returns, ddof=0) * np.sqrt(252)",
    },
    {
        "id": "cvar_wrong_percentile_tail",
        "name": "CVaR uses high percentile (profit tail) not low tail",
        "description": (
            "CVaR/Expected Shortfall at confidence p averages returns BELOW the (1-p) "
            "percentile. np.percentile(returns, 95) selects the profit tail, not the loss tail."
        ),
        "surface_pattern": r"np\.percentile\s*\([^,]+,\s*9[0-9]\.?\d*\s*\)",
        "neg_pattern": r"np\.percentile\s*\([^,]+,\s*[0-9]\.?\d*\s*\)",
        "task_keywords": ["cvar", "expected shortfall", "conditional var", "es", "tail risk"],
        "confidence": 0.83,
        "recommendation": (
            "cvar = np.mean(returns[returns <= np.percentile(returns, (1-confidence)*100)])"
        ),
    },
    {
        "id": "beta_std_ratio_not_covariance",
        "name": "Beta computed as std ratio instead of cov/var",
        "description": (
            "Beta = cov(asset, market) / var(market). Using std(asset)/std(market) gives "
            "the ratio of volatilities (≈correlation when combined with correlation coefficient), "
            "not market sensitivity."
        ),
        "surface_pattern": r"\bstd\s*\([^)]*\)\s*/\s*(?:np\.)?std\s*\(",
        "neg_pattern": r"\bcov\s*\(|np\.cov\s*\(",
        "task_keywords": ["beta", "capm", "market return", "systematic risk", "benchmark"],
        "confidence": 0.84,
        "recommendation": "beta = np.cov(asset_returns, market_returns)[0,1] / np.var(market_returns, ddof=0)",
    },
]



# ══════════════════════════════════════════════════════════════════════════════
#  HIDDEN VERIFIERS  (edge cases the model consistently misses)
# ══════════════════════════════════════════════════════════════════════════════



def _exec_ns(code: str) -> dict | str:
    ns: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<vs>", "exec"), ns)  # noqa: S102
        return ns
    except Exception as e:
        return str(e)


class T:
    """Minimal Transaction-like object for checks.
    Supports both attribute access (t.amount) and dict access (t["amount"]).
    """
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def _d(y, m, dd): return datetime.date(y, m, dd)


# ── Phase 1: Data model ───────────────────────────────────────────────────────

def _check_transaction(ns) -> tuple[bool, str]:
    Tc = ns.get("Transaction"); S = ns.get("TransactionStore")
    if not Tc or not S:
        return False, "Transaction or TransactionStore not defined"
    try:
        t1 = Tc(id="a", date=_d(2024,1,1), amount=100.0, category="food",
                description="x", type="expense")
        t2 = Tc(id="b", date=_d(2024,1,2), amount=500.0, category="salary",
                description="y", type="income")
        s = S(); s.add(t1); s.add(t2)
        assert len(s.get_all()) == 2
        assert s.get_by_category("food") == [t1]
        assert s.get_by_type("income") == [t2]
        assert s.get_by_category("none") == []
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_running_balance(ns) -> tuple[bool, str]:
    fn = ns.get("running_balance")
    if not fn: return False, "running_balance not defined"
    try:
        txns = [
            T(date=_d(2024,1,3), amount=200.0, type="expense"),
            T(date=_d(2024,1,1), amount=1000.0, type="income"),
            T(date=_d(2024,1,2), amount=300.0, type="expense"),
        ]
        r = fn(txns)
        assert r, "empty result"
        dates, bals = zip(*r)
        assert list(dates) == sorted(dates), "not sorted by date"
        assert abs(bals[0] - 1000.0) < 1e-9, f"after income: {bals[0]}≠1000"
        assert abs(bals[1] - 700.0)  < 1e-9, f"after jan2 expense: {bals[1]}≠700"
        assert abs(bals[2] - 500.0)  < 1e-9, f"after jan3 expense: {bals[2]}≠500"
        assert fn([]) == [], "empty input must return []"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_monthly_summary(ns) -> tuple[bool, str]:
    fn = ns.get("monthly_summary")
    if not fn: return False, "monthly_summary not defined"
    try:
        txns = [
            T(date=_d(2024,1,5),  amount=500.0,  type="income"),
            T(date=_d(2024,1,20), amount=100.0,  type="expense"),
            T(date=_d(2024,2,10), amount=1000.0, type="income"),
            T(date=_d(2024,2,15), amount=300.0,  type="expense"),
        ]
        r = fn(txns)
        assert (2024, 1) in r, f"key (2024,1) missing — got: {list(r.keys())[:3]}"
        jan = r[(2024, 1)]
        assert abs(jan["income"]   - 500.0) < 1e-9
        assert abs(jan["expenses"] - 100.0) < 1e-9
        assert abs(jan["net"]      - 400.0) < 1e-9
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_savings_rate(ns) -> tuple[bool, str]:
    fn = ns.get("savings_rate")
    if not fn: return False, "savings_rate not defined"
    try:
        txns = [T(amount=1000.0, type="income", date=_d(2024,1,1)),
                T(amount=300.0,  type="expense", date=_d(2024,1,1))]
        r = fn(txns)
        assert abs(r - 0.7) < 1e-9, f"expected 0.7, got {r} (divide by income not expenses)"
        assert fn([]) == 0.0, "empty → 0.0"
        assert fn([T(amount=100.0, type="expense", date=_d(2024,1,1))]) == 0.0, "no income → 0.0"
        r2 = fn([T(amount=200.0, type="income", date=_d(2024,1,1)),
                 T(amount=50.0,  type="expense", date=_d(2024,1,1))])
        assert r2 > 0, f"must be positive when saving, got {r2}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_expense_ratios(ns) -> tuple[bool, str]:
    fn = ns.get("expense_ratios")
    if not fn: return False, "expense_ratios not defined"
    try:
        txns = [T(amount=400.0, category="food",      type="expense"),
                T(amount=400.0, category="rent",      type="expense"),
                T(amount=200.0, category="transport", type="expense"),
                T(amount=999.0, category="salary",    type="income")]
        r = fn(txns)
        assert "salary" not in r, "income must be excluded from expense ratios"
        assert abs(r["food"]      - 40.0) < 1e-9, f"food: {r['food']}≠40"
        assert abs(r["transport"] - 20.0) < 1e-9
        total = sum(r.values())
        assert abs(total - 100.0) < 1e-6, f"ratios sum to {total}≠100"
        assert fn([]) == {}, "empty → {}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_budget(ns) -> tuple[bool, str]:
    B = ns.get("Budget")
    if not B: return False, "Budget class not defined"
    try:
        b = B(limits={"food": 200.0, "transport": 100.0})
        txns = [T(amount=150.0, category="food",      type="expense"),
                T(amount=120.0, category="transport", type="expense"),
                T(amount=500.0, category="salary",    type="income")]
        r = b.track(txns)
        spent = r.get("spent_per_category") or r.get("spent", {})
        over  = r.get("over_budget", [])
        assert abs(spent.get("food", -1) - 150.0) < 1e-9
        assert "transport" in over, f"transport over budget, got: {over}"
        assert "food" not in over, f"food under budget, should not be over: {over}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_date_filter(ns) -> tuple[bool, str]:
    fn = ns.get("filter_by_date")
    if not fn: return False, "filter_by_date not defined"
    try:
        txns = [T(date=_d(2024,3,1)), T(date=_d(2024,3,15)), T(date=_d(2024,3,31))]
        r = fn(txns, _d(2024,3,1), _d(2024,3,31))
        assert len(r) == 3, f"inclusive range: expected 3, got {len(r)}"
        r2 = fn(txns, _d(2024,3,15), _d(2024,3,31))
        assert len(r2) == 2, f"from mar15: expected 2, got {len(r2)}"
        r3 = fn(txns, _d(2024,3,1), _d(2024,3,1))
        assert len(r3) == 1, f"single day: expected 1, got {len(r3)}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_moving_avg(ns) -> tuple[bool, str]:
    fn = ns.get("moving_avg_spend")
    if not fn: return False, "moving_avg_spend not defined"
    try:
        txns = [
            T(date=_d(2024,1,1), amount=30.0, type="expense"),
            T(date=_d(2024,1,2), amount=60.0, type="expense"),
            T(date=_d(2024,1,3), amount=90.0, type="expense"),
            T(date=_d(2024,1,1), amount=500.0, type="income"),
        ]
        r = fn(txns, window=3)
        assert len(r) >= 3, f"expected ≥3 entries"
        _, avgs = zip(*r)
        assert abs(avgs[-1] - 60.0) < 1e-9, f"3-day avg: {avgs[-1]}≠60"
        r2 = fn([T(date=_d(2024,1,1), amount=1000.0, type="income")], window=3)
        _, a2 = zip(*r2) if r2 else ([], [0.0])
        assert all(a == 0.0 for a in a2), "income should not count as spend"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_cash_flow(ns) -> tuple[bool, str]:
    fn = ns.get("forecast")
    if not fn: return False, "forecast not defined"
    try:
        txns = [
            T(date=_d(2024,1,1), amount=100.0, type="income"),
            T(date=_d(2024,1,1), amount=50.0,  type="expense"),
            T(date=_d(2024,1,2), amount=100.0, type="income"),
            T(date=_d(2024,1,2), amount=50.0,  type="expense"),
        ]
        r = fn(txns, days=5)
        assert isinstance(r, list) and len(r) == 5, f"expected 5 items, got {r}"
        _, bals = zip(*r)
        assert all(bals[i] < bals[i+1] for i in range(len(bals)-1)), \
            "balances should increase when net>0"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_compound(ns) -> tuple[bool, str]:
    fn = ns.get("compound_interest")
    if not fn: return False, "compound_interest not defined"
    try:
        r = fn(1000, 0.05, 1, n=12)
        assert abs(r - 1051.162) < 0.01, f"monthly: expected ~1051.16, got {r}"
        r2 = fn(1000, 0.10, 1, n=1)
        assert abs(r2 - 1100.0) < 1e-6, f"annual: expected 1100, got {r2}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_loan(ns) -> tuple[bool, str]:
    fn = ns.get("loan_amortization")
    if not fn: return False, "loan_amortization not defined"
    try:
        sched = fn(1000, 0.12, 12)
        assert isinstance(sched, list) and len(sched) == 12, f"expected 12 rows"
        final_balance = sched[-1].get("balance", -1)
        assert final_balance < 1.0, f"final balance should be ~0, got {final_balance}"
        assert sum(r.get("interest", 0) for r in sched) > 0
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_allocation(ns) -> tuple[bool, str]:
    fn = ns.get("portfolio_allocation")
    if not fn: return False, "portfolio_allocation not defined"
    try:
        holdings = {"AAPL": 1000.0, "MSFT": 1000.0, "GOOGL": 2000.0}
        r = fn(holdings)
        assert abs(r["AAPL"]  - 25.0) < 1e-9, f"AAPL: {r['AAPL']}≠25"
        assert abs(r["GOOGL"] - 50.0) < 1e-9, f"GOOGL: {r['GOOGL']}≠50"
        assert abs(sum(r.values()) - 100.0) < 1e-6
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_rebalance(ns) -> tuple[bool, str]:
    fn = ns.get("rebalance_trades")
    if not fn: return False, "rebalance_trades not defined"
    try:
        current = {"AAPL": 3000.0, "MSFT": 1000.0}
        target  = {"AAPL": 0.5, "MSFT": 0.5}
        trades  = fn(current, target)
        aapl = trades.get("AAPL", 0)
        msft = trades.get("MSFT", 0)
        assert aapl < 0, f"AAPL should be sold (negative), got {aapl}"
        assert msft > 0, f"MSFT should be bought (positive), got {msft}"
        assert abs(abs(aapl) - abs(msft)) < 1e-6
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_twr(ns) -> tuple[bool, str]:
    fn = ns.get("time_weighted_return")
    if not fn: return False, "time_weighted_return not defined"
    try:
        periods = [
            {"start_value": 1000.0, "end_value": 1200.0, "cashflow": 0.0},
            {"start_value": 1200.0, "end_value": 1080.0, "cashflow": 0.0},
        ]
        r = fn(periods)
        assert abs(r - 0.08) < 1e-9, f"TWR: expected 0.08, got {r}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_weighted_vol(ns) -> tuple[bool, str]:
    fn = ns.get("weighted_volatility")
    if not fn: return False, "weighted_volatility not defined"
    try:
        import math
        weights = {"A": 0.6, "B": 0.4}
        asset_returns = {
            "A": [0.01, -0.02, 0.03, -0.01, 0.02],
            "B": [0.02, -0.01, 0.01, -0.03, 0.03],
        }
        port = [0.6*a + 0.4*b for a, b in zip(asset_returns["A"], asset_returns["B"])]
        n = len(port); mu = sum(port) / n
        pop_std = math.sqrt(sum((r - mu) ** 2 for r in port) / n)
        expected = pop_std * math.sqrt(252)
        r = fn(weights, asset_returns)
        assert abs(r - expected) < 0.001, f"expected {expected:.6f} (population std), got {r:.6f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_rolling_sharpe(ns) -> tuple[bool, str]:
    fn = ns.get("rolling_sharpe")
    if not fn: return False, "rolling_sharpe not defined"
    try:
        import math
        returns = [0.002, -0.001, 0.003, -0.002, 0.001, 0.002, -0.001, 0.003, -0.002, 0.001]
        window = 4
        result = fn(returns, window=window, risk_free_daily=0.0)
        assert len(result) == len(returns), f"length must be {len(returns)}, got {len(result)}"
        for i in range(window - 1):
            assert result[i] is None or result[i] == 0.0, f"result[{i}] should be None before first window"
        w = returns[0:window]; mu_w = sum(w) / window
        std_w = math.sqrt(sum((x - mu_w) ** 2 for x in w) / window)
        expected_3 = (mu_w / std_w) * math.sqrt(252)
        assert abs(result[window - 1] - expected_3) < 0.01, \
            f"rolling_sharpe[{window-1}]: expected {expected_3:.4f}, got {result[window-1]:.4f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_sharpe_strict(ns) -> tuple[bool, str]:
    fn = ns.get("sharpe_ratio")
    if not fn: return False, "sharpe_ratio not defined"
    try:
        import math
        # rf=0.50 makes compound vs rf/252 differ by ~0.74 Sharpe units
        returns = [0.01, -0.005, 0.015, -0.01, 0.008, 0.003, -0.007, 0.012, 0.002, 0.007]
        rf_annual = 0.50
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        excess = [r - rf_daily for r in returns]
        n = len(excess); mu = sum(excess) / n
        pop_std = math.sqrt(sum((e - mu) ** 2 for e in excess) / n)
        expected = (mu / pop_std) * math.sqrt(252)
        r = fn(returns, risk_free_annual=rf_annual)
        assert abs(r - expected) < 0.01, \
            f"sharpe(rf=0.50): expected {expected:.4f} (compound rf), got {r:.4f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_sortino_strict(ns) -> tuple[bool, str]:
    fn = ns.get("sortino_ratio")
    if not fn: return False, "sortino_ratio not defined"
    try:
        import math
        returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01, 0.01, -0.01]
        mar = 0.0
        excess = [r - mar for r in returns]
        n = len(excess)
        downside_var = sum(min(e, 0) ** 2 for e in excess) / n
        downside_dev = math.sqrt(downside_var)
        mu_excess = sum(excess) / n
        expected = (mu_excess * math.sqrt(252)) / downside_dev
        r = fn(returns, mar=mar)
        assert abs(r - expected) < 0.01, \
            f"sortino: expected {expected:.4f} (div by n_total={n}), got {r:.4f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_information_ratio(ns) -> tuple[bool, str]:
    fn = ns.get("information_ratio")
    if not fn: return False, "information_ratio not defined"
    try:
        import math
        pr = [0.002, 0.003, -0.001, 0.004, 0.001, -0.001, 0.003]
        br = [0.001, 0.002,  0.001, 0.002, 0.000,  0.001, 0.001]
        active = [p - b for p, b in zip(pr, br)]
        n = len(active); mu = sum(active) / n
        pop_std = math.sqrt(sum((a - mu) ** 2 for a in active) / n)
        expected = (mu / pop_std) * math.sqrt(252)
        r = fn(pr, br)
        assert abs(r - expected) < 0.01, f"IR: expected {expected:.4f}, got {r:.4f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_drawdown_series(ns) -> tuple[bool, str]:
    fn = ns.get("drawdown_series")
    if not fn: return False, "drawdown_series not defined"
    try:
        returns = [0.10, 0.05, -0.20, -0.10, 0.15]
        vals = [1.0]
        for r in returns:
            vals.append(vals[-1] * (1 + r))
        peak = 1.0; expected = []
        for v in vals[1:]:
            peak = max(peak, v)
            expected.append((peak - v) / peak)
        result = fn(returns)
        assert len(result) == 5, f"expected length 5, got {len(result)}"
        for i, (got, exp) in enumerate(zip(result, expected)):
            assert abs(got - exp) < 1e-9, f"drawdown[{i}]: expected {exp:.6f}, got {got:.6f}"
        assert fn([0.01, 0.02, 0.03]) == [0.0, 0.0, 0.0], "monotone increase must give zero drawdown"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_calmar(ns) -> tuple[bool, str]:
    fn = ns.get("calmar_ratio")
    if not fn: return False, "calmar_ratio not defined"
    try:
        import math
        returns = [0.05, -0.10, 0.08, -0.15, 0.12, -0.05, 0.10]
        n = len(returns)
        vals = [1.0]
        for r in returns:
            vals.append(vals[-1] * (1 + r))
        total_return = vals[-1] - 1.0
        ann_return = (1 + total_return) ** (252 / n) - 1
        peak = vals[0]; max_dd = 0.0
        for v in vals[1:]:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)
        expected = ann_return / max_dd
        r = fn(returns)
        assert abs(r - expected) < 0.001, \
            f"calmar: expected {expected:.4f} (compound annualisation), got {r:.4f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_capm_metrics(ns) -> tuple[bool, str]:
    fn = ns.get("capm_metrics")
    if not fn: return False, "capm_metrics not defined"
    try:
        pr = [0.002, -0.001, 0.003, -0.002, 0.001, 0.002, -0.001]
        mr = [0.001, -0.001, 0.002, -0.001, 0.001, 0.001, -0.001]
        rf = 0.0001
        ep = [r - rf for r in pr]; em = [r - rf for r in mr]
        n = len(ep); mu_ep = sum(ep) / n; mu_em = sum(em) / n
        cov   = sum((e - mu_ep) * (m - mu_em) for e, m in zip(ep, em)) / n
        var_m = sum((m - mu_em) ** 2 for m in em) / n
        beta  = cov / var_m
        alpha_annual = (mu_ep - beta * mu_em) * 252
        r = fn(pr, mr, risk_free_daily=rf)
        assert abs(r["beta"]         - beta)         < 0.001, f"beta: expected {beta:.4f}, got {r['beta']}"
        assert abs(r["alpha_annual"] - alpha_annual) < 0.001, f"alpha_annual: expected {alpha_annual:.4f}, got {r['alpha_annual']}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_tracking_error(ns) -> tuple[bool, str]:
    fn = ns.get("tracking_error")
    if not fn: return False, "tracking_error not defined"
    try:
        import math
        pr = [0.002, -0.001, 0.003, -0.002, 0.001, 0.002, -0.001, 0.003]
        br = [0.001,  0.000, 0.002, -0.001, 0.001, 0.001,  0.000, 0.002]
        active = [p - b for p, b in zip(pr, br)]
        n = len(active); mu = sum(active) / n
        pop_std = math.sqrt(sum((a - mu) ** 2 for a in active) / n)
        r_ann = fn(pr, br, annualize=True)
        assert abs(r_ann - pop_std * math.sqrt(252)) < 0.001, \
            f"annualized TE: expected {pop_std*math.sqrt(252):.6f}, got {r_ann:.6f}"
        r_raw = fn(pr, br, annualize=False)
        assert abs(r_raw - pop_std) < 1e-6, f"raw TE: expected {pop_std:.6f}, got {r_raw:.6f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_expected_shortfall(ns) -> tuple[bool, str]:
    fn = ns.get("expected_shortfall")
    if not fn: return False, "expected_shortfall not defined"
    try:
        returns = [0.01, -0.02, 0.03, -0.05, 0.02, -0.01, 0.04, -0.03, 0.01, -0.04]
        sorted_r = sorted(returns); n = len(sorted_r)
        n_tail_95 = max(1, int((1 - 0.95) * n))
        expected_95 = -sum(sorted_r[:n_tail_95]) / n_tail_95
        r95 = fn(returns, confidence=0.95)
        assert abs(r95 - expected_95) < 1e-9, f"CVaR 95%: expected {expected_95:.4f}, got {r95:.4f}"
        n_tail_80 = max(1, int((1 - 0.80) * n))
        expected_80 = -sum(sorted_r[:n_tail_80]) / n_tail_80
        r80 = fn(returns, confidence=0.80)
        assert abs(r80 - expected_80) < 1e-9, f"CVaR 80%: expected {expected_80:.4f}, got {r80:.4f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_parametric_var(ns) -> tuple[bool, str]:
    fn = ns.get("parametric_var")
    if not fn: return False, "parametric_var not defined"
    try:
        import math
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.015, 0.025, -0.005]
        n = len(returns); mu = sum(returns) / n
        pop_std = math.sqrt(sum((r - mu) ** 2 for r in returns) / n)
        exp_95 = -(mu - 1.6449 * pop_std)
        exp_99 = -(mu - 2.3263 * pop_std)
        r95 = fn(returns, confidence=0.95)
        r99 = fn(returns, confidence=0.99)
        assert abs(r95 - exp_95) < 0.001, f"pVaR 95%: expected {exp_95:.6f} (z=1.6449), got {r95:.6f}"
        assert abs(r99 - exp_99) < 0.001, f"pVaR 99%: expected {exp_99:.6f} (z=2.3263), got {r99:.6f}"
        assert r99 > r95, "99% VaR must exceed 95% VaR"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_omega_ratio(ns) -> tuple[bool, str]:
    fn = ns.get("omega_ratio")
    if not fn: return False, "omega_ratio not defined"
    try:
        # 3 gainers, 5 losers at threshold=0.01 → wrong denominator is detectable
        returns = [0.05, -0.02, 0.03, -0.01, -0.03, -0.01, 0.02, -0.01]
        n = len(returns)
        for thresh in [0.0, 0.01]:
            g = sum(max(r - thresh, 0) for r in returns) / n
            l = sum(max(thresh - r, 0) for r in returns) / n
            expected = g / l
            r = fn(returns, threshold=thresh)
            assert abs(r - expected) < 0.001, \
                f"omega (thresh={thresh}): expected {expected:.4f}, got {r:.4f}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_gbm_mc(ns) -> tuple[bool, str]:
    fn = ns.get("monte_carlo_gbm")
    if not fn: return False, "monte_carlo_gbm not defined"
    try:
        r = fn(initial_value=10000.0, mu_annual=0.10, sigma_annual=0.20,
               days=252, n_sims=1000, seed=42)
        assert isinstance(r, dict), "must return dict"
        med = r.get("median") or r.get("p50")
        assert med is not None, f"missing 'median' key, got: {list(r.keys())}"
        p5 = r.get("p5"); p95 = r.get("p95")
        assert p5 is not None and p95 is not None, "missing p5/p95"
        assert p5 < med < p95, "must have p5 < median < p95"
        # WITH Ito: ~10812-10920; WITHOUT Ito: ~11031-11141. Cutoff 10980 separates them.
        assert 8000 < med < 10980, \
            f"median {med:.0f} suggests missing Ito correction (drift must be (mu-0.5σ²)/252)"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_full_risk_report(ns) -> tuple[bool, str]:
    fn = ns.get("risk_report")
    if not fn: return False, "risk_report not defined"
    try:
        import math
        portfolio = {"A": 6000.0, "B": 4000.0}
        asset_returns = {
            "A": [0.01, -0.02, 0.03, -0.01, 0.02, -0.01, 0.02],
            "B": [0.02, -0.01, 0.01, -0.03, 0.03,  0.01, -0.01],
        }
        benchmark = [0.01, -0.01, 0.02, -0.01, 0.02, 0.00, 0.01]
        r = fn(portfolio, asset_returns, benchmark)
        assert isinstance(r, dict)
        for key in ["total_value", "volatility", "max_drawdown", "sharpe", "beta"]:
            assert key in r, f"missing key: {key}"
        assert r["total_value"] == 10000.0
        port = [0.6*a + 0.4*b for a, b in zip(asset_returns["A"], asset_returns["B"])]
        n = len(port); mu = sum(port) / n
        pop_std = math.sqrt(sum((x - mu) ** 2 for x in port) / n)
        assert abs(r["volatility"] - pop_std * math.sqrt(252)) < 0.01, \
            f"volatility: expected {pop_std*math.sqrt(252):.4f} (population std on weighted returns), got {r['volatility']:.4f}"
        vals = [1.0]
        for x in port: vals.append(vals[-1] * (1 + x))
        peak = 1.0; max_dd = 0.0
        for v in vals[1:]:
            peak = max(peak, v); max_dd = max(max_dd, (peak - v) / peak)
        assert abs(r["max_drawdown"] - max_dd) < 0.01, \
            f"max_drawdown: expected {max_dd:.4f} (price-series), got {r['max_drawdown']:.4f}"
        mu_b = sum(benchmark) / n
        cov   = sum((p - mu) * (b - mu_b) for p, b in zip(port, benchmark)) / n
        var_b = sum((b - mu_b) ** 2 for b in benchmark) / n
        beta  = cov / var_b
        assert abs(r["beta"] - beta) < 0.05, f"beta: expected {beta:.4f}, got {r['beta']:.4f}"
        return True, ""
    except Exception as e:
        return False, str(e)


# ── keep legacy names so the old TASKS entries still compile ───────────────────
def _check_inflation(ns) -> tuple[bool, str]:
    fn = ns.get("inflation_adjusted")
    if not fn: return False, "inflation_adjusted not defined"
    try:
        r = fn(1000.0, rate=0.02, years=10)
        expected = 1000.0 / (1.02 ** 10)
        assert abs(r - expected) < 0.01, f"expected {expected:.2f}, got {r}"
        r2 = fn(1000.0, rate=0.05, years=1)
        assert abs(r2 - 1000/1.05) < 0.01
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_var(ns) -> tuple[bool, str]:
    fn = ns.get("value_at_risk")
    if not fn: return False, "value_at_risk not defined"
    try:
        returns = sorted([0.01, -0.02, 0.03, -0.05, 0.02, -0.01, 0.04, -0.03, 0.01, -0.04])
        r = fn(returns, confidence=0.95)
        assert r >= 0, f"VaR should be positive loss, got {r}"
        assert abs(r - 0.05) < 0.01, f"95% VaR: expected ~0.05, got {r}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_stress_test(ns) -> tuple[bool, str]:
    fn = ns.get("stress_test")
    if not fn: return False, "stress_test not defined"
    try:
        r = fn({"AAPL": 10000.0, "MSFT": 5000.0}, {"AAPL": -0.20, "MSFT": -0.30})
        assert isinstance(r, dict)
        total_loss = r.get("total_loss") or r.get("loss")
        assert total_loss is not None, f"missing 'total_loss', got: {list(r.keys())}"
        assert abs(total_loss - 3500.0) < 1e-6, f"expected 3500 loss, got {total_loss}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_correlation(ns) -> tuple[bool, str]:
    fn = ns.get("returns_correlation")
    if not fn: return False, "returns_correlation not defined"
    try:
        a = [0.01, 0.02, -0.01, 0.03]
        assert abs(fn(a, a) - 1.0) < 1e-9
        assert abs(fn(a, [-x for x in a]) - (-1.0)) < 1e-9
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_kelly(ns) -> tuple[bool, str]:
    fn = ns.get("kelly_fraction")
    if not fn: return False, "kelly_fraction not defined"
    try:
        assert abs(fn(win_prob=0.6, win_loss_ratio=1.0) - 0.2) < 1e-9
        assert fn(win_prob=0.3, win_loss_ratio=1.0) <= 0
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_drawdown(ns) -> tuple[bool, str]:
    fn = ns.get("max_drawdown")
    if not fn: return False, "max_drawdown not defined"
    try:
        r = fn([100.0, 150.0, 90.0, 120.0, 80.0])
        expected = (150.0 - 80.0) / 150.0
        assert abs(r - expected) < 1e-9, f"expected {expected:.4f}, got {r}"
        assert fn([100.0, 110.0, 120.0]) == 0.0
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_sortino(ns) -> tuple[bool, str]:
    fn = ns.get("sortino_ratio")
    if not fn: return False, "sortino_ratio not defined"
    try:
        import math
        returns = [0.05, -0.02, 0.03, -0.04, 0.01]
        r = fn(returns, risk_free=0.0)
        neg = [-0.02, -0.04]
        downside_std = math.sqrt(sum(x**2 for x in neg) / len(returns))
        expected = (sum(returns) / len(returns)) / downside_std
        assert abs(r - expected) < 0.01, f"sortino: expected {expected:.4f}, got {r}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_sharpe(ns) -> tuple[bool, str]:
    fn = ns.get("sharpe_ratio")
    if not fn: return False, "sharpe_ratio not defined"
    try:
        import math
        returns = [0.01, 0.02, -0.01, 0.03, 0.0]
        mean_r  = sum(returns) / len(returns)
        std_r   = math.sqrt(sum((r - mean_r)**2 for r in returns) / len(returns))
        expected = mean_r / std_r if std_r > 0 else 0.0
        r = fn(returns, risk_free=0.0)
        assert abs(r - expected) < 0.01, f"sharpe: expected {expected:.4f}, got {r}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_ewma(ns) -> tuple[bool, str]:
    fn = ns.get("ewma_volatility")
    if not fn: return False, "ewma_volatility not defined"
    try:
        r = fn([0.01, 0.02, -0.01, 0.03], lam=0.94)
        assert isinstance(r, float) and r > 0
        r2 = fn([0.10, -0.10, 0.10, -0.10], lam=0.94)
        assert r2 > r, "high-var series should have higher EWMA vol"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_monte_carlo(ns) -> tuple[bool, str]:
    fn = ns.get("monte_carlo_portfolio")
    if not fn: return False, "monte_carlo_portfolio not defined"
    try:
        result = fn(
            initial_value=10000.0, daily_return=0.001,
            daily_vol=0.01, days=252, n_simulations=200, seed=42
        )
        assert isinstance(result, dict)
        med = result.get("median") or result.get("p50")
        assert med is not None, f"missing 'median', got: {list(result.keys())}"
        assert 9000 < med < 50000
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_risk_report(ns) -> tuple[bool, str]:
    fn = ns.get("risk_report")
    if not fn: return False, "risk_report not defined"
    try:
        r = fn(
            {"AAPL": 5000.0, "MSFT": 5000.0},
            {"AAPL": [0.01, -0.02, 0.03, -0.01, 0.02],
             "MSFT": [0.02, -0.01, 0.01, -0.03, 0.03]},
        )
        assert isinstance(r, dict)
        for key in ["total_value", "volatility", "max_drawdown"]:
            assert key in r, f"missing key: {key}"
        assert r["total_value"] == 10000.0
        return True, ""
    except Exception as e:
        return False, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

TASKS = [
    {
        "name": "Transaction model",
        "check": _check_transaction,
        "prompt": (
            "Write a `Transaction` dataclass: id (str), date (datetime.date), amount (float), "
            "category (str), description (str), type ('income'|'expense'). "
            "Write `TransactionStore` with add(t), get_all(), get_by_category(cat), get_by_type(t). "
            + NO_EVAL
        ),
    },
    {
        "name": "Running balance",
        "check": _check_running_balance,
        "prompt": (
            "Write `running_balance(transactions) -> list[(date, balance)]`. "
            "Income ADDS to balance; expenses SUBTRACT. Sort by date. Return [] for empty input. "
            "Start at balance=0." + NO_EVAL
        ),
    },
    {
        "name": "Monthly rollup",
        "check": _check_monthly_summary,
        "prompt": (
            "Write `monthly_summary(transactions) -> dict[(year, month): dict]`. "
            "Key MUST be (year, month) integer tuple. "
            "Each value: {'income': float, 'expenses': float, 'net': float}." + NO_EVAL
        ),
    },
    {
        "name": "Savings rate",
        "check": _check_savings_rate,
        "prompt": (
            "Write `savings_rate(transactions) -> float`. "
            "Formula: (total_income - total_expenses) / total_income. "
            "Return 0.0 if no income. Return 0.0 for empty list." + NO_EVAL
        ),
    },
    {
        "name": "Expense ratios",
        "check": _check_expense_ratios,
        "prompt": (
            "Write `expense_ratios(transactions) -> dict[str, float]`. "
            "Each category's share of total expenses as a percentage (0-100). "
            "Exclude income transactions. Percentages must sum to exactly 100.0. "
            "Return {} if no expenses." + NO_EVAL
        ),
    },
    {
        "name": "Budget tracker",
        "check": _check_budget,
        "prompt": (
            "Write `Budget(limits: dict[str, float])` with `track(transactions)` method. "
            "Returns dict with: spent_per_category (dict), remaining_per_category (dict), "
            "over_budget (list of category NAMES). Ignore income transactions." + NO_EVAL
        ),
    },
    {
        "name": "Date range filter",
        "check": _check_date_filter,
        "prompt": (
            "Write `filter_by_date(transactions, start, end) -> list`. "
            "Return transactions where start <= date <= end (INCLUSIVE on BOTH ends)." + NO_EVAL
        ),
    },
    {
        "name": "Moving average spend",
        "check": _check_moving_avg,
        "prompt": (
            "Write `moving_avg_spend(transactions, window=30) -> list[(date, float)]`. "
            "For each date in the dataset, compute average DAILY spend over the rolling window. "
            "Only count expense transactions. Income is excluded." + NO_EVAL
        ),
    },
    {
        "name": "Cash flow forecast",
        "check": _check_cash_flow,
        "prompt": (
            "Write `forecast(transactions, days=30) -> list[(date, float)]`. "
            "Compute average daily net income from the historical data. "
            "Return `days` projected (future_date, cumulative_balance) tuples "
            "starting the day after the last transaction date." + NO_EVAL
        ),
    },
    {
        "name": "Compound interest",
        "check": _check_compound,
        "prompt": (
            "Write `compound_interest(principal, annual_rate, years, n=12) -> float`. "
            "Formula: A = P * (1 + r/n)^(n*t)." + NO_EVAL
        ),
    },
    {
        "name": "Loan amortization",
        "check": _check_loan,
        "prompt": (
            "Write `loan_amortization(principal, annual_rate, months) -> list[dict]`. "
            "Each dict: {'month': int, 'payment': float, 'principal': float, "
            "'interest': float, 'balance': float}. Final balance must be ~0." + NO_EVAL
        ),
    },
    {
        "name": "Portfolio allocation",
        "check": _check_allocation,
        "prompt": (
            "Write `portfolio_allocation(holdings: dict[str, float]) -> dict[str, float]`. "
            "Return each asset's percentage of total portfolio value. "
            "Values must sum to exactly 100.0." + NO_EVAL
        ),
    },
    {
        "name": "Rebalancing trades",
        "check": _check_rebalance,
        "prompt": (
            "Write `rebalance_trades(current: dict[str, float], "
            "target: dict[str, float]) -> dict[str, float]`. "
            "current: asset→current_value. target: asset→target_fraction. "
            "Return asset→trade_amount (positive=buy, negative=sell)." + NO_EVAL
        ),
    },
    {
        "name": "Time-weighted return",
        "check": _check_twr,
        "prompt": (
            "Write `time_weighted_return(periods: list[dict]) -> float`. "
            "Each period has start_value, end_value, cashflow. "
            "TWR = product of (1 + sub_period_return) for each period, minus 1. "
            "Sub-period return = (end_value - start_value - cashflow) / (start_value + cashflow)." + NO_EVAL
        ),
    },
    # ── Phase 4b — Multi-asset analytics ──────────────────────────────────────
    {
        "name": "Weighted volatility",
        "check": _check_weighted_vol,
        "prompt": (
            "Write `weighted_volatility(weights: dict[str, float], "
            "asset_returns: dict[str, list[float]]) -> float`. "
            "weights and asset_returns share the same asset keys; daily returns. "
            "Return annualised portfolio volatility." + NO_EVAL
        ),
    },
    {
        "name": "Rolling Sharpe",
        "check": _check_rolling_sharpe,
        "prompt": (
            "Write `rolling_sharpe(returns: list[float], window: int, "
            "risk_free_daily: float = 0.0) -> list`. "
            "Returns are daily. For each rolling window compute the annualised Sharpe ratio. "
            "Positions before the first full window should be None." + NO_EVAL
        ),
    },
    {
        "name": "Sharpe ratio",
        "check": _check_sharpe_strict,
        "prompt": (
            "Write `sharpe_ratio(returns: list[float], risk_free_annual: float = 0.0) -> float`. "
            "Returns are daily. risk_free_annual is the annual risk-free rate. "
            "Return the annualised Sharpe ratio." + NO_EVAL
        ),
    },
    {
        "name": "Sortino ratio",
        "check": _check_sortino_strict,
        "prompt": (
            "Write `sortino_ratio(returns: list[float], mar: float = 0.0) -> float`. "
            "Returns are daily; mar is the minimum acceptable daily return. "
            "Return the annualised Sortino ratio." + NO_EVAL
        ),
    },
    {
        "name": "Information ratio",
        "check": _check_information_ratio,
        "prompt": (
            "Write `information_ratio(portfolio_returns: list[float], "
            "benchmark_returns: list[float]) -> float`. "
            "Returns are daily. Return the annualised information ratio." + NO_EVAL
        ),
    },
    # ── Phase 5 — Return-based risk ────────────────────────────────────────────
    {
        "name": "Drawdown series",
        "check": _check_drawdown_series,
        "prompt": (
            "Write `drawdown_series(returns: list[float]) -> list[float]`. "
            "Returns are daily. For each period return the drawdown from the running peak "
            "as a fraction (0.0 when at peak, positive when below it)." + NO_EVAL
        ),
    },
    {
        "name": "Calmar ratio",
        "check": _check_calmar,
        "prompt": (
            "Write `calmar_ratio(returns: list[float]) -> float`. "
            "Returns are daily. Return the Calmar ratio: annualised return / max drawdown." + NO_EVAL
        ),
    },
    {
        "name": "CAPM metrics",
        "check": _check_capm_metrics,
        "prompt": (
            "Write `capm_metrics(portfolio_returns: list[float], "
            "market_returns: list[float], risk_free_daily: float = 0.0) -> dict`. "
            "Returns are daily. Return {'beta': float, 'alpha_annual': float}." + NO_EVAL
        ),
    },
    {
        "name": "Tracking error",
        "check": _check_tracking_error,
        "prompt": (
            "Write `tracking_error(portfolio_returns: list[float], "
            "benchmark_returns: list[float], annualize: bool = True) -> float`. "
            "Return the tracking error (std of active return series). "
            "Annualise if annualize=True." + NO_EVAL
        ),
    },
    {
        "name": "Expected shortfall",
        "check": _check_expected_shortfall,
        "prompt": (
            "Write `expected_shortfall(returns: list[float], confidence: float = 0.95) -> float`. "
            "Return the Expected Shortfall (CVaR) as a positive number (it is a loss). "
            "Support both confidence=0.95 and confidence=0.80." + NO_EVAL
        ),
    },
    # ── Phase 6 — Advanced ─────────────────────────────────────────────────────
    {
        "name": "Parametric VaR",
        "check": _check_parametric_var,
        "prompt": (
            "Write `parametric_var(returns: list[float], confidence: float = 0.95) -> float`. "
            "Use the parametric (Gaussian) method. Do not use scipy — hardcode the z-scores. "
            "Return the VaR as a positive number. Must support confidence=0.95 and 0.99." + NO_EVAL
        ),
    },
    {
        "name": "Omega ratio",
        "check": _check_omega_ratio,
        "prompt": (
            "Write `omega_ratio(returns: list[float], threshold: float = 0.0) -> float`. "
            "Return the Omega ratio for the given return series and threshold." + NO_EVAL
        ),
    },
    {
        "name": "GBM Monte Carlo",
        "check": _check_gbm_mc,
        "prompt": (
            "Write `monte_carlo_gbm(initial_value, mu_annual, sigma_annual, "
            "days, n_sims, seed=None) -> dict`. "
            "Simulate Geometric Brownian Motion paths. "
            "Return dict with 'median', 'p5', 'p95' of final portfolio values." + NO_EVAL
        ),
    },
    {
        "name": "Risk report",
        "check": _check_full_risk_report,
        "prompt": (
            "Write `risk_report(portfolio: dict[str, float], "
            "asset_returns: dict[str, list[float]], "
            "benchmark_returns: list[float]) -> dict`. "
            "portfolio maps asset name → current value; asset_returns maps asset → daily return series. "
            "Return dict with: total_value, volatility (annualised), sharpe, max_drawdown, beta." + NO_EVAL
        ),
    },
]



# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _verify(text: str, check_fn) -> tuple[bool, str]:
    code = _extract_tool_code(text)
    if not code:
        return False, "no code block extracted"
    ns = _exec_ns(code)
    if isinstance(ns, str):
        return False, f"exec error: {ns}"
    try:
        return check_fn(ns)
    except Exception as e:
        return False, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    embedder = build_embedder()
    brain    = BrainAgent(embedder, k=5, threshold=0.45)

    # Pre-seed brain with 6 abstract failure motifs — one per recurring trap.
    # Each motif has a regex surface_pattern that fires when code matches,
    # regardless of task domain. No trace strings needed.
    console.print("[dim]Seeding brain with abstract failure motifs…[/]")
    brain.seed_motifs(_BRAIN_MOTIFS)
    console.print(f"[dim]Brain seeded: {len(_BRAIN_MOTIFS)} motifs loaded.[/]\n")

    console.print(Panel(
        "[bold cyan]VibeSession — 28-task Personal Finance Toolkit[/]\n\n"
        "Brain pre-seeded with 6 named failure motifs (regex-based, no false positives).\n"
        "Fires only on concrete evidence: motif match, constraint violation, plan-code mismatch.\n\n"
        "Tasks 15-28 share recurring failure patterns:\n"
        "  Phase 4b (15-19): population std, compound rf, Sortino denominator\n"
        "  Phase 5  (20-24): compound price series, Calmar annualization\n"
        "  Phase 6  (25-28): parametric z-scores, Ito, risk report combos\n\n"
        "[dim]BrainAgent  •  tool_agent(python_exec)  •  threshold=0.45[/]",
        box=box.ROUNDED,
    ))

    rows:    list[tuple] = []
    n_pass:  int         = 0
    n_fires: int         = 0
    N = len(TASKS)

    for i, task in enumerate(TASKS):
        n      = i + 1
        t0     = time.perf_counter()
        fired  = False
        p_fail_shown: float | None = None

        brain.set_task(i, task=task["prompt"])
        brain.reset()

        agent = tool_agent(
            ["python_exec"],
            model=HAIKU,
            max_tokens=4096,
            max_turns=MAX_TURNS,
        )
        agent.monitor = brain

        console.rule(f"[bold][{n:02d}/{N}] {task['name']}[/]")

        try:
            result = agent(task["prompt"])
            trace  = result[0] if isinstance(result, tuple) else str(result)
            tokens = result[1] if isinstance(result, tuple) else 0
        except Exception as e:
            trace  = ""
            tokens = 0
            console.print(f"  [red]Agent error: {e}[/]")

        elapsed = time.perf_counter() - t0

        # Fire detection: use last_fire (set only when a hook RETURNED a STOP message)
        # NOT trajectory pt.fired — that could be stale from a previous task even
        # though reset() now clears _trajectory. Using last_fire is the canonical source.
        if brain.last_fire is not None:
            fired        = True
            n_fires     += 1
            combined     = brain.last_fire.get("combined") or brain.last_p_fail
            p_fail_shown = combined
            pf_str       = f"  p_fail={combined:.2f}" if combined else ""
            console.print(f"  [yellow]⚡ BRAIN fired{pf_str}[/]")
            # Show full warning including evidence bullets (not just first 3 lines)
            if brain.last_warning:
                lines_shown = 0
                for ln in brain.last_warning.splitlines():
                    if ln.strip():
                        console.print(f"    [dim]{ln.strip()[:120]}[/]")
                        lines_shown += 1
                        if lines_shown >= 8:
                            break

        passed, detail = _verify(trace, task["check"])
        if passed:
            n_pass += 1

        brain.store(trace, int(passed), detail[:200] if not passed else "")

        status = "PASS" if passed else "FAIL"
        color  = "green" if passed else "red"
        pf_s   = f"{p_fail_shown:.2f}" if p_fail_shown is not None else "—"
        tok_s  = f"{tokens:,}" if tokens else "—"
        if not passed and detail:
            console.print(f"  [{color}]{status}[/{color}]  {detail[:80]}")
        else:
            console.print(f"  [{color}]{status}[/{color}]")
        console.print(f"  [dim]{tok_s}tok  {elapsed:.0f}s[/]")

        rows.append((n, task["name"], status, pf_s, fired, detail))

    # Summary
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1))
    table.add_column("#",       width=4,  style="dim")
    table.add_column("Task",    width=22)
    table.add_column("Result",  width=6)
    table.add_column("p_fail",  width=7)
    table.add_column("Brain",   width=5)
    table.add_column("Detail",  width=52)

    for n, name, status, pf_s, f, detail in rows:
        c    = "green" if status == "PASS" else "red"
        fire = "[yellow]⚡[/]" if f else "—"
        table.add_row(
            str(n), name[:22], f"[{c}]{status}[/{c}]",
            pf_s, fire, (detail or "")[:52],
        )

    console.rule("[bold]Session Summary[/]")
    console.print(table)
    console.print(
        f"\n  [bold]{n_pass}/{N} passed[/]  "
        f"{n_fires} brain fires  "
        f"[dim]{brain.n_stored} stored "
        f"({brain.n_pass}✓ {brain.n_fail}✗)[/]"
    )


if __name__ == "__main__":
    main()
