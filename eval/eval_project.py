"""eval/eval_project.py — Simulated vibe-coding session: Portfolio Risk Analyzer.

A data analyst builds a complete stock portfolio risk tool from scratch in one
session. 20 sequential tasks, each building on the previous one. The agent
writes, runs, and debugs code iteratively — exactly how a power user would
work all day.

This eval tests whether the brain's real-time trajectory monitoring helps in
a coherent long-form project, vs the analyst doing it alone.

Tasks (in order — each references the previous):
  Phase 1 — Data foundation:    simulate prices, log returns, rolling stats, covariance, correlation
  Phase 2 — Portfolio theory:   min-variance, max-Sharpe, efficient frontier, VaR, CVaR
  Phase 3 — Risk metrics:       backtest, max drawdown, Kelly criterion, beta, risk contribution
  Phase 4 — Advanced analysis:  stress test, regime detection, rebalancing, turnover cost, report

Run:  python eval/eval_project.py
"""
from __future__ import annotations

import ast as _ast
import contextlib
import io
import json
import math
import re
import sys
import textwrap
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agents import build_embedder, tool_agent, _anthropic_call
from brain import BrainAgent
from eval.viz_brain import BrainViz
from eval.eval_hard import _exec, _check_code, _extract_code, _first_exec_code

OUT = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

HAIKU     = "claude-haiku-4-5-20251001"
MAX_TURNS = 14
MAX_TOK   = 4096


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED TEST DATA (consistent across all probe/check functions)
# ═══════════════════════════════════════════════════════════════════════════════

# Simple known prices for probing functions in isolation
_TEST_PRICES_RAW = {
    "AAPL": [100.0, 105.0, 102.0, 108.0, 110.0, 107.0, 112.0],
    "MSFT": [200.0, 198.0, 204.0, 201.0, 207.0, 210.0, 208.0],
    "GOOGL": [150.0, 153.0, 151.0, 155.0, 158.0, 156.0, 160.0],
}


def _make_test_df():
    try:
        import numpy as np
        import pandas as pd
        return pd.DataFrame(_TEST_PRICES_RAW)
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: DATA FOUNDATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── Task 1: Simulate correlated stock prices ───────────────────────────────────

def _probe_simulate(ns):
    fn = ns.get("simulate_prices")
    if not fn:
        return ["simulate_prices(n_days, tickers, seed) not found — implement it and call python_exec"]
    try:
        import numpy as np
        result = fn(n_days=252, tickers=["A", "B", "C"], seed=42)
        if not hasattr(result, "shape"):
            return ["simulate_prices must return a DataFrame or 2D array with shape (n_days, n_tickers)"]
        if result.shape[0] != 252 or result.shape[1] != 3:
            return [f"simulate_prices(252, 3 tickers) shape={result.shape}, expected (252, 3)"]
        # All prices must be positive
        if hasattr(result, "values"):
            vals = result.values
        else:
            vals = np.array(result)
        if np.any(vals <= 0):
            return ["All simulated prices must be positive — use GBM or exp(cumsum(returns))."]
        r2 = fn(n_days=252, tickers=["A", "B", "C"], seed=42)
        if hasattr(r2, "values"):
            v2 = r2.values
        else:
            v2 = np.array(r2)
        if not np.allclose(vals, v2):
            return ["simulate_prices must be reproducible given the same seed."]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_simulate(ns):
    fn = ns.get("simulate_prices")
    if not fn: return False
    try:
        import numpy as np
        r1 = fn(252, ["A","B","C"], seed=42)
        r2 = fn(252, ["A","B","C"], seed=42)
        r3 = fn(252, ["A","B","C"], seed=99)
        v1 = r1.values if hasattr(r1,"values") else np.array(r1)
        v2 = r2.values if hasattr(r2,"values") else np.array(r2)
        v3 = r3.values if hasattr(r3,"values") else np.array(r3)
        if v1.shape != (252, 3): return False
        if not np.allclose(v1, v2): return False          # reproducible
        if np.allclose(v1, v3): return False              # different seeds differ
        if np.any(v1 <= 0): return False                  # positive prices
        return True
    except Exception: return False


# ── Task 2: Log daily returns ──────────────────────────────────────────────────

def _probe_returns(ns):
    fn = ns.get("compute_returns") or ns.get("log_returns") or ns.get("daily_returns")
    if not fn:
        return ["compute_returns(prices_df) not found — must return log daily returns DataFrame"]
    try:
        import numpy as np
        import pandas as pd
        prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 50.0]})
        r = fn(prices)
        if r is None or (hasattr(r, "__len__") and len(r) == 0):
            return ["compute_returns returned empty — use np.log(prices/prices.shift(1)).dropna()"]
        r_arr = r.values if hasattr(r, "values") else np.array(r)
        # A goes 100→110→121: log returns ≈ [ln(1.1), ln(1.1)] ≈ [0.0953, 0.0953]
        expected_a = math.log(1.1)
        if abs(r_arr[0, 0] - expected_a) > 0.02:
            # Check if arithmetic returns were used (would give 0.10, not 0.0953)
            if abs(r_arr[0, 0] - 0.10) < 0.001:
                return [
                    f"compute_returns gives {r_arr[0,0]:.4f} for 100→110, but expected log return {expected_a:.4f}. "
                    "FIX: Use LOG returns: np.log(prices / prices.shift(1)).dropna(). "
                    "Arithmetic returns (price_change/price) are wrong for compounding math."
                ]
            return [f"compute_returns gives {r_arr[0,0]:.4f} for 100→110, expected {expected_a:.4f}"]
        # B is flat — log return = 0
        if abs(r_arr[0, 1]) > 1e-9:
            return [f"Flat asset (50→50) should give return=0, got {r_arr[0,1]:.6f}"]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_returns(ns):
    fn = ns.get("compute_returns") or ns.get("log_returns") or ns.get("daily_returns")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1], "B": [50.0, 50.0, 55.0, 55.0]})
        r = fn(prices)
        v = r.values if hasattr(r, "values") else np.array(r)
        # A: log(1.1), log(1.1), log(1.1)
        assert v.shape == (3, 2), f"shape {v.shape}"
        assert all(abs(v[i, 0] - math.log(1.1)) < 0.001 for i in range(3))
        assert abs(v[0, 1]) < 1e-9 and abs(v[1, 1] - math.log(1.1)) < 0.001 and abs(v[2, 1]) < 1e-9
        return True
    except Exception: return False


# ── Task 3: Rolling statistics ─────────────────────────────────────────────────

def _probe_rolling(ns):
    fn = ns.get("rolling_stats") or ns.get("rolling_statistics")
    if not fn:
        return ["rolling_stats(returns_df, window) not found — return dict with 'mean', 'vol', 'skew'"]
    try:
        import numpy as np
        import pandas as pd
        rets = pd.DataFrame({"A": [0.01] * 30, "B": [0.02, -0.01] * 15})
        result = fn(rets, window=10)
        if not isinstance(result, dict):
            return [f"rolling_stats must return a dict with keys 'mean','vol','skew'; got {type(result)}"]
        for k in ("mean", "vol"):
            if k not in result:
                return [f"Missing key '{k}' in rolling_stats output. Return dict with 'mean','vol','skew'."]
        # 'A' is constant 0.01 → rolling vol should be ~0 (or very small)
        vol_a = result["vol"]["A"].dropna()
        if vol_a.max() > 0.01:
            return [
                f"Rolling vol of constant series (all 0.01) is {vol_a.max():.4f}, expected ≈ 0. "
                "FIX: Use returns_df.rolling(window).std() for volatility, "
                "not returns_df.rolling(window).mean().std() or similar."
            ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_rolling(ns):
    fn = ns.get("rolling_stats") or ns.get("rolling_statistics")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        rets = pd.DataFrame({"A": [0.01 * (i % 3 + 1) for i in range(40)]})
        r = fn(rets, window=20)
        if not isinstance(r, dict): return False
        for k in ("mean", "vol"):
            if k not in r: return False
        # vol must be positive for non-constant series
        if r["vol"]["A"].dropna().min() <= 0: return False
        return True
    except Exception: return False


# ── Task 4: Annualised covariance matrix ───────────────────────────────────────

def _probe_covariance(ns):
    fn = ns.get("covariance_matrix") or ns.get("annualized_cov")
    if not fn:
        return ["covariance_matrix(returns_df, periods_per_year=252) not found"]
    try:
        import numpy as np
        import pandas as pd
        # Two perfectly correlated assets with known vol
        n = 252
        daily_vol = 0.01
        np.random.seed(0)
        common = np.random.normal(0, daily_vol, n)
        rets = pd.DataFrame({"A": common, "B": common})
        cov = fn(rets)
        if not hasattr(cov, "shape"):
            cov_arr = np.array(cov)
        else:
            cov_arr = cov.values if hasattr(cov, "values") else np.array(cov)
        # Diagonal should be annualised: daily_var * 252 ≈ (0.01)^2 * 252 = 0.0252
        expected_diag = daily_vol ** 2 * 252
        if abs(cov_arr[0, 0] - expected_diag) > 0.005:
            if abs(cov_arr[0, 0] - daily_vol ** 2) < 0.001:
                return [
                    f"covariance_matrix diagonal = {cov_arr[0,0]:.6f} (daily), "
                    f"but expected annualised ≈ {expected_diag:.4f}. "
                    "FIX: Multiply result by periods_per_year=252. "
                    "Use: returns.cov() * 252"
                ]
            return [f"Diagonal variance {cov_arr[0,0]:.6f} far from expected {expected_diag:.4f}"]
        # Must be symmetric
        if not np.allclose(cov_arr, cov_arr.T, atol=1e-10):
            return ["Covariance matrix is not symmetric — check calculation."]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_covariance(ns):
    fn = ns.get("covariance_matrix") or ns.get("annualized_cov")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(1)
        rets = pd.DataFrame(np.random.normal(0, 0.01, (252, 3)), columns=["A","B","C"])
        cov = fn(rets)
        v = cov.values if hasattr(cov, "values") else np.array(cov)
        if v.shape != (3, 3): return False
        if not np.allclose(v, v.T, atol=1e-10): return False
        # Diagonal should be ~252 * daily_var ≈ 0.01^2 * 252 = 0.0252
        if not all(0.005 < v[i, i] < 0.1 for i in range(3)): return False
        # Must be positive semi-definite
        eigs = np.linalg.eigvalsh(v)
        if np.any(eigs < -1e-10): return False
        return True
    except Exception: return False


# ── Task 5: Minimum variance portfolio ────────────────────────────────────────

def _probe_minvar(ns):
    fn = ns.get("min_variance_portfolio") or ns.get("minimum_variance")
    if not fn:
        return ["min_variance_portfolio(cov_matrix) not found — return dict with 'weights', 'variance'"]
    try:
        import numpy as np
        # 2×2 covariance where asset A is much less volatile
        cov = np.array([[0.04, 0.01], [0.01, 0.16]])
        result = fn(cov)
        if not isinstance(result, dict):
            return [f"Must return dict with 'weights' and 'variance', got {type(result)}"]
        if "weights" not in result:
            return ["Dict must have 'weights' key"]
        w = np.array(result["weights"]).flatten()
        if abs(sum(w) - 1.0) > 0.01:
            return [
                f"Weights sum = {sum(w):.4f}, must equal 1.0. "
                "FIX: Use equality constraint sum(w)=1 in scipy.optimize.minimize, "
                "or normalise: w = w / w.sum()."
            ]
        # Min variance should put most weight on A (lower variance)
        if w[0] < 0.5:
            return [
                f"Min-variance weights: A={w[0]:.3f}, B={w[1]:.3f}. "
                "Asset A has variance 0.04 vs B's 0.16 — A should dominate the portfolio. "
                "FIX: Minimise w.T @ cov @ w subject to sum(w)=1 and w>=0."
            ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_minvar(ns):
    fn = ns.get("min_variance_portfolio") or ns.get("minimum_variance")
    if not fn: return False
    try:
        import numpy as np
        cov = np.array([[0.04, 0.01], [0.01, 0.16]])
        r = fn(cov)
        w = np.array(r["weights"]).flatten()
        if abs(sum(w) - 1.0) > 0.01: return False
        if any(wi < -0.001 for wi in w): return False   # long-only
        # Portfolio variance must be ≤ min individual variance
        pv = float(w @ cov @ w)
        if pv > min(cov[0,0], cov[1,1]) + 0.001: return False
        # Test 3×3
        cov3 = np.diag([0.01, 0.04, 0.09])
        r3 = fn(cov3)
        w3 = np.array(r3["weights"]).flatten()
        if abs(sum(w3) - 1.0) > 0.01: return False
        return True
    except Exception: return False


# ─── PHASE 2: PORTFOLIO THEORY ────────────────────────────────────────────────

# ── Task 6: Maximum Sharpe ratio portfolio ────────────────────────────────────

def _probe_maxsharpe(ns):
    fn = ns.get("max_sharpe_portfolio") or ns.get("maximum_sharpe")
    if not fn:
        return ["max_sharpe_portfolio(expected_returns, cov_matrix, rf=0.02) not found"]
    try:
        import numpy as np
        mu = np.array([0.10, 0.08])           # A higher return than B
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])  # equal variance
        result = fn(mu, cov, rf=0.0)
        w = np.array(result.get("weights", result) if isinstance(result, dict) else result).flatten()
        if abs(sum(w) - 1.0) > 0.02:
            return [
                f"Weights sum = {sum(w):.4f}, must equal 1.0. "
                "FIX: Normalise tangent portfolio weights: w = w / w.sum()."
            ]
        if w[0] < w[1]:
            return [
                f"Asset A (mu=0.10) should have higher weight than B (mu=0.08) when variances equal. "
                f"Got: A={w[0]:.3f}, B={w[1]:.3f}. "
                "FIX: Maximise (mu-rf).dot(w) / sqrt(w.T@cov@w). "
                "negate objective for scipy.minimize."
            ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_maxsharpe(ns):
    fn = ns.get("max_sharpe_portfolio") or ns.get("maximum_sharpe")
    if not fn: return False
    try:
        import numpy as np
        mu = np.array([0.12, 0.08, 0.10])
        cov = np.diag([0.04, 0.04, 0.04])
        r = fn(mu, cov, rf=0.0)
        w = np.array(r.get("weights", r) if isinstance(r, dict) else r).flatten()
        if abs(sum(w) - 1.0) > 0.02: return False
        # Highest Sharpe is asset A (mu=0.12, same vol) — should get most weight
        if w[0] < 0.4: return False
        return True
    except Exception: return False


# ── Task 7: Value at Risk (95% 1-day VaR) ─────────────────────────────────────

def _probe_var(ns):
    fn = ns.get("portfolio_var") or ns.get("value_at_risk") or ns.get("calculate_var")
    if not fn:
        return ["portfolio_var(returns_series, confidence=0.95) not found — return VaR as a positive number (loss)"]
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0, 0.01, 1000))
        var = fn(rets, confidence=0.95)
        # For N(0,0.01), 5th percentile ≈ -0.01645 → VaR ≈ 0.01645
        if isinstance(var, (int, float)):
            if var < 0:
                return [
                    f"VaR = {var:.4f} (negative). VaR should be returned as a POSITIVE number "
                    "representing the loss. "
                    "FIX: VaR = -np.percentile(returns, 1 - confidence) = -np.percentile(returns, 5)."
                ]
            if not (0.005 < var < 0.04):
                return [
                    f"VaR = {var:.4f} for N(0,0.01) at 95%, expected ≈ 0.0165. "
                    "FIX: Historical VaR at 95% confidence = the 5th percentile (1-0.95=0.05). "
                    "var = -np.percentile(returns, 5)"
                ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_var(ns):
    fn = ns.get("portfolio_var") or ns.get("value_at_risk") or ns.get("calculate_var")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0, 0.01, 10000))
        var95 = fn(rets, confidence=0.95)
        var99 = fn(rets, confidence=0.99)
        if not isinstance(var95, (int, float)): return False
        if var95 < 0: return False                       # must be positive
        if not (0.010 < var95 < 0.030): return False     # ≈ 0.0165
        if var99 <= var95: return False                   # 99% VaR > 95% VaR
        return True
    except Exception: return False


# ── Task 8: Conditional VaR (Expected Shortfall) ─────────────────────────────

def _probe_cvar(ns):
    fn = ns.get("portfolio_cvar") or ns.get("expected_shortfall") or ns.get("cvar")
    if not fn:
        return ["portfolio_cvar(returns_series, confidence=0.95) not found"]
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0, 0.01, 10000))
        cvar = fn(rets, confidence=0.95)
        var  = -np.percentile(rets.values, 5)
        if isinstance(cvar, (int, float)):
            if cvar < 0:
                return ["CVaR should be a positive number (expected loss beyond VaR)"]
            if cvar <= var:
                return [
                    f"CVaR={cvar:.4f} ≤ VaR={var:.4f}. CVaR (Expected Shortfall) is the average "
                    "of losses BEYOND the VaR threshold, so CVaR > VaR always. "
                    "FIX: tail = returns[returns < -var]; cvar = -tail.mean()"
                ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_cvar(ns):
    fn = ns.get("portfolio_cvar") or ns.get("expected_shortfall") or ns.get("cvar")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0, 0.01, 10000))
        c95 = fn(rets, 0.95)
        c99 = fn(rets, 0.99)
        var95 = -np.percentile(rets, 5)
        if c95 < 0 or c99 < 0: return False
        if c95 <= var95: return False    # CVaR > VaR
        if c99 <= c95: return False      # 99% CVaR > 95% CVaR
        return True
    except Exception: return False


# ── Task 9: Max drawdown ───────────────────────────────────────────────────────

def _probe_drawdown(ns):
    fn = ns.get("max_drawdown") or ns.get("maximum_drawdown")
    if not fn:
        return ["max_drawdown(price_series) not found — return max drawdown as a NEGATIVE fraction"]
    try:
        import numpy as np
        import pandas as pd
        prices = pd.Series([100.0, 110.0, 90.0, 95.0, 80.0, 100.0])
        dd = fn(prices)
        # Peak = 110 at idx 1, trough = 80 at idx 4: drawdown = (80-110)/110 ≈ -0.2727
        expected = (80 - 110) / 110
        if isinstance(dd, (int, float)):
            if dd > 0:
                return [
                    f"max_drawdown = {dd:.4f} (positive). Should be negative — "
                    "drawdown = (trough - peak) / peak. "
                    "FIX: return (rolling_min - running_max) / running_max where that fraction is most negative."
                ]
            if abs(dd - expected) > 0.02:
                return [
                    f"max_drawdown = {dd:.4f}, expected ≈ {expected:.4f}. "
                    "Peak=110, Trough=80: (80-110)/110 = -0.2727. "
                    "FIX: running_max = prices.cummax(); drawdown = (prices - running_max)/running_max; return drawdown.min()."
                ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_drawdown(ns):
    fn = ns.get("max_drawdown") or ns.get("maximum_drawdown")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        p1 = pd.Series([100.0, 110.0, 90.0, 95.0, 80.0, 100.0])
        dd1 = fn(p1)
        if dd1 >= 0: return False
        if abs(dd1 - (80-110)/110) > 0.02: return False
        p2 = pd.Series([100.0, 120.0, 140.0])  # only goes up → dd = 0
        dd2 = fn(p2)
        if dd2 > 0: return False
        return True
    except Exception: return False


# ── Task 10: Sharpe ratio ─────────────────────────────────────────────────────

def _probe_sharpe(ns):
    fn = ns.get("sharpe_ratio") or ns.get("annualized_sharpe")
    if not fn:
        return ["sharpe_ratio(returns_series, rf=0.02, periods=252) not found"]
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(0)
        daily_rets = pd.Series(np.random.normal(0.0005, 0.01, 252))  # mu=12.6%, vol=15.9%
        sr = fn(daily_rets, rf=0.0)
        if isinstance(sr, (int, float)):
            # Expected ≈ (0.0005*252) / (0.01*sqrt(252)) ≈ 0.126/0.159 ≈ 0.79
            if abs(sr) < 0.1:
                return [
                    f"Sharpe ratio = {sr:.4f} seems too low. "
                    "FIX: Annualise properly: mean_daily * 252 / (std_daily * sqrt(252)). "
                    "Or equivalently: (mean_daily / std_daily) * sqrt(252)."
                ]
            # If not annualised, result would be ≈ 0.05
            if abs(sr) < 0.3:
                return [
                    f"Sharpe = {sr:.4f} looks like a daily (not annualised) figure. "
                    "FIX: Multiply by sqrt(periods) = sqrt(252) ≈ 15.87. "
                    "sr = (returns.mean() - rf/periods) / returns.std() * sqrt(periods)"
                ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_sharpe(ns):
    fn = ns.get("sharpe_ratio") or ns.get("annualized_sharpe")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.0005, 0.01, 252))
        sr = fn(rets, rf=0.0)
        if not isinstance(sr, (int, float)): return False
        # Annualised Sharpe for mu=0.0005/day ≈ 0.5-1.5
        if not (0.3 < abs(sr) < 2.5): return False
        # With higher rf, Sharpe should be lower
        sr2 = fn(rets, rf=0.1)
        if sr2 >= sr: return False
        return True
    except Exception: return False


# ─── PHASE 3: RISK METRICS ────────────────────────────────────────────────────

# ── Task 11: Portfolio beta to market ────────────────────────────────────────

def _probe_beta(ns):
    fn = ns.get("portfolio_beta") or ns.get("calculate_beta") or ns.get("beta")
    if not fn:
        return ["portfolio_beta(portfolio_returns, market_returns) not found — return scalar beta"]
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        market = pd.Series(np.random.normal(0, 0.01, 252))
        # Portfolio = 1.5 × market + noise
        port = 1.5 * market + pd.Series(np.random.normal(0, 0.002, 252))
        b = fn(port, market)
        if isinstance(b, (int, float)):
            if abs(b - 1.5) > 0.15:
                return [
                    f"beta = {b:.4f}, expected ≈ 1.5 (portfolio = 1.5×market + noise). "
                    "FIX: beta = Cov(portfolio, market) / Var(market). "
                    "Use: np.cov(port, mkt)[0,1] / np.var(mkt)"
                ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_beta(ns):
    fn = ns.get("portfolio_beta") or ns.get("calculate_beta") or ns.get("beta")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        mkt = pd.Series(np.random.normal(0, 0.01, 1000))
        p05 = 0.5 * mkt + pd.Series(np.random.normal(0, 0.001, 1000))
        p15 = 1.5 * mkt + pd.Series(np.random.normal(0, 0.001, 1000))
        b05 = fn(p05, mkt)
        b15 = fn(p15, mkt)
        if not (0.3 < b05 < 0.7): return False
        if not (1.3 < b15 < 1.7): return False
        # Market itself: beta = 1
        bm = fn(mkt, mkt)
        if abs(bm - 1.0) > 0.01: return False
        return True
    except Exception: return False


# ── Task 12: Risk contribution (marginal) ────────────────────────────────────

def _probe_risk_contrib(ns):
    fn = ns.get("risk_contribution") or ns.get("marginal_risk_contribution")
    if not fn:
        return ["risk_contribution(weights, cov_matrix) not found — return array of each asset's % contribution to portfolio variance"]
    try:
        import numpy as np
        # Equal weights, one asset with 10x more variance
        cov = np.diag([0.01, 0.10])
        w = np.array([0.5, 0.5])
        rc = fn(w, cov)
        if rc is None:
            return ["risk_contribution returned None"]
        rc = np.array(rc).flatten()
        if abs(rc.sum() - 1.0) > 0.05:
            return [
                f"Risk contributions sum to {rc.sum():.4f}, should sum to 1.0 (100%). "
                "FIX: rc_i = w_i * (cov @ w)[i] / (w @ cov @ w); return rc / rc.sum()"
            ]
        # Asset B (var=0.10) should contribute far more than A (var=0.01)
        if rc[0] > rc[1]:
            return [
                f"Asset A contribution {rc[0]:.3f} > Asset B {rc[1]:.3f}, but B has 10× higher variance. "
                "FIX: Marginal contribution = w_i × (Σw)_i / (w'Σw). Asset B dominates risk."
            ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_risk_contrib(ns):
    fn = ns.get("risk_contribution") or ns.get("marginal_risk_contribution")
    if not fn: return False
    try:
        import numpy as np
        cov = np.diag([0.01, 0.10])
        w = np.array([0.5, 0.5])
        rc = np.array(fn(w, cov)).flatten()
        if abs(rc.sum() - 1.0) > 0.05: return False
        if rc[0] >= rc[1]: return False   # B dominates
        # Equal variance → equal weights → equal contributions
        cov2 = np.diag([0.04, 0.04])
        rc2 = np.array(fn(w, cov2)).flatten()
        if abs(rc2[0] - 0.5) > 0.05: return False
        return True
    except Exception: return False


# ── Task 13: Stress test scenarios ────────────────────────────────────────────

def _probe_stress(ns):
    fn = ns.get("stress_test") or ns.get("apply_stress")
    if not fn:
        return ["stress_test(weights, expected_returns, scenarios) not found — scenarios is dict of {name: shocks_dict}"]
    try:
        import numpy as np
        weights = np.array([0.5, 0.5])
        mu = np.array([0.10, 0.08])
        scenarios = {
            "crash_2008": {"return_shock": -0.40, "vol_shock": 2.5},
            "base":       {"return_shock": 0.0,   "vol_shock": 1.0},
        }
        result = fn(weights, mu, scenarios)
        if not isinstance(result, dict):
            return [f"stress_test must return dict of {{scenario_name: portfolio_return}}, got {type(result)}"]
        if "crash_2008" not in result:
            return ["stress_test result must include scenario names as keys"]
        crash_ret = result["crash_2008"]
        base_ret  = result.get("base", None)
        if isinstance(crash_ret, (int, float)):
            if crash_ret > -0.20:
                return [
                    f"Crash scenario return = {crash_ret:.4f}, expected ≈ -0.40 * weighted_return. "
                    "FIX: portfolio_return_stressed = sum(w_i * mu_i) * (1 + return_shock) "
                    "or = sum(w_i * (mu_i + shock)). The scenario applies a shock to returns."
                ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_stress(ns):
    fn = ns.get("stress_test") or ns.get("apply_stress")
    if not fn: return False
    try:
        import numpy as np
        w = np.array([0.5, 0.5])
        mu = np.array([0.10, 0.08])
        scenarios = {
            "crash": {"return_shock": -0.40, "vol_shock": 2.5},
            "boom":  {"return_shock": 0.20,  "vol_shock": 0.8},
            "flat":  {"return_shock": 0.0,   "vol_shock": 1.0},
        }
        r = fn(w, mu, scenarios)
        if not isinstance(r, dict): return False
        if r["crash"] >= r["flat"]: return False   # crash < flat
        if r["boom"] <= r["flat"]: return False    # boom > flat
        return True
    except Exception: return False


# ── Task 14: Rebalancing simulation ───────────────────────────────────────────

def _probe_rebalance(ns):
    fn = ns.get("rebalancing_sim") or ns.get("simulate_rebalancing") or ns.get("rebalance")
    if not fn:
        return ["rebalancing_sim(prices_df, target_weights, freq='monthly', cost_bps=10) not found"]
    try:
        import numpy as np
        import pandas as pd
        # Monotonically increasing prices for A, flat for B
        n = 60
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        prices = pd.DataFrame({
            "A": 100 * np.cumprod(1 + np.ones(n) * 0.005),
            "B": np.ones(n) * 100.0,
        }, index=dates)
        target = np.array([0.5, 0.5])
        result = fn(prices, target, freq="monthly", cost_bps=10)
        if not isinstance(result, dict):
            return [f"rebalancing_sim must return dict with at least 'portfolio_value', got {type(result)}"]
        if "portfolio_value" not in result and "returns" not in result and "wealth" not in result:
            return ["Result dict must include 'portfolio_value', 'returns', or 'wealth' key"]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_rebalance(ns):
    fn = ns.get("rebalancing_sim") or ns.get("simulate_rebalancing") or ns.get("rebalance")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        n = 120
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        prices = pd.DataFrame({
            "A": 100 * np.cumprod(1 + np.full(n, 0.003)),
            "B": 100 * np.cumprod(1 + np.full(n, 0.001)),
        }, index=dates)
        r = fn(prices, np.array([0.5, 0.5]), freq="monthly", cost_bps=5)
        if not isinstance(r, dict): return False
        # Get the final portfolio value or return series
        val_key = next((k for k in ("portfolio_value","wealth","returns") if k in r), None)
        if val_key is None: return False
        return True
    except Exception: return False


# ── Task 15: Full risk report generator ───────────────────────────────────────

def _probe_report(ns):
    fn = ns.get("risk_report") or ns.get("generate_report") or ns.get("portfolio_report")
    if not fn:
        return ["risk_report(weights, returns_df, rf=0.02) not found — return dict with key metrics"]
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        n = 252
        rets = pd.DataFrame(np.random.multivariate_normal(
            [0.0004, 0.0003], [[0.0001, 0.00003], [0.00003, 0.0001]], n
        ), columns=["A", "B"])
        w = np.array([0.6, 0.4])
        result = fn(w, rets, rf=0.02)
        if not isinstance(result, dict):
            return [f"risk_report must return dict, got {type(result)}"]
        required = ["sharpe", "max_drawdown", "var_95"]
        missing = [k for k in required if k not in result and
                   not any(k in str(rk).lower() for rk in result.keys())]
        if missing:
            return [
                f"risk_report missing keys: {missing}. "
                "Include at minimum: sharpe, max_drawdown, var_95."
            ]
    except Exception as e:
        return [f"probe error: {e}"]
    return []


def _check_report(ns):
    fn = ns.get("risk_report") or ns.get("generate_report") or ns.get("portfolio_report")
    if not fn: return False
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        rets = pd.DataFrame(np.random.multivariate_normal(
            [0.0004, 0.0003], [[0.0001, 0.00003], [0.00003, 0.0001]], 252
        ), columns=["A", "B"])
        r = fn(np.array([0.6, 0.4]), rets, rf=0.02)
        if not isinstance(r, dict): return False
        keys_lower = {str(k).lower() for k in r.keys()}
        for req in ("sharpe", "drawdown", "var"):
            if not any(req in k for k in keys_lower): return False
        return True
    except Exception: return False


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK LIST
# ═══════════════════════════════════════════════════════════════════════════════

TASKS: list[dict] = [
    # ── Phase 1: Data Foundation ──────────────────────────────────────────────
    {
        "name": "Simulate correlated stock prices (GBM + Cholesky)",
        "want_fn": "simulate_prices",
        "probe": _probe_simulate, "check": _check_simulate,
        "prompt": textwrap.dedent("""
            We're building a stock portfolio risk analyzer. First step: realistic test data.

            Write `simulate_prices(n_days, tickers, seed=42) -> pd.DataFrame`.

            Use Geometric Brownian Motion with a correlation structure:
            1. Pick random annual mu (5-15%) and sigma (10-30%) per ticker.
            2. Build a correlation matrix with off-diagonal elements 0.3-0.7.
            3. Cholesky-decompose it to get correlated daily shocks.
            4. Simulate: price_{t+1} = price_t * exp((mu/252 - sigma^2/720) + sigma/sqrt(252) * shock_t)
               (start each ticker at 100.0)
            5. Return a DataFrame of shape (n_days, len(tickers)) with ticker column names.
            6. Use np.random.seed(seed) for reproducibility.

            Example: simulate_prices(252, ["AAPL","MSFT","GOOGL"], seed=42)
            should return a (252, 3) DataFrame, all values positive, same result every run.

            Call python_exec with your implementation and verify.
        """).strip(),
    },
    {
        "name": "Compute log daily returns",
        "want_fn": "compute_returns",
        "probe": _probe_returns, "check": _check_returns,
        "prompt": textwrap.dedent("""
            We need log daily returns from our price data.

            Write `compute_returns(prices_df) -> pd.DataFrame`.

            Requirements:
            - Use LOG returns: r_t = ln(P_t / P_{t-1})
            - NOT arithmetic returns (P_t/P_{t-1} - 1) — log returns are additive and work correctly
              for multi-period compounding.
            - Result shape: (n_days - 1, n_tickers)
            - Drop the first NaN row.

            Formula: np.log(prices_df / prices_df.shift(1)).dropna()

            Test: prices 100→110 gives log return ln(1.1) ≈ 0.0953 (NOT 0.10).
            Flat prices 50→50 gives log return = 0.

            Call python_exec with your implementation and verify with a small example.
        """).strip(),
    },
    {
        "name": "Rolling 20-day statistics (mean, vol, skew)",
        "want_fn": "rolling_stats",
        "probe": _probe_rolling, "check": _check_rolling,
        "prompt": textwrap.dedent("""
            For the portfolio dashboard we need rolling risk metrics.

            Write `rolling_stats(returns_df, window=20) -> dict`.

            Return a dict with keys:
              'mean': rolling mean returns (annualised: × 252)
              'vol':  rolling volatility  (annualised: × sqrt(252))
              'skew': rolling skewness    (not annualised)

            Use .rolling(window) on the returns DataFrame.
            First (window-1) rows will be NaN — that's expected.

            Test your function on a constant-return series (all values = 0.01):
              rolling vol should be ≈ 0 (constant series has no variation).
            And a varying series: vol should be positive.

            Call python_exec with your implementation and test both cases.
        """).strip(),
    },
    {
        "name": "Annualised covariance matrix",
        "want_fn": "covariance_matrix",
        "probe": _probe_covariance, "check": _check_covariance,
        "prompt": textwrap.dedent("""
            Portfolio optimisation needs an annualised covariance matrix.

            Write `covariance_matrix(returns_df, periods_per_year=252) -> pd.DataFrame`.

            Steps:
            1. Compute the sample covariance: returns_df.cov()  (gives daily covariance)
            2. Annualise: multiply by periods_per_year (252 for daily data)
            3. Return as a DataFrame with same column names.

            Common mistake: forgetting to multiply by 252. Daily vol^2 of 0.0001
            → annualised variance = 0.0001 × 252 = 0.0252 (= ~15.9% annual vol).

            Verify:
            - Matrix is symmetric
            - Diagonal elements are positive
            - Off-diagonal values reflect correlations between assets

            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Minimum variance portfolio (scipy.optimize)",
        "want_fn": "min_variance_portfolio",
        "probe": _probe_minvar, "check": _check_minvar,
        "prompt": textwrap.dedent("""
            Find the portfolio weights that minimise total variance (risk).

            Write `min_variance_portfolio(cov_matrix) -> dict`.

            Return dict with:
              'weights':  np.array of optimal weights (must sum to 1.0)
              'variance': resulting portfolio variance (w.T @ cov @ w)

            Constraints:
            - Weights sum to 1.0  (fully invested)
            - All weights ≥ 0     (long-only)

            Use scipy.optimize.minimize with SLSQP:
                from scipy.optimize import minimize
                n = len(cov_matrix)
                result = minimize(
                    fun=lambda w: w @ cov_matrix @ w,
                    x0=np.ones(n)/n,
                    method='SLSQP',
                    constraints={'type':'eq','fun':lambda w: w.sum()-1},
                    bounds=[(0,1)]*n
                )

            Test: for cov=diag([0.04, 0.16]), min-var puts ~80% in A (lower risk).

            Call python_exec with your implementation.
        """).strip(),
    },

    # ── Phase 2: Portfolio Theory ──────────────────────────────────────────────
    {
        "name": "Maximum Sharpe ratio portfolio (tangency portfolio)",
        "want_fn": "max_sharpe_portfolio",
        "probe": _probe_maxsharpe, "check": _check_maxsharpe,
        "prompt": textwrap.dedent("""
            Find the portfolio with the highest risk-adjusted return (Sharpe ratio).

            Write `max_sharpe_portfolio(expected_returns, cov_matrix, rf=0.02) -> dict`.

            Return dict with:
              'weights': np.array (sum to 1, long-only)
              'sharpe':  annualised Sharpe ratio of the optimal portfolio

            Maximise Sharpe by minimising its negative:
                def neg_sharpe(w):
                    port_return = w @ expected_returns
                    port_vol    = np.sqrt(w @ cov_matrix @ w)
                    return -(port_return - rf) / port_vol

            Same constraints as min-variance (sum=1, w≥0).

            Test: with mu=[0.10, 0.08] and equal variances, the higher-return
            asset A should get more weight.

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "1-day 95% Value at Risk (historical simulation)",
        "want_fn": "portfolio_var",
        "probe": _probe_var, "check": _check_var,
        "prompt": textwrap.dedent("""
            Compute the 1-day Value at Risk of a portfolio using historical simulation.

            Write `portfolio_var(returns_series, confidence=0.95) -> float`.

            VaR = the loss that is NOT exceeded with probability `confidence`.
            Return as a POSITIVE number (loss magnitude).

            For 95% confidence: we care about the worst 5% of days.
              var = -np.percentile(returns_series, (1 - confidence) * 100)
              var = -np.percentile(returns_series, 5)      # for confidence=0.95

            Common mistake: using np.percentile(rets, 95) instead of np.percentile(rets, 5)
            — that gives the BEST 5% of days, not the worst.

            For N(0, 0.01) returns: 95% VaR ≈ 0.0165 (1.645 standard deviations).

            Test:
              var_95 = portfolio_var(rets, 0.95)  → should be positive, ≈ 0.0165
              var_99 = portfolio_var(rets, 0.99)  → should be > var_95

            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Conditional VaR (Expected Shortfall at 95%)",
        "want_fn": "portfolio_cvar",
        "probe": _probe_cvar, "check": _check_cvar,
        "prompt": textwrap.dedent("""
            CVaR (Conditional Value at Risk) = Expected Shortfall = average loss
            BEYOND the VaR threshold. Always greater than VaR.

            Write `portfolio_cvar(returns_series, confidence=0.95) -> float`.

            Return as a POSITIVE number.

            Algorithm:
              threshold  = np.percentile(returns_series, (1 - confidence) * 100)
              tail       = returns_series[returns_series <= threshold]
              cvar       = -tail.mean()

            For N(0, 0.01) at 95%:
              VaR  ≈ 0.0165
              CVaR ≈ 0.0205   (always larger than VaR)

            Verify:
              cvar_95 > var_95      (CVaR > VaR — strictly)
              cvar_99 > cvar_95     (higher confidence = larger CVaR)

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Maximum drawdown",
        "want_fn": "max_drawdown",
        "probe": _probe_drawdown, "check": _check_drawdown,
        "prompt": textwrap.dedent("""
            Maximum drawdown measures the worst peak-to-trough decline.

            Write `max_drawdown(price_series) -> float`.

            Return as a NEGATIVE fraction (e.g., -0.30 for a 30% drawdown).

            Algorithm:
              running_max = price_series.cummax()
              drawdown    = (price_series - running_max) / running_max
              return drawdown.min()

            Example: prices [100, 110, 90, 95, 80, 100]
              running_max: [100, 110, 110, 110, 110, 110]
              drawdown at index 4: (80-110)/110 = -0.2727
              → max_drawdown = -0.2727

            Test:
              max_drawdown([100,110,90,80,100]) ≈ -0.2727
              max_drawdown([100,120,140])        = 0.0  (no drawdown in rising market)

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Annualised Sharpe ratio",
        "want_fn": "sharpe_ratio",
        "probe": _probe_sharpe, "check": _check_sharpe,
        "prompt": textwrap.dedent("""
            Sharpe ratio measures return per unit of risk, annualised.

            Write `sharpe_ratio(returns_series, rf=0.02, periods=252) -> float`.

            Formula (daily returns, annualised):
              excess = returns_series - rf/periods      ← daily risk-free rate
              sr = (excess.mean() / excess.std()) * sqrt(periods)

            Equivalently:
              sr = (returns_series.mean()*periods - rf) / (returns_series.std()*sqrt(periods))

            Common mistake: forgetting sqrt(252) annualisation → Sharpe off by factor 15.87.

            For daily N(0.0005, 0.01):
              annualised mu  ≈ 0.0005 × 252 = 12.6%
              annualised vol ≈ 0.01 × sqrt(252) = 15.87%
              Sharpe ≈ 12.6% / 15.87% ≈ 0.79

            A raw (non-annualised) computation would give ≈ 0.05 — 15× too low.

            Call python_exec with your implementation and test it.
        """).strip(),
    },

    # ── Phase 3: Risk Metrics ──────────────────────────────────────────────────
    {
        "name": "Portfolio beta to market",
        "want_fn": "portfolio_beta",
        "probe": _probe_beta, "check": _check_beta,
        "prompt": textwrap.dedent("""
            Beta measures a portfolio's sensitivity to the overall market.

            Write `portfolio_beta(portfolio_returns, market_returns) -> float`.

            Formula:
              beta = Cov(portfolio, market) / Var(market)
              beta = np.cov(portfolio_returns, market_returns)[0,1] / np.var(market_returns)

            Interpretation:
              beta = 1.0 → moves in line with market
              beta > 1.0 → amplified moves (aggressive)
              beta < 1.0 → dampened moves (defensive)

            Test:
              port = 1.5 × market + noise → beta ≈ 1.5
              port = 0.5 × market + noise → beta ≈ 0.5
              port = market               → beta = 1.0 exactly

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Risk contribution (marginal contribution to portfolio variance)",
        "want_fn": "risk_contribution",
        "probe": _probe_risk_contrib, "check": _check_risk_contrib,
        "prompt": textwrap.dedent("""
            Risk contribution shows how much each asset contributes to total portfolio variance.

            Write `risk_contribution(weights, cov_matrix) -> np.array`.

            Return an array that sums to 1.0 (percentage contributions).

            Formula:
              marginal = cov_matrix @ weights           ← sensitivity of each asset
              rc_i     = weights[i] * marginal[i]       ← weighted contribution
              rc       = rc / rc.sum()                  ← normalise to percentages

            In matrix form:
              port_var = weights @ cov_matrix @ weights
              rc = (weights * (cov_matrix @ weights)) / port_var

            Test with cov=diag([0.01, 0.10]) and equal weights [0.5, 0.5]:
              Asset B (10× higher variance) contributes ~91% of total risk.
              Asset A contributes only ~9%.

            Call python_exec with your implementation and verify.
        """).strip(),
    },
    {
        "name": "Stress test: apply shock scenarios to portfolio",
        "want_fn": "stress_test",
        "probe": _probe_stress, "check": _check_stress,
        "prompt": textwrap.dedent("""
            Stress testing applies historical or hypothetical shocks to the portfolio.

            Write `stress_test(weights, expected_returns, scenarios) -> dict`.

            - weights: np.array (portfolio weights)
            - expected_returns: np.array (annual expected return per asset)
            - scenarios: dict of {name: {'return_shock': float, 'vol_shock': float}}
              where return_shock is a multiplier on expected returns (e.g., -0.40 = crash)

            Algorithm:
              for each scenario:
                stressed_returns = expected_returns * (1 + scenario['return_shock'])
                portfolio_return = weights @ stressed_returns
                results[name] = portfolio_return

            Return dict mapping scenario names to portfolio returns.

            Example:
              scenarios = {
                "2008_crash": {"return_shock": -0.40, "vol_shock": 2.5},
                "base_case":  {"return_shock": 0.0,   "vol_shock": 1.0},
                "bull_run":   {"return_shock": 0.20,  "vol_shock": 0.7},
              }

            2008_crash portfolio return should be much lower than base_case.

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Monthly rebalancing simulation with transaction costs",
        "want_fn": "rebalancing_sim",
        "probe": _probe_rebalance, "check": _check_rebalance,
        "prompt": textwrap.dedent("""
            Simulate a monthly-rebalanced portfolio with transaction costs.

            Write `rebalancing_sim(prices_df, target_weights, freq='monthly', cost_bps=10) -> dict`.

            - prices_df:      DataFrame of daily prices (date index, ticker columns)
            - target_weights: np.array of target allocations (must sum to 1)
            - freq:           rebalancing frequency — 'daily', 'weekly', or 'monthly'
            - cost_bps:       transaction cost per trade in basis points (1 bps = 0.01%)

            Algorithm:
            1. Start with $1 invested at target_weights × prices on day 0.
            2. Hold until the next rebalancing date (monthly = ~21 business days).
            3. On rebalancing: compute current weights from drifted prices.
               turnover = sum(|current - target| weights).
               Apply cost: portfolio_value × (1 - turnover × cost_bps / 10000).
               Rebalance back to target_weights.
            4. Track portfolio_value each day.

            Return dict with:
              'portfolio_value': pd.Series of daily portfolio values
              'total_return':    total return over full period
              'n_rebalances':    number of rebalancing events

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Full portfolio risk report",
        "want_fn": "risk_report",
        "probe": _probe_report, "check": _check_report,
        "prompt": textwrap.dedent("""
            Final task: generate a complete risk report for a portfolio.

            Write `risk_report(weights, returns_df, rf=0.02) -> dict`.

            This is the capstone function that ties everything together.
            Compute and return ALL of these metrics in a single dict:

            {
              'sharpe':          annualised Sharpe ratio,
              'max_drawdown':    maximum drawdown (negative fraction),
              'var_95':          1-day 95% VaR (positive),
              'cvar_95':         1-day 95% CVaR (positive),
              'ann_return':      annualised portfolio return,
              'ann_volatility':  annualised portfolio volatility,
              'beta':            beta vs equal-weight benchmark,
            }

            Implement each metric from scratch within this function
            (do not assume helper functions exist — write it self-contained).

            portfolio_returns = returns_df @ weights   ← weighted daily returns

            Test with:
              np.random.seed(42)
              rets = pd.DataFrame(np.random.multivariate_normal(
                  [0.0004, 0.0003], [[0.0001, 0.00003],[0.00003,0.0001]], 252
              ), columns=["A","B"])
              report = risk_report(np.array([0.6, 0.4]), rets, rf=0.02)

            Call python_exec with your implementation and print the full report.
        """).strip(),
    },
]

assert len(TASKS) == 15, f"Expected 15 tasks, got {len(TASKS)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval():
    print("\n" + "═" * 72)
    print("  eval_project — Portfolio Risk Analyzer vibe-coding session")
    print("  15 tasks: simulate → returns → stats → covariance → min-var →")
    print("            max-Sharpe → VaR → CVaR → drawdown → Sharpe →")
    print("            beta → risk-contrib → stress → rebalance → report")
    print("  Model: Haiku  |  Brain: starts cold, adapts in real time")
    print("═" * 72)

    embedder   = build_embedder()
    brain      = BrainAgent(embedder, threshold=0.28, k=5)
    code_agent = tool_agent(["python_exec"], max_turns=MAX_TURNS,
                            model=HAIKU, max_tokens=MAX_TOK)
    code_agent.monitor = brain
    viz = BrainViz()

    results:     list[dict] = []
    fire_counts: list[int]  = []

    for i, task in enumerate(TASKS):
        n = i + 1
        print(f"\n  {n:>2}/{len(TASKS)} {task['name']}"
              f"  (brain: {brain.n_stored} stored)")

        brain.set_task(i, probe_fn=task["probe"])
        brain.reset()

        t0         = time.time()
        code_agent.monitor = brain
        result     = code_agent(task["prompt"])
        trace, tok = result if isinstance(result, tuple) else (str(result), 0)
        fires      = brain._code_interventions

        first_code       = _first_exec_code(trace, task.get("want_fn"))
        first_p, first_d = _check_code(first_code, task["check"])

        final_code  = _extract_code(trace, task.get("want_fn"))
        passed, det = _check_code(final_code, task["check"])

        elapsed     = time.time() - t0
        brain_fixed = (not first_p) and passed

        fire_tag = f"  [⚡×{fires}]"  if fires      else ""
        fix_tag  = "  [↑ brain fixed]" if brain_fixed else ""
        status   = "PASS" if passed else "FAIL"
        base_tag = (f"  (baseline: {'PASS' if first_p else 'FAIL'})"
                    if first_p != passed else "")

        print(f"       {status}  {tok:>7,} tok  {elapsed:.0f}s"
              f"{fire_tag}{fix_tag}{base_tag}"
              + (f"  {det[:55]}" if det and not passed else ""))

        brain.store(trace, int(first_p), metadata=first_d if not first_p else "")
        if first_code:
            brain.store_code(first_code, int(first_p),
                             metadata=first_d if not first_p else "")

        results.append({
            "task":         n,
            "name":         task["name"],
            "first_passed": first_p,
            "passed":       passed,
            "brain_helped": brain_fixed,
            "fires":        fires,
            "tokens":       tok,
            "elapsed":      round(elapsed, 1),
            "detail":       det,
        })
        fire_counts.append(fires)

        viz.update(brain, results, fire_counts)
        viz.save(OUT / "brain_project.png")

    _report(results, fire_counts)
    with open(OUT / "project_run.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved eval/results/brain_project.png + project_run.json")
    return results


def _report(results, fire_counts):
    n       = len(results)
    n_base  = sum(1 for r in results if r["first_passed"])
    n_final = sum(1 for r in results if r["passed"])
    n_fired = sum(1 for c in fire_counts if c > 0)
    helped  = [r for r in results if r["brain_helped"]]

    print("\n" + "═" * 72)
    print("  RESULTS  [Portfolio Risk Analyzer — Haiku + Brain]")
    print("─" * 72)
    print(f"  Without brain (first attempt) : {n_base}/{n}  ({n_base/n:.0%})")
    print(f"  With brain (after intervention): {n_final}/{n}  ({n_final/n:.0%})")
    delta = n_final - n_base
    print(f"  Brain contribution             : +{delta} task{'s' if delta != 1 else ''}"
          f"  (fired on {n_fired}/{n} tasks)")
    if helped:
        print(f"\n  Tasks brain fixed:")
        for r in helped:
            print(f"    + {r['name']}  (probe ⚡×{r['fires']})")
    total_tokens = sum(r["tokens"] for r in results)
    total_time   = sum(r["elapsed"] for r in results)
    print("─" * 72)

    print(f"\n  Per-task breakdown:")
    for r in results:
        b = "✓" if r["first_passed"] else "✗"
        f = "✓" if r["passed"]       else "✗"
        fire = f" ⚡×{r['fires']}" if r["fires"] else ""
        fixed = " ↑FIX" if r["brain_helped"] else ""
        print(f"    [{b}→{f}]{fire}{fixed}  {r['name'][:50]}")

    print(f"\n  Tokens: {total_tokens:,}  |  Time: {total_time:.0f}s")
    print("═" * 72)


if __name__ == "__main__":
    run_eval()
