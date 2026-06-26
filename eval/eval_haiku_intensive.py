"""eval/eval_haiku_intensive.py — 30-task intensive haiku evaluation.

Single-pass design: brain starts cold and learns in real time.
Tasks ordered so the brain accumulates failure patterns before similar later tasks.

15 code tasks — chosen for haiku's one-shot failure modes:
  LRU, slide-win-max, histogram, regex, burst balloons, trie,
  edit distance, coin-change-2 (combinations), decode ways,
  trapping rain water, jump game II, word break,
  union-find, course schedule, N-queens count.

15 text tasks — physics/probability/math haiku reliably gets wrong:
  Bayesian, rolling sphere, Hβ line, buoyancy, Simpson's, Bertrand box,
  Monty Hall, Carnot efficiency (needs Kelvin!), Snell's critical angle,
  capacitor energy (½CV²), perfectly-inelastic collision, geometric dist.,
  Doppler approaching source, electric field (parallel plates), birthday paradox.

Run:  python eval/eval_haiku_intensive.py
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
import time
import threading
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from trace_use import build_embedder, tool_agent
from trace_use.agents import _anthropic_call
from trace_use import BrainAgent
from eval.viz_brain import BrainViz

# Import shared extraction utilities and existing probes/checks from eval_hard
from eval.eval_hard import (
    _exec, _check_code, _check_text, _extract_code, _first_exec_code,
    _probe_lru,  _check_lru,
    _probe_swmax, _check_swmax,
    _probe_histogram, _check_histogram,
    _probe_regex, _check_regex,
    _probe_bank,  _check_bank,
    _probe_burst, _check_burst,
    _probe_trie,  _check_trie,
    _check_bayes, _check_rolling, _check_twin, _check_hline,
    _check_buoyancy, _check_simpsons, _check_bertrand,
)

OUT   = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

HAIKU     = "claude-haiku-4-5-20251001"
MAX_TURNS = 14
MAX_TOK   = 4096


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW CODE PROBES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Edit distance ──────────────────────────────────────────────────────────────

def _probe_edit(ns):
    fn = ns.get("minDistance") or ns.get("editDistance") or ns.get("edit_distance")
    if not fn:
        return ["minDistance(word1, word2) not defined — implement it and call python_exec"]
    fails = []
    try:
        r = fn("horse", "ros")
        if r != 3:
            fails.append(
                f"minDistance('horse','ros')={r}, expected 3. "
                "FIX: dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, "
                "dp[i-1][j-1]+(0 if s1[i-1]==s2[j-1] else 1)). "
                "Base cases: dp[i][0]=i and dp[0][j]=j (essential!)."
            )
            return fails
        if fn("", "abc") != 3:
            fails.append(f"minDistance('','abc')={fn('','abc')}, expected 3. FIX: dp[0][j]=j.")
        if fn("intention", "execution") != 5:
            fails.append(f"minDistance('intention','execution')={fn('intention','execution')}, expected 5.")
        if fn("abc", "abc") != 0:
            fails.append(f"minDistance('abc','abc')={fn('abc','abc')}, expected 0.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_edit(ns):
    fn = ns.get("minDistance") or ns.get("editDistance") or ns.get("edit_distance")
    if not fn: return False
    try:
        return all(fn(a, b) == e for a, b, e in [
            ("horse", "ros", 3), ("intention", "execution", 5),
            ("", "abc", 3), ("abc", "", 3), ("abc", "abc", 0),
            ("a", "b", 1), ("kitten", "sitting", 3), ("sunday", "saturday", 3),
        ])
    except Exception: return False


# ── Coin Change 2 (number of combinations) ───────────────────────────────────

def _probe_coin2(ns):
    fn = ns.get("change") or ns.get("coinChange2") or ns.get("coin_change_2")
    if not fn:
        return ["change(amount, coins) not defined — implement and call python_exec"]
    fails = []
    try:
        r = fn(5, [1, 2, 5])
        if r != 4:
            fails.append(
                f"change(5, [1,2,5])={r}, expected 4. "
                "FIX: to count COMBINATIONS (not permutations) loop COINS outer, AMOUNTS inner. "
                "dp = [0]*(amount+1); dp[0]=1. "
                "for coin in coins: for a in range(coin, amount+1): dp[a] += dp[a-coin]. "
                "Reversing the loop order counts ordered sequences (permutations) instead."
            )
            return fails
        if fn(3, [2]) != 0:
            fails.append(f"change(3, [2])={fn(3,[2])}, expected 0 — 3 is not reachable with 2s only.")
        if fn(10, [10]) != 1:
            fails.append(f"change(10, [10])={fn(10,[10])}, expected 1.")
        if fn(0, [1, 2]) != 1:
            fails.append(f"change(0, [1,2])={fn(0,[1,2])}, expected 1 (empty combo).")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_coin2(ns):
    fn = ns.get("change") or ns.get("coinChange2") or ns.get("coin_change_2")
    if not fn: return False
    try:
        return all(fn(a, c) == e for a, c, e in [
            (5,   [1,2,5], 4), (3, [2], 0), (10, [10], 1),
            (0,   [1,2],   1), (500, [1,2,5], 12701),
        ])
    except Exception: return False


# ── Decode ways ────────────────────────────────────────────────────────────────

def _probe_decode(ns):
    fn = ns.get("numDecodings") or ns.get("decode_ways") or ns.get("num_decodings")
    if not fn:
        return ["numDecodings(s) not defined — implement and call python_exec"]
    fails = []
    try:
        if fn("12") != 2:
            fails.append(f"numDecodings('12')={fn('12')}, expected 2 (decode as 1+2 or 12).")
            return fails
        if fn("226") != 3:
            fails.append(f"numDecodings('226')={fn('226')}, expected 3 (2+2+6, 22+6, 2+26).")
        if fn("06") != 0:
            fails.append(
                f"numDecodings('06')={fn('06')}, expected 0. "
                "FIX: a digit '0' standing alone is invalid (no letter maps to 0). "
                "Single-digit decode: valid only if s[i] != '0'. "
                "Two-digit decode: valid only if '10' <= s[i-1:i+1] <= '26'."
            )
        if fn("10") != 1:
            fails.append(
                f"numDecodings('10')={fn('10')}, expected 1 (only '10'=J is valid; '1'+'0' is not). "
                "FIX: '0' alone = 0 ways; only the two-digit path dp[i] += dp[i-2] contributes."
            )
        if fn("0") != 0:
            fails.append(f"numDecodings('0')={fn('0')}, expected 0.")
        if fn("11106") != 2:
            fails.append(f"numDecodings('11106')={fn('11106')}, expected 2.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_decode(ns):
    fn = ns.get("numDecodings") or ns.get("decode_ways") or ns.get("num_decodings")
    if not fn: return False
    try:
        return all(fn(s) == e for s, e in [
            ("12", 2), ("226", 3), ("06", 0), ("10", 1), ("0", 0),
            ("1", 1), ("11106", 2), ("2101", 1), ("111", 3),
        ])
    except Exception: return False


# ── Trapping rain water ────────────────────────────────────────────────────────

def _probe_trap(ns):
    fn = ns.get("trap") or ns.get("trappingRainWater") or ns.get("trapping_rain")
    if not fn:
        return ["trap(height) not defined — implement and call python_exec"]
    fails = []
    try:
        r = fn([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
        if r != 6:
            fails.append(
                f"trap([0,1,0,2,1,0,1,3,2,1,2,1])={r}, expected 6. "
                "FIX: two-pointer approach. left=0, right=n-1, lmax=rmax=0. "
                "While left<right: if height[left]<=height[right]: "
                "if height[left]>=lmax: lmax=height[left] else water+=lmax-height[left]; left++. "
                "Else: mirror logic on right side."
            )
            return fails
        if fn([4, 2, 0, 3, 2, 5]) != 9:
            fails.append(f"trap([4,2,0,3,2,5])={fn([4,2,0,3,2,5])}, expected 9.")
        if fn([3, 0, 2, 0, 4]) != 7:
            fails.append(f"trap([3,0,2,0,4])={fn([3,0,2,0,4])}, expected 7.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_trap(ns):
    fn = ns.get("trap") or ns.get("trappingRainWater") or ns.get("trapping_rain")
    if not fn: return False
    try:
        return all(fn(h) == e for h, e in [
            ([0,1,0,2,1,0,1,3,2,1,2,1], 6), ([4,2,0,3,2,5], 9),
            ([3,0,2,0,4], 7), ([1,0,1], 1), ([0], 0), ([3,1,2,4,0,1,3,2], 8),
        ])
    except Exception: return False


# ── Jump Game II (minimum jumps) ──────────────────────────────────────────────

def _probe_jump(ns):
    fn = ns.get("jump") or ns.get("jumpGame2") or ns.get("jump_game_2")
    if not fn:
        return ["jump(nums) not defined — implement and call python_exec"]
    fails = []
    try:
        r = fn([2, 3, 1, 1, 4])
        if r != 2:
            fails.append(
                f"jump([2,3,1,1,4])={r}, expected 2. "
                "FIX: greedy. Track cur_end=0, farthest=0, jumps=0. "
                "For i in range(len-1): farthest=max(farthest, i+nums[i]). "
                "If i==cur_end: jumps++; cur_end=farthest. Return jumps."
            )
            return fails
        if fn([2, 3, 0, 1, 4]) != 2:
            fails.append(f"jump([2,3,0,1,4])={fn([2,3,0,1,4])}, expected 2.")
        if fn([0]) != 0:
            fails.append(f"jump([0])={fn([0])}, expected 0 (already at end).")
        if fn([1, 1, 1, 1]) != 3:
            fails.append(f"jump([1,1,1,1])={fn([1,1,1,1])}, expected 3.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_jump(ns):
    fn = ns.get("jump") or ns.get("jumpGame2") or ns.get("jump_game_2")
    if not fn: return False
    try:
        return all(fn(a) == e for a, e in [
            ([2,3,1,1,4], 2), ([2,3,0,1,4], 2), ([0], 0),
            ([1,1,1,1], 3), ([1,2,3], 2), ([5,4,3,2,1,0], 1),
        ])
    except Exception: return False


# ── Word Break ────────────────────────────────────────────────────────────────

def _probe_wordbreak(ns):
    fn = ns.get("wordBreak") or ns.get("word_break")
    if not fn:
        return ["wordBreak(s, wordDict) not defined — implement and call python_exec"]
    fails = []
    try:
        if not fn("leetcode", ["leet", "code"]):
            fails.append(
                "wordBreak('leetcode', ['leet','code'])=False, expected True. "
                "FIX: dp[i] = True if any dp[j] is True and s[j:i] in wordSet. "
                "dp[0]=True (empty string = base case). "
                "for i in range(1, len+1): for j in range(i): if dp[j] and s[j:i] in wordSet: dp[i]=True."
            )
            return fails
        if fn("catsandog", ["cats", "dog", "sand", "and", "cat"]):
            fails.append(
                "wordBreak('catsandog', ...)=True, expected False — "
                "'catsandog' cannot be fully segmented."
            )
        if not fn("applepenapple", ["apple", "pen"]):
            fails.append(
                f"wordBreak('applepenapple', ['apple','pen'])=False, expected True."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_wordbreak(ns):
    fn = ns.get("wordBreak") or ns.get("word_break")
    if not fn: return False
    try:
        cases = [
            ("leetcode",      ["leet","code"],                      True),
            ("catsandog",     ["cats","dog","sand","and","cat"],     False),
            ("applepenapple", ["apple","pen"],                       True),
            ("a",             ["b"],                                 False),
            ("",              ["a"],                                 True),
            ("aaaaaaa",       ["aaaa","aaa"],                        True),
        ]
        return all(fn(s, d) == e for s, d, e in cases)
    except Exception: return False


# ── Union-Find with path compression ─────────────────────────────────────────

def _probe_uf(ns):
    UF = ns.get("UnionFind") or ns.get("DSU") or ns.get("DisjointSet")
    if not UF:
        return ["UnionFind(n) class not defined — implement with path compression + union by rank, call python_exec"]
    fails = []
    try:
        uf = UF(5)
        uf.union(0, 1); uf.union(1, 2); uf.union(3, 4)
        if uf.find(0) != uf.find(2):
            fails.append(
                "find(0) != find(2) after union(0,1) and union(1,2). "
                "FIX: union by rank — always attach smaller tree under root of larger. "
                "Path compression — in find(), set parent[x]=parent[parent[x]] (or full compression)."
            )
            return fails
        if uf.find(0) == uf.find(3):
            fails.append("find(0) should NOT equal find(3) — they are in different components.")
            return fails
        uf.union(2, 3)
        if uf.find(0) != uf.find(4):
            fails.append("After union(2,3), all 5 nodes should be in one component.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_uf(ns):
    UF = ns.get("UnionFind") or ns.get("DSU") or ns.get("DisjointSet")
    if not UF: return False
    try:
        uf = UF(6)
        uf.union(0,1); uf.union(2,3); uf.union(4,5); uf.union(1,2)
        if uf.find(0) != uf.find(3): return False
        if uf.find(0) == uf.find(4): return False
        # path compression: repeated finds should be stable
        r = uf.find(0)
        return uf.find(0) == r == uf.find(1) == uf.find(2) == uf.find(3)
    except Exception: return False


# ── Course schedule (cycle detection via DFS) ─────────────────────────────────

def _probe_courses(ns):
    fn = ns.get("canFinish") or ns.get("can_finish")
    if not fn:
        return ["canFinish(numCourses, prerequisites) not defined — implement and call python_exec"]
    fails = []
    try:
        if not fn(2, [[1, 0]]):
            fails.append("canFinish(2, [[1,0]])=False, expected True. Course 1 requires 0, no cycle.")
            return fails
        if fn(2, [[1, 0], [0, 1]]):
            fails.append(
                "canFinish(2, [[1,0],[0,1]])=True, expected False (0→1→0 cycle). "
                "FIX: DFS with 3-state coloring: 0=unvisited, 1=in-progress, 2=done. "
                "If you reach a node in state 1 during DFS, there's a cycle → return False."
            )
            return fails
        if not fn(4, [[1,0],[2,0],[3,1],[3,2]]):
            fails.append("canFinish(4, [[1,0],[2,0],[3,1],[3,2]])=False, expected True (DAG).")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_courses(ns):
    fn = ns.get("canFinish") or ns.get("can_finish")
    if not fn: return False
    try:
        return all(fn(n, p) == e for n, p, e in [
            (2, [[1,0]], True),
            (2, [[1,0],[0,1]], False),
            (4, [[1,0],[2,0],[3,1],[3,2]], True),
            (1, [], True),
            (3, [[0,1],[0,2],[1,2]], True),
            (3, [[0,1],[1,2],[2,0]], False),
        ])
    except Exception: return False


# ── N-Queens (count solutions) ────────────────────────────────────────────────

def _probe_nqueens(ns):
    fn = ns.get("totalNQueens") or ns.get("total_n_queens") or ns.get("solveNQueens")
    if not fn:
        return ["totalNQueens(n) not defined — implement and call python_exec"]
    fails = []
    try:
        r1 = fn(1)
        expected1 = 1 if not isinstance(r1, list) else len(r1)
        r4 = fn(4)
        expected4 = 2 if not isinstance(r4, list) else len(r4)
        r8 = fn(8)
        expected8 = 92 if not isinstance(r8, list) else len(r8)

        if expected4 != 2:
            fails.append(
                f"totalNQueens(4)={expected4}, expected 2. "
                "FIX: backtrack with sets for columns, diagonals (r-c), anti-diagonals (r+c). "
                "At each row, try each column: if not in any set, place queen, recurse, backtrack."
            )
            return fails
        if expected1 != 1:
            fails.append(f"totalNQueens(1)={expected1}, expected 1.")
        if expected8 != 92:
            fails.append(f"totalNQueens(8)={expected8}, expected 92.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_nqueens(ns):
    fn = ns.get("totalNQueens") or ns.get("total_n_queens") or ns.get("solveNQueens")
    if not fn: return False
    try:
        def count(r):
            return r if not isinstance(r, list) else len(r)
        return all(count(fn(n)) == e for n, e in [
            (1,1), (2,0), (3,0), (4,2), (5,10), (6,4), (8,92),
        ])
    except Exception: return False


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW TEXT CHECK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _check_monty(resp: str) -> bool:
    r = resp.lower()
    has_right  = bool(re.search(r'\b2/3\b|two.thirds?|66\.?6?%|0\.6+7?\b', r))
    has_switch = bool(re.search(r'switch|change', r))
    wrong_half = bool(re.search(r'\b1/2\b|\b50\s*%\b', r)) and not has_right
    return has_right and has_switch and not wrong_half


def _check_carnot(resp: str) -> bool:
    # η ≈ 65.6% (T_hot=600°C=873K, T_cold=27°C=300K)
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*%', r):
        if 63.0 <= float(n) <= 68.0:
            return True
    # Kelvin mention is also a strong signal
    if re.search(r'873\s*k|300\s*k', r):
        for n in re.findall(r'0\.\d+', r):
            if 0.63 <= float(n) <= 0.68:
                return True
    return False


def _check_snell(resp: str) -> bool:
    # critical angle ≈ 41.8° for n1=1.5, n2=1.0
    r = resp.lower()
    for n in re.findall(r'(\d+\.?\d*)\s*°', r):
        if 40.5 <= float(n) <= 43.0:
            return True
    # Also accept arcsin(2/3) or ~41.8 mentioned as degrees
    if re.search(r'41\.8|arcsin.*2.*3|sin.*0\.66|sin.*0\.67', r):
        return True
    return False


def _check_capacitor(resp: str) -> bool:
    # E = ½ × 47e-6 × 230² = 1.243 J
    r = resp.lower().replace(",", "")
    # Check for ½CV² formula first (more important than value)
    has_formula = bool(re.search(r'½\s*cv|1/2\s*cv|0\.5\s*cv|half.*cv', r))
    # Accept 1.2–1.3 J range
    for n in re.findall(r'(\d+\.?\d*)\s*j\b', r):
        if 1.1 <= float(n) <= 1.35:
            return True
    # Accept millijoule notation: 1243 mJ or ~1240 mJ
    for n in re.findall(r'(\d{3,4})\s*mj', r):
        if 1100 <= int(n) <= 1350:
            return True
    return has_formula and bool(re.search(r'1\.2|1\.24|1243', r))


def _check_collision(resp: str) -> bool:
    # v' = 3 m/s, energy lost = 60 J
    r = resp.lower().replace(",", "")
    has_v   = bool(re.search(r'\b3\s*(m/s|m s)', r)) or "3.0 m/s" in r
    has_ke  = bool(re.search(r'\b60\s*j\b|60\.0\s*j', r))
    return has_v and has_ke


def _check_geometric(resp: str) -> bool:
    # E[X] = 2 flips, Var = 2
    r = resp.lower()
    has_e   = bool(re.search(r'expected.*\b2\b|\b2\b.*flip|mean.*\b2\b|e\[x\].*=.*2|= 2 flip', r))
    has_var = bool(re.search(r'var.*=.*2|variance.*2|σ².*=.*2', r))
    return has_e or has_var


def _check_doppler(resp: str) -> bool:
    # f' ≈ 767.7 Hz (ambulance at 30 m/s, f=700, v=340)
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*hz', r):
        if 760 <= float(n) <= 775:
            return True
    if re.search(r'767|768', r):
        return True
    return False


def _check_efield(resp: str) -> bool:
    # E = V/d = 120 / 0.005 = 24,000 V/m = 24 kV/m
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*kv/m', r):
        if 22 <= float(n) <= 26:
            return True
    # 24000 V/m
    if re.search(r'24\s*000|24000|2\.4\s*×?\s*10\^?4', r):
        return True
    return False


def _check_birthday(resp: str) -> bool:
    # P(≥2 same birthday, n=23) ≈ 50.73%
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*%', r):
        if 49.0 <= float(n) <= 52.0:
            return True
    # Accept "23 people" + mention of ~50%
    has_23 = "23" in r
    has_50 = bool(re.search(r'\b50\b|0\.507|0\.5073', r))
    return has_23 and has_50


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK LIST  —  15 code  +  15 text  =  30 tasks
# ═══════════════════════════════════════════════════════════════════════════════

TASKS: list[dict] = [
    # ── CODE ──────────────────────────────────────────────────────────────────
    {
        "name": "LRU Cache (O(1) get/put)",
        "type": "code", "want_fn": "LRUCache",
        "probe": _probe_lru, "check": _check_lru,
        "prompt": textwrap.dedent("""
            Implement an LRU (Least Recently Used) cache.
            Class: LRUCache(capacity)  with  get(key)->int  and  put(key, value).
            - get/put must both be O(1)
            - Do NOT use OrderedDict or functools.lru_cache
            - Implement doubly-linked list + hashmap yourself

            Test capacity=2 (access-order eviction) and capacity=1.
            Call python_exec with your complete implementation.
        """).strip(),
    },
    {
        "name": "Sliding window maximum (monotone deque)",
        "type": "code", "want_fn": "maxSlidingWindow",
        "probe": _probe_swmax, "check": _check_swmax,
        "prompt": textwrap.dedent("""
            Write `maxSlidingWindow(nums: list[int], k: int) -> list[int]`.
            Return the max in each sliding window of size k.
            Example: maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3) → [3,3,5,5,6,7]
            Requirement: O(n) time using a monotone deque of indices (no max() per window).
            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Edit distance (Levenshtein)",
        "type": "code", "want_fn": "minDistance",
        "probe": _probe_edit, "check": _check_edit,
        "prompt": textwrap.dedent("""
            Write `minDistance(word1: str, word2: str) -> int`.
            Return the minimum edit distance (insert, delete, or replace one character).
            Examples:
              minDistance("horse", "ros") = 3
              minDistance("intention", "execution") = 5
              minDistance("", "abc") = 3

            Use bottom-up DP (2D table, size (m+1)×(n+1)).
            Base cases: dp[i][0]=i  and  dp[0][j]=j  (these are commonly missed!).
            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Coin Change 2 — count combinations",
        "type": "code", "want_fn": "change",
        "probe": _probe_coin2, "check": _check_coin2,
        "prompt": textwrap.dedent("""
            Write `change(amount: int, coins: list[int]) -> int`.
            Count the number of COMBINATIONS (not ordered sequences) of coins
            that sum to amount.
            Example: change(5, [1,2,5]) = 4
              (1+1+1+1+1, 1+1+1+2, 1+2+2, 5)

            IMPORTANT: outer loop = coins, inner loop = amounts.
            Reversing the loops counts permutations instead of combinations.
            dp[0]=1, rest=0.  dp[a] += dp[a - coin].
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Decode Ways (DP with zero-handling)",
        "type": "code", "want_fn": "numDecodings",
        "probe": _probe_decode, "check": _check_decode,
        "prompt": textwrap.dedent("""
            Write `numDecodings(s: str) -> int`.
            Count ways to decode a digit string where 'A'=1, 'B'=2, ..., 'Z'=26.
            Examples:
              "12"  → 2  (AB or L)
              "226" → 3  (BBF, BZ, VF)
              "06"  → 0  (invalid: '0' alone, and "06">26)
              "10"  → 1  (J only; '1'+'0' is invalid since '0' alone = nothing)

            Key edge cases:
            - Single digit valid: s[i] != '0'
            - Two digit valid: '10' <= s[i-1:i+1] <= '26'
            - dp[0]=1 (empty string), dp[1]=1 if s[0]!='0' else 0
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Trapping rain water (two pointers)",
        "type": "code", "want_fn": "trap",
        "probe": _probe_trap, "check": _check_trap,
        "prompt": textwrap.dedent("""
            Write `trap(height: list[int]) -> int`.
            Compute how much water can be trapped between bars.
            Example: trap([0,1,0,2,1,0,1,3,2,1,2,1]) = 6

            Use the O(n) two-pointer approach:
              left=0, right=n-1, left_max=0, right_max=0
              While left < right:
                if height[left] <= height[right]:
                  if height[left] >= left_max: left_max = height[left]
                  else: water += left_max - height[left]
                  left++
                else: mirror logic on right side

            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Largest rectangle in histogram (stack)",
        "type": "code", "want_fn": "largestRectangleArea",
        "probe": _probe_histogram, "check": _check_histogram,
        "prompt": textwrap.dedent("""
            Write `largestRectangleArea(heights: list[int]) -> int`.
            Example: largestRectangleArea([2,1,5,6,2,3]) = 10
            O(n) monotone stack: append sentinel 0 so remaining stack is flushed at end.
            When popping h: width = i - stack[-1] - 1  (or i if stack empty).
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Word Break (DP)",
        "type": "code", "want_fn": "wordBreak",
        "probe": _probe_wordbreak, "check": _check_wordbreak,
        "prompt": textwrap.dedent("""
            Write `wordBreak(s: str, wordDict: list[str]) -> bool`.
            Return True if s can be segmented into words from wordDict.
            Examples:
              wordBreak("leetcode", ["leet","code"]) = True
              wordBreak("catsandog", ["cats","dog","sand","and","cat"]) = False

            Use 1D DP: dp[i] = True if s[:i] can be segmented.
            dp[0] = True.
            For i in 1..len(s): for j in 0..i: if dp[j] and s[j:i] in wordSet: dp[i]=True.
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Jump Game II (min jumps)",
        "type": "code", "want_fn": "jump",
        "probe": _probe_jump, "check": _check_jump,
        "prompt": textwrap.dedent("""
            Write `jump(nums: list[int]) -> int`.
            Return minimum number of jumps to reach the last index.
            nums[i] = max jump length from index i. Always reachable.
            Example: jump([2,3,1,1,4]) = 2

            Greedy: track current_end and farthest.
            for i in range(len(nums)-1):
              farthest = max(farthest, i + nums[i])
              if i == current_end: jumps++; current_end = farthest
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Regex matching (. and *)",
        "type": "code", "want_fn": "isMatch",
        "probe": _probe_regex, "check": _check_regex,
        "prompt": textwrap.dedent("""
            Write `isMatch(s: str, p: str) -> bool`.
            Implement regex where '.' matches one char and '*' = zero or more of preceding.
            isMatch("aa","a*")=True, isMatch("aab","c*a*b")=True, isMatch("aa",".")=False
            Use bottom-up DP table. When p[j-1]=='*':
              dp[i][j] = dp[i][j-2]  (zero occurrences)
                      OR dp[i-1][j] if s[i-1]==p[j-2] or p[j-2]=='.'
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Burst balloons (interval DP)",
        "type": "code", "want_fn": "maxCoins",
        "probe": _probe_burst, "check": _check_burst,
        "prompt": textwrap.dedent("""
            Write `maxCoins(nums: list[int]) -> int`.
            Burst all balloons. Bursting i gives nums[i-1]*nums[i]*nums[i+1] coins.
            Example: maxCoins([3,1,5,8]) = 167.
            Interval DP: pad [1]+nums+[1]. dp[i][j]=max coins strictly between i and j.
            k = LAST balloon burst in (i,j): dp[i][j]=max(dp[i][k]+nums[i]*nums[k]*nums[j]+dp[k][j]).
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Union-Find (path compression + rank)",
        "type": "code", "want_fn": "UnionFind",
        "probe": _probe_uf, "check": _check_uf,
        "prompt": textwrap.dedent("""
            Implement UnionFind (Disjoint Set Union):
              class UnionFind:
                  def __init__(self, n: int)
                  def find(self, x: int) -> int    # returns root; use path compression
                  def union(self, x: int, y: int)  # merge by rank

            Path compression: in find(), set parent[x] = parent[parent[x]] (path halving)
            or recursively flatten.
            Union by rank: attach root of smaller rank under root of larger rank.

            Call python_exec with your implementation and test:
            UnionFind(5); union(0,1); union(1,2); union(3,4) → 0,1,2 connected; 3,4 connected.
        """).strip(),
    },
    {
        "name": "Course schedule (cycle detection)",
        "type": "code", "want_fn": "canFinish",
        "probe": _probe_courses, "check": _check_courses,
        "prompt": textwrap.dedent("""
            Write `canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool`.
            Return True if all courses can be finished (no circular dependency).
            [a, b] means course a requires b first.
            Example: canFinish(2, [[1,0],[0,1]]) = False  (cycle: 0↔1)

            DFS with 3-state coloring: 0=unvisited, 1=in-progress, 2=done.
            If DFS reaches a node in state 1 → cycle → return False.
            Build adjacency list from prerequisites.
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Trie (insert / search / startsWith)",
        "type": "code", "want_fn": "Trie",
        "probe": _probe_trie, "check": _check_trie,
        "prompt": textwrap.dedent("""
            Implement:
              class Trie:
                  def insert(self, word: str) -> None
                  def search(self, word: str) -> bool   # exact word only
                  def startsWith(self, prefix: str) -> bool

            If only "apple" inserted: search("app")=False, startsWith("app")=True.
            Use TrieNode with dict children and is_end flag.
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "N-Queens (count solutions)",
        "type": "code", "want_fn": "totalNQueens",
        "probe": _probe_nqueens, "check": _check_nqueens,
        "prompt": textwrap.dedent("""
            Write `totalNQueens(n: int) -> int`.
            Return the number of distinct N-Queens solutions on an n×n board.
            totalNQueens(4)=2, totalNQueens(8)=92.

            Backtrack row by row. Track three sets:
              cols — occupied columns
              diag — occupied diagonals (row - col)
              anti — occupied anti-diagonals (row + col)

            For each row, try each column not in any set, recurse, backtrack.
            Call python_exec with your implementation.
        """).strip(),
    },

    # ── TEXT ──────────────────────────────────────────────────────────────────
    {
        "name": "Bayesian disease screening (base-rate neglect)",
        "type": "text", "check": _check_bayes,
        "prompt": textwrap.dedent("""
            Disease prevalence: 0.1% (1 in 1000).
            Test sensitivity: 99%. Test specificity: 95% (false positive rate 5%).
            You test positive. What is the probability you have the disease?
            Show full Bayes' theorem calculation. Give the answer as a percentage.
        """).strip(),
        "retry_hint": (
            "P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|¬D)P(¬D)]. "
            "= 0.99×0.001 / (0.99×0.001 + 0.05×0.999) = 0.00099/0.05094 ≈ 1.94%."
        ),
    },
    {
        "name": "Rolling sphere on incline (rotational inertia)",
        "type": "text", "check": _check_rolling,
        "prompt": textwrap.dedent("""
            A solid sphere (I = 2MR²/5) rolls without slipping down a 30° incline.
            Derive the linear acceleration using Newton's 2nd law for both translation
            and rotation. Give the symbolic result a=f(g,θ) and the numerical value
            for θ=30°, g=9.8 m/s².
        """).strip(),
        "retry_hint": (
            "a = (5/7)g sinθ — NOT g sinθ (that ignores rolling). "
            "Rotation: fR=(2MR²/5)(a/R) → f=2Ma/5. "
            "Translation: Mg sinθ - 2Ma/5 = Ma → a(7/5)=g sinθ → a=(5/7)g sinθ ≈ 3.5 m/s²."
        ),
    },
    {
        "name": "Monty Hall problem",
        "type": "text", "check": _check_monty,
        "prompt": textwrap.dedent("""
            You are on a game show. There are 3 doors. Behind one is a car; behind
            the other two are goats. You pick door 1. The host (who knows what's
            behind each door) opens door 3, revealing a goat. He then asks:
            "Do you want to switch to door 2?"

            Should you switch? What is the probability of winning if you switch
            vs. if you stay? Show your full reasoning. Many people intuitively
            say it's 50/50 — prove whether that is correct or not.
        """).strip(),
        "retry_hint": (
            "The 50/50 intuition is wrong. "
            "P(car behind door 1 | you picked 1) = 1/3. "
            "P(car behind door 2 or 3 | you picked 1) = 2/3. "
            "The host ALWAYS opens a goat door. So all of that 2/3 probability "
            "collapses onto door 2 after he opens door 3. "
            "P(win by switching) = 2/3."
        ),
    },
    {
        "name": "Carnot efficiency (must use Kelvin!)",
        "type": "text", "check": _check_carnot,
        "prompt": textwrap.dedent("""
            A heat engine operates between a hot reservoir at 600°C and a cold
            reservoir at 27°C. Calculate the maximum (Carnot) efficiency.

            Formula: η_Carnot = 1 − T_cold / T_hot
            where temperatures are in KELVIN.

            Show:
            1. Conversion of both temperatures to Kelvin
            2. The Carnot efficiency as a fraction and as a percentage
            3. If the engine absorbs 10 kJ of heat per cycle, how much useful
               work can it produce at maximum efficiency?
        """).strip(),
        "retry_hint": (
            "CRITICAL ERROR if using Celsius directly. Must convert: "
            "T_hot = 600 + 273.15 = 873.15 K, T_cold = 27 + 273.15 = 300.15 K. "
            "η = 1 − 300/873 = 0.656 = 65.6%. Work = 0.656 × 10 kJ = 6.56 kJ."
        ),
    },
    {
        "name": "Snell's law — critical angle (glass to air)",
        "type": "text", "check": _check_snell,
        "prompt": textwrap.dedent("""
            Light travels from glass (refractive index n₁ = 1.5) into air (n₂ = 1.0).

            1. State the condition for total internal reflection.
            2. Derive the formula for the critical angle θ_c.
            3. Calculate θ_c for this glass-air interface.
            4. What happens to light hitting the interface at 50°? At 35°?
        """).strip(),
        "retry_hint": (
            "At critical angle: refracted ray is at 90°, so Snell's law gives "
            "n₁ sin(θ_c) = n₂ sin(90°) = n₂. "
            "sin(θ_c) = n₂/n₁ = 1.0/1.5 = 2/3. "
            "θ_c = arcsin(2/3) ≈ 41.8°. "
            "At 50° > 41.8°: total internal reflection. At 35° < 41.8°: light refracts into air."
        ),
    },
    {
        "name": "Capacitor energy (½CV²)",
        "type": "text", "check": _check_capacitor,
        "prompt": textwrap.dedent("""
            A capacitor of C = 47 μF is charged to a voltage of V = 230 V.

            1. Write the formula for the energy stored in a capacitor.
            2. Calculate the stored energy in joules.
            3. If the capacitor discharges through a resistor in 0.1 s, what
               is the average power dissipated?

            Show all unit conversions clearly (μF → F).
        """).strip(),
        "retry_hint": (
            "E = ½CV² (NOT CV² — the ½ is essential). "
            "C = 47 × 10⁻⁶ F = 4.7 × 10⁻⁵ F. "
            "E = 0.5 × 4.7 × 10⁻⁵ × 230² = 0.5 × 4.7 × 10⁻⁵ × 52900 = 1.243 J. "
            "Average power = E/t = 1.243/0.1 = 12.43 W."
        ),
    },
    {
        "name": "Perfectly inelastic collision (momentum only)",
        "type": "text", "check": _check_collision,
        "prompt": textwrap.dedent("""
            A 3 kg ball moving at 8 m/s collides head-on with a stationary 5 kg ball.
            They stick together (perfectly inelastic collision).

            Calculate:
            1. The velocity of the combined mass immediately after collision
            2. The kinetic energy before and after the collision
            3. The energy lost in the collision and where it goes

            Make clear which conservation law applies (and why KE is NOT conserved).
        """).strip(),
        "retry_hint": (
            "Use conservation of MOMENTUM only (inelastic = KE not conserved). "
            "p = m₁v₁ = 3×8 = 24 kg·m/s. "
            "v' = p/(m₁+m₂) = 24/8 = 3 m/s. "
            "KE_before = ½×3×8² = 96 J. KE_after = ½×8×3² = 36 J. "
            "Energy lost = 60 J (converted to heat/sound/deformation)."
        ),
    },
    {
        "name": "Twin paradox (special relativity)",
        "type": "text", "check": _check_twin,
        "prompt": textwrap.dedent("""
            Twin A travels at v = 0.6c to a star 3 light-years away (Earth frame)
            and returns at the same speed. Twin B stays on Earth.
            Calculate: Earth-frame trip time, γ, ship clock time, age difference.
        """).strip(),
        "retry_hint": (
            "Earth time: 2×(3/0.6) = 10 yr. "
            "γ = 1/√(1−0.36) = 1/0.8 = 1.25. "
            "Ship time = 10/1.25 = 8 yr. Age diff = 2 yr."
        ),
    },
    {
        "name": "Expected flips to first heads (geometric distribution)",
        "type": "text", "check": _check_geometric,
        "prompt": textwrap.dedent("""
            You flip a fair coin repeatedly until you get the first heads.
            Let X = number of flips needed.

            1. What distribution does X follow? State its PMF P(X=k).
            2. Calculate E[X] — the expected number of flips.
            3. Calculate Var[X] — the variance.
            4. What is P(X > 4) — probability you need more than 4 flips?
            5. Given you've already flipped 3 tails, what is the expected
               number of additional flips needed? Explain why.

            Show all derivations.
        """).strip(),
        "retry_hint": (
            "X ~ Geometric(p=0.5). P(X=k) = (1-p)^(k-1) × p = (0.5)^k. "
            "E[X] = 1/p = 2. Var[X] = (1-p)/p² = 0.5/0.25 = 2. "
            "P(X>4) = (1-p)^4 = (0.5)^4 = 1/16 = 6.25%. "
            "Memoryless property: given 3 tails, still E[additional] = 2."
        ),
    },
    {
        "name": "Doppler effect (approaching source)",
        "type": "text", "check": _check_doppler,
        "prompt": textwrap.dedent("""
            An ambulance siren emits sound at 700 Hz. The ambulance is
            approaching a stationary observer at 30 m/s.
            Speed of sound in air: 340 m/s.

            1. Write the Doppler formula for a moving source.
            2. Calculate the frequency heard by the observer.
            3. After the ambulance passes and is now moving away at the same
               speed, what frequency does the observer hear?
            4. Calculate the ratio of the two frequencies.
        """).strip(),
        "retry_hint": (
            "Source approaching: f' = f × v/(v − v_s) = 700 × 340/(340−30) = 700 × 340/310 ≈ 767.7 Hz. "
            "NOT v+v_s in denominator — that formula is for receding source. "
            "Source receding: f' = 700 × 340/(340+30) = 700 × 340/370 ≈ 643.2 Hz. "
            "Ratio: 767.7/643.2 ≈ 1.19."
        ),
    },
    {
        "name": "Hydrogen Hβ emission line (Rydberg formula)",
        "type": "text", "check": _check_hline,
        "prompt": textwrap.dedent("""
            A hydrogen atom transitions from n=4 to n=2.
            Use: 1/λ = R_H (1/n_f² − 1/n_i²), R_H = 1.097×10⁷ m⁻¹.
            Calculate λ in nm, name the spectral series and this specific line.
        """).strip(),
        "retry_hint": (
            "1/4 − 1/16 = 3/16. 1/λ = 1.097×10⁷ × 3/16 = 2.057×10⁶ m⁻¹. "
            "λ = 4.861×10⁻⁷ m = 486.1 nm. Balmer series, H-beta (Hβ) line."
        ),
    },
    {
        "name": "Electric field between parallel plates",
        "type": "text", "check": _check_efield,
        "prompt": textwrap.dedent("""
            Two parallel conducting plates are separated by d = 5 mm and
            connected to a 120 V battery.

            1. What is the electric field between the plates?
               Give the formula, show unit conversions, give the answer in V/m.
            2. What is the force on a proton (charge e = 1.6×10⁻¹⁹ C) placed
               midway between the plates?
            3. What is the capacitance per unit area (in F/m²)?
               (Use ε₀ = 8.85×10⁻¹² F/m)
        """).strip(),
        "retry_hint": (
            "E = V/d. MUST convert d to metres: 5 mm = 0.005 m. "
            "E = 120/0.005 = 24,000 V/m = 24 kV/m. "
            "F on proton = eE = 1.6×10⁻¹⁹ × 24000 = 3.84×10⁻¹⁵ N. "
            "C/A = ε₀/d = 8.85×10⁻¹²/0.005 = 1.77×10⁻⁹ F/m²."
        ),
    },
    {
        "name": "Buoyancy paradox (steel ball in boat)",
        "type": "text", "check": _check_buoyancy,
        "prompt": textwrap.dedent("""
            A rubber dinghy floats in a bathtub with a heavy steel ball inside it.
            Someone picks up the ball and drops it into the water, where it sinks.
            Does the water level rise, fall, or stay the same?
            Explain rigorously using Archimedes' principle for both cases.
        """).strip(),
        "retry_hint": (
            "FALLS. (a) Ball in floating dinghy: displaces water by WEIGHT of (dinghy+ball). "
            "Steel is ~7.8× denser than water, so weight-displacement >> volume-displacement. "
            "(b) Sunken ball: displaces water only by its VOLUME. "
            "Since ρ_steel >> ρ_water, volume displacement < weight displacement. "
            "Water level falls when the ball moves from dinghy to water."
        ),
    },
    {
        "name": "Birthday paradox (n=23 → ~50%)",
        "type": "text", "check": _check_birthday,
        "prompt": textwrap.dedent("""
            The birthday problem: how many people must be in a room before there
            is at least a 50% chance that two of them share a birthday?

            1. Derive the formula for P(at least 2 people share a birthday | n people).
            2. Calculate this probability step by step for n = 23.
            3. Give the exact percentage to two decimal places for n = 23.
            4. Why is this result so counterintuitive?

            Assume 365 days, birthdays uniformly distributed, no leap years.
        """).strip(),
        "retry_hint": (
            "P(no match with n people) = 365/365 × 364/365 × 363/365 × ... × (365-n+1)/365. "
            "For n=23: multiply out 23 terms. "
            "P(no match) ≈ 0.4927. P(at least one match) = 1 − 0.4927 ≈ 50.73%."
        ),
    },
    {
        "name": "Simpson's paradox (hospital case-mix)",
        "type": "text", "check": _check_simpsons,
        "prompt": textwrap.dedent("""
            Hospital A: 900 mild patients (90% survival) + 100 severe (30%) = 84% overall.
            Hospital B: 100 mild patients (90% survival) + 900 severe (30%) = 36% overall.
            Both hospitals have identical per-category rates. Why does overall differ so dramatically?
            Name the phenomenon and explain the practical danger.
        """).strip(),
        "retry_hint": (
            "Simpson's Paradox. Hospital A treats mostly easy cases (90% mild), "
            "Hospital B mostly hard ones (90% severe). The 'overall' rate is a weighted average "
            "with very different weights. Comparing overall rates is meaningless here — "
            "you must stratify by case severity."
        ),
    },
]

assert len(TASKS) == 30, f"Expected 30 tasks, got {len(TASKS)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval():
    print("\n" + "═" * 72)
    print("  eval_haiku_intensive — 30 tasks — Haiku — brain adapts in real time")
    print("  Code (15): LRU, slide-win-max, edit-dist, coin2, decode, trap,")
    print("             histogram, word-break, jump, regex, burst, UF, courses, trie, nqueens")
    print("  Text (15): Bayes, sphere, Monty Hall, Carnot, Snell, capacitor,")
    print("             collision, twin, geometric, Doppler, Hβ, E-field,")
    print("             buoyancy, birthday, Simpson's")
    print("═" * 72)

    embedder   = build_embedder()
    brain      = BrainAgent(embedder, threshold=0.30, k=5)
    code_agent = tool_agent(["python_exec"], max_turns=MAX_TURNS,
                            model=HAIKU, max_tokens=MAX_TOK)
    code_agent.monitor = brain
    viz = BrainViz()

    results:     list[dict] = []
    fire_counts: list[int]  = []

    for i, task in enumerate(TASKS):
        n     = i + 1
        ttype = task["type"]
        print(f"\n  {n:>2}/{len(TASKS)} [{ttype:4}] {task['name']}"
              f"  (brain: {brain.n_stored} stored)")

        brain.set_task(i, probe_fn=task.get("probe") if ttype == "code" else None)
        brain.reset()

        t0 = time.time()

        if ttype == "code":
            result     = code_agent(task["prompt"])
            trace, tok = result if isinstance(result, tuple) else (str(result), 0)
            fires      = brain._code_interventions

            first_code       = _first_exec_code(trace, task.get("want_fn"))
            first_p, first_d = _check_code(first_code, task["check"])

            final_code    = _extract_code(trace, task.get("want_fn"))
            passed, det   = _check_code(final_code, task["check"])

        else:
            code_agent.monitor = None
            result     = _anthropic_call(HAIKU, task["prompt"], max_tokens=MAX_TOK)
            trace, tok = result if isinstance(result, tuple) else (str(result), 0)
            fires      = 0

            first_p, first_d = _check_text(trace, task["check"])
            passed, det      = first_p, first_d

            if not passed:
                hint     = task.get("retry_hint", "")
                feedback = (
                    "\n\nYour previous answer was incomplete or incorrect.\n"
                    + (f"Correction: {hint}\n\n" if hint else "")
                    + "Redo with the correct approach shown above."
                )
                r2, tok2 = _anthropic_call(HAIKU, task["prompt"] + feedback,
                                           max_tokens=MAX_TOK)
                tok      += tok2
                p2, _     = _check_text(r2, task["check"])
                if p2:
                    passed = True
                    det    = "corrected on retry"
            code_agent.monitor = brain

        elapsed     = time.time() - t0
        brain_fixed = (not first_p) and passed

        fire_tag  = f"  [⚡×{fires}]"   if fires       else ""
        fix_tag   = "  [↑ brain fixed]"  if brain_fixed  else ""
        status    = "PASS" if passed else "FAIL"
        base_tag  = (f"  (baseline: {'PASS' if first_p else 'FAIL'})"
                     if first_p != passed else "")

        print(f"       {status}  {tok:>7,} tok  {elapsed:.0f}s"
              f"{fire_tag}{fix_tag}{base_tag}"
              + (f"  {det[:55]}" if det and not passed else ""))

        brain.store(trace, int(first_p), metadata=first_d if not first_p else "")
        if ttype == "code" and first_code:
            brain.store_code(first_code, int(first_p),
                             metadata=first_d if not first_p else "")

        results.append({
            "task":         n,
            "name":         task["name"],
            "type":         ttype,
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
        viz.save(OUT / "brain_haiku_intensive.png")

    _report(results, fire_counts)
    with open(OUT / "haiku_intensive_run.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved eval/results/brain_haiku_intensive.png + haiku_intensive_run.json")
    return results


def _report(results, fire_counts):
    n       = len(results)
    n_base  = sum(1 for r in results if r["first_passed"])
    n_final = sum(1 for r in results if r["passed"])
    n_fired = sum(1 for c in fire_counts if c > 0)
    helped  = [r for r in results if r["brain_helped"]]

    print("\n" + "═" * 72)
    print("  RESULTS  [Haiku — 30 tasks]")
    print("─" * 72)
    print(f"  Without brain (first attempt) : {n_base}/{n}  ({n_base/n:.0%})")
    print(f"  With brain (after intervention): {n_final}/{n}  ({n_final/n:.0%})")
    delta = n_final - n_base
    print(f"  Brain contribution             : +{delta} task{'s' if delta != 1 else ''}"
          f"  (brain fired on {n_fired}/{n} tasks)")
    if helped:
        print(f"\n  Tasks brain fixed:")
        for r in helped:
            how = f"probe ⚡×{r['fires']}" if r["fires"] else "retry with hint"
            print(f"    + [{r['type']:4}] {r['name']}  ({how})")
    print()
    for ttype in ("code", "text"):
        rs = [r for r in results if r["type"] == ttype]
        if not rs: continue
        b  = sum(1 for r in rs if r["first_passed"])
        f  = sum(1 for r in rs if r["passed"])
        nf = sum(1 for r in rs if r["fires"] > 0)
        print(f"  {ttype:4}  baseline {b}/{len(rs)}  →  final {f}/{len(rs)}"
              f"  (brain fired on {nf}/{len(rs)})")
    print("─" * 72)
    print(f"  Tokens: {sum(r['tokens'] for r in results):,}  |  "
          f"Time: {sum(r['elapsed'] for r in results):.0f}s")
    print("═" * 72)


if __name__ == "__main__":
    run_eval()
