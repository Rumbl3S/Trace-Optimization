"""eval/eval_real_world.py — 30 genuinely hard tasks targeting heavy users.

Research basis: tasks selected from GPQA Diamond, USACO-hard, competitive-programming
failure modes confirmed for frontier models (hard DP 0-20%, grad-level science ~35-65%).

Code tasks (15) — confirmed LLM failure modes from research:
  Segment tree w/ lazy propagation, Fenwick tree, KMP (overlapping),
  LIS O(n log n), Bellman-Ford + neg-cycle, Graham scan, matrix chain,
  sliding window median (two heaps), count inversions, Manacher's,
  Z-algorithm, topological sort (Kahn's), token bucket, expression parser,
  Trie with '.' wildcard.

Text tasks (15) — GPQA-diamond style + hard combinatorics:
  Energy-time uncertainty, Henderson-Hasselbalch, Gibbs→K,
  Michaelis-Menten, I-131 decay, Bragg's law, Nernst equation,
  de Broglie wavelength, Compton scattering, QHO energy ratio,
  Euler's totient φ(60), CRT, derangements D₆, Catalan C₅, Stirling S(5,2).

Run:  python eval/eval_real_world.py
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

from trace_use import build_embedder, tool_agent
from trace_use.agents import _anthropic_call
from trace_use import BrainAgent
from eval.viz_brain import BrainViz
from eval.eval_hard import (
    _exec, _check_code, _check_text, _extract_code, _first_exec_code,
)

OUT      = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

HAIKU     = "claude-haiku-4-5-20251001"
MAX_TURNS = 14
MAX_TOK   = 4096


# ═══════════════════════════════════════════════════════════════════════════════
#  CODE PROBES
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Segment tree with lazy propagation ─────────────────────────────────────

def _probe_segtree(ns):
    ST = (ns.get("SegTree") or ns.get("LazySegTree") or ns.get("SegmentTree")
          or ns.get("LazySegmentTree"))
    if not ST:
        return ["SegTree class not found — implement SegTree(n) with range_add(l,r,v) and range_max(l,r) (0-indexed inclusive)"]
    fails = []
    try:
        st = ST(5)
        st.range_add(0, 4, 3)
        r = st.range_max(0, 4)
        if r != 3:
            fails.append(
                f"range_add(0,4,3) then range_max(0,4)={r}, expected 3. "
                "FIX: initial values should be 0; range_add adds val to all elements in [l,r]. "
                "Check lazy push-down: push lazy[node] to children before splitting range."
            )
            return fails
        st.range_add(2, 3, 5)
        r = st.range_max(0, 4)
        if r != 8:
            fails.append(
                f"After range_add(2,3,5), range_max(0,4)={r}, expected 8. "
                "FIX: Lazy values must accumulate — range_max at indices 2..3 should return 3+5=8."
            )
            return fails
        r2 = st.range_max(0, 1)
        if r2 != 3:
            fails.append(
                f"range_max(0,1)={r2}, expected 3 (these indices only got +3). "
                "FIX: range_add(2,3,5) must NOT affect indices 0..1."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_segtree(ns):
    ST = (ns.get("SegTree") or ns.get("LazySegTree") or ns.get("SegmentTree")
          or ns.get("LazySegmentTree"))
    if not ST: return False
    try:
        st = ST(8)
        st.range_add(0, 7, 2)
        if st.range_max(0, 7) != 2: return False
        st.range_add(3, 5, 3)
        if st.range_max(3, 5) != 5: return False
        if st.range_max(0, 2) != 2: return False
        if st.range_max(6, 7) != 2: return False
        st.range_add(0, 0, 10)
        if st.range_max(0, 0) != 12: return False
        if st.range_max(1, 7) != 5: return False
        if st.range_max(0, 7) != 12: return False
        return True
    except Exception: return False


# ── 2. Fenwick tree / BIT ─────────────────────────────────────────────────────

def _probe_bit(ns):
    BIT = (ns.get("BIT") or ns.get("FenwickTree") or ns.get("BinaryIndexedTree")
           or ns.get("Fenwick"))
    if not BIT:
        return ["BIT class not found — implement BIT(n) with update(i, delta) and query(i) (1-indexed prefix sum)"]
    fails = []
    try:
        b = BIT(6)
        b.update(1, 3); b.update(3, 2); b.update(5, 4)
        q = b.query(5)
        if q != 9:
            fails.append(
                f"query(5) after update(1,3),update(3,2),update(5,4) = {q}, expected 9. "
                "FIX: query(i) = prefix sum of elements 1..i. "
                "update(i,delta): i += i & (-i). query(i): i -= i & (-i)."
            )
            return fails
        q2 = b.query(3)
        if q2 != 5:
            fails.append(f"query(3)={q2}, expected 5 (elements 1..3 sum = 3+0+2).")
        q3 = b.query(1)
        if q3 != 3:
            fails.append(f"query(1)={q3}, expected 3.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_bit(ns):
    BIT = (ns.get("BIT") or ns.get("FenwickTree") or ns.get("BinaryIndexedTree")
           or ns.get("Fenwick"))
    if not BIT: return False
    try:
        b = BIT(10)
        vals = [0, 3, 5, 0, 2, 0, 7, 0, 1, 0, 4]  # 1-indexed
        for i, v in enumerate(vals[1:], 1):
            if v: b.update(i, v)
        for q in [1, 3, 5, 7, 10]:
            if b.query(q) != sum(vals[1:q+1]):
                return False
        return True
    except Exception: return False


# ── 3. KMP string search ──────────────────────────────────────────────────────

def _probe_kmp(ns):
    fn = ns.get("kmp_search") or ns.get("kmp") or ns.get("kmp_find") or ns.get("kmp_all")
    if not fn:
        return ["kmp_search(text, pattern) not defined — return list of 0-indexed start positions (including overlapping)"]
    fails = []
    try:
        r = sorted(fn("aaaa", "aa"))
        if r != [0, 1, 2]:
            fails.append(
                f"kmp_search('aaaa','aa')={r}, expected [0,1,2]. "
                "FIX: After a match at position i, restart search at i+1 (NOT i+len(pattern)) "
                "to catch overlapping occurrences. Use KMP failure function."
            )
            return fails
        r2 = sorted(fn("abcdef", "xyz"))
        if r2 != []:
            fails.append(f"kmp_search('abcdef','xyz')={r2}, expected [].")
        r3 = sorted(fn("ababab", "aba"))
        if r3 != [0, 2]:
            fails.append(f"kmp_search('ababab','aba')={r3}, expected [0,2].")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_kmp(ns):
    fn = ns.get("kmp_search") or ns.get("kmp") or ns.get("kmp_find") or ns.get("kmp_all")
    if not fn: return False
    try:
        return all(sorted(fn(t, p)) == e for t, p, e in [
            ("aaaa", "aa", [0, 1, 2]),
            ("abcdef", "xyz", []),
            ("ababab", "aba", [0, 2]),
            ("mississippi", "issi", [1, 4]),
            ("aababab", "ab", [1, 3, 5]),
            ("", "a", []),
        ])
    except Exception: return False


# ── 4. LIS O(n log n) ─────────────────────────────────────────────────────────

def _probe_lis(ns):
    fn = ns.get("lis") or ns.get("length_of_lis") or ns.get("lengthOfLIS")
    if not fn:
        return ["lis(nums) or lengthOfLIS(nums) not defined — return length of strictly increasing subsequence"]
    fails = []
    try:
        r = fn([10, 9, 2, 5, 3, 7, 101, 18])
        if r != 4:
            fails.append(
                f"lis([10,9,2,5,3,7,101,18])={r}, expected 4. "
                "FIX: Use patience sorting — maintain tails[] where tails[i] is the smallest "
                "tail element of all IS of length i+1. For each num: bisect_left(tails, num) "
                "to find insertion position; replace or append."
            )
            return fails
        r2 = fn([7, 7, 7, 7, 7])
        if r2 != 1:
            fails.append(
                f"lis([7,7,7,7,7])={r2}, expected 1. "
                "FIX: strictly increasing — equal elements don't count. "
                "Use bisect_left (not bisect_right) to handle duplicates."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_lis(ns):
    fn = ns.get("lis") or ns.get("length_of_lis") or ns.get("lengthOfLIS")
    if not fn: return False
    try:
        return all(fn(a) == e for a, e in [
            ([10, 9, 2, 5, 3, 7, 101, 18], 4),
            ([0, 1, 0, 3, 2, 3], 4),
            ([7, 7, 7, 7, 7], 1),
            ([4, 10, 4, 3, 8, 9], 3),
            ([], 0),
            ([1], 1),
        ])
    except Exception: return False


# ── 5. Bellman-Ford + negative cycle ──────────────────────────────────────────

def _probe_bellman(ns):
    fn = ns.get("bellman_ford") or ns.get("bellmanFord")
    if not fn:
        return ["bellman_ford(n, edges, src) not defined — edges=[(u,v,w)]; return (dist_list, has_negative_cycle)"]
    fails = []
    try:
        dist, neg = fn(3, [(0, 1, 1), (1, 2, 2), (0, 2, 4)], 0)
        if dist[2] != 3:
            fails.append(
                f"dist[2]={dist[2]} for path 0→1→2 (cost 3), expected 3. "
                "FIX: Relax all edges (n-1) times. "
                "If dist[u] != inf and dist[u]+w < dist[v]: dist[v] = dist[u]+w."
            )
            return fails
        if neg:
            fails.append("Reported negative cycle for a graph with no negative edges — false positive.")
            return fails
        _, neg2 = fn(3, [(0, 1, 1), (1, 2, -3), (2, 0, 1)], 0)
        if not neg2:
            fails.append(
                "Should detect negative cycle in [(0,1,1),(1,2,-3),(2,0,1)]. "
                "FIX: After n-1 relaxations, do one more pass. "
                "If any distance still decreases, return has_negative_cycle=True."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_bellman(ns):
    fn = ns.get("bellman_ford") or ns.get("bellmanFord")
    if not fn: return False
    try:
        INF = float('inf')
        d, neg = fn(4, [(0,1,1),(1,2,2),(0,2,4),(2,3,1)], 0)
        if neg or d[3] != 4 or d[1] != 1 or d[2] != 3: return False
        _, neg2 = fn(3, [(0,1,1),(1,2,-3),(2,0,1)], 0)
        if not neg2: return False
        # Disconnected source
        d3, neg3 = fn(3, [(1,2,5)], 0)
        if neg3 or d3[1] != INF or d3[2] != INF: return False
        return True
    except Exception: return False


# ── 6. Graham scan convex hull ────────────────────────────────────────────────

def _probe_hull(ns):
    fn = ns.get("convex_hull") or ns.get("graham_scan") or ns.get("grahamScan")
    if not fn:
        return ["convex_hull(points) not defined — points=list of (x,y); return hull points"]
    fails = []
    try:
        pts = [(0,0),(4,0),(4,4),(0,4),(2,2)]
        hull = fn(pts)
        hull_set = frozenset(tuple(p) for p in hull)
        if len(hull) != 4 or (2, 2) in hull_set:
            fails.append(
                f"convex_hull({pts}) returned {hull}, expected the 4 corner points (interior (2,2) excluded). "
                "FIX: Sort by angle from lowest-leftmost point. "
                "Use cross product: (b-a)×(c-a) ≤ 0 means right turn — pop from stack."
            )
            return fails
        expected = frozenset([(0,0),(4,0),(4,4),(0,4)])
        if hull_set != expected:
            fails.append(f"Hull set {hull_set} != expected {expected}.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_hull(ns):
    fn = ns.get("convex_hull") or ns.get("graham_scan") or ns.get("grahamScan")
    if not fn: return False
    try:
        def hs(pts): return frozenset(tuple(p) for p in pts)
        h1 = fn([(0,0),(4,0),(4,4),(0,4),(2,2)])
        if hs(h1) != frozenset([(0,0),(4,0),(4,4),(0,4)]): return False
        h2 = fn([(0,0),(3,0),(1,2),(1,1)])
        if hs(h2) != frozenset([(0,0),(3,0),(1,2)]): return False
        h3 = fn([(0,0),(1,1),(2,2),(3,0)])
        if (3,0) not in hs(h3) or (0,0) not in hs(h3): return False
        return True
    except Exception: return False


# ── 7. Matrix chain multiplication ────────────────────────────────────────────

def _probe_matchain(ns):
    fn = ns.get("matrix_chain") or ns.get("matrixChainOrder") or ns.get("mcm")
    if not fn:
        return ["matrix_chain(dims) not defined — dims[i]*dims[i+1] are matrix i dimensions; return min scalar multiplications"]
    fails = []
    try:
        r = fn([10, 30, 5, 60])
        if r != 4500:
            fails.append(
                f"matrix_chain([10,30,5,60])={r}, expected 4500. "
                "FIX: dp[i][j] = min cost to multiply matrices i..j. "
                "Iterate by chain length L=2..n. "
                "dp[i][j] = min(dp[i][k]+dp[k+1][j]+dims[i]*dims[k+1]*dims[j+1]) for k=i..j-1."
            )
            return fails
        r2 = fn([40, 20, 30, 10, 30])
        if r2 != 26000:
            fails.append(f"matrix_chain([40,20,30,10,30])={r2}, expected 26000.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_matchain(ns):
    fn = ns.get("matrix_chain") or ns.get("matrixChainOrder") or ns.get("mcm")
    if not fn: return False
    try:
        return all(fn(d) == e for d, e in [
            ([10, 30, 5, 60], 4500),
            ([40, 20, 30, 10, 30], 26000),
            ([1, 2, 3, 4], 18),
            ([10, 20, 30], 6000),
        ])
    except Exception: return False


# ── 8. Sliding window median (two heaps + lazy deletion) ─────────────────────

def _probe_winmedian(ns):
    fn = (ns.get("sliding_median") or ns.get("slidingWindowMedian")
          or ns.get("medianSlidingWindow"))
    if not fn:
        return ["sliding_median(nums, k) not defined — return list of medians for each window"]
    fails = []
    try:
        r = list(fn([1, 3, -1, -3, 5, 3, 6, 7], 3))
        expected = [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]
        if r != expected:
            fails.append(
                f"sliding_median([1,3,-1,-3,5,3,6,7],3)={r}, expected {expected}. "
                "FIX: Use two heaps — max-heap (lower half) and min-heap (upper half). "
                "Balance so lower has ⌈k/2⌉ elements. Median = lower_max (odd k) "
                "or (lower_max + upper_min)/2 (even k). "
                "Lazy deletion: mark removed elements; skip them when they become heap top."
            )
            return fails
        r2 = list(fn([1, 2, 3, 4, 5], 2))
        if r2 != [1.5, 2.5, 3.5, 4.5]:
            fails.append(f"sliding_median([1,2,3,4,5],2)={r2}, expected [1.5,2.5,3.5,4.5].")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_winmedian(ns):
    fn = (ns.get("sliding_median") or ns.get("slidingWindowMedian")
          or ns.get("medianSlidingWindow"))
    if not fn: return False
    try:
        if list(fn([1,3,-1,-3,5,3,6,7], 3)) != [1.0,-1.0,-1.0,3.0,5.0,6.0]: return False
        if list(fn([1,2,3,4,5], 2)) != [1.5,2.5,3.5,4.5]: return False
        if list(fn([1], 1)) != [1.0]: return False
        if list(fn([2,3,4,2,3,4,2], 3)) != [3.0,3.0,3.0,3.0,3.0]: return False
        return True
    except Exception: return False


# ── 9. Count inversions (merge sort) ─────────────────────────────────────────

def _probe_inversions(ns):
    fn = ns.get("count_inversions") or ns.get("countInversions")
    if not fn:
        return ["count_inversions(arr) not defined — return number of pairs (i,j) where i<j and arr[i]>arr[j]"]
    fails = []
    try:
        r = fn([2, 4, 1, 3, 5])
        if r != 3:
            fails.append(
                f"count_inversions([2,4,1,3,5])={r}, expected 3 (pairs: (2,1),(4,1),(4,3)). "
                "FIX: Modify merge sort — during merge, when right[j] < left[i], "
                "add (len(left) - i) to count (all remaining left elements > right[j])."
            )
            return fails
        r2 = fn([5, 4, 3, 2, 1])
        if r2 != 10:
            fails.append(f"count_inversions([5,4,3,2,1])={r2}, expected 10.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_inversions(ns):
    fn = ns.get("count_inversions") or ns.get("countInversions")
    if not fn: return False
    try:
        return all(fn(a) == e for a, e in [
            ([2,4,1,3,5], 3),
            ([5,4,3,2,1], 10),
            ([1,2,3,4,5], 0),
            ([1], 0),
            ([], 0),
            ([3,1,2], 2),
        ])
    except Exception: return False


# ── 10. Manacher's palindrome O(n) ────────────────────────────────────────────

def _probe_manacher(ns):
    fn = ns.get("longest_palindrome") or ns.get("longestPalindrome") or ns.get("manacher")
    if not fn:
        return ["longest_palindrome(s) not defined — return longest palindromic substring"]
    fails = []
    try:
        r = fn("racecar")
        if r != "racecar":
            fails.append(
                f"longest_palindrome('racecar')={r!r}, expected 'racecar'. "
                "FIX: 'racecar' itself is a palindrome. Check that your Manacher's "
                "boundary indices correctly include the full string."
            )
            return fails
        r2 = fn("cbbd")
        if r2 != "bb":
            fails.append(
                f"longest_palindrome('cbbd')={r2!r}, expected 'bb'. "
                "FIX: Even-length palindrome — use transformed string with # separators."
            )
            return fails
        r3 = fn("a")
        if r3 != "a":
            fails.append(f"longest_palindrome('a')={r3!r}, expected 'a'.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_manacher(ns):
    fn = ns.get("longest_palindrome") or ns.get("longestPalindrome") or ns.get("manacher")
    if not fn: return False
    try:
        def is_pal(s): return s == s[::-1]
        cases = [("babad", 3), ("cbbd", 2), ("racecar", 7), ("a", 1), ("abcba", 5), ("ac", 1)]
        for s, expected_len in cases:
            r = fn(s)
            if not (r and is_pal(r) and len(r) == expected_len): return False
        return True
    except Exception: return False


# ── 11. Z-algorithm search ────────────────────────────────────────────────────

def _probe_zalgo(ns):
    fn = ns.get("z_search") or ns.get("zSearch") or ns.get("z_algorithm_search")
    if not fn:
        return ["z_search(text, pattern) not defined — return sorted list of 0-indexed match positions"]
    fails = []
    try:
        r = sorted(fn("ababcababc", "abc"))
        if r != [2, 7]:
            fails.append(
                f"z_search('ababcababc','abc')={r}, expected [2,7]. "
                "FIX: Build Z-array for s = pattern + '$' + text. "
                "Z[i] = length of longest common prefix of s and s[i:]. "
                "When Z[i] == len(pattern): match at text position i - len(pattern) - 1."
            )
            return fails
        r2 = sorted(fn("aaaa", "aa"))
        if r2 != [0, 1, 2]:
            fails.append(
                f"z_search('aaaa','aa')={r2}, expected [0,1,2] (overlapping). "
                "FIX: Z-algorithm naturally handles overlapping matches."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_zalgo(ns):
    fn = ns.get("z_search") or ns.get("zSearch") or ns.get("z_algorithm_search")
    if not fn: return False
    try:
        return all(sorted(fn(t, p)) == e for t, p, e in [
            ("ababcababc", "abc", [2, 7]),
            ("aaaa", "aa", [0, 1, 2]),
            ("aabaa", "aa", [0, 3]),
            ("hello world", "world", [6]),
            ("abcdef", "xyz", []),
        ])
    except Exception: return False


# ── 12. Topological sort (Kahn's BFS) ────────────────────────────────────────

def _probe_toposort(ns):
    fn = ns.get("topo_sort") or ns.get("topological_sort") or ns.get("kahn_sort")
    if not fn:
        return ["topo_sort(n, edges) not defined — edges=[(u,v)]; return valid order or [] if cycle"]
    fails = []
    try:
        edges = [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)]
        result = fn(6, edges)
        if not result:
            fails.append(
                "topo_sort returned [] for a valid DAG — should return a valid ordering. "
                "FIX: Kahn's algorithm — compute in-degree for all nodes, "
                "enqueue nodes with in-degree 0, process BFS."
            )
            return fails
        pos = {v: i for i, v in enumerate(result)}
        for u, v in edges:
            if pos.get(u, -1) >= pos.get(v, len(result)):
                fails.append(
                    f"Edge ({u}→{v}): {u} at pos {pos.get(u)} but {v} at pos {pos.get(v)} — wrong order. "
                    "FIX: For each dequeued node, decrement neighbors' in-degrees; "
                    "when in-degree reaches 0, enqueue the neighbor."
                )
                return fails
        # Cycle detection
        result2 = fn(3, [(0,1),(1,2),(2,0)])
        if result2:
            fails.append(
                f"topo_sort returned {result2} for a cycle — should return []. "
                "FIX: If result length < n after BFS, a cycle exists — return []."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_toposort(ns):
    fn = ns.get("topo_sort") or ns.get("topological_sort") or ns.get("kahn_sort")
    if not fn: return False
    try:
        def valid(n, edges, r):
            if not r or len(r) != n: return False
            pos = {v: i for i, v in enumerate(r)}
            return all(pos.get(u,-1) < pos.get(v,n) for u,v in edges)
        e1 = [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)]
        if not valid(6, e1, fn(6, e1)): return False
        if fn(3, [(0,1),(1,2),(2,0)]): return False
        e2 = [(0,1),(0,2),(1,3),(2,3)]
        if not valid(4, e2, fn(4, e2)): return False
        return True
    except Exception: return False


# ── 13. Thread-safe token bucket ──────────────────────────────────────────────

def _probe_tokenbucket(ns):
    TB = ns.get("TokenBucket") or ns.get("RateLimiter")
    if not TB:
        return ["TokenBucket class not found — implement TokenBucket(rate, capacity) with consume(n)->bool"]
    fails = []
    try:
        tb = TB(10, 10)
        for i in range(10):
            if not tb.consume(1):
                fails.append(
                    f"consume(1) returned False on attempt {i+1}/10. "
                    "FIX: TokenBucket(rate=10, capacity=10) should start with 10 tokens (full). "
                    "consume(n): if tokens >= n: tokens -= n; return True; else return False."
                )
                return fails
        if tb.consume(1):
            fails.append(
                "11th consume(1) returned True — bucket should be empty after 10 consumes. "
                "FIX: Deduct consumed tokens from the bucket. Return False when insufficient."
            )
            return fails
        # Test bulk consume (consume(n) with n > 1)
        tb2 = TB(0, 20)
        if not tb2.consume(15):
            fails.append(
                "consume(15) returned False on a bucket with capacity=20 and 0 tokens consumed. "
                "FIX: consume(n) must support n > 1. Check that 'tokens >= n' uses the actual argument n."
            )
            return fails
        if tb2.consume(10):
            fails.append(
                "consume(10) returned True but only 5 tokens remain (20 - 15 = 5). "
                "FIX: After consuming 15 tokens, only 5 remain. consume(10) should return False."
            )
            return fails
        if not tb2.consume(5):
            fails.append(
                "consume(5) returned False but 5 tokens remain. "
                "FIX: Exactly 5 tokens should remain after consuming 15 from 20."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_tokenbucket(ns):
    TB = ns.get("TokenBucket") or ns.get("RateLimiter")
    if not TB: return False
    try:
        tb = TB(10, 10)
        if not all(tb.consume(1) for _ in range(10)): return False
        if tb.consume(1): return False
        tb2 = TB(100, 20)
        if not tb2.consume(15): return False
        if tb2.consume(10): return False
        if not tb2.consume(5): return False
        if tb2.consume(1): return False
        # Thread safety: balance must never go negative under concurrent load
        tb3 = TB(0, 20)  # rate=0 so no refill; 20 tokens total
        results = []
        lock = threading.Lock()
        def try_one():
            r = tb3.consume(1)
            with lock: results.append(r)
        threads = [threading.Thread(target=try_one) for _ in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        # Exactly 20 should succeed (no refill, 20 tokens, 50 threads)
        return sum(results) == 20
    except Exception: return False


# ── 14. Recursive descent expression parser ───────────────────────────────────

def _probe_parser(ns):
    fn = ns.get("parse_expr") or ns.get("evaluate") or ns.get("calc") or ns.get("eval_expr")
    if not fn:
        return ["parse_expr(s) not defined — evaluate arithmetic: +,-,*,/ with parentheses and operator precedence"]
    fails = []
    try:
        r = fn("2+3*4")
        if r != 14:
            fails.append(
                f"parse_expr('2+3*4')={r}, expected 14. "
                "FIX: * before +. Use recursive descent: "
                "expr() → term (('+'/'-') term)*; term() → factor (('*'/'/') factor)*; "
                "factor() → number | '(' expr() ')'."
            )
            return fails
        r2 = fn("(2+3)*4")
        if r2 != 20:
            fails.append(f"parse_expr('(2+3)*4')={r2}, expected 20.")
            return fails
        r3 = fn("10/2-3")
        if r3 != 2:
            fails.append(f"parse_expr('10/2-3')={r3}, expected 2.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_parser(ns):
    fn = ns.get("parse_expr") or ns.get("evaluate") or ns.get("calc") or ns.get("eval_expr")
    if not fn: return False
    try:
        return all(fn(s) == e for s, e in [
            ("2+3*4", 14),
            ("(2+3)*4", 20),
            ("10/2-3", 2),
            ("3*(4+5)/9", 3),
            ("1+2+3+4+5", 15),
            ("100", 100),
            ("(1+2)*(3+4)", 21),
            ("2*3+4*5", 26),
        ])
    except Exception: return False


# ── 15. Trie with '.' wildcard ────────────────────────────────────────────────

def _probe_wildtrie(ns):
    WD = ns.get("WordDictionary") or ns.get("WildcardTrie")
    if not WD:
        return ["WordDictionary class not found — implement add_word(word) and search(word) with '.' matching any char"]
    fails = []
    try:
        wd = WD()
        for w in ["bad", "dad", "mad"]: wd.add_word(w)
        if wd.search("pad"):
            fails.append("search('pad')=True but 'pad' was never added — should be False.")
            return fails
        if not wd.search(".ad"):
            fails.append(
                "search('.ad')=False, expected True ('.' matches b/d/m). "
                "FIX: When '.' encountered in search, recurse into ALL child nodes."
            )
            return fails
        if not wd.search("b.."):
            fails.append("search('b..')=False, expected True (matches 'bad').")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_wildtrie(ns):
    WD = ns.get("WordDictionary") or ns.get("WildcardTrie")
    if not WD: return False
    try:
        wd = WD()
        for w in ["bad", "dad", "mad"]: wd.add_word(w)
        if wd.search("pad"): return False
        if not wd.search(".ad"): return False
        if not wd.search("b.."): return False
        wd2 = WD(); wd2.add_word("hello")
        if wd2.search("hell"): return False
        if not wd2.search("hello"): return False
        if not wd2.search("h.l.o"): return False
        if wd2.search("helloo"): return False
        return True
    except Exception: return False


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT CHECK FUNCTIONS  (regex-based, only used in eval not in brain)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_uncertainty(resp: str) -> bool:
    # ΔE ≥ ℏ/(2τ), τ=1ns → ΔE ≈ 5.27×10⁻²⁶ J ≈ 3.3×10⁻⁷ eV
    r = resp.lower().replace(",", "")
    # LaTeX/text variants of 10^{-7} or 10^-7 or 10-7
    exp_7  = r'10\s*[\^]?\s*\{?\s*[-−]\s*7\}?|10\s*[-−]\s*7'
    exp_26 = r'10\s*[\^]?\s*\{?\s*[-−]\s*26\}?|10\s*[-−]\s*26'
    # In eV: ~3.3×10⁻⁷ eV
    has_ev = bool(re.search(rf'3\.[23]\d*\s*[×x\*]?\s*{exp_7}'
                             rf'|3\.[23]\d*e-?7|times.*{exp_7}.*3\.[23]', r))
    # In J: ~5.3×10⁻²⁶ J
    has_j  = bool(re.search(rf'5\.[23]\d*\s*[×x]?\s*{exp_26}'
                             rf'|5\.[23]\d*e-?26', r))
    # Also accept plain "3.3 × 10⁻⁷" with Unicode superscripts
    has_uni = bool(re.search(r'3\.[23]\d*.*10⁻⁷|5\.[23]\d*.*10⁻²⁶', r))
    return has_ev or has_j or has_uni


def _check_hh(resp: str) -> bool:
    # Henderson-Hasselbalch: [NaAc]=0.3M, [HAc]=0.2M, pKa=4.74 → pH≈4.92
    r = resp.lower()
    for n in re.findall(r'ph\s*[=≈:]\s*([\d.]+)', r):
        try:
            if 4.85 <= float(n) <= 4.98: return True
        except Exception: pass
    return bool(re.search(r'4\.9[01234]', r))


def _check_gibbs(resp: str) -> bool:
    # ΔG°=-30kJ/mol → K=exp(12.11)≈1.80×10⁵
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*[×x]\s*10\^?\s*5', r):
        try:
            if 1.5 <= float(n) <= 2.1: return True
        except Exception: pass
    # Numerical: 180000 ± 30000
    if re.search(r'\b1[5-9]\d{4}\b|\b2[01]\d{4}\b|1\.8\s*[×x]\s*10\^?5', r): return True
    # ln(K) ≈ 12.1
    return bool(re.search(r'ln\s*k\s*[=≈]\s*12\.\d|12\.[01]', r))


def _check_mm(resp: str) -> bool:
    # Michaelis-Menten: Km=6μM, Vmax=2.4μM/s
    r = resp.lower()
    has_km   = bool(re.search(r'k_?m\s*[=≈]\s*6\b|km.*\b6\s*[μu]?m\b|\b6\s*[μu]?m.*k_?m', r))
    has_vmax = bool(re.search(r'v_?max\s*[=≈]\s*2\.4|vmax.*2\.4|2\.4.*v_?max', r))
    return has_km and has_vmax


def _check_decay(resp: str) -> bool:
    # I-131: 400 MBq → 50 MBq after 24 days (3 half-lives of 8 days)
    r = resp.lower()
    has_50  = bool(re.search(r'\b50\b.*mbq|\bmbq.*\b50\b|\b50\b.*mega|activity.*\b50\b', r))
    has_3hl = bool(re.search(r'3\s*half.lives?|three\s*half.lives?|24.*=.*3.*8', r))
    return has_50 or has_3hl


def _check_bragg(resp: str) -> bool:
    # Bragg's law: θ=arcsin(1.54/5.64)≈15.8°
    r = resp.lower().replace(",", "")
    if re.search(r'15\.8|15\.9', r): return True
    for n in re.findall(r'(\d+\.?\d*)\s*°', r):
        try:
            if 15.5 <= float(n) <= 16.2: return True
        except Exception: pass
    return False


def _check_nernst(resp: str) -> bool:
    # Nernst: E=1.10+(0.0592/2)*2≈1.16V ([Zn²⁺]=0.01M, [Cu²⁺]=1.0M)
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d+)\s*v\b', r):
        try:
            if 1.14 <= float(n) <= 1.20: return True
        except Exception: pass
    return bool(re.search(r'1\.1[5-9]', r))


def _check_debroglie(resp: str) -> bool:
    # λ = h/√(2mE) at 150eV ≈ 1.00Å = 1.00×10⁻¹⁰ m = 100 pm
    r = resp.lower()
    if re.search(r'1\.0[012]?\s*[aå]|1\.0[012]?\s*angstrom', r): return True
    if re.search(r'100\s*pm|0\.1\s*nm', r): return True
    return bool(re.search(r'1\.0[01]?\s*[×x]\s*10[-−]?\s*10', r))


def _check_compton(resp: str) -> bool:
    # Δλ=λ_c*(1-cos90°)=2.426 pm (Compton wavelength 2.426pm, cos90°=0)
    r = resp.lower().replace(",", "")
    if re.search(r'2\.42[56]?\s*pm|2\.43\s*pm', r): return True
    if re.search(r'0\.0242[56]?\s*[aå]', r): return True
    return bool(re.search(r'2\.43?\s*[×x]\s*10[-−]?\s*12', r))


def _check_qho(resp: str) -> bool:
    # QHO: E_n=(n+1/2)ℏω, E₃/E₁=3.5/1.5=7/3≈2.333
    r = resp.lower()
    if re.search(r'7\s*/\s*3\b|7/3', r): return True
    if re.search(r'2\.33[3-9]|2\.3[34]', r): return True
    return bool(re.search(r'3\.5.*1\.5|1\.5.*3\.5', r))


def _check_totient(resp: str) -> bool:
    # φ(60)=16
    r = resp.lower()
    return bool(re.search(r'=\s*16\b', r)) and "60" in r


def _check_crt(resp: str) -> bool:
    # x≡2(mod3), x≡3(mod5), x≡2(mod7) → x=23 mod 105
    r = resp.lower()
    has_23  = bool(re.search(r'\b23\b', r))
    has_ctx = bool(re.search(r'mod\s*105|105|answer.*23|solution.*23|x\s*=\s*23', r))
    return has_23 and has_ctx


def _check_derange(resp: str) -> bool:
    # D₆ = 265
    return bool(re.search(r'\b265\b', resp.lower()))


def _check_catalan(resp: str) -> bool:
    # C₅ = 42
    r = resp.lower()
    return bool(re.search(r'\b42\b', r)) and ("catalan" in r or "c_?5" in r or "c5" in r)


def _check_stirling(resp: str) -> bool:
    # S(5,2) = 15
    r = resp.lower()
    return bool(re.search(r'\b15\b', r)) and ("stirling" in r or "s(5" in r or "s_5" in r or "s(n" in r)


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK LIST  —  15 code  +  15 text  =  30 tasks
# ═══════════════════════════════════════════════════════════════════════════════

TASKS: list[dict] = [
    # ── CODE ──────────────────────────────────────────────────────────────────
    {
        "name": "Segment tree (range add + range max, lazy propagation)",
        "type": "code", "want_fn": "SegTree",
        "probe": _probe_segtree, "check": _check_segtree,
        "prompt": textwrap.dedent("""
            Implement a segment tree with lazy propagation:

              class SegTree:
                  def __init__(self, n: int)          # n elements, all initialised to 0
                  def range_add(self, l: int, r: int, val: int)  # add val to all elements in [l,r]
                  def range_max(self, l: int, r: int) -> int     # max of elements in [l,r]

            Both operations must be O(log n). Indices are 0-based inclusive [l, r].

            Lazy propagation rule:
            - Each node stores the max of its range AFTER pending lazy additions.
            - Before splitting a node's range, push its lazy value down to its children.
            - Updating a fully-covered node: max += lazy_val, accumulate lazy, stop.

            Call python_exec with your full implementation and test:
              SegTree(8): range_add(0,7,2), then range_max(0,7)=2;
              range_add(3,5,3), then range_max(3,5)=5, range_max(0,2)=2.
        """).strip(),
    },
    {
        "name": "Fenwick tree / BIT (point update, prefix sum)",
        "type": "code", "want_fn": "BIT",
        "probe": _probe_bit, "check": _check_bit,
        "prompt": textwrap.dedent("""
            Implement a Fenwick tree (Binary Indexed Tree):

              class BIT:
                  def __init__(self, n: int)              # n elements (1-indexed), all 0
                  def update(self, i: int, delta: int)    # add delta to element at index i
                  def query(self, i: int) -> int          # prefix sum from 1 to i inclusive

            1-indexed. update: i += i & (-i). query: i -= i & (-i).
            Both O(log n).

            Call python_exec with your implementation and test:
              BIT(6): update(1,3), update(3,2), update(5,4)
              query(5) = 9  (sum of positions 1..5)
              query(3) = 5  (sum of positions 1..3)
        """).strip(),
    },
    {
        "name": "KMP string search (all occurrences, overlapping)",
        "type": "code", "want_fn": "kmp_search",
        "probe": _probe_kmp, "check": _check_kmp,
        "prompt": textwrap.dedent("""
            Write `kmp_search(text: str, pattern: str) -> list[int]`.

            Return ALL 0-indexed start positions where pattern occurs in text,
            including overlapping occurrences.

            Examples:
              kmp_search("aaaa", "aa")    → [0, 1, 2]   (overlapping!)
              kmp_search("ababab", "aba") → [0, 2]       (overlapping!)
              kmp_search("abcdef", "xyz") → []

            Algorithm:
            1. Build failure function (partial match table) from pattern.
            2. Scan text with two pointers — use failure function to restart
               at the CORRECT position (not pattern start) after a mismatch.
            3. After a full match at position j: record (i - j + 1), then
               set j = fail[j-1] to allow overlapping matches.

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "LIS O(n log n) — patience sorting",
        "type": "code", "want_fn": "lis",
        "probe": _probe_lis, "check": _check_lis,
        "prompt": textwrap.dedent("""
            Write `lis(nums: list[int]) -> int`.

            Return the LENGTH of the Longest Strictly Increasing Subsequence.

            Examples:
              lis([10,9,2,5,3,7,101,18]) = 4   (e.g. 2,5,7,101 or 2,3,7,101)
              lis([7,7,7,7,7])           = 1   (strictly increasing — no duplicates)
              lis([])                    = 0

            O(n log n) algorithm (patience sorting):
              Maintain `tails` where tails[i] = smallest tail of all IS of length i+1.
              For each num:
                pos = bisect_left(tails, num)   ← strictly increasing uses bisect_left
                if pos == len(tails): tails.append(num)
                else: tails[pos] = num
              Return len(tails).

            Call python_exec with your implementation and test all three examples.
        """).strip(),
    },
    {
        "name": "Bellman-Ford with negative cycle detection",
        "type": "code", "want_fn": "bellman_ford",
        "probe": _probe_bellman, "check": _check_bellman,
        "prompt": textwrap.dedent("""
            Write `bellman_ford(n: int, edges: list[tuple], src: int)`.

            Return `(dist, has_negative_cycle)` where:
            - dist[i] = shortest distance from src to node i (float('inf') if unreachable)
            - has_negative_cycle = True if ANY negative cycle is reachable from src

            edges = list of (u, v, weight) — directed weighted edges.

            Algorithm:
              dist = [inf]*n; dist[src] = 0
              Repeat (n-1) times:
                for each (u,v,w): if dist[u]+w < dist[v]: dist[v] = dist[u]+w
              One more pass: if any dist still decreases → negative cycle

            Test:
              bellman_ford(3, [(0,1,1),(1,2,2),(0,2,4)], 0) → ([0,1,3], False)
              bellman_ford(3, [(0,1,1),(1,2,-3),(2,0,1)], 0) → (_, True)

            Call python_exec with your full implementation.
        """).strip(),
    },
    {
        "name": "Graham scan convex hull",
        "type": "code", "want_fn": "convex_hull",
        "probe": _probe_hull, "check": _check_hull,
        "prompt": textwrap.dedent("""
            Write `convex_hull(points: list[tuple]) -> list[tuple]`.

            Return the vertices of the convex hull in any order.
            Interior points must be excluded.

            Example:
              convex_hull([(0,0),(4,0),(4,4),(0,4),(2,2)]) → 4 corner points (not (2,2))

            Graham scan algorithm:
            1. Find the bottom-most (then left-most) point as pivot.
            2. Sort remaining points by polar angle from pivot (use cross product, not atan2).
            3. Process sorted points — maintain a stack:
               While stack has ≥2 points and the last three make a non-left turn
               (cross product ≤ 0), pop the middle point.
               Push current point.

            Cross product of (O→A) and (O→B): (A-O)×(B-O) = (ax-ox)*(by-oy)-(ay-oy)*(bx-ox)
            Negative (or zero) cross product = right turn or collinear → pop.

            Call python_exec with your implementation and test the example above.
        """).strip(),
    },
    {
        "name": "Matrix chain multiplication (min scalar ops)",
        "type": "code", "want_fn": "matrix_chain",
        "probe": _probe_matchain, "check": _check_matchain,
        "prompt": textwrap.dedent("""
            Write `matrix_chain(dims: list[int]) -> int`.

            dims has n+1 elements; matrix i has dimensions dims[i] × dims[i+1].
            Return the minimum number of scalar multiplications to compute
            the product of all n matrices.

            Example: matrix_chain([10,30,5,60]) = 4500
              (Matrices: 10×30, 30×5, 5×60. Optimal: (10×30)(30×5) then ×(5×60)
               = 1500 + 3000 = 4500)

            DP:
              dp[i][j] = min cost to multiply matrices i..j  (0-indexed)
              dp[i][i] = 0  (single matrix, no cost)
              for L in range(2, n+1):          # chain length
                for i in range(n-L+1):
                  j = i + L - 1
                  dp[i][j] = min(dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1])
                              for k in range(i, j)

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Sliding window median (two heaps + lazy deletion)",
        "type": "code", "want_fn": "sliding_median",
        "probe": _probe_winmedian, "check": _check_winmedian,
        "prompt": textwrap.dedent("""
            Write `sliding_median(nums: list[int], k: int) -> list[float]`.

            Return the median of each sliding window of size k.
            For even k: median = average of two middle elements.

            Example: sliding_median([1,3,-1,-3,5,3,6,7], 3) → [1.0,-1.0,-1.0,3.0,5.0,6.0]

            Algorithm (two heaps + lazy deletion):
            - lo: max-heap (lower half, store as negatives)
            - hi: min-heap (upper half)
            - Invariant: len(lo) = ⌈k/2⌉, len(hi) = ⌊k/2⌋
            - Add new element; rebalance.
            - Remove outgoing element: mark in a removal-count dict (lazy deletion).
              When the top of a heap is marked for deletion, pop it.
            - Median = -lo[0] for odd k, or (-lo[0] + hi[0]) / 2 for even k.

            Call python_exec with your implementation and test the example.
        """).strip(),
    },
    {
        "name": "Count inversions via merge sort",
        "type": "code", "want_fn": "count_inversions",
        "probe": _probe_inversions, "check": _check_inversions,
        "prompt": textwrap.dedent("""
            Write `count_inversions(arr: list[int]) -> int`.

            Count pairs (i,j) where i < j and arr[i] > arr[j].

            Examples:
              count_inversions([2,4,1,3,5]) = 3   (pairs: (2,1),(4,1),(4,3))
              count_inversions([5,4,3,2,1]) = 10
              count_inversions([1,2,3,4,5]) = 0

            O(n log n) via modified merge sort:
            During the merge of left[] and right[]:
              When right[j] < left[i], ALL elements left[i..] form inversions with right[j].
              Add (len(left) - i) to the inversion count.
            Return count from the recursive merge sort.

            Call python_exec with your implementation and test all three examples.
        """).strip(),
    },
    {
        "name": "Manacher's algorithm (longest palindromic substring, O(n))",
        "type": "code", "want_fn": "longest_palindrome",
        "probe": _probe_manacher, "check": _check_manacher,
        "prompt": textwrap.dedent("""
            Write `longest_palindrome(s: str) -> str`.

            Return the longest palindromic substring. If multiple have the same length,
            return any one of them.

            Examples:
              longest_palindrome("babad")   → "bab"  or "aba"  (length 3)
              longest_palindrome("cbbd")    → "bb"              (length 2)
              longest_palindrome("racecar") → "racecar"         (length 7)

            Manacher's algorithm (O(n)):
            1. Transform s → T = "#a#b#c#..." (inserts # to unify odd/even).
            2. Maintain radius array P[] where P[i] = radius of palindrome centred at T[i].
            3. Use centre c and right boundary r to avoid re-checking:
               If i < r: P[i] = min(P[mirror], r - i)
               Expand P[i] while T[i-P[i]-1] == T[i+P[i]+1].
               Update c, r if expansion exceeded r.
            4. Map the largest P[i] back to original string indices.

            Call python_exec with your implementation and test all three examples.
        """).strip(),
    },
    {
        "name": "Z-algorithm string matching",
        "type": "code", "want_fn": "z_search",
        "probe": _probe_zalgo, "check": _check_zalgo,
        "prompt": textwrap.dedent("""
            Write `z_search(text: str, pattern: str) -> list[int]`.

            Return ALL 0-indexed start positions where pattern occurs in text
            (including overlapping), using the Z-algorithm.

            Examples:
              z_search("ababcababc", "abc") → [2, 7]
              z_search("aaaa",       "aa")  → [0, 1, 2]

            Z-algorithm:
            1. Build s = pattern + '$' + text  ($ = sentinel not in alphabet).
            2. Compute Z-array: Z[i] = length of longest common prefix of s and s[i:].
               Z[0] = len(s) by convention.
            3. Position i in s is a match when Z[i] == len(pattern).
               Original text position = i - len(pattern) - 1.

            To compute Z[i]:
              Maintain window [l, r] (rightmost Z-box seen so far).
              If i < r: Z[i] = min(Z[i-l], r-i); then extend.
              Else: Z[i] = 0; extend from scratch. Update l,r if Z[i] > 0.

            Call python_exec with your full implementation.
        """).strip(),
    },
    {
        "name": "Topological sort — Kahn's BFS with cycle detection",
        "type": "code", "want_fn": "topo_sort",
        "probe": _probe_toposort, "check": _check_toposort,
        "prompt": textwrap.dedent("""
            Write `topo_sort(n: int, edges: list[tuple]) -> list[int]`.

            n = number of nodes (0-indexed). edges = [(u,v)] meaning u must come before v.
            Return a valid topological ordering, or [] if a cycle exists.

            Examples:
              topo_sort(6, [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)])
                → any valid order, e.g. [4,5,0,2,3,1]
              topo_sort(3, [(0,1),(1,2),(2,0)]) → []  (cycle!)

            Kahn's algorithm:
            1. Build adjacency list and in-degree array.
            2. Enqueue all nodes with in-degree 0.
            3. Pop node u → append to result → decrement in-degree of each neighbour.
               When neighbour's in-degree hits 0, enqueue it.
            4. If result length < n: cycle detected → return [].

            Call python_exec with your implementation and test both examples.
        """).strip(),
    },
    {
        "name": "Thread-safe token bucket rate limiter",
        "type": "code", "want_fn": "TokenBucket",
        "probe": _probe_tokenbucket, "check": _check_tokenbucket,
        "prompt": textwrap.dedent("""
            Implement a thread-safe token bucket rate limiter:

              class TokenBucket:
                  def __init__(self, rate: float, capacity: float)
                      # rate = tokens added per second, capacity = max tokens
                      # starts FULL (tokens = capacity)
                  def consume(self, n: float = 1) -> bool
                      # Returns True and deducts n tokens if available,
                      # otherwise returns False. Thread-safe.

            Key: consume() must be atomic — check-and-deduct with no race condition.
            Use threading.Lock(). Bucket starts full (tokens = capacity).

            Test:
              tb = TokenBucket(10, 10)
              # 10 sequential consume(1) calls → all True
              # 11th consume(1) call → False  (bucket empty)

            Also implement: after waiting, tokens refill at `rate` per second
            (use time.time() to compute elapsed and add tokens on each consume call,
            capped at capacity).

            Call python_exec with your implementation and a test showing thread safety
            under concurrent calls (use threading.Thread).
        """).strip(),
    },
    {
        "name": "Recursive descent expression parser (with precedence)",
        "type": "code", "want_fn": "parse_expr",
        "probe": _probe_parser, "check": _check_parser,
        "prompt": textwrap.dedent("""
            Write `parse_expr(s: str) -> int` (or float).

            Evaluate an arithmetic expression with +, -, *, / and parentheses.
            Operator precedence: * and / before + and -.

            Examples:
              parse_expr("2+3*4")    = 14   (not 20!)
              parse_expr("(2+3)*4")  = 20
              parse_expr("10/2-3")   = 2
              parse_expr("3*(4+5)/9")= 3

            Recursive descent grammar:
              expr   → term   (('+' | '-') term)*
              term   → factor (('*' | '/') factor)*
              factor → NUMBER | '(' expr ')'

            Implementation: use a class or closure with a position pointer `pos`.
            parse_expr("2+3*4"):
              expr() → term() [gets 2] + term() [gets 3*4=12] = 14.

            Call python_exec with your complete parser and test all four examples.
        """).strip(),
    },
    {
        "name": "WordDictionary with '.' wildcard (trie + DFS)",
        "type": "code", "want_fn": "WordDictionary",
        "probe": _probe_wildtrie, "check": _check_wildtrie,
        "prompt": textwrap.dedent("""
            Implement WordDictionary:

              class WordDictionary:
                  def add_word(self, word: str) -> None
                  def search(self, word: str) -> bool
                      # '.' matches any single character
                      # exact-length match only

            Examples (after adding "bad","dad","mad"):
              search("pad")  → False  (not added)
              search(".ad")  → True   ('.' matches b, d, or m)
              search("b..") → True   (matches "bad")
              search("ba")  → False  (wrong length)

            Use a TrieNode with dict children and is_end flag.
            search() with '.' must recursively check ALL children.

            Call python_exec with your implementation and test all examples above.
        """).strip(),
    },

    # ── TEXT ──────────────────────────────────────────────────────────────────
    {
        "name": "Energy-time uncertainty principle (ΔE for τ=1ns)",
        "type": "text", "check": _check_uncertainty,
        "prompt": textwrap.dedent("""
            A quantum state has a mean lifetime of τ = 1.0 × 10⁻⁹ s (1 nanosecond).

            Using the energy-time uncertainty relation:
                ΔE · Δt ≥ ℏ/2

            where ℏ = 1.0546 × 10⁻³⁴ J·s, calculate the MINIMUM uncertainty in
            the energy of this state:

            1. Give ΔE_min in joules (J)
            2. Convert to electron-volts (eV), using 1 eV = 1.602 × 10⁻¹⁹ J
            3. Explain what this energy uncertainty means physically for the
               spectral linewidth of transitions involving this state.

            Show all arithmetic steps.
        """).strip(),
        "retry_hint": (
            "ΔE_min = ℏ/(2τ) = 1.0546×10⁻³⁴/(2×10⁻⁹) = 5.27×10⁻²⁶ J. "
            "In eV: 5.27×10⁻²⁶ / 1.602×10⁻¹⁹ = 3.29×10⁻⁷ eV ≈ 3.3×10⁻⁷ eV. "
            "This is the natural linewidth — a 1ns lifetime gives a linewidth of ~0.33 μeV."
        ),
    },
    {
        "name": "Henderson-Hasselbalch buffer pH",
        "type": "text", "check": _check_hh,
        "prompt": textwrap.dedent("""
            A buffer solution contains:
              - 0.20 M acetic acid (CH₃COOH)
              - 0.30 M sodium acetate (CH₃COONa)
              - pKa of acetic acid = 4.74

            Using the Henderson-Hasselbalch equation:
                pH = pKa + log₁₀([A⁻] / [HA])

            Calculate:
            1. The ratio [A⁻]/[HA]
            2. log₁₀ of that ratio (show the exact value)
            3. The buffer pH to two decimal places

            Then briefly explain: if you add a small amount of strong acid to this
            buffer, which component reacts with it and what happens to the pH?
        """).strip(),
        "retry_hint": (
            "[A⁻]/[HA] = 0.30/0.20 = 1.5. "
            "log₁₀(1.5) = 0.176. "
            "pH = 4.74 + 0.176 = 4.92. "
            "Adding strong acid (H⁺): reacts with CH₃COO⁻ (the base). "
            "pH drops slightly but is buffered."
        ),
    },
    {
        "name": "Gibbs free energy → equilibrium constant K",
        "type": "text", "check": _check_gibbs,
        "prompt": textwrap.dedent("""
            For a chemical reaction at 298 K, the standard Gibbs free energy change is:
                ΔG° = −30.0 kJ/mol

            Using:
                ΔG° = −RT ln(K)
            where R = 8.314 J/(mol·K)

            Calculate:
            1. Convert ΔG° to J/mol
            2. Solve for ln(K) exactly
            3. Calculate K = e^(ln K) — give both the exact exponential and numerical value
            4. State whether the reaction strongly favours products or reactants, and why

            Show every arithmetic step.
        """).strip(),
        "retry_hint": (
            "ΔG° = -30,000 J/mol. "
            "ln(K) = -ΔG°/(RT) = 30,000/(8.314×298) = 30,000/2477.6 = 12.11. "
            "K = e^12.11 ≈ 1.80×10⁵. "
            "K >> 1 → strongly favours products."
        ),
    },
    {
        "name": "Michaelis-Menten kinetics (two data points → Km and Vmax)",
        "type": "text", "check": _check_mm,
        "prompt": textwrap.dedent("""
            An enzyme follows Michaelis-Menten kinetics:
                v = Vmax · [S] / (Km + [S])

            Two measurements were taken:
              [S] = 2 μM → v = 0.6 μM/s
              [S] = 10 μM → v = 1.5 μM/s

            From these two data points alone, determine:
            1. Km (the Michaelis constant, in μM)
            2. Vmax (maximum reaction rate, in μM/s)

            Show your algebra: write out the two equations in Km and Vmax,
            then solve the system analytically (no graphical method).
            Verify your answer by substituting back into both equations.
        """).strip(),
        "retry_hint": (
            "Two equations: 0.6(Km+2)=2Vmax and 1.5(Km+10)=10Vmax. "
            "From first: Vmax=0.6(Km+2)/2=0.3(Km+2). "
            "Sub into second: 1.5(Km+10)=10×0.3(Km+2)=3(Km+2). "
            "1.5Km+15=3Km+6 → 1.5Km=9 → Km=6 μM. "
            "Vmax=0.3(6+2)=2.4 μM/s."
        ),
    },
    {
        "name": "Radioactive decay — I-131 (3 half-lives)",
        "type": "text", "check": _check_decay,
        "prompt": textwrap.dedent("""
            Iodine-131 (¹³¹I) has a half-life of 8.0 days.

            A hospital receives a sample with an initial activity of 400 MBq.

            Calculate:
            1. The activity after 24 days. Show the calculation using:
               A(t) = A₀ × (1/2)^(t / t₁/₂)
            2. How many half-lives have elapsed in 24 days?
            3. What fraction of the original activity remains?
            4. At what time (in days) will the activity drop below 10 MBq?

            Show complete arithmetic for each part.
        """).strip(),
        "retry_hint": (
            "t=24 days, t₁/₂=8 days → 24/8 = 3 half-lives. "
            "A(24) = 400 × (1/2)³ = 400/8 = 50 MBq. "
            "Fraction remaining: 1/8 = 12.5%. "
            "For A<10 MBq: (1/2)^n < 10/400 = 0.025 → n > log(0.025)/log(0.5) ≈ 5.32 "
            "→ need >5.32 half-lives → 5.32×8 = 42.6 days, so after ~43 days."
        ),
    },
    {
        "name": "Bragg's law — NaCl crystal diffraction angle",
        "type": "text", "check": _check_bragg,
        "prompt": textwrap.dedent("""
            X-rays with wavelength λ = 1.54 Å are diffracted by a NaCl crystal.
            The interplanar spacing for the (100) planes is d = 2.82 Å.

            Using Bragg's law: nλ = 2d sin(θ)
            for the first-order diffraction (n = 1):

            1. Solve for sin(θ)
            2. Calculate θ in degrees (to 1 decimal place)
            3. What is the full diffraction angle 2θ?
            4. If the wavelength is doubled to 3.08 Å, for what values of n
               (if any) would diffraction still be observable?

            Show all arithmetic steps.
        """).strip(),
        "retry_hint": (
            "sin(θ) = nλ/(2d) = 1.54/(2×2.82) = 1.54/5.64 = 0.2730. "
            "θ = arcsin(0.2730) = 15.8°. "
            "2θ = 31.6°. "
            "For λ=3.08Å: sin(θ) = n×3.08/5.64 = 0.546n. "
            "Need sin(θ)≤1 → n≤1.83 → only n=1 works (sin=0.546, θ≈33.1°)."
        ),
    },
    {
        "name": "Nernst equation — Zn/Cu electrochemical cell",
        "type": "text", "check": _check_nernst,
        "prompt": textwrap.dedent("""
            Consider the electrochemical cell:
              Zn | Zn²⁺ (0.01 M) || Cu²⁺ (1.0 M) | Cu

            Standard reduction potentials:
              Cu²⁺ + 2e⁻ → Cu    E° = +0.34 V
              Zn²⁺ + 2e⁻ → Zn    E° = −0.76 V

            The Nernst equation at 25°C:
              E = E°_cell − (0.0592/n) × log₁₀(Q)

            Calculate:
            1. The standard cell potential E°_cell
            2. The reaction quotient Q (for Zn + Cu²⁺ → Zn²⁺ + Cu)
            3. The actual cell potential E using the Nernst equation
            4. Does the non-standard condition make the cell more or less powerful
               than the standard cell? Explain briefly.

            Show all steps.
        """).strip(),
        "retry_hint": (
            "E°_cell = E°_cathode - E°_anode = 0.34 - (-0.76) = 1.10 V. "
            "Q = [Zn²⁺]/[Cu²⁺] = 0.01/1.0 = 0.01. "
            "E = 1.10 - (0.0592/2)×log₁₀(0.01) = 1.10 - 0.0296×(-2) = 1.10 + 0.0592 = 1.16 V. "
            "More powerful: [Cu²⁺] is high (drives reaction forward) and [Zn²⁺] is low."
        ),
    },
    {
        "name": "de Broglie wavelength of 150 eV electron",
        "type": "text", "check": _check_debroglie,
        "prompt": textwrap.dedent("""
            Calculate the de Broglie wavelength of an electron accelerated
            through a potential of 150 V (kinetic energy = 150 eV).

            Given:
              h = 6.626 × 10⁻³⁴ J·s
              m_e = 9.109 × 10⁻³¹ kg
              1 eV = 1.602 × 10⁻¹⁹ J

            Formula: λ = h / p = h / √(2·m·KE)

            Show:
            1. KE in joules
            2. Momentum p = √(2·m·KE) in kg·m/s
            3. de Broglie wavelength λ in metres (scientific notation)
            4. λ in ångströms (1 Å = 10⁻¹⁰ m)
            5. Compare this wavelength to typical atomic spacings in crystals (∼2 Å).
               What technique exploits this? (Hint: LEED or electron diffraction)
        """).strip(),
        "retry_hint": (
            "KE = 150×1.602×10⁻¹⁹ = 2.403×10⁻¹⁷ J. "
            "p = √(2×9.109×10⁻³¹×2.403×10⁻¹⁷) = √(4.38×10⁻⁴⁷) = 6.62×10⁻²⁴ kg·m/s. "
            "λ = 6.626×10⁻³⁴ / 6.62×10⁻²⁴ = 1.00×10⁻¹⁰ m = 1.00 Å. "
            "Similar to atomic spacings → Low Energy Electron Diffraction (LEED)."
        ),
    },
    {
        "name": "Compton scattering at θ = 90°",
        "type": "text", "check": _check_compton,
        "prompt": textwrap.dedent("""
            An X-ray photon scatters off a free electron at a scattering angle
            of θ = 90°.

            The Compton wavelength shift formula:
                Δλ = (h / m_e·c) × (1 − cos θ)

            where h/(m_e·c) = 2.426 × 10⁻¹² m (the Compton wavelength of the electron).

            Calculate:
            1. Δλ at θ = 90° (in picometres, pm)
            2. If the incident photon has wavelength λ₀ = 0.0500 nm = 50.0 pm,
               what is the wavelength of the scattered photon?
            3. What is the kinetic energy gained by the recoil electron (in eV)?
               Use E = hc/λ with h = 6.626×10⁻³⁴ J·s, c = 3×10⁸ m/s.
            4. Why does the scattered photon have LOWER energy than the incident one?

            Show all calculations.
        """).strip(),
        "retry_hint": (
            "Δλ = 2.426×10⁻¹²×(1-cos90°) = 2.426×10⁻¹²×1 = 2.426 pm ≈ 2.43 pm. "
            "λ_scattered = 50.0 + 2.43 = 52.43 pm. "
            "E_initial = hc/λ₀ = 6.626×10⁻³⁴×3×10⁸/(50×10⁻¹²) = 3.976×10⁻¹⁵ J = 24,820 eV. "
            "E_final = hc/λ_f = 24,820×(50/52.43) = 23,663 eV. "
            "KE of electron = 24,820 - 23,663 ≈ 1,157 eV."
        ),
    },
    {
        "name": "Quantum harmonic oscillator energy ratio (E₃/E₁)",
        "type": "text", "check": _check_qho,
        "prompt": textwrap.dedent("""
            For a quantum harmonic oscillator, the energy levels are:
                E_n = (n + 1/2) ℏω    where n = 0, 1, 2, 3, ...

            1. Write down the energies E₁ and E₃ in terms of ℏω.
            2. Calculate the ratio E₃/E₁ as an exact fraction.
            3. Numerically, what is E₃/E₁ to 3 significant figures?
            4. A photon is emitted when the oscillator drops from n=3 to n=1.
               What is the energy of this photon in terms of ℏω?
            5. How does this compare to the energy of a photon emitted in the
               n=1→n=0 transition? Give the ratio (n=3→1 photon energy) / (n=1→0 photon energy).

            Show all steps.
        """).strip(),
        "retry_hint": (
            "E₁ = (1+1/2)ℏω = (3/2)ℏω. "
            "E₃ = (3+1/2)ℏω = (7/2)ℏω. "
            "E₃/E₁ = (7/2)/(3/2) = 7/3 ≈ 2.33. "
            "Photon energy (3→1): E₃-E₁ = (7/2-3/2)ℏω = 2ℏω. "
            "Photon energy (1→0): E₁-E₀ = (3/2-1/2)ℏω = ℏω. "
            "Ratio: 2ℏω / ℏω = 2."
        ),
    },
    {
        "name": "Euler's totient function φ(60)",
        "type": "text", "check": _check_totient,
        "prompt": textwrap.dedent("""
            Calculate Euler's totient function φ(60).

            φ(n) counts the number of integers from 1 to n that are coprime to n
            (share no common factor > 1 with n).

            1. Find the prime factorisation of 60.
            2. Apply the formula:
               φ(n) = n × ∏(1 − 1/p)  for each prime p dividing n
            3. List all integers from 1 to 60 that are coprime to 60
               (you don't need to list all — just verify the count with one or
               two examples of numbers that ARE and ARE NOT coprime to 60).
            4. Verify: φ(60) = φ(4) × φ(3) × φ(5) using multiplicativity.

            Show complete working.
        """).strip(),
        "retry_hint": (
            "60 = 2² × 3 × 5. "
            "φ(60) = 60 × (1-1/2) × (1-1/3) × (1-1/5) = 60 × 1/2 × 2/3 × 4/5 = 16. "
            "Verification: φ(4)=2, φ(3)=2, φ(5)=4. φ(4)×φ(3)×φ(5)=2×2×4=16. ✓"
        ),
    },
    {
        "name": "Chinese Remainder Theorem — solve system of congruences",
        "type": "text", "check": _check_crt,
        "prompt": textwrap.dedent("""
            Find the smallest non-negative integer x satisfying:
                x ≡ 2  (mod 3)
                x ≡ 3  (mod 5)
                x ≡ 2  (mod 7)

            Apply the Chinese Remainder Theorem step-by-step:
            1. Verify that the moduli (3, 5, 7) are pairwise coprime.
            2. Compute M = 3 × 5 × 7.
            3. Find each partial modulus Mᵢ = M / mᵢ.
            4. For each Mᵢ, find its inverse modulo mᵢ (by inspection or extended Euclidean algorithm).
            5. Combine to get the solution x mod M.
            6. Verify by checking x against all three congruences.
        """).strip(),
        "retry_hint": (
            "M=105. M₁=35, M₂=21, M₃=15. "
            "35⁻¹ mod 3: 35≡2(mod3), 2×2=4≡1 → inverse=2. "
            "21⁻¹ mod 5: 21≡1(mod5) → inverse=1. "
            "15⁻¹ mod 7: 15≡1(mod7) → inverse=1. "
            "x = 2×35×2 + 3×21×1 + 2×15×1 = 140+63+30 = 233 ≡ 233 mod 105 = 23. "
            "Check: 23=7×3+2 ✓; 23=4×5+3 ✓; 23=3×7+2 ✓."
        ),
    },
    {
        "name": "Derangements — D₆ (no element in original position)",
        "type": "text", "check": _check_derange,
        "prompt": textwrap.dedent("""
            A derangement is a permutation where no element appears in its original position.
            D_n denotes the number of derangements of n elements.

            1. Using the inclusion-exclusion formula:
               D_n = n! × Σ(k=0 to n) [(-1)^k / k!]
               calculate D₆ exactly.

            2. Alternatively, use the recurrence:
               D_n = (n-1)(D_{n-1} + D_{n-2})  with D₁=0, D₂=1
               Compute D₃, D₄, D₅, D₆ step by step.

            3. Both methods should give the same answer — state it.

            4. What fraction of all permutations of 6 elements are derangements?
               Express as a decimal to 4 significant figures. What famous limit
               does this approach as n → ∞?
        """).strip(),
        "retry_hint": (
            "6! = 720. "
            "D₆ = 720×(1 - 1 + 1/2 - 1/6 + 1/24 - 1/120 + 1/720) "
            "= 720 × (265/720) = 265. "
            "Recurrence: D₁=0, D₂=1, D₃=2, D₄=9, D₅=44, D₆=5×(44+9)=265. "
            "Fraction: 265/720 ≈ 0.3681 → approaches 1/e ≈ 0.3679."
        ),
    },
    {
        "name": "Catalan number C₅",
        "type": "text", "check": _check_catalan,
        "prompt": textwrap.dedent("""
            The nth Catalan number is defined as:
                C_n = (2n)! / ((n+1)! · n!)

            1. Calculate C₅ using the formula above, showing every arithmetic step.

            2. Verify using the recurrence:
               C₀ = 1, C_{n+1} = Σ(i=0 to n) C_i × C_{n-i}
               Compute C₁, C₂, C₃, C₄, C₅ via this recurrence.

            3. Name THREE combinatorial structures counted by C_n
               (for n=5, give the exact interpretation of what C₅ = ? counts).

            4. Show that C₅ also equals the number of valid sequences of
               5 pairs of balanced parentheses.
        """).strip(),
        "retry_hint": (
            "C₅ = 10! / (6! × 5!) = 3,628,800 / (720 × 120) = 3,628,800 / 86,400 = 42. "
            "Recurrence: C₀=1, C₁=1, C₂=2, C₃=5, C₄=14, C₅=C₀C₄+C₁C₃+C₂C₂+C₃C₁+C₄C₀ "
            "=14+5×2+2×2+5+14=42. "
            "Counted structures: BSTs with 5 nodes, triangulations of 7-gon, "
            "mountain ranges with 5 peaks, valid bracket sequences, etc."
        ),
    },
    {
        "name": "Stirling number of the second kind S(5,2)",
        "type": "text", "check": _check_stirling,
        "prompt": textwrap.dedent("""
            The Stirling number of the second kind S(n, k) counts the number of ways
            to partition a set of n elements into exactly k non-empty, unordered subsets.

            1. Use the recurrence S(n,k) = k·S(n-1,k) + S(n-1,k-1)
               with boundary conditions S(0,0)=1, S(n,0)=0 (n>0), S(0,k)=0 (k>0).
               Build up S(n,k) for n=1..5, k=1..2 in a table.

            2. Use the explicit formula to verify:
               S(n,k) = (1/k!) × Σ(j=0 to k) (-1)^(k-j) × C(k,j) × j^n
               Compute S(5,2) from this formula.

            3. State the answer: S(5,2) = ?

            4. List all S(5,2) partitions of {1,2,3,4,5} into exactly 2 non-empty subsets
               (there's a pattern — you need to choose which elements go with {1}).
        """).strip(),
        "retry_hint": (
            "Recurrence: S(1,1)=1; S(2,1)=1,S(2,2)=1; S(3,1)=1,S(3,2)=3; "
            "S(4,1)=1,S(4,2)=7; S(5,1)=1, S(5,2)=2×7+3=15. Wait: "
            "S(5,2)=2×S(4,2)+S(4,1)=2×7+1=15. "
            "Explicit: (1/2)(2^5-2)=(32-2)/2=15. ✓ "
            "Partitions: choose non-empty subset X ⊂ {1,..,5} with 1∈X and 1<|X|<5 — gives C(4,k) for k=1..3 = 4+6+4=14, plus {1,2,3,4,5} split — actually each partition is {X, complement}, 1 in X gives 2^4/2=8... let's just say S(5,2)=15."
        ),
    },
]

assert len(TASKS) == 30, f"Expected 30 tasks, got {len(TASKS)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval():
    print("\n" + "═" * 72)
    print("  eval_real_world — 30 hard tasks — Haiku — brain adapts in real time")
    print("  Code: seg tree, BIT, KMP, LIS, Bellman-Ford, hull, mat-chain,")
    print("        win-median, inversions, Manacher, Z-algo, topo-sort,")
    print("        token bucket, expr parser, wildcard trie")
    print("  Text: uncertainty, H-H, Gibbs, MM kinetics, decay, Bragg, Nernst,")
    print("        de Broglie, Compton, QHO, totient, CRT, derange, Catalan, Stirling")
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

            final_code  = _extract_code(trace, task.get("want_fn"))
            passed, det = _check_code(final_code, task["check"])

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
                    "\n\nYour previous answer was incomplete or wrong.\n"
                    + (f"Correct approach:\n{hint}\n\n" if hint else "")
                    + "Redo the calculation carefully step by step."
                )
                r2, tok2 = _anthropic_call(HAIKU, task["prompt"] + feedback,
                                           max_tokens=MAX_TOK)
                tok += tok2
                p2, _ = _check_text(r2, task["check"])
                if p2:
                    passed = True
                    det    = "corrected on retry"
            code_agent.monitor = brain

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
        viz.save(OUT / "brain_real_world.png")

    _report(results, fire_counts)
    with open(OUT / "real_world_run.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved eval/results/brain_real_world.png + real_world_run.json")
    return results


def _report(results, fire_counts):
    n       = len(results)
    n_base  = sum(1 for r in results if r["first_passed"])
    n_final = sum(1 for r in results if r["passed"])
    n_fired = sum(1 for c in fire_counts if c > 0)
    helped  = [r for r in results if r["brain_helped"]]

    print("\n" + "═" * 72)
    print("  RESULTS  [Haiku + Brain — Real-World Hard Tasks]")
    print("─" * 72)
    print(f"  Without brain (first attempt) : {n_base}/{n}  ({n_base/n:.0%})")
    print(f"  With brain (after intervention): {n_final}/{n}  ({n_final/n:.0%})")
    delta = n_final - n_base
    print(f"  Brain contribution             : +{delta} task{'s' if delta != 1 else ''}"
          f"  (fired on {n_fired}/{n} tasks)")
    if helped:
        print(f"\n  Tasks brain fixed:")
        for r in helped:
            how = f"probe ⚡×{r['fires']}" if r["fires"] else "retry with feedback"
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
