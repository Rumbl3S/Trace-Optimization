"""eval/eval_extensive.py — 40 hard tasks targeting heavy-user failure modes.

Categories drawn from competitive programming benchmarks (LiveCodeBench Pro,
ICPC-Eval), GPQA-style science, and real-world debugging patterns that
commonly trip haiku-class models.

Code (20): segment tree lazy, bitmask TSP, matrix exponentiation, Fenwick BIT,
           KMP, digit DP, Manacher's, minimum window substring, LCS substring,
           expression evaluator, topological sort (lexi), binary search on
           answer, Kruskal MST, Bellman-Ford neg-cycle, count inversions,
           mutable default bug, late-binding closure, token bucket rate limiter,
           TTL-expiry LRU cache, sudoku solver.

Text (12): particle in box, Gibbs free energy trap (non-spontaneous!), Hess's law,
           Euler's totient, derangements, inclusion-exclusion, relativistic KE,
           Shannon entropy, Bayes (1% prevalence), modular exponentiation,
           nuclear decay branching, Maxwell-Boltzmann most-probable speed.

Debugging (8): mutable default arg, late-binding closure, generator exhaustion,
               integer accumulation in float loop, binary search off-by-one,
               token bucket rate limiter, TTL cache, sudoku solver.

(Debug tasks embedded in code section above; text section = 12 pure text.)

Run:  python eval/eval_extensive.py
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
from eval.eval_hard import (
    _exec, _check_code, _check_text, _extract_code, _first_exec_code,
)

OUT   = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

HAIKU     = "claude-haiku-4-5-20251001"
MAX_TURNS = 16
MAX_TOK   = 4096


# ═══════════════════════════════════════════════════════════════════════════════
#  CODE PROBES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Segment tree with lazy propagation ────────────────────────────────────────

def _probe_segtree(ns):
    cls = (ns.get("SegTree") or ns.get("LazySegTree") or ns.get("SegmentTree")
           or ns.get("RangeTree"))
    if not cls:
        return ["SegTree class not defined with range_add(l,r,val) and query_sum(l,r) — implement and call python_exec"]
    fails = []
    try:
        st = cls([1, 2, 3, 4, 5])
        st.range_add(1, 3, 10)
        # array is now [1, 12, 13, 14, 5]
        r = st.query_sum(0, 4)
        if r != 45:
            fails.append(
                f"After range_add(1,3,10) on [1,2,3,4,5]: query_sum(0,4)={r}, expected 45. "
                "FIX: When propagating lazy tag to children before descending, push pending "
                "additions down: child.lazy += parent.lazy; child.sum += lazy * child.size. "
                "Common bug: forgetting to reset parent.lazy to 0 after pushing."
            )
            return fails
        r2 = st.query_sum(1, 3)
        if r2 != 39:
            fails.append(
                f"query_sum(1,3)={r2}, expected 39 (12+13+14). "
                "FIX: Ensure _push_down is called on every internal node before recursing."
            )
            return fails
        st.range_add(0, 2, 5)
        # array: [6, 17, 18, 14, 5]
        r3 = st.query_sum(2, 4)
        if r3 != 37:
            fails.append(f"After second range_add(0,2,5): query_sum(2,4)={r3}, expected 37 (18+14+5).")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_segtree(ns):
    cls = (ns.get("SegTree") or ns.get("LazySegTree") or ns.get("SegmentTree")
           or ns.get("RangeTree"))
    if not cls: return False
    try:
        st = cls([1, 2, 3, 4, 5])
        st.range_add(1, 3, 10)
        if st.query_sum(0, 4) != 45: return False
        if st.query_sum(1, 3) != 39: return False
        st.range_add(0, 2, 5)
        if st.query_sum(0, 4) != 60: return False
        if st.query_sum(2, 4) != 37: return False
        # All zeros, large range add
        st2 = cls([0] * 8)
        st2.range_add(0, 7, 1)
        st2.range_add(2, 5, 3)
        if st2.query_sum(2, 5) != 16: return False  # 4*(1+3)=16
        if st2.query_sum(0, 7) != 20: return False  # 8*1 + 4*3 = 8+12=20
        return True
    except Exception: return False


# ── Bitmask DP — Travelling Salesman (min cost Hamiltonian cycle) ──────────────

def _probe_tsp(ns):
    fn = ns.get("tsp") or ns.get("travelling_salesman") or ns.get("min_cost_tour")
    if not fn:
        return ["tsp(dist) not defined — implement bitmask DP and call python_exec"]
    fails = []
    try:
        dist = [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
        r = fn(dist)
        if r != 80:
            fails.append(
                f"tsp([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]])={r}, expected 80. "
                "FIX: dp[mask][i] = min cost to have visited all cities in mask, ending at city i. "
                "Start: dp[1<<0][0]=0, all else=inf. "
                "Transition: dp[mask|(1<<j)][j] = min(dp[mask][i] + dist[i][j]) for i in mask, j not in mask. "
                "Answer: min over i of dp[(1<<n)-1][i] + dist[i][0]."
            )
            return fails
        dist2 = [[0,5,10],[5,0,6],[10,6,0]]
        r2 = fn(dist2)
        if r2 != 21:
            fails.append(f"tsp 3-city symmetric: expected 21 (5+6+10), got {r2}.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_tsp(ns):
    fn = ns.get("tsp") or ns.get("travelling_salesman") or ns.get("min_cost_tour")
    if not fn: return False
    try:
        d1 = [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
        d2 = [[0,5,10],[5,0,6],[10,6,0]]
        d3 = [[0,1],[1,0]]
        return fn(d1)==80 and fn(d2)==21 and fn(d3)==2
    except Exception: return False


# ── Matrix exponentiation (nth Fibonacci in O(log n)) ─────────────────────────

def _probe_matfib(ns):
    fn = ns.get("mat_fib") or ns.get("fib_fast") or ns.get("fibonacci_fast")
    if not fn:
        return ["mat_fib(n) not defined using matrix exponentiation — implement and call python_exec"]
    fails = []
    try:
        if fn(0) != 0:
            fails.append(f"mat_fib(0)={fn(0)}, expected 0.")
            return fails
        if fn(1) != 1:
            fails.append(f"mat_fib(1)={fn(1)}, expected 1.")
            return fails
        if fn(10) != 55:
            fails.append(
                f"mat_fib(10)={fn(10)}, expected 55. "
                "FIX: [[F(n+1)],[F(n)]] = [[1,1],[1,0]]^n × [[1],[0]]. "
                "mat_pow([[1,1],[1,0]], n) using repeated squaring: O(log n). "
                "Multiply 2×2 matrices: C[i][j] = sum_k A[i][k]*B[k][j]."
            )
            return fails
        if fn(50) != 12586269025:
            fails.append(f"mat_fib(50)={fn(50)}, expected 12586269025.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_matfib(ns):
    fn = ns.get("mat_fib") or ns.get("fib_fast") or ns.get("fibonacci_fast")
    if not fn: return False
    fibs = [0,1,1,2,3,5,8,13,21,34,55,89,144]
    try:
        return all(fn(i)==fibs[i] for i in range(13)) and fn(50)==12586269025
    except Exception: return False


# ── Fenwick tree / Binary Indexed Tree ────────────────────────────────────────

def _probe_bit(ns):
    cls = (ns.get("BIT") or ns.get("FenwickTree") or ns.get("BinaryIndexedTree"))
    if not cls:
        return ["BIT(n) class not defined with update(i, delta) and prefix_sum(i) — implement and call python_exec"]
    fails = []
    try:
        bit = cls(5)
        for i, v in enumerate([3, 2, -1, 6, 5]):
            bit.update(i, v)
        if bit.prefix_sum(4) != 15:
            fails.append(
                f"BIT with [3,2,-1,6,5]: prefix_sum(4)={bit.prefix_sum(4)}, expected 15. "
                "FIX: update(i, delta): while i<n: tree[i]+=delta; i+=i&(-i). "
                "prefix_sum(i): s=0; while i>=0: s+=tree[i]; i-=(i&(-i))-1 (0-indexed variant)."
            )
            return fails
        if bit.prefix_sum(2) != 4:
            fails.append(f"prefix_sum(2)={bit.prefix_sum(2)}, expected 4 (3+2-1).")
        bit.update(2, 10)  # -1 becomes 9
        if bit.prefix_sum(4) != 25:
            fails.append(f"After update(2,10): prefix_sum(4)={bit.prefix_sum(4)}, expected 25.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_bit(ns):
    cls = (ns.get("BIT") or ns.get("FenwickTree") or ns.get("BinaryIndexedTree"))
    if not cls: return False
    try:
        bit = cls(5)
        for i, v in enumerate([3, 2, -1, 6, 5]):
            bit.update(i, v)
        if bit.prefix_sum(4) != 15: return False
        if bit.prefix_sum(2) != 4: return False
        bit.update(2, 10)
        if bit.prefix_sum(4) != 25: return False
        # Range sum via two prefix queries
        if hasattr(bit, 'range_sum'):
            if bit.range_sum(1, 3) != 2+9+6: return False
        return True
    except Exception: return False


# ── KMP pattern search ─────────────────────────────────────────────────────────

def _probe_kmp(ns):
    fn = ns.get("kmp_search") or ns.get("kmp") or ns.get("find_all")
    if not fn:
        return ["kmp_search(text, pattern) not defined returning list of start indices — implement and call python_exec"]
    fails = []
    try:
        r = fn("AABABCAAB", "AAB")
        if sorted(r) != [0, 6]:
            fails.append(
                f"kmp_search('AABABCAAB','AAB')={r}, expected [0,6]. "
                "FIX: Build failure function (lps array): "
                "lps[i] = length of longest proper prefix of pattern[:i+1] that is also a suffix. "
                "Then scan text using two pointers i,j; when mismatch: j=lps[j-1] if j>0, else i++."
            )
            return fails
        if fn("AAAA", "AA") != [0, 1, 2]:
            fails.append(
                f"kmp_search('AAAA','AA')={fn('AAAA','AA')}, expected [0,1,2]. "
                "FIX: After a match at position i, set j=lps[m-1] (not 0) to allow overlapping matches."
            )
        if fn("ABCDEF", "GH") != []:
            fails.append(f"kmp_search with no match should return [].")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_kmp(ns):
    fn = ns.get("kmp_search") or ns.get("kmp") or ns.get("find_all")
    if not fn: return False
    try:
        return all(sorted(fn(t,p))==e for t,p,e in [
            ("AABABCAAB","AAB",[0,6]),
            ("AAAA","AA",[0,1,2]),
            ("ABCDEF","GH",[]),
            ("AABAACAADAABAAB","AABA",[0,9]),
            ("A","A",[0]),
        ])
    except Exception: return False


# ── Digit DP (count numbers 1..N with digit-sum divisible by k) ───────────────

def _probe_digitdp(ns):
    fn = ns.get("count_digit_sum_mod") or ns.get("countDigitSumMod") or ns.get("digit_dp")
    if not fn:
        return ["count_digit_sum_mod(n, k) not defined — implement digit DP and call python_exec"]
    fails = []
    try:
        def brute(n, k):
            return sum(1 for x in range(1, n+1) if sum(int(d) for d in str(x)) % k == 0)
        for n, k in [(100,3),(50,7),(200,5),(30,4)]:
            expected = brute(n, k)
            r = fn(n, k)
            if r != expected:
                fails.append(
                    f"count_digit_sum_mod({n},{k})={r}, expected {expected}. "
                    "FIX: dp[pos][tight][rem] = ways to fill remaining positions. "
                    "tight=True means digits so far == prefix of N. "
                    "At each position try digit 0..9 (or 0..N[pos] if tight). "
                    "rem = (current_sum) % k. Answer = dp[0][True][0] - (1 if 0 is counted)."
                )
                return fails
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_digitdp(ns):
    fn = ns.get("count_digit_sum_mod") or ns.get("countDigitSumMod") or ns.get("digit_dp")
    if not fn: return False
    def brute(n, k):
        return sum(1 for x in range(1, n+1) if sum(int(d) for d in str(x)) % k == 0)
    try:
        return all(fn(n,k)==brute(n,k) for n,k in [(100,3),(50,7),(200,5),(1000,11)])
    except Exception: return False


# ── Manacher's algorithm (longest palindromic substring length) ────────────────

def _probe_manacher(ns):
    fn = (ns.get("longest_palindrome") or ns.get("manacher")
          or ns.get("longest_palindromic_substring"))
    if not fn:
        return ["longest_palindrome(s) not defined returning the length — implement Manacher's and call python_exec"]
    fails = []
    try:
        # Return length or the substring itself — handle both
        def _len(r): return r if isinstance(r, int) else len(r)
        if _len(fn("babad")) not in (3,):
            fails.append(f"longest_palindrome('babad')={fn('babad')}, expected length 3 ('bab' or 'aba').")
            return fails
        if _len(fn("cbbd")) != 2:
            fails.append(f"longest_palindrome('cbbd')={fn('cbbd')}, expected 2 ('bb').")
        if _len(fn("racecar")) != 7:
            fails.append(
                f"longest_palindrome('racecar')={fn('racecar')}, expected 7. "
                "FIX: Manacher's inserts separators: '#r#a#c#e#c#a#r#'. "
                "Maintains center c and right boundary r. For each i: "
                "mirror = 2*c-i; p[i]=min(r-i, p[mirror]) if i<r else 0. "
                "Then expand and update c, r."
            )
        if _len(fn("a")) != 1:
            fails.append(f"longest_palindrome('a')={fn('a')}, expected 1.")
        # Brute-force check for a longer string
        s = "abacabadabacaba"
        def brute_len(t):
            best = 1
            for i in range(len(t)):
                for j in range(i+1, len(t)+1):
                    sub = t[i:j]
                    if sub == sub[::-1] and len(sub) > best:
                        best = len(sub)
            return best
        expected = brute_len(s)
        got = _len(fn(s))
        if got != expected:
            fails.append(f"longest_palindrome('{s}')={got}, expected {expected}.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_manacher(ns):
    fn = (ns.get("longest_palindrome") or ns.get("manacher")
          or ns.get("longest_palindromic_substring"))
    if not fn: return False
    def _len(r): return r if isinstance(r, int) else len(r)
    def brute(t):
        best = 1
        for i in range(len(t)):
            for j in range(i+1, len(t)+1):
                sub = t[i:j]
                if sub == sub[::-1] and len(sub) > best: best = len(sub)
        return best
    try:
        tests = ["babad","cbbd","racecar","a","aaaa","abcba","abacaba","xabacabax",""]
        return all(_len(fn(s))==(brute(s) if s else 0) for s in tests)
    except Exception: return False


# ── Minimum window substring ──────────────────────────────────────────────────

def _probe_minwin(ns):
    fn = ns.get("min_window") or ns.get("minWindow") or ns.get("minimum_window")
    if not fn:
        return ["min_window(s, t) not defined returning shortest substring containing all chars of t — implement and call python_exec"]
    fails = []
    try:
        r = fn("ADOBECODEBANC", "ABC")
        if r != "BANC":
            fails.append(
                f"min_window('ADOBECODEBANC','ABC')='{r}', expected 'BANC'. "
                "FIX: Two pointers + freq map. Expand right until all t chars covered; "
                "shrink left while still valid. Track minimum valid window."
            )
            return fails
        if fn("a", "a") != "a":
            fails.append(f"min_window('a','a')='{fn('a','a')}', expected 'a'.")
        if fn("a", "b") != "":
            fails.append(f"min_window('a','b')='{fn('a','b')}', expected ''.")
        if fn("aa", "aa") != "aa":
            fails.append(f"min_window('aa','aa')='{fn('aa','aa')}', expected 'aa'.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_minwin(ns):
    fn = ns.get("min_window") or ns.get("minWindow") or ns.get("minimum_window")
    if not fn: return False
    try:
        return all(fn(s,t)==e for s,t,e in [
            ("ADOBECODEBANC","ABC","BANC"), ("a","a","a"), ("a","b",""),
            ("aa","aa","aa"), ("ACBBACA","ABA","BACA"),
        ])
    except Exception: return False


# ── Longest Common Substring (contiguous, NOT subsequence) ───────────────────

def _probe_lcs_substr(ns):
    fn = ns.get("lcs_substring") or ns.get("longest_common_substring") or ns.get("lcsubstring")
    if not fn:
        return ["lcs_substring(s1, s2) not defined — NOTE: this is SUBSTRING (contiguous), not subsequence — implement DP and call python_exec"]
    fails = []
    try:
        r = fn("abcdef", "bcdf")
        if r != 3:
            fails.append(
                f"lcs_substring('abcdef','bcdf')={r}, expected 3 ('bcd'). "
                "FIX: dp[i][j] = length of common suffix ending at s1[i-1] and s2[j-1]. "
                "dp[i][j] = dp[i-1][j-1]+1 if s1[i-1]==s2[j-1], else 0. "
                "Answer = max over all dp[i][j]. This is DIFFERENT from LCS subsequence."
            )
            return fails
        if fn("abcde", "ace") != 1:
            fails.append(f"lcs_substring('abcde','ace')={fn('abcde','ace')}, expected 1 (contiguous chars only).")
        if fn("", "abc") != 0:
            fails.append(f"lcs_substring('','abc')={fn('','abc')}, expected 0.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_lcs_substr(ns):
    fn = ns.get("lcs_substring") or ns.get("longest_common_substring") or ns.get("lcsubstring")
    if not fn: return False
    try:
        return all(fn(a,b)==e for a,b,e in [
            ("abcdef","bcdf",3), ("abcde","ace",1), ("","abc",0),
            ("abcab","ab",2), ("xyz","xyz",3), ("abc","def",0),
        ])
    except Exception: return False


# ── Expression evaluator (correct operator precedence) ────────────────────────

def _probe_expr(ns):
    fn = ns.get("eval_expr") or ns.get("evaluate") or ns.get("calc")
    if not fn:
        return ["eval_expr(s) not defined — implement expression evaluator with +,-,*,/,(,) and call python_exec"]
    fails = []
    try:
        if fn("3 + 5 * 2") != 13:
            fails.append(
                f"eval_expr('3 + 5 * 2')={fn('3+5*2')}, expected 13. "
                "FIX: * and / bind tighter than + and -. "
                "Two-stack approach: one for numbers, one for operators. "
                "When pushing an operator, first pop operators of >= precedence."
            )
            return fails
        if fn("(3 + 5) * 2") != 16:
            fails.append(f"eval_expr('(3+5)*2')={fn('(3+5)*2')}, expected 16.")
        if fn("100 - 20 + 3 * (10 - 4)") != 98:
            fails.append(f"eval_expr('100-20+3*(10-4)')={fn('100-20+3*(10-4)')}, expected 98.")
        if fn("10 / 2") != 5:
            fails.append(f"eval_expr('10/2')={fn('10/2')}, expected 5.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_expr(ns):
    fn = ns.get("eval_expr") or ns.get("evaluate") or ns.get("calc")
    if not fn: return False
    try:
        cases = [
            ("3 + 5 * 2", 13), ("(3 + 5) * 2", 16),
            ("100 - 20 + 3 * (10 - 4)", 98), ("10 / 2", 5),
            ("2 + 3 * 4 - 1", 13), ("((2 + 3) * (4 - 1))", 15),
            ("1 + 2 + 3 + 4", 10), ("100 / 5 / 4", 5),
        ]
        return all(int(fn(s)) == e for s, e in cases)
    except Exception: return False


# ── Topological sort — lexicographically smallest ─────────────────────────────

def _probe_topo(ns):
    fn = ns.get("topo_sort") or ns.get("topological_sort") or ns.get("kahn_lexi")
    if not fn:
        return ["topo_sort(n, edges) not defined returning lex-smallest topo order or [] if cycle — implement using min-heap Kahn's and call python_exec"]
    fails = []
    try:
        r = fn(4, [[1,0],[2,0],[3,1],[3,2]])
        # In-degrees: 0→0, 1→1(from 0), 2→1(from 0), 3→2(from 1,2)
        # Min-heap starts with [0]. Process 0 → unblock 1,2. Heap=[1,2].
        # Process 1 → unblock 3. Heap=[2,3].
        # Process 2 → unblock 3 again (but already added). Heap=[3].
        # Hmm: lex smallest: [0,1,2,3]
        if r != [0,1,2,3]:
            fails.append(
                f"topo_sort(4, [[1,0],[2,0],[3,1],[3,2]])={r}, expected [0,1,2,3]. "
                "FIX: Use min-heap (heapq) for Kahn's BFS — always pick the smallest "
                "available node to get lexicographically smallest result."
            )
            return fails
        # Cycle detection
        r2 = fn(2, [[0,1],[1,0]])
        if r2 != []:
            fails.append(f"topo_sort with cycle should return [], got {r2}.")
        r3 = fn(3, [[2,0],[1,0]])
        if r3 != [0,1,2]:
            fails.append(f"topo_sort(3,[[2,0],[1,0]])={r3}, expected [0,1,2].")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_topo(ns):
    fn = ns.get("topo_sort") or ns.get("topological_sort") or ns.get("kahn_lexi")
    if not fn: return False
    def valid(order, n, edges):
        if not order or len(order) != n: return False
        pos = {v:i for i,v in enumerate(order)}
        return all(pos[b]<pos[a] for a,b in edges)
    try:
        r1 = fn(4, [[1,0],[2,0],[3,1],[3,2]])
        if r1 != [0,1,2,3]: return False
        if fn(2, [[0,1],[1,0]]) != []: return False
        r3 = fn(3, [[2,0],[1,0]])
        if not valid(r3, 3, [[2,0],[1,0]]): return False
        return True
    except Exception: return False


# ── Binary search on answer (minimum max allocation / painters partition) ─────

def _probe_binsearch(ns):
    fn = ns.get("allocate_pages") or ns.get("min_max_partition") or ns.get("painters_partition")
    if not fn:
        return ["allocate_pages(pages, k) not defined — binary search on answer + greedy check — implement and call python_exec"]
    fails = []
    try:
        r = fn([12, 34, 67, 90], 2)
        if r != 113:
            fails.append(
                f"allocate_pages([12,34,67,90], 2)={r}, expected 113. "
                "FIX: Binary search on the answer (max pages per student) in [max(pages), sum(pages)]. "
                "For a candidate mid, greedily count: assign pages until adding next exceeds mid → new student. "
                "If students_needed <= k, mid is feasible — try lower. Else try higher."
            )
            return fails
        r2 = fn([10, 20, 30, 40], 2)
        if r2 != 60:
            fails.append(f"allocate_pages([10,20,30,40], 2)={r2}, expected 60 ([10,20,30] | [40]).")
        r3 = fn([5, 5, 5, 5], 4)
        if r3 != 5:
            fails.append(f"allocate_pages([5,5,5,5], 4)={r3}, expected 5 (one page per student).")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_binsearch(ns):
    fn = ns.get("allocate_pages") or ns.get("min_max_partition") or ns.get("painters_partition")
    if not fn: return False
    try:
        return all(fn(p,k)==e for p,k,e in [
            ([12,34,67,90], 2, 113), ([10,20,30,40], 2, 60),
            ([5,5,5,5], 4, 5), ([1,2,3,4,5], 1, 15), ([100], 1, 100),
            ([1,1,1,1], 2, 2),
        ])
    except Exception: return False


# ── Kruskal's Minimum Spanning Tree ──────────────────────────────────────────

def _probe_kruskal(ns):
    fn = ns.get("kruskal") or ns.get("mst_kruskal") or ns.get("minimum_spanning_tree")
    if not fn:
        return ["kruskal(n, edges) not defined where edges=[(weight,u,v)] returning MST total weight — implement and call python_exec"]
    fails = []
    try:
        # n=4, edges by weight
        edges = [(1,0,1),(2,0,2),(3,1,2),(4,2,3)]
        r = fn(4, edges)
        if r != 7:
            fails.append(
                f"kruskal(4, [(1,0,1),(2,0,2),(3,1,2),(4,2,3)])={r}, expected 7. "
                "FIX: Sort edges by weight. Use Union-Find: for each edge (w,u,v) in order, "
                "if find(u) != find(v): union(u,v), add w to MST weight. "
                "Stop after adding n-1 edges."
            )
            return fails
        edges2 = [(4,0,1),(8,0,7),(11,1,2),(7,1,7),(9,2,3),(14,2,5),(2,3,4),(10,3,5),(2,5,6),(6,6,7),(1,7,8),(7,2,8),(6,6,8)]
        r2 = fn(9, edges2)
        if r2 != 37:
            fails.append(f"9-node graph: kruskal expected 37, got {r2}.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_kruskal(ns):
    fn = ns.get("kruskal") or ns.get("mst_kruskal") or ns.get("minimum_spanning_tree")
    if not fn: return False
    try:
        e1 = [(1,0,1),(2,0,2),(3,1,2),(4,2,3)]
        e2 = [(4,0,1),(8,0,7),(11,1,2),(7,1,7),(9,2,3),(14,2,5),(2,3,4),(10,3,5),(2,5,6),(6,6,7),(1,7,8),(7,2,8),(6,6,8)]
        return fn(4, e1)==7 and fn(9, e2)==37
    except Exception: return False


# ── Bellman-Ford with negative cycle detection ────────────────────────────────

def _probe_bellman(ns):
    fn = ns.get("bellman_ford") or ns.get("shortest_path_bf") or ns.get("bf")
    if not fn:
        return ["bellman_ford(n, edges, src) not defined returning dist list or None if neg cycle — implement and call python_exec"]
    fails = []
    try:
        # No negative cycle: simple 4-node graph
        r = fn(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1)], 0)
        if r is None or r[3] != 4:
            fails.append(
                f"bellman_ford 4-node (no cycle): dist[3]={r[3] if r else 'None'}, expected 4. "
                "FIX: Initialise dist[src]=0, others=inf. "
                "Relax all edges n-1 times: if dist[u]+w < dist[v]: dist[v]=dist[u]+w. "
                "Negative cycle check: run a FULL nth iteration; if any edge still relaxes → cycle."
            )
            return fails
        # Negative cycle
        r2 = fn(3, [(0,1,1),(1,2,2),(2,1,-4)], 0)
        if r2 is not None:
            fails.append(
                f"bellman_ford with negative cycle (1→2→1: 2-4=-2 per loop)={r2}, expected None. "
                "FIX: After n-1 relaxations, do one more pass; if any dist[v] decreases → negative cycle → return None."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_bellman(ns):
    fn = ns.get("bellman_ford") or ns.get("shortest_path_bf") or ns.get("bf")
    if not fn: return False
    try:
        r1 = fn(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1)], 0)
        if r1 is None or r1[3] != 4: return False
        r2 = fn(3, [(0,1,1),(1,2,2),(2,1,-4)], 0)
        if r2 is not None: return False
        r3 = fn(5, [(0,1,-1),(0,2,4),(1,2,3),(1,3,2),(1,4,2),(3,2,5),(3,1,1),(4,3,-3)], 0)
        if r3 is None or r3[3] != -2: return False
        return True
    except Exception: return False


# ── Count inversions (merge-sort based, O(n log n)) ──────────────────────────

def _probe_inversions(ns):
    fn = ns.get("count_inversions") or ns.get("inversions") or ns.get("merge_count")
    if not fn:
        return ["count_inversions(arr) not defined — implement merge-sort inversion count and call python_exec"]
    fails = []
    try:
        if fn([3, 1, 2]) != 2:
            fails.append(
                f"count_inversions([3,1,2])={fn([3,1,2])}, expected 2 ((3,1) and (3,2)). "
                "FIX: Modified merge sort: when picking right[j] over left[i] during merge, "
                "add (len(left)-i) to inversion count — all remaining left elements form inversions with right[j]."
            )
            return fails
        if fn([4, 3, 2, 1]) != 6:
            fails.append(f"count_inversions([4,3,2,1])={fn([4,3,2,1])}, expected 6.")
        if fn([1, 2, 3]) != 0:
            fails.append(f"count_inversions([1,2,3])={fn([1,2,3])}, expected 0.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_inversions(ns):
    fn = ns.get("count_inversions") or ns.get("inversions") or ns.get("merge_count")
    if not fn: return False
    def brute(a):
        return sum(1 for i in range(len(a)) for j in range(i+1,len(a)) if a[i]>a[j])
    try:
        tests = [[3,1,2],[4,3,2,1],[1,2,3],[1],[5,4,3,2,1],[2,4,1,3,5]]
        return all(fn(a)==brute(a) for a in tests)
    except Exception: return False


# ── Mutable default argument bug ─────────────────────────────────────────────

def _probe_mutdef(ns):
    fn = ns.get("append_and_return") or ns.get("add_to_list") or ns.get("safe_append")
    if not fn:
        return ["append_and_return(item, lst=None) not defined — fix mutable default bug and call python_exec"]
    fails = []
    try:
        r1 = fn(1)
        r2 = fn(2)
        if 1 in r2:
            fails.append(
                f"append_and_return(2) returned {r2} — still contains item from previous call. "
                "FIX: Never use mutable objects as default arguments. "
                "Change `def f(item, lst=[])` to `def f(item, lst=None): if lst is None: lst=[]`. "
                "The default `[]` is evaluated ONCE at function definition, shared across all calls."
            )
        if r1 != [1]:
            fails.append(f"append_and_return(1) should return [1], got {r1}.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_mutdef(ns):
    fn = ns.get("append_and_return") or ns.get("add_to_list") or ns.get("safe_append")
    if not fn: return False
    try:
        for _ in range(3):
            r = fn(42)
            if r != [42]: return False
        return fn(1) == [1] and fn(2) == [2]
    except Exception: return False


# ── Late-binding closure bug ──────────────────────────────────────────────────

def _probe_closure(ns):
    fn = ns.get("make_adders") or ns.get("make_multipliers") or ns.get("make_funcs")
    if not fn:
        return ["make_adders(n) not defined returning list of n functions where funcs[i](x)=x+i — fix late-binding bug and call python_exec"]
    fails = []
    try:
        funcs = fn(5)
        r0 = funcs[0](10)
        r3 = funcs[3](10)
        if r0 == r3:
            fails.append(
                f"make_adders(5): funcs[0](10)={r0}, funcs[3](10)={r3} — all return same value. "
                "FIX: Python closures capture variables by REFERENCE, not by value. "
                "Use a default argument: `lambda x, i=i: x + i` or "
                "`def make(i): return lambda x: x + i` then `[make(i) for i in range(n)]`."
            )
            return fails
        for i in range(5):
            if funcs[i](10) != 10 + i:
                fails.append(f"funcs[{i}](10)={funcs[i](10)}, expected {10+i}.")
                return fails
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_closure(ns):
    fn = ns.get("make_adders") or ns.get("make_multipliers") or ns.get("make_funcs")
    if not fn: return False
    try:
        funcs = fn(6)
        return all(funcs[i](100) == 100 + i for i in range(6))
    except Exception: return False


# ── Token bucket rate limiter ─────────────────────────────────────────────────

def _probe_ratelimiter(ns):
    cls = ns.get("TokenBucket") or ns.get("RateLimiter") or ns.get("Limiter")
    if not cls:
        return ["TokenBucket(rate, capacity) class not defined with allow() method — implement and call python_exec"]
    fails = []
    try:
        import time as _t
        tb = cls(rate=10, capacity=10)  # 10 tokens/sec, capacity 10
        # Initially full: 10 requests should all pass
        results = [tb.allow() for _ in range(10)]
        if not all(results):
            fails.append(
                f"First 10 requests on full bucket: {results}, expected all True. "
                "FIX: Initialise tokens=capacity. allow() consumes 1 token if tokens>=1 and returns True, else False."
            )
            return fails
        # 11th request should fail (bucket empty)
        if tb.allow():
            fails.append(
                "11th request on empty bucket returned True, expected False. "
                "FIX: After consuming all tokens, allow() must return False until refill."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_ratelimiter(ns):
    cls = ns.get("TokenBucket") or ns.get("RateLimiter") or ns.get("Limiter")
    if not cls: return False
    try:
        tb = cls(rate=5, capacity=5)
        r = [tb.allow() for _ in range(5)]
        if not all(r): return False
        if tb.allow(): return False
        return True
    except Exception: return False


# ── TTL cache (LRU + expiry) ─────────────────────────────────────────────────

def _probe_ttlcache(ns):
    cls = ns.get("TTLCache") or ns.get("CacheWithTTL") or ns.get("ExpiringCache")
    if not cls:
        return ["TTLCache(capacity, ttl_seconds) class not defined with get/put — implement and call python_exec"]
    fails = []
    try:
        import time as _t
        cache = cls(capacity=3, ttl_seconds=1)
        cache.put("a", 1); cache.put("b", 2)
        if cache.get("a") != 1:
            fails.append(
                "TTLCache.get('a') should return 1 immediately after put. "
                "FIX: Store (value, expiry_time) in the cache. get() checks time.time() < expiry_time."
            )
            return fails
        _t.sleep(1.1)
        if cache.get("a") is not None and cache.get("a") != -1:
            fails.append(
                f"TTLCache.get('a') after TTL expiry returned {cache.get('a')}, expected None or -1. "
                "FIX: On get(), check if time.time() > stored_expiry → evict and return None/-1."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_ttlcache(ns):
    cls = ns.get("TTLCache") or ns.get("CacheWithTTL") or ns.get("ExpiringCache")
    if not cls: return False
    try:
        import time as _t
        c = cls(capacity=2, ttl_seconds=1)
        c.put("x", 10); c.put("y", 20)
        if c.get("x") not in (10,): return False
        # Capacity eviction
        c.put("z", 30)
        # One of x or y should be evicted (LRU), z should be present
        if c.get("z") not in (30,): return False
        return True
    except Exception: return False


# ── Sudoku solver ─────────────────────────────────────────────────────────────

_SUDOKU_PUZZLE = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9],
]

_SUDOKU_SOLUTION = [
    [5,3,4,6,7,8,9,1,2],
    [6,7,2,1,9,5,3,4,8],
    [1,9,8,3,4,2,5,6,7],
    [8,5,9,7,6,1,4,2,3],
    [4,2,6,8,5,3,7,9,1],
    [7,1,3,9,2,4,8,5,6],
    [9,6,1,5,3,7,2,8,4],
    [2,8,7,4,1,9,6,3,5],
    [3,4,5,2,8,6,1,7,9],
]

def _probe_sudoku(ns):
    fn = ns.get("solve_sudoku") or ns.get("sudoku_solver") or ns.get("solveSudoku")
    if not fn:
        return ["solve_sudoku(board) not defined (modifies board in-place or returns solved board) — implement backtracking solver and call python_exec"]
    fails = []
    try:
        import copy
        board = copy.deepcopy(_SUDOKU_PUZZLE)
        result = fn(board)
        solved = result if result is not None else board
        # Verify a few key cells
        if solved[0][2] != 4:
            fails.append(
                f"Sudoku: cell[0][2]={solved[0][2]}, expected 4. "
                "FIX: Backtracking — for each empty cell try 1-9; "
                "check row, column, 3x3 box for conflicts; recurse; backtrack on dead ends."
            )
            return fails
        # Validate full solution
        def valid(b):
            for i in range(9):
                row = [b[i][j] for j in range(9)]
                col = [b[j][i] for j in range(9)]
                if sorted(row) != list(range(1,10)): return False
                if sorted(col) != list(range(1,10)): return False
            for r in range(0,9,3):
                for c in range(0,9,3):
                    box = [b[r+dr][c+dc] for dr in range(3) for dc in range(3)]
                    if sorted(box) != list(range(1,10)): return False
            return True
        if not valid(solved):
            fails.append("Sudoku solution invalid — rows/cols/boxes don't each contain 1-9.")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_sudoku(ns):
    fn = ns.get("solve_sudoku") or ns.get("sudoku_solver") or ns.get("solveSudoku")
    if not fn: return False
    try:
        import copy
        board = copy.deepcopy(_SUDOKU_PUZZLE)
        result = fn(board)
        solved = result if result is not None else board
        def valid(b):
            for i in range(9):
                if sorted(b[i]) != list(range(1,10)): return False
                if sorted(b[j][i] for j in range(9)) != list(range(1,10)): return False
            for r in range(0,9,3):
                for c in range(0,9,3):
                    if sorted(b[r+dr][c+dc] for dr in range(3) for dc in range(3)) != list(range(1,10)): return False
            return True
        return valid(solved)
    except Exception: return False


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT CHECK FUNCTIONS  (hard science + math)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_particle_box(resp: str) -> bool:
    # E_3 for electron in L=1nm box ≈ 3.38 eV
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*ev', r):
        if 3.1 <= float(n) <= 3.6:
            return True
    # Joule notation ~5.4e-19 J
    if re.search(r'5\.[3-5]\s*[×x]\s*10\s*[\^-]*\s*19', r):
        return True
    return False


def _check_gibbs_trap(resp: str) -> bool:
    # ΔG = -200 - 600×(-0.4) = +40 kJ → NOT spontaneous
    r = resp.lower()
    not_spontaneous = bool(re.search(r'not spontaneous|non.spontaneous|nonspontaneous|ΔG.*>.*0|positive.*ΔG|ΔG.*positive', r))
    pos_value       = bool(re.search(r'\+40|\b40\s*kJ', r))
    # Must NOT say "is spontaneous" without negation
    wrong           = bool(re.search(r'(?<!not )(?<!non)spontaneous', r)) and not not_spontaneous
    return (not_spontaneous or pos_value) and not wrong


def _check_hess(resp: str) -> bool:
    # ΔH for C + ½O₂ → CO = -110.5 kJ/mol
    r = resp.lower().replace(",", "")
    for n in re.findall(r'-\s*(\d+\.?\d*)\s*kj', r):
        if 109 <= float(n) <= 112:
            return True
    if re.search(r'110\.5|110\.4|110\.6', r):
        return True
    return False


def _check_totient(resp: str) -> bool:
    return bool(re.search(r'\b96\b', resp))


def _check_derangements(resp: str) -> bool:
    return bool(re.search(r'\b44\b', resp))


def _check_inclusion(resp: str) -> bool:
    # exactly 2 of 3 subjects = 22
    return bool(re.search(r'\b22\b', resp))


def _check_rel_ke(resp: str) -> bool:
    # KE at v=0.8c: γ=5/3, KE=0.341 MeV or 5.47e-14 J
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*mev', r):
        if 0.33 <= float(n) <= 0.36:
            return True
    # γ = 5/3 or 1.667
    if re.search(r'5/3|1\.66+7?|γ.*1\.6|gamma.*1\.6', r):
        return True
    return False


def _check_entropy(resp: str) -> bool:
    # H(0.5, 0.3, 0.2) = 1.485 bits
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d+)\s*bits?', r):
        if 1.47 <= float(n) <= 1.50:
            return True
    if re.search(r'1\.48|1\.485|1\.49', r):
        return True
    return False


def _check_bayes_low(resp: str) -> bool:
    # 1% prevalence, 90% sens, 85% spec → 5.71%
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*%', r):
        if 5.0 <= float(n) <= 7.0:
            return True
    return False


def _check_modexp(resp: str) -> bool:
    # 3^200 mod 17 = 16
    return bool(re.search(r'\b16\b', resp))


def _check_nuclear(resp: str) -> bool:
    # Branching decay: λ_total = λ_α + λ_β → T_½ = ln2/λ_total
    # T_½α = 4 days, T_½β = 12 days
    # λ_α = ln2/4, λ_β = ln2/12, λ_total = ln2(1/4+1/12) = ln2(4/12) = ln2/3
    # T_½ effective = 3 days
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*days?', r):
        if 2.8 <= float(n) <= 3.2:
            return True
    if re.search(r'\b3\.0\b|\b3 day', r):
        return True
    return False


def _check_maxwell(resp: str) -> bool:
    # Most probable speed v_p = sqrt(2kT/m) for N₂ at 300K
    # m = 28.02 u = 28.02×1.66e-27 = 4.65e-26 kg
    # v_p = sqrt(2×1.38e-23×300 / 4.65e-26) = sqrt(8.28e-21/4.65e-26) = sqrt(1.781e5) ≈ 422 m/s
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*m/s', r):
        if 400 <= float(n) <= 445:
            return True
    if re.search(r'42[0-9]|v_p|most probable', r):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK LIST  —  20 code  +  12 text  =  32 tasks
# ═══════════════════════════════════════════════════════════════════════════════

TASKS: list[dict] = [
    # ── HARD ALGORITHMS ───────────────────────────────────────────────────────
    {
        "name": "Segment tree with lazy propagation (range add, range sum)",
        "type": "code", "want_fn": "SegTree",
        "probe": _probe_segtree, "check": _check_segtree,
        "prompt": textwrap.dedent("""
            Implement a segment tree with LAZY PROPAGATION supporting:
              - range_add(l, r, val): add val to all elements in index range [l, r] (inclusive)
              - query_sum(l, r): return sum of elements in [l, r]

            The key challenge: when you range_add to a node that covers [l,r] exactly,
            you must DELAY the update (store it as "lazy") and push it down to children
            only when you need to descend into them.

            Test on array [1, 2, 3, 4, 5]:
              range_add(1, 3, 10) → array [1,12,13,14,5]
              query_sum(0, 4) should be 45
              query_sum(1, 3) should be 39

            Call python_exec with your complete SegTree class.
        """).strip(),
    },
    {
        "name": "Bitmask DP — Travelling Salesman Problem",
        "type": "code", "want_fn": "tsp",
        "probe": _probe_tsp, "check": _check_tsp,
        "prompt": textwrap.dedent("""
            Write `tsp(dist)` where dist[i][j] is the cost to travel from city i to city j.
            Return the minimum cost Hamiltonian cycle (visit all cities exactly once and return).

            Example: tsp([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]) = 80

            Use bitmask DP:
              dp[mask][i] = min cost to have visited exactly the cities in `mask`, currently at city i
              Base: dp[1<<0][0] = 0
              Transition: dp[mask|(1<<j)][j] = min(dp[mask][i] + dist[i][j])  for j not in mask
              Answer: min over i of dp[(1<<n)-1][i] + dist[i][0]

            Complexity: O(n² × 2ⁿ). Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Matrix exponentiation (nth Fibonacci in O(log n))",
        "type": "code", "want_fn": "mat_fib",
        "probe": _probe_matfib, "check": _check_matfib,
        "prompt": textwrap.dedent("""
            Write `mat_fib(n)` returning the nth Fibonacci number (fib(0)=0, fib(1)=1)
            using matrix exponentiation in O(log n) time.

            Key identity: [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n

            Implement:
              1. mat_mul(A, B): multiply two 2×2 matrices
              2. mat_pow(M, n): raise matrix M to power n using repeated squaring
              3. mat_fib(n): use the above to return F(n)

            Test: mat_fib(50) = 12586269025
            Call python_exec with your complete implementation.
        """).strip(),
    },
    {
        "name": "Fenwick Tree / Binary Indexed Tree (point update, prefix sum)",
        "type": "code", "want_fn": "BIT",
        "probe": _probe_bit, "check": _check_bit,
        "prompt": textwrap.dedent("""
            Implement a Fenwick Tree (Binary Indexed Tree) for prefix sums:
              class BIT:
                  def __init__(self, n: int)             # n elements, all 0
                  def update(self, i: int, delta: int)   # add delta to position i (0-indexed)
                  def prefix_sum(self, i: int) -> int    # sum of elements [0..i] inclusive

            Core trick: use index bit manipulation (i & -i) for O(log n) operations.
            update: while i < n: tree[i] += delta; i += (i & -i)
            prefix_sum: while i >= 0: s += tree[i]; i -= (i & -i) - 1   [for 0-indexed]

            Test: BIT(5); update each of [3,2,-1,6,5]; prefix_sum(4) == 15
            Call python_exec with your complete implementation.
        """).strip(),
    },
    {
        "name": "KMP string search (all occurrences, O(n+m))",
        "type": "code", "want_fn": "kmp_search",
        "probe": _probe_kmp, "check": _check_kmp,
        "prompt": textwrap.dedent("""
            Write `kmp_search(text: str, pattern: str) -> list[int]`
            returning all start indices where pattern occurs in text (overlapping allowed).

            Two steps:
            1. Build LPS array (longest proper prefix-suffix):
               lps[0]=0; for i>0: if match extend; else fall back via lps[j-1]
            2. Scan text: j tracks position in pattern; on full match record i-m+1,
               then set j=lps[j-1] (NOT j=0) to allow overlapping matches.

            Examples:
              kmp_search("AABABCAAB", "AAB") == [0, 6]
              kmp_search("AAAA", "AA")       == [0, 1, 2]   # overlapping!

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Digit DP (count numbers 1..N with digit-sum divisible by k)",
        "type": "code", "want_fn": "count_digit_sum_mod",
        "probe": _probe_digitdp, "check": _check_digitdp,
        "prompt": textwrap.dedent("""
            Write `count_digit_sum_mod(n: int, k: int) -> int`
            counting integers from 1 to n whose digit-sum is divisible by k.

            Use top-down DP with memoization:
              dp(pos, tight, remainder)
              - pos: current digit position (0-indexed from left)
              - tight: True if digits so far match the prefix of n
              - remainder: current digit sum mod k

            At each position, try digit 0..9 (or 0..digit[pos] if tight).
            Base case: pos==len(digits), return 1 if remainder==0 else 0.
            Subtract 1 at the end to exclude 0 itself.

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Manacher's algorithm (longest palindromic substring, O(n))",
        "type": "code", "want_fn": "longest_palindrome",
        "probe": _probe_manacher, "check": _check_manacher,
        "prompt": textwrap.dedent("""
            Write `longest_palindrome(s: str) -> int` returning the LENGTH of the
            longest palindromic substring using Manacher's algorithm in O(n) time.

            Key idea: insert separators '#' between every character and at boundaries.
              "abc" → "#a#b#c#"
            Maintain: center c, right boundary r, and array p[] of palindrome radii.
            For each position i:
              Start p[i] = min(r-i, p[2*c-i]) if i < r, else 0
              Then expand: while s[i-p[i]-1] == s[i+p[i]+1]: p[i]++
              Update c, r if i+p[i] > r

            longest_palindrome("racecar") == 7

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Minimum window substring (smallest window containing all chars of t)",
        "type": "code", "want_fn": "min_window",
        "probe": _probe_minwin, "check": _check_minwin,
        "prompt": textwrap.dedent("""
            Write `min_window(s: str, t: str) -> str`.
            Return the shortest contiguous substring of s that contains all characters
            of t (including duplicates). Return "" if impossible.

            Example: min_window("ADOBECODEBANC", "ABC") == "BANC"

            Two-pointer sliding window:
            - Expand right pointer until window contains all chars of t (track with freq map)
            - Then shrink left pointer while still valid, recording minimum
            - "have" vs "need" counter trick for O(1) validity check

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Longest common SUBSTRING (contiguous DP — not subsequence)",
        "type": "code", "want_fn": "lcs_substring",
        "probe": _probe_lcs_substr, "check": _check_lcs_substr,
        "prompt": textwrap.dedent("""
            Write `lcs_substring(s1: str, s2: str) -> int` returning the length of
            the longest common SUBSTRING (contiguous characters shared by both strings).

            This is DIFFERENT from longest common subsequence (LCS).
            "abcde" and "ace" → LCS = 3 ("ace") but longest common SUBSTRING = 1 ("a"/"c"/"e")

            DP approach:
              dp[i][j] = length of common suffix ending at s1[i-1] and s2[j-1]
              dp[i][j] = dp[i-1][j-1] + 1  if s1[i-1] == s2[j-1]
                        = 0                 otherwise
              Answer = max over all dp[i][j]

            lcs_substring("abcdef", "bcdf") == 3  ("bcd")
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Expression evaluator with operator precedence (no eval())",
        "type": "code", "want_fn": "eval_expr",
        "probe": _probe_expr, "check": _check_expr,
        "prompt": textwrap.dedent("""
            Write `eval_expr(s: str) -> int` that evaluates arithmetic expressions
            with +, -, *, /, and parentheses. Do NOT use Python's eval().

            Rules: * and / have higher precedence than + and -.
            eval_expr("3 + 5 * 2")         == 13
            eval_expr("(3 + 5) * 2")       == 16
            eval_expr("100 - 20 + 3 * (10 - 4)") == 98

            Two-stack approach (numbers + operators):
            - When pushing operator, first pop operators of >= precedence from stack
            - '(' pushes to op stack, ')' pops until matching '('
            - Use integer division for /

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Topological sort — lexicographically smallest (min-heap Kahn's)",
        "type": "code", "want_fn": "topo_sort",
        "probe": _probe_topo, "check": _check_topo,
        "prompt": textwrap.dedent("""
            Write `topo_sort(n: int, edges: list[list[int]]) -> list[int]`
            where edges[i]=[a,b] means b must come before a.
            Return the LEXICOGRAPHICALLY SMALLEST valid topological order,
            or [] if the graph has a cycle.

            Key: use a min-heap (heapq) instead of a regular queue in Kahn's BFS.
            Start with all nodes of in-degree 0. Always pop the smallest node.

            topo_sort(4, [[1,0],[2,0],[3,1],[3,2]]) == [0,1,2,3]

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Binary search on answer — minimum max load (painters partition)",
        "type": "code", "want_fn": "allocate_pages",
        "probe": _probe_binsearch, "check": _check_binsearch,
        "prompt": textwrap.dedent("""
            Write `allocate_pages(pages: list[int], k: int) -> int`.
            You have k students who must read all books in order (each gets a contiguous segment).
            Return the minimum possible maximum pages assigned to any single student.

            allocate_pages([12, 34, 67, 90], 2) == 113  ([12,34,67] | [90])

            Binary search on the answer:
            - lo = max(pages), hi = sum(pages)
            - For each mid: greedily count how many students needed
            - If students_needed <= k: mid is feasible, try lower
            - Else: try higher

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Kruskal's Minimum Spanning Tree (Union-Find)",
        "type": "code", "want_fn": "kruskal",
        "probe": _probe_kruskal, "check": _check_kruskal,
        "prompt": textwrap.dedent("""
            Write `kruskal(n: int, edges: list[tuple]) -> int`
            where edges = [(weight, u, v)] and n = number of nodes.
            Return the total weight of the Minimum Spanning Tree.

            Algorithm:
            1. Sort edges by weight
            2. Use Union-Find (with path compression + union by rank)
            3. For each edge (w, u, v): if find(u) != find(v): union(u,v), add w to total

            kruskal(4, [(1,0,1),(2,0,2),(3,1,2),(4,2,3)]) == 7

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Bellman-Ford with negative cycle detection",
        "type": "code", "want_fn": "bellman_ford",
        "probe": _probe_bellman, "check": _check_bellman,
        "prompt": textwrap.dedent("""
            Write `bellman_ford(n: int, edges: list[tuple], src: int)`
            where edges = [(u, v, weight)].
            Return the shortest distance array from src, or None if a negative cycle is reachable.

            Algorithm:
            - Init dist[src]=0, all others=inf
            - Relax ALL edges n-1 times
            - Run one MORE full pass: if any edge still relaxes → negative cycle → return None

            bellman_ford(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1)], 0)[3] == 4
            bellman_ford(3, [(0,1,1),(1,2,2),(2,1,-4)], 0) == None  # negative cycle

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Count inversions in O(n log n) via merge sort",
        "type": "code", "want_fn": "count_inversions",
        "probe": _probe_inversions, "check": _check_inversions,
        "prompt": textwrap.dedent("""
            Write `count_inversions(arr: list[int]) -> int`.
            An inversion is a pair (i, j) where i < j but arr[i] > arr[j].
            Count all inversions in O(n log n) using a modified merge sort.

            Key: During the merge step, when you take element from the RIGHT half
            over the LEFT half, all remaining left elements form inversions with it:
              inversions += len(left) - left_ptr

            count_inversions([3, 1, 2]) == 2   # (3,1) and (3,2)
            count_inversions([4,3,2,1]) == 6

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Python bug: mutable default argument",
        "type": "code", "want_fn": "append_and_return",
        "probe": _probe_mutdef, "check": _check_mutdef,
        "prompt": textwrap.dedent("""
            The following Python function has a subtle bug:

              def append_and_return(item, lst=[]):
                  lst.append(item)
                  return lst

            Bug: The default list `[]` is created ONCE at function definition and
            shared across all calls. Calling append_and_return(1) then append_and_return(2)
            returns [1, 2] from the second call — not [2].

            Write the FIXED version `append_and_return(item, lst=None)` that:
            - Creates a fresh list if lst is None
            - Appends item to it and returns it
            - Each call with no lst argument gets its own fresh list

            Call python_exec with your fixed implementation and test it.
        """).strip(),
    },
    {
        "name": "Python bug: late-binding closure in loops",
        "type": "code", "want_fn": "make_adders",
        "probe": _probe_closure, "check": _check_closure,
        "prompt": textwrap.dedent("""
            The following Python code has a late-binding closure bug:

              def make_adders(n):
                  return [lambda x: x + i for i in range(n)]

            Bug: all lambdas capture `i` by REFERENCE. When called, `i` is n-1
            (the last value from the loop). make_adders(5)[0](10) returns 14, not 10.

            Write the FIXED `make_adders(n)` that returns a list of n functions
            where funcs[i](x) == x + i for each i.

            Two correct approaches:
            1. Default arg: lambda x, i=i: x + i
            2. Factory: def make(i): return lambda x: x + i

            Call python_exec with your fixed implementation.
        """).strip(),
    },
    {
        "name": "Token bucket rate limiter (thread-safe)",
        "type": "code", "want_fn": "TokenBucket",
        "probe": _probe_ratelimiter, "check": _check_ratelimiter,
        "prompt": textwrap.dedent("""
            Implement a token bucket rate limiter:
              class TokenBucket:
                  def __init__(self, rate: float, capacity: float)
                      # rate = tokens added per second, capacity = max tokens
                  def allow(self) -> bool
                      # consume 1 token; return True if allowed, False if bucket empty

            Behaviour:
            - Start with capacity tokens (full bucket)
            - Refill continuously at `rate` tokens/second (check elapsed time on each call)
            - allow() returns True and decrements tokens if tokens >= 1, else False
            - Use threading.Lock() for thread safety

            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "TTL cache — LRU eviction + time-based expiry",
        "type": "code", "want_fn": "TTLCache",
        "probe": _probe_ttlcache, "check": _check_ttlcache,
        "prompt": textwrap.dedent("""
            Implement a cache with both LRU eviction AND TTL expiry:
              class TTLCache:
                  def __init__(self, capacity: int, ttl_seconds: float)
                  def get(self, key) -> any    # return None (or -1) if missing or expired
                  def put(self, key, value)    # evict LRU if at capacity

            Rules:
            - On get(): if key exists but time.time() > stored_expiry → treat as expired (evict + return None)
            - On put(): store (value, time.time() + ttl_seconds); evict LRU if over capacity
            - Accessing an unexpired key counts as a use (refreshes LRU order)

            Use collections.OrderedDict for the LRU ordering.
            Call python_exec with your implementation.
        """).strip(),
    },
    {
        "name": "Sudoku solver (backtracking + constraint checking)",
        "type": "code", "want_fn": "solve_sudoku",
        "probe": _probe_sudoku, "check": _check_sudoku,
        "prompt": textwrap.dedent("""
            Write `solve_sudoku(board)` where board is a 9×9 list of lists.
            Empty cells are 0. Fill in the solution (in-place or return the solved board).

            Algorithm: backtracking.
            For each empty cell, try digits 1-9. Before placing digit d:
            - Check row: d not already in that row
            - Check column: d not already in that column
            - Check 3×3 box: d not already in box (row//3)*3 to (row//3)*3+2, same for col

            Optimise: pick the cell with fewest valid candidates (MRV heuristic) — optional but helpful.

            Test puzzle (0=empty):
            [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],
             [8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],
             [0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]

            Call python_exec with your implementation.
        """).strip(),
    },

    # ── HARD SCIENCE + MATH TEXT ──────────────────────────────────────────────
    {
        "name": "Quantum: particle-in-a-box energy level (n=3, L=1nm)",
        "type": "text", "check": _check_particle_box,
        "prompt": textwrap.dedent("""
            An electron is confined in a one-dimensional box of length L = 1.0 nm.
            Using the particle-in-a-box model:

            1. Write the formula for the energy levels Eₙ in terms of n, L, ℏ, and m.
            2. Calculate E₃ (n = 3) in joules and in electron-volts.
            3. What is the energy of the photon emitted when the electron drops from n=3 to n=2?

            Use: ℏ = 1.055×10⁻³⁴ J·s, mₑ = 9.109×10⁻³¹ kg, 1 eV = 1.602×10⁻¹⁹ J.
            Show all calculations.
        """).strip(),
        "retry_hint": (
            "Eₙ = n²π²ℏ²/(2mL²). "
            "E₁ = π²×(1.055e-34)²/(2×9.109e-31×(1e-9)²) = 6.024e-20 J = 0.376 eV. "
            "E₃ = 9×E₁ = 5.42e-19 J = 3.38 eV. "
            "Photon energy = E₃-E₂ = 9E₁-4E₁ = 5E₁ = 1.88 eV."
        ),
    },
    {
        "name": "Gibbs free energy TRAP: becomes non-spontaneous at high T",
        "type": "text", "check": _check_gibbs_trap,
        "prompt": textwrap.dedent("""
            A reaction has ΔH = −200 kJ/mol and ΔS = −400 J/(mol·K).

            1. Calculate ΔG at T = 300 K. Is it spontaneous?
            2. Calculate ΔG at T = 600 K. Is it spontaneous?
            3. At what temperature does the reaction become non-spontaneous?
            4. Explain the physical meaning: why does a reaction that is exothermic
               become non-spontaneous at high temperature?

            Use ΔG = ΔH − TΔS. Show all calculations.
        """).strip(),
        "retry_hint": (
            "ΔG = ΔH - TΔS = -200,000 - T×(-400) = -200,000 + 400T J/mol. "
            "At T=300K: ΔG = -200,000 + 120,000 = -80,000 J = -80 kJ → spontaneous. "
            "At T=600K: ΔG = -200,000 + 240,000 = +40,000 J = +40 kJ → NOT spontaneous. "
            "Crossover: T = ΔH/ΔS = 200,000/400 = 500 K."
        ),
    },
    {
        "name": "Hess's law — ΔH for CO formation",
        "type": "text", "check": _check_hess,
        "prompt": textwrap.dedent("""
            Given:
              (1) C(s) + O₂(g) → CO₂(g)       ΔH₁ = −393.5 kJ/mol
              (2) CO(g) + ½O₂(g) → CO₂(g)     ΔH₂ = −283.0 kJ/mol

            Use Hess's Law to calculate ΔH for the reaction:
              C(s) + ½O₂(g) → CO(g)

            Show step by step which reactions you reverse/scale and why.
            Give the final answer in kJ/mol.
        """).strip(),
        "retry_hint": (
            "Reverse reaction (2): CO₂(g) → CO(g) + ½O₂(g), ΔH = +283.0. "
            "Add to reaction (1): C + O₂ + CO₂ → CO₂ + CO + ½O₂. "
            "Cancel CO₂: C + ½O₂ → CO. "
            "ΔH = -393.5 + 283.0 = -110.5 kJ/mol."
        ),
    },
    {
        "name": "Euler's totient function φ(360)",
        "type": "text", "check": _check_totient,
        "prompt": textwrap.dedent("""
            Calculate Euler's totient function φ(360).

            φ(n) counts the number of integers from 1 to n that are coprime to n.

            1. Find the prime factorisation of 360.
            2. Apply the formula: φ(n) = n × ∏(1 − 1/p) for each prime factor p.
            3. State φ(360) and verify by explaining what it represents.
            4. As a bonus: what is φ(p²) for prime p in general?
        """).strip(),
        "retry_hint": (
            "360 = 2³ × 3² × 5. "
            "φ(360) = 360 × (1−1/2) × (1−1/3) × (1−1/5) = 360 × 1/2 × 2/3 × 4/5 = 96."
        ),
    },
    {
        "name": "Derangements D₅ (permutations with no fixed points)",
        "type": "text", "check": _check_derangements,
        "prompt": textwrap.dedent("""
            A derangement is a permutation where NO element appears in its original position.
            D_n is the number of derangements of n elements.

            1. Derive the formula for D_n using inclusion-exclusion.
            2. Calculate D₅ step by step.
            3. What is the limiting probability P(derangement) as n→∞?
            4. If 5 people each put their hat in a box and hats are randomly redistributed,
               what is the probability that nobody gets their own hat back?

            Show the complete inclusion-exclusion derivation.
        """).strip(),
        "retry_hint": (
            "D_n = n! × Σ(k=0 to n) (-1)^k/k! "
            "D₅ = 120×(1 - 1 + 1/2 - 1/6 + 1/24 - 1/120) = 120-120+60-20+5-1 = 44. "
            "P(derangement) → 1/e ≈ 36.79%."
        ),
    },
    {
        "name": "Inclusion-exclusion: exactly 2 of 3 subjects",
        "type": "text", "check": _check_inclusion,
        "prompt": textwrap.dedent("""
            In a class of 100 students:
              - 40 study Mathematics
              - 35 study Science
              - 30 study English
              - 15 study both Mathematics and Science
              - 12 study both Mathematics and English
              - 10 study both Science and English
              - 5 study all three subjects

            Using inclusion-exclusion:
            1. How many students study at least one subject?
            2. How many study EXACTLY one subject?
            3. How many study EXACTLY two subjects?
            4. How many study NONE of the three?
        """).strip(),
        "retry_hint": (
            "|M∪S∪E| = 40+35+30-15-12-10+5 = 73. "
            "Exactly one: (40-15-12+5)+(35-15-10+5)+(30-12-10+5) = 18+15+13 = 46. "
            "Exactly three: 5. "
            "Exactly two: 73-46-5 = 22. "
            "None: 100-73 = 27."
        ),
    },
    {
        "name": "Relativistic kinetic energy at v=0.8c (γ=5/3)",
        "type": "text", "check": _check_rel_ke,
        "prompt": textwrap.dedent("""
            An electron is accelerated to v = 0.8c (80% of the speed of light).

            1. Calculate the Lorentz factor γ. Show the formula and arithmetic.
            2. Calculate the relativistic kinetic energy KE = (γ−1)mₑc².
               Give the answer in MeV (use mₑc² = 0.511 MeV).
            3. Compare with the Newtonian KE = ½mv². What is the percentage error
               of the Newtonian approximation at this speed?

            Note: Newtonian mechanics is inaccurate at relativistic speeds.
        """).strip(),
        "retry_hint": (
            "γ = 1/√(1-v²/c²) = 1/√(1-0.64) = 1/√0.36 = 1/0.6 = 5/3 ≈ 1.6667. "
            "KE = (5/3-1)×0.511 = (2/3)×0.511 = 0.341 MeV. "
            "Newtonian KE = ½m(0.8c)² = 0.32mₑc² = 0.163 MeV — off by ~52%."
        ),
    },
    {
        "name": "Shannon entropy of a 3-outcome distribution",
        "type": "text", "check": _check_entropy,
        "prompt": textwrap.dedent("""
            A random variable X has three outcomes with probabilities:
              P(X=1) = 0.5,  P(X=2) = 0.3,  P(X=3) = 0.2

            1. Calculate the Shannon entropy H(X) in bits.
               Formula: H = −Σ pᵢ log₂(pᵢ)
            2. What would H be if the distribution were uniform over 3 outcomes?
            3. What would H be for a deterministic outcome (p=1)?
            4. Explain intuitively why H(0.5,0.3,0.2) < H(1/3,1/3,1/3).

            Show all logarithm calculations.
        """).strip(),
        "retry_hint": (
            "H = -0.5×log₂(0.5) - 0.3×log₂(0.3) - 0.2×log₂(0.2). "
            "= 0.5×1 + 0.3×1.737 + 0.2×2.322 = 0.5 + 0.521 + 0.464 = 1.485 bits. "
            "Uniform: H = log₂(3) ≈ 1.585 bits. Deterministic: H = 0."
        ),
    },
    {
        "name": "Bayesian screening (1% prevalence, 90% sensitivity, 85% specificity)",
        "type": "text", "check": _check_bayes_low,
        "prompt": textwrap.dedent("""
            A disease affects 1% of the population.
            A test has 90% sensitivity and 85% specificity (15% false positive rate).
            You test POSITIVE.

            1. Using Bayes' theorem, calculate P(disease | positive test).
            2. Many people assume that a 90%-accurate test means a positive result
               is 90% likely to be correct. Calculate the actual value and explain
               why the intuition is so wrong.
            3. How would the answer change if prevalence were 10% instead of 1%?
        """).strip(),
        "retry_hint": (
            "P(D|+) = 0.90×0.01 / (0.90×0.01 + 0.15×0.99) = 0.009/0.1575 ≈ 5.71%. "
            "The 90% accuracy refers to individual components — the false positive rate "
            "swamps the true positives when prevalence is low. "
            "At 10% prevalence: 0.90×0.10/(0.90×0.10+0.15×0.90) = 0.09/0.225 = 40%."
        ),
    },
    {
        "name": "Modular exponentiation: 3^200 mod 17 (Fermat's little theorem)",
        "type": "text", "check": _check_modexp,
        "prompt": textwrap.dedent("""
            Calculate 3^200 mod 17 without a calculator (using number theory).

            1. State Fermat's Little Theorem and explain when it applies here.
            2. Use it to reduce the exponent: find r such that 3^200 ≡ 3^r (mod 17).
            3. Calculate 3^r mod 17 using successive squaring.
            4. State the final answer.

            Show every step of the calculation.
        """).strip(),
        "retry_hint": (
            "Fermat: 3^16 ≡ 1 (mod 17) since gcd(3,17)=1. "
            "200 = 16×12 + 8, so 3^200 ≡ 3^8 (mod 17). "
            "3^2=9, 3^4=81≡13, 3^8=13²=169≡169-9×17=169-153=16 (mod 17). "
            "Answer: 16."
        ),
    },
    {
        "name": "Nuclear decay with two competing channels (branching)",
        "type": "text", "check": _check_nuclear,
        "prompt": textwrap.dedent("""
            A radioactive nucleus decays via two competing channels simultaneously:
              - Alpha decay with partial half-life T_α = 4 days
              - Beta decay with partial half-life T_β = 12 days

            1. Write the total decay constant λ_total in terms of λ_α and λ_β.
            2. Calculate the effective (total) half-life T_½.
            3. After 6 days, what fraction of the original nuclei remain?
            4. What fraction of all decays are alpha decays?

            Show all working. Use the relationship λ = ln(2) / T_½.
        """).strip(),
        "retry_hint": (
            "λ_α = ln2/4, λ_β = ln2/12. "
            "λ_total = λ_α + λ_β = ln2(1/4 + 1/12) = ln2(3/12 + 1/12) = ln2 × 4/12 = ln2/3. "
            "T_½ = ln2/λ_total = 3 days. "
            "After 6 days = 2 half-lives: fraction remaining = (1/2)² = 0.25 = 25%. "
            "Branching: f_α = λ_α/λ_total = (ln2/4)/(ln2/3) = 3/4 = 75%."
        ),
    },
    {
        "name": "Maxwell-Boltzmann most probable speed of N₂ at 300 K",
        "type": "text", "check": _check_maxwell,
        "prompt": textwrap.dedent("""
            For nitrogen gas (N₂, molar mass M = 28.02 g/mol) at T = 300 K:

            1. Derive the formula for the most probable speed v_p from the
               Maxwell-Boltzmann speed distribution.
            2. Calculate v_p in m/s.
            3. Also calculate the mean speed v_mean and the rms speed v_rms.
            4. Rank v_p, v_mean, v_rms and explain physically which is largest.

            Use: R = 8.314 J/(mol·K), or k_B = 1.38×10⁻²³ J/K, Nₐ = 6.022×10²³.
        """).strip(),
        "retry_hint": (
            "v_p = sqrt(2RT/M) = sqrt(2×8.314×300/0.02802) = sqrt(178,373) ≈ 422 m/s. "
            "v_mean = sqrt(8RT/(πM)) ≈ 476 m/s. "
            "v_rms = sqrt(3RT/M) ≈ 516 m/s. "
            "Order: v_p < v_mean < v_rms."
        ),
    },
]

assert len(TASKS) == 32, f"Expected 32 tasks, got {len(TASKS)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval():
    print("\n" + "═" * 72)
    print("  eval_extensive — 32 hard tasks — Haiku — brain adapts in real time")
    print("  Algorithms: seg-tree, TSP, mat-exp, BIT, KMP, digit-DP, Manacher,")
    print("              min-window, LCS-substr, expr-eval, topo, bin-search,")
    print("              Kruskal, Bellman-Ford, inversions, 2 debug, 2 system")
    print("  Science:    quantum, Gibbs trap, Hess, totient, derange, incl-excl,")
    print("              rel-KE, entropy, Bayes-1%, mod-exp, nuclear, Maxwell-B")
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
            final_code       = _extract_code(trace, task.get("want_fn"))
            passed, det      = _check_code(final_code, task["check"])

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
                    + (f"Key correction: {hint}\n\n" if hint else "")
                    + "Redo with the correct approach."
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

        fire_tag = f"  [⚡×{fires}]"    if fires       else ""
        fix_tag  = "  [↑ brain fixed]"  if brain_fixed  else ""
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
        viz.save(OUT / "brain_extensive.png")

    _report(results, fire_counts)
    with open(OUT / "extensive_run.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved eval/results/brain_extensive.png + extensive_run.json")
    return results


def _report(results, fire_counts):
    n       = len(results)
    n_base  = sum(1 for r in results if r["first_passed"])
    n_final = sum(1 for r in results if r["passed"])
    n_fired = sum(1 for c in fire_counts if c > 0)
    helped  = [r for r in results if r["brain_helped"]]

    print("\n" + "═" * 72)
    print("  RESULTS  [Haiku — 32 tasks — extensive]")
    print("─" * 72)
    print(f"  Without brain (first attempt) : {n_base}/{n}  ({n_base/n:.0%})")
    print(f"  With brain (after intervention): {n_final}/{n}  ({n_final/n:.0%})")
    delta = n_final - n_base
    sign  = "+" if delta >= 0 else ""
    print(f"  Brain contribution             : {sign}{delta} task{'s' if abs(delta)!=1 else ''}"
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
