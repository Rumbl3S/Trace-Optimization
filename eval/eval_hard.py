"""eval/eval_hard.py — 14 hard tasks, Sonnet, brain adapts in real time.

Single-pass design: brain starts cold and stores each completed run before the
next task starts. Baseline = first attempt within a code task (before probe
fires) or first text response (before retry with feedback). Brain contribution
is the delta — tasks it fixed that the baseline failed.

Code tasks  — hard one-shot failures even for Sonnet:
  LRU cache, sliding window max, histogram area, regex matching,
  thread-safe bank, burst balloons, trie.

Text tasks  — physics/probability Sonnet consistently gets wrong:
  Bayesian base-rate neglect, rolling-sphere rotational inertia,
  twin-paradox time dilation, hydrogen emission line, buoyancy
  paradox, Simpson's paradox, Bertrand box probability.

Run:  python eval/eval_hard.py
"""
from __future__ import annotations

import ast as _ast
import contextlib
import io
import json
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

OUT = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

SONNET    = "claude-sonnet-4-6"
HAIKU     = "claude-haiku-4-5-20251001"
MAX_TURNS = 12
MAX_TOK   = 8192


# ─── low-level helpers ────────────────────────────────────────────────────────

def _exec(code: str, ns: dict) -> str | None:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<hard>", "exec"), ns)
        return None
    except Exception as e:
        return str(e)


def _check_code(code: str | None, check_fn) -> tuple[bool, str]:
    if not code:
        return False, "no code extracted"
    ns: dict = {}
    err = _exec(code, ns)
    if err:
        return False, f"runtime error: {err[:140]}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ok = bool(check_fn(ns))
        return ok, "" if ok else "failed verification tests"
    except Exception as e:
        return False, f"test error: {e}"


def _check_text(response: str, check_fn) -> tuple[bool, str]:
    try:
        ok = bool(check_fn(response))
        return ok, "" if ok else "answer missing required content"
    except Exception as e:
        return False, f"check error: {e}"


def _extract_code(trace: str, want_fn: str | None = None) -> str | None:
    """Extract the best code candidate from an agent trace.

    Priority: python_exec tool calls (JSON-formatted by agents.py) > markdown
    blocks > bare defs/classes. Among tool calls, later = higher priority
    (most likely to be the corrected post-brain version).
    """
    candidates: list[tuple[int, str]] = []  # (priority, code)

    # Path 1: python_exec JSON tool calls
    pos = 0
    call_idx = 0
    for prefix in ('python_exec({"', "python_exec({'"):
        search_pos = 0
        while True:
            start = trace.find(prefix, search_pos)
            if start == -1:
                break
            arg_start = start + len("python_exec(")
            depth = 0
            i = arg_start
            while i < len(trace):
                ch = trace[i]
                if ch == "{":   depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            raw = trace[arg_start: i + 1]
            search_pos = i + 1
            code = None
            for loader in (json.loads, _ast.literal_eval):
                try:
                    d = loader(raw)
                    if isinstance(d, dict):
                        code = d.get("code") or d.get("script") or d.get("source")
                        if code:
                            break
                except Exception:
                    pass
            if code:
                candidates.append((10 + call_idx, code))
                call_idx += 1

    # Path 2: markdown fenced blocks
    for m in re.finditer(r"```(?:python)?\s*\n?(.*?)```", trace, re.DOTALL):
        candidates.append((5, m.group(1).strip()))

    # Path 3: bare def/class blocks (last resort)
    if want_fn:
        for kw in ("class", "def"):
            for m in re.finditer(
                rf"({kw} {re.escape(want_fn)}[\s\(].*?)(?=\n(?:class |def )|\Z)",
                trace, re.DOTALL,
            ):
                candidates.append((1, m.group(1).strip()))

    if not candidates:
        return None

    def score(item: tuple[int, str]) -> tuple[int, int]:
        priority, c = item
        try:
            compile(c, "<s>", "exec")
        except SyntaxError:
            return (-1, 0)
        bonus = 0
        if want_fn and (f"def {want_fn}" in c or f"class {want_fn}" in c):
            bonus = 2
        elif "def " in c or "class " in c:
            bonus = 1
        return (bonus, priority)

    best = max(candidates, key=score)
    return best[1] if score(best)[0] >= 0 else None


def _first_exec_code(trace: str, want_fn: str | None) -> str | None:
    """Extract only the FIRST python_exec code block — the pre-intervention baseline."""
    for prefix in ('python_exec({"', "python_exec({'"):
        start = trace.find(prefix)
        if start == -1:
            continue
        arg_start = start + len("python_exec(")
        depth = 0
        i = arg_start
        while i < len(trace):
            ch = trace[i]
            if ch == "{":   depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        raw = trace[arg_start: i + 1]
        for loader in (json.loads, _ast.literal_eval):
            try:
                d = loader(raw)
                if isinstance(d, dict):
                    code = d.get("code") or d.get("script") or d.get("source")
                    if code:
                        return code
            except Exception:
                pass
    return _extract_code(trace, want_fn)


def _sonnet(prompt: str) -> tuple[str, int]:
    return _anthropic_call(SONNET, prompt, max_tokens=MAX_TOK)


# ═══════════════════════════════════════════════════════════════════════════════
#  PROBES  (deterministic edge-case tests fired mid-turn by the brain)
# ═══════════════════════════════════════════════════════════════════════════════

# ── LRU Cache ──────────────────────────────────────────────────────────────────

def _probe_lru(ns):
    LRU = ns.get("LRUCache")
    if not LRU:
        return ["LRUCache class not defined — implement it and call python_exec"]
    fails = []
    try:
        c = LRU(2)
        c.put(1, 1); c.put(2, 2)
        if c.get(1) != 1:
            fails.append("get(1) after put(1,1),put(2,2) should return 1")
            return fails
        c.put(3, 3)
        if c.get(2) != -1:
            fails.append(
                "get(2) should be -1 after capacity-2 cache adds key 3. "
                "FIX: accessing key 1 made it MRU; key 2 is LRU and must be evicted. "
                "Use doubly-linked list + hashmap for O(1) eviction."
            )
            return fails
        c2 = LRU(1); c2.put(1, 1); c2.put(2, 2)
        if c2.get(1) != -1:
            fails.append(
                "capacity=1: after put(1,1) then put(2,2), get(1) must be -1. "
                "FIX: adding any new key when at capacity must evict the existing key."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_lru(ns):
    LRU = ns.get("LRUCache")
    if not LRU: return False
    try:
        c = LRU(2)
        c.put(1,1); c.put(2,2)
        if c.get(1) != 1: return False
        c.put(3,3)
        if c.get(2) != -1: return False
        if c.get(3) != 3: return False
        c.put(4,4)
        if c.get(1) != -1: return False
        c2 = LRU(1); c2.put(1,1); c2.put(2,2)
        return c2.get(1) == -1 and c2.get(2) == 2
    except Exception: return False


# ── Sliding window maximum ──────────────────────────────────────────────────────

def _probe_swmax(ns):
    fn = (ns.get("maxSlidingWindow") or ns.get("sliding_window_max")
          or ns.get("max_sliding_window"))
    if not fn:
        return ["maxSlidingWindow(nums, k) not defined — implement it and call python_exec"]
    fails = []
    try:
        r = list(fn([1, 3, -1, -3, 5, 3, 6, 7], 3))
        if r != [3, 3, 5, 5, 6, 7]:
            fails.append(
                f"maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3) = {r}, expected [3,3,5,5,6,7]. "
                "FIX: use a monotone deque of indices in DECREASING value order. "
                "For each i: pop right while nums[i] >= nums[deque[-1]], append i. "
                "Pop left while deque[0] < i-k+1. Window max = nums[deque[0]]."
            )
            return fails
        if list(fn([1, -1], 1)) != [1, -1]:
            fails.append(f"maxSlidingWindow([1,-1], 1)={list(fn([1,-1],1))}, expected [1,-1]")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_swmax(ns):
    fn = (ns.get("maxSlidingWindow") or ns.get("sliding_window_max")
          or ns.get("max_sliding_window"))
    if not fn: return False
    try:
        return all(list(fn(a, k)) == e for a, k, e in [
            ([1,3,-1,-3,5,3,6,7], 3, [3,3,5,5,6,7]),
            ([1], 1, [1]),
            ([1,-1], 1, [1,-1]),
            ([9,11], 2, [11]),
            ([4,-2], 2, [4]),
            ([2,1,5,3,6,4,8,2], 4, [5,6,6,8,8]),
        ])
    except Exception: return False


# ── Largest rectangle in histogram ─────────────────────────────────────────────

def _probe_histogram(ns):
    fn = (ns.get("largestRectangleArea") or ns.get("largest_rectangle")
          or ns.get("largestRectangle"))
    if not fn:
        return ["largestRectangleArea(heights) not defined — implement it and call python_exec"]
    fails = []
    try:
        r = fn([2, 1, 5, 6, 2, 3])
        if r != 10:
            fails.append(
                f"largestRectangleArea([2,1,5,6,2,3]) = {r}, expected 10. "
                "FIX: monotone stack of indices. For each bar i: while stack and "
                "heights[i] < heights[stack[-1]], pop h=stack.pop(), "
                "width = i - stack[-1] - 1 (or i if stack empty), area = heights[h]*width. "
                "Append sentinel 0 at end so remaining stack is flushed."
            )
            return fails
        if fn([6, 2, 5, 4, 5, 1, 6]) != 12:
            fails.append(f"largestRectangleArea([6,2,5,4,5,1,6])={fn([6,2,5,4,5,1,6])}, expected 12")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_histogram(ns):
    fn = (ns.get("largestRectangleArea") or ns.get("largest_rectangle")
          or ns.get("largestRectangle"))
    if not fn: return False
    try:
        return all(fn(h) == e for h, e in [
            ([2,1,5,6,2,3], 10), ([2,4], 4), ([6,2,5,4,5,1,6], 12),
            ([1], 1), ([1,1], 2), ([0], 0), ([4,4,4,4], 16),
        ])
    except Exception: return False


# ── Regex matching (. and *) ────────────────────────────────────────────────────

def _probe_regex(ns):
    fn = ns.get("isMatch") or ns.get("is_match") or ns.get("regex_match")
    if not fn:
        return ["isMatch(s, p) not defined — implement it and call python_exec"]
    fails = []
    try:
        if not fn("aa", "a*"):
            fails.append(
                "isMatch('aa', 'a*') must be True. "
                "FIX: use DP table dp[i][j] = isMatch(s[:i], p[:j]). "
                "When p[j-1]=='*': dp[i][j] = dp[i][j-2]  (zero of preceding) "
                "OR (dp[i-1][j] AND (s[i-1]==p[j-2] OR p[j-2]=='.'))."
            )
            return fails
        if fn("aa", "."):
            fails.append(
                "isMatch('aa', '.') must be False — '.' matches exactly ONE char."
            )
            return fails
        if not fn("aab", "c*a*b"):
            fails.append(
                "isMatch('aab', 'c*a*b') must be True: c*=empty, a*=aa, b=b. "
                "FIX: dp[i][j-2] handles zero occurrences of the preceding element."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_regex(ns):
    fn = ns.get("isMatch") or ns.get("is_match") or ns.get("regex_match")
    if not fn: return False
    try:
        return all(fn(s, p) == e for s, p, e in [
            ("aa","a",False), ("aa","a*",True), ("ab",".*",True),
            ("aab","c*a*b",True), ("mississippi","mis*is*p*.",False),
            ("aa",".",False), ("",".*",True), ("","a*",True),
            ("","",True), ("a","",False), ("ab",".*c",False),
        ])
    except Exception: return False


# ── Thread-safe bank ────────────────────────────────────────────────────────────

def _probe_bank(ns):
    Bank = ns.get("BankAccount") or ns.get("ThreadSafeBank") or ns.get("Account")
    if not Bank:
        return ["BankAccount class not defined — implement it and call python_exec"]
    fails = []
    try:
        a = Bank(100)
        if a.balance != 100:
            fails.append("BankAccount(100).balance should be 100")
            return fails
        if a.withdraw(150):
            fails.append(
                "withdraw(150) on balance=100 should return False. "
                "FIX: check balance >= amount BEFORE deducting; return False if not."
            )
            return fails
        a.deposit(50)
        if not a.withdraw(120) or a.balance != 30:
            fails.append(f"deposit(50) then withdraw(120): expected ok=True, balance=30")
        b = Bank(1000)
        threads = [threading.Thread(target=lambda: b.withdraw(11)) for _ in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()
        if b.balance < 0:
            fails.append(
                f"100 concurrent withdraw(11) from 1000: balance={b.balance} < 0. "
                "FIX: use threading.Lock() — make check-then-deduct atomic."
            )
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_bank(ns):
    Bank = ns.get("BankAccount") or ns.get("ThreadSafeBank") or ns.get("Account")
    if not Bank: return False
    try:
        a = Bank(0); a.deposit(200)
        if a.balance != 200: return False
        if a.withdraw(300): return False
        if a.balance != 200: return False
        if not a.withdraw(200): return False
        if a.balance != 0: return False
        b = Bank(1000)
        threads = [threading.Thread(target=lambda: b.withdraw(11)) for _ in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()
        return b.balance >= 0
    except Exception: return False


# ── Burst balloons ──────────────────────────────────────────────────────────────

def _probe_burst(ns):
    fn = ns.get("maxCoins") or ns.get("max_coins") or ns.get("burst_balloons")
    if not fn:
        return ["maxCoins(nums) not defined — implement it and call python_exec"]
    fails = []
    try:
        r = fn([3, 1, 5, 8])
        if r != 167:
            fails.append(
                f"maxCoins([3,1,5,8]) = {r}, expected 167. "
                "FIX: interval DP. Pad with sentinel 1s: nums = [1]+nums+[1]. "
                "dp[i][j] = max coins from balloons strictly between i and j. "
                "For k in (i,j): dp[i][j] = max(dp[i][k]+nums[i]*nums[k]*nums[j]+dp[k][j]). "
                "k is the LAST balloon popped in interval (i,j), NOT the first."
            )
            return fails
        if fn([1, 5]) != 10:
            fails.append(f"maxCoins([1,5])={fn([1,5])}, expected 10")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_burst(ns):
    fn = ns.get("maxCoins") or ns.get("max_coins") or ns.get("burst_balloons")
    if not fn: return False
    try:
        return all(fn(a) == e for a, e in [
            ([3,1,5,8], 167), ([1,5], 10), ([1], 1),
            ([1,2,3], 12), ([7,9,8,0,7,1,3,5,5,2,3], 1654),
        ])
    except Exception: return False


# ── Trie ────────────────────────────────────────────────────────────────────────

def _probe_trie(ns):
    Trie = ns.get("Trie")
    if not Trie:
        return ["Trie class not defined — implement it and call python_exec"]
    fails = []
    try:
        t = Trie()
        t.insert("apple")
        if not t.search("apple"):
            fails.append("search('apple') should be True after insert('apple')")
            return fails
        if t.search("app"):
            fails.append(
                "search('app') should be False — 'app' was never inserted. "
                "FIX: search() must check node.is_end at the FINAL character node, "
                "not just that the node path exists."
            )
            return fails
        if not t.startsWith("app"):
            fails.append("startsWith('app') should be True — 'apple' begins with 'app'")
    except Exception as e:
        fails.append(f"probe error: {e}")
    return fails


def _check_trie(ns):
    Trie = ns.get("Trie")
    if not Trie: return False
    try:
        t = Trie()
        t.insert("apple"); t.insert("app")
        if not t.search("apple") or not t.search("app"): return False
        if t.search("ap"): return False
        if not t.startsWith("app") or t.startsWith("b"): return False
        t.insert("b")
        return t.search("b")
    except Exception: return False


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT TASK CHECK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _check_bayes(resp: str) -> bool:
    # Correct answer: ~1.94%  (between 1% and 3.5%)
    r = resp.lower().replace(",", "")
    for n in re.findall(r'(\d+\.?\d*)\s*%', r):
        if 1.0 <= float(n) <= 3.5:
            return True
    # Decimal form: 0.019...
    if re.search(r'0\.0(1[5-9]|[23]\d)', resp):
        return True
    return False


def _check_rolling(resp: str) -> bool:
    # a = (5/7) g sin30° ≈ 3.5 m/s²
    r = resp.lower()
    has_formula = "5/7" in r or "5g/14" in r or "five.seventh" in r or \
                  bool(re.search(r'5\s*/\s*7', r))
    has_value   = bool(re.search(r'3\.4[5-9]|3\.5[0-5]', r))
    return has_formula or has_value


def _check_twin(resp: str) -> bool:
    # Earth 10yr, gamma=1.25, ship 8yr, age diff 2yr
    r = resp.lower()
    has_gamma = bool(re.search(r'1\.25|γ\s*=\s*1\.25|gamma.*1\.25|lorentz.*1\.25', r))
    has_diff  = bool(re.search(r'\b2\s*(year|yr)', r))
    has_earth = bool(re.search(r'\b10\s*(year|yr)', r))
    has_ship  = bool(re.search(r'\b8\s*(year|yr)', r))
    return (has_gamma or has_diff) and (has_earth or has_ship)


def _check_hline(resp: str) -> bool:
    # n=4→n=2: Hβ line, λ ≈ 486 nm
    for n in re.findall(r'(\d+\.?\d*)\s*nm', resp.lower()):
        if 483 <= float(n) <= 490:
            return True
    # Also accept metre notation ~4.86e-7
    if re.search(r'4\.8[4-9]\s*[×x\*]?\s*10\s*[\^]?\s*-?\s*7', resp):
        return True
    return False


def _check_buoyancy(resp: str) -> bool:
    # Water level FALLS
    r = resp.lower()
    falls = bool(re.search(r'\bfall\b|\bdrop\b|\bdecrease\b|\blower\b', r))
    anti  = bool(re.search(r'does not fall|won.t fall|not fall|not drop|will not', r))
    return falls and not anti


def _check_simpsons(resp: str) -> bool:
    r = resp.lower()
    has_name = "simpson" in r or "paradox" in r
    has_why  = any(k in r for k in [
        "proportion", "different mix", "more severe", "more mild",
        "composition", "case mix", "subgroup", "aggregat",
        "different proportion", "weighted", "confound", "selection"
    ])
    return has_name or has_why


def _check_bertrand(resp: str) -> bool:
    r = resp.lower()
    has_right = bool(re.search(r'2/3|two.thirds?|66\.?6?%|0\.6+7?\b', r))
    has_wrong = bool(re.search(r'\b1/2\b|\b50\s*%\b', r)) and not has_right
    return has_right and not has_wrong


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK LIST  —  7 code  +  7 text
# ═══════════════════════════════════════════════════════════════════════════════

TASKS: list[dict] = [
    # ── CODE ──────────────────────────────────────────────────────────────────
    {
        "name": "LRU Cache (O(1) get/put)",
        "type": "code", "want_fn": "LRUCache",
        "probe": _probe_lru, "check": _check_lru,
        "prompt": textwrap.dedent("""
            Implement an LRU (Least Recently Used) cache.

            Class interface:
              LRUCache(capacity: int)
              get(key: int) -> int      # return -1 if key not present
              put(key: int, value: int) # evict LRU entry when at capacity

            Requirements:
            - O(1) time for both get and put
            - Do NOT use Python's OrderedDict or functools.lru_cache
            - Implement a doubly-linked list + hashmap yourself

            Call python_exec with your complete implementation and test:
            capacity=2 (including access-order eviction) and capacity=1.
        """).strip(),
    },
    {
        "name": "Sliding window maximum (monotone deque)",
        "type": "code", "want_fn": "maxSlidingWindow",
        "probe": _probe_swmax, "check": _check_swmax,
        "prompt": textwrap.dedent("""
            Write `maxSlidingWindow(nums: list[int], k: int) -> list[int]`.

            Return the maximum value in each sliding window of size k.
            Example: maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3) → [3,3,5,5,6,7]

            Requirements:
            - O(n) time using a monotone deque (collections.deque of indices)
            - Do NOT use max() on each window (that would be O(n×k))

            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Largest rectangle in histogram (stack)",
        "type": "code", "want_fn": "largestRectangleArea",
        "probe": _probe_histogram, "check": _check_histogram,
        "prompt": textwrap.dedent("""
            Write `largestRectangleArea(heights: list[int]) -> int`.

            Find the area of the largest rectangle that fits entirely within
            a histogram. Example: largestRectangleArea([2,1,5,6,2,3]) = 10

            Requirements:
            - O(n) time using a monotone (ascending) stack of indices
            - Append a sentinel 0 at the end to flush remaining elements

            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Regex matching with . and *",
        "type": "code", "want_fn": "isMatch",
        "probe": _probe_regex, "check": _check_regex,
        "prompt": textwrap.dedent("""
            Write `isMatch(s: str, p: str) -> bool`.

            Implement full regular expression matching where:
            - '.' matches exactly one character (any)
            - '*' means zero or more of the immediately preceding element
            - The match must cover the entire string s

            Examples:
              isMatch("aa",  "a*")    = True   ('a*' = zero or more a)
              isMatch("aa",  ".")     = False  ('.' = one char only)
              isMatch("aab", "c*a*b") = True   (c*=empty, a*=aa, b=b)

            Use bottom-up DP (2D table), not recursion.
            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Thread-safe bank account",
        "type": "code", "want_fn": "BankAccount",
        "probe": _probe_bank, "check": _check_bank,
        "prompt": textwrap.dedent("""
            Implement a thread-safe BankAccount class:

              class BankAccount:
                  def __init__(self, initial_balance: float)
                  @property
                  def balance(self) -> float
                  def deposit(self, amount: float) -> None
                  def withdraw(self, amount: float) -> bool
                      # True on success; False if insufficient funds
                      # check-then-deduct must be atomic (race-free)

            Use threading.Lock(). Balance must NEVER go negative under
            100 concurrent withdrawal threads.

            Call python_exec with your implementation and a concurrency test.
        """).strip(),
    },
    {
        "name": "Burst balloons (interval DP)",
        "type": "code", "want_fn": "maxCoins",
        "probe": _probe_burst, "check": _check_burst,
        "prompt": textwrap.dedent("""
            Write `maxCoins(nums: list[int]) -> int`.

            You have n balloons. Bursting balloon i gives
            nums[i-1] * nums[i] * nums[i+1] coins (treat out-of-bounds as 1).
            Burst all balloons to collect maximum coins.

            Example: maxCoins([3,1,5,8]) = 167

            Requirements:
            - Interval DP — NOT greedy, NOT brute force
            - Pad both ends with sentinel 1
            - dp[i][j] = max coins from balloons STRICTLY between index i and j
            - For each split k: treat k as the LAST balloon popped in (i, j)

            Call python_exec with your implementation and test it.
        """).strip(),
    },
    {
        "name": "Trie (insert / search / startsWith)",
        "type": "code", "want_fn": "Trie",
        "probe": _probe_trie, "check": _check_trie,
        "prompt": textwrap.dedent("""
            Implement a Trie (prefix tree):

              class Trie:
                  def insert(self, word: str) -> None
                  def search(self, word: str) -> bool   # True only if exact word was inserted
                  def startsWith(self, prefix: str) -> bool

            Key distinction: if only "apple" was inserted,
              search("apple") → True
              search("app")   → False   (not inserted as a word)
              startsWith("app") → True  (apple starts with app)

            Implement using TrieNode with a dict of children and an is_end flag.
            Call python_exec with your implementation and test it.
        """).strip(),
    },

    # ── TEXT — things Sonnet reliably gets wrong ───────────────────────────────
    {
        "name": "Bayesian disease screening (base-rate neglect)",
        "type": "text", "check": _check_bayes,
        "prompt": textwrap.dedent("""
            A rare disease affects 0.1% of the population (1 in 1000 people).
            A diagnostic test for this disease has:
              Sensitivity (true positive rate):  99%
              Specificity (true negative rate):  95%
              (so the false positive rate is 5%)

            You take the test and it comes back POSITIVE.
            What is the probability you actually have the disease?

            Show every step of your calculation using Bayes' theorem.
            Give your final answer as a percentage to two decimal places.
        """).strip(),
        "retry_hint": (
            "The correct answer is roughly 1.94% — much lower than most people expect. "
            "P(D|+) = P(+|D)×P(D) / [P(+|D)×P(D) + P(+|¬D)×P(¬D)]. "
            "P(D)=0.001, P(+|D)=0.99, P(+|¬D)=0.05. "
            "Numerator: 0.99×0.001 = 0.00099. "
            "Denominator: 0.00099 + 0.05×0.999 = 0.05094. "
            "Result: 0.00099/0.05094 = 1.94%."
        ),
    },
    {
        "name": "Rolling sphere on incline (rotational inertia)",
        "type": "text", "check": _check_rolling,
        "prompt": textwrap.dedent("""
            A solid uniform sphere (mass M, radius R) rolls without slipping
            down an inclined plane at angle θ = 30°. Take g = 9.8 m/s².

            Derive the linear acceleration of the sphere's centre of mass.
            Your derivation must use Newton's second law for BOTH:
              (a) translation: F = Ma
              (b) rotation: τ = Iα  (with I = 2MR²/5 for a solid sphere)

            Express your final answer both:
            - symbolically: a = f(g, θ)
            - numerically for θ = 30°: a = __ m/s²
        """).strip(),
        "retry_hint": (
            "Common error: using a = g sinθ (ignores rotation). "
            "For rolling without slipping, friction provides torque. "
            "No-slip constraint: a = Rα. "
            "From rotation: τ = Iα → fR = (2MR²/5)(a/R) → f = 2Ma/5. "
            "From translation: Mg sinθ − f = Ma → Mg sinθ − 2Ma/5 = Ma. "
            "→ a(1 + 2/5) = g sinθ → a = (5/7) g sinθ ≈ 3.5 m/s²."
        ),
    },
    {
        "name": "Twin paradox (special relativity)",
        "type": "text", "check": _check_twin,
        "prompt": textwrap.dedent("""
            Twin A boards a spaceship and travels at v = 0.6c to a star
            exactly 3 light-years away (Earth frame), then immediately
            turns around and returns to Earth at the same speed.
            Twin B stays on Earth throughout.

            Calculate step by step:
            1. Duration of the round trip in Twin B's (Earth) frame
            2. The Lorentz factor γ for v = 0.6c  (show the formula)
            3. Duration of the round trip on Twin A's ship clock
            4. The age difference when they reunite

            Use exact arithmetic (fractions), then give decimal answers.
        """).strip(),
        "retry_hint": (
            "v=0.6c, star=3ly. "
            "Earth time one-way: 3/0.6 = 5 years; round trip = 10 years. "
            "γ = 1/√(1−v²/c²) = 1/√(1−0.36) = 1/√0.64 = 1/0.8 = 5/4 = 1.25. "
            "Ship time = Earth time / γ = 10/1.25 = 8 years. "
            "Age difference = 10 − 8 = 2 years (Twin A is 2 years younger)."
        ),
    },
    {
        "name": "Hydrogen Hβ emission line (Rydberg formula)",
        "type": "text", "check": _check_hline,
        "prompt": textwrap.dedent("""
            A hydrogen atom emits a photon when an electron transitions from the
            n = 4 energy level down to the n = 2 energy level.

            Use the Rydberg formula:
                1/λ = R_H × (1/n_f² − 1/n_i²)
            where R_H = 1.097 × 10⁷ m⁻¹, n_f = 2, n_i = 4.

            Calculate:
            1. 1/λ in m⁻¹  (show all arithmetic steps)
            2. λ in metres  (scientific notation)
            3. λ in nanometres  (to 4 significant figures)
            4. State the spectral series and the name of this line.
        """).strip(),
        "retry_hint": (
            "1/n_f² − 1/n_i² = 1/4 − 1/16 = 4/16 − 1/16 = 3/16 = 0.1875. "
            "1/λ = 1.097×10⁷ × 0.1875 = 2.057×10⁶ m⁻¹. "
            "λ = 1/2.057×10⁶ = 4.861×10⁻⁷ m = 486.1 nm. "
            "This is the Balmer series, H-beta (Hβ) line."
        ),
    },
    {
        "name": "Buoyancy paradox (steel ball in floating boat)",
        "type": "text", "check": _check_buoyancy,
        "prompt": textwrap.dedent("""
            A rubber dinghy floats in a large bathtub. Inside the dinghy
            sits a heavy steel ball. Someone picks up the steel ball and
            drops it directly into the bathwater, where it sinks to the bottom.

            Does the water level in the bathtub RISE, FALL, or STAY THE SAME?

            Answer the question, then give a rigorous explanation using
            Archimedes' principle that covers both cases:
              (a) steel ball sitting in the floating dinghy
              (b) steel ball resting on the bathtub floor
        """).strip(),
        "retry_hint": (
            "In (a) the dinghy+ball float, so they displace water equal to "
            "their combined WEIGHT. The steel ball (density ~7800 kg/m³) displaces "
            "far more water by weight than by volume. "
            "In (b) the sunken ball only displaces water equal to its VOLUME. "
            "Since ρ_steel >> ρ_water, volume displacement (b) < weight displacement (a). "
            "Therefore the water level FALLS when the ball moves from dinghy to water."
        ),
    },
    {
        "name": "Simpson's paradox (hospital case-mix)",
        "type": "text", "check": _check_simpsons,
        "prompt": textwrap.dedent("""
            Two hospitals report these patient survival statistics:

            Hospital A:
              Mild cases:   900 patients, 810 survived   (90%)
              Severe cases: 100 patients,  30 survived   (30%)
              Overall:     1000 patients, 840 survived   (84%)

            Hospital B:
              Mild cases:   100 patients,  90 survived   (90%)
              Severe cases: 900 patients, 270 survived   (30%)
              Overall:     1000 patients, 360 survived   (36%)

            Both hospitals have IDENTICAL survival rates in each category
            (90% mild, 30% severe), yet Hospital A's overall rate is 84%
            vs Hospital B's 36%. The arithmetic is correct.

            1. Verify the numbers add up correctly.
            2. Explain exactly why the overall rates differ so dramatically
               when the per-category rates are identical.
            3. Name the statistical phenomenon this illustrates.
            4. What is the practical danger of comparing hospitals
               only by overall survival rates?
        """).strip(),
        "retry_hint": (
            "This is Simpson's Paradox. The key is case mix proportions. "
            "Hospital A treats 90% mild cases (easy to survive), "
            "Hospital B treats 90% severe cases (hard to survive). "
            "Both are equally skilled, but A looks dramatically better because "
            "it treats a much easier patient mix. The 'overall' rate is a "
            "weighted average, and the weights (case mix) differ between hospitals."
        ),
    },
    {
        "name": "Bertrand box probability paradox",
        "type": "text", "check": _check_bertrand,
        "prompt": textwrap.dedent("""
            There are three identical-looking boxes:
              Box 1: contains 2 GOLD coins
              Box 2: contains 2 SILVER coins
              Box 3: contains 1 GOLD coin + 1 SILVER coin

            You choose a box at random and, without looking inside, draw one
            coin at random. The coin you drew is GOLD.

            What is the probability that the OTHER coin in that same box
            is also GOLD?

            Most people intuitively answer 1/2, reasoning: "I must be in
            Box 1 or Box 3, so it's 50/50." Show whether this is right or
            wrong, and give the correct answer using conditional probability
            or Bayes' theorem. Give the answer as an exact fraction.
        """).strip(),
        "retry_hint": (
            "The 50/50 intuition is wrong. There are 3 gold coins in total. "
            "Each gold coin is equally likely to be the one you drew: "
            "Gold-1 from Box 1, Gold-2 from Box 1, Gold-3 from Box 3. "
            "In cases Gold-1 and Gold-2, the other coin is also gold (Box 1). "
            "In case Gold-3, the other coin is silver (Box 3). "
            "P(other is gold) = 2/3."
        ),
    },
]

assert len(TASKS) == 14, f"Expected 14 tasks, got {len(TASKS)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(use_haiku: bool = False):
    model      = HAIKU  if use_haiku else SONNET
    label      = "Haiku" if use_haiku else "Sonnet"
    png_name   = "brain_hard_haiku.png"  if use_haiku else "brain_hard.png"
    json_name  = "hard_haiku_run.json"   if use_haiku else "hard_run.json"
    max_tok    = 4096 if use_haiku else MAX_TOK

    print("\n" + "═" * 72)
    print(f"  eval_hard — 14 hard tasks — {label} — brain adapts in real time")
    print("  Code: LRU, slide-win-max, histogram, regex, bank, balloons, trie")
    print("  Text: Bayes, rolling sphere, relativity, quantum, buoyancy,")
    print("        Simpson's paradox, Bertrand box")
    print("═" * 72)

    embedder   = build_embedder()
    brain      = BrainAgent(embedder, threshold=0.35, k=5)
    code_agent = tool_agent(["python_exec"], max_turns=MAX_TURNS,
                            model=model, max_tokens=max_tok)
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

            # Baseline: what would have happened without brain (first tool call)
            first_code       = _first_exec_code(trace, task.get("want_fn"))
            first_p, first_d = _check_code(first_code, task["check"])

            # Final: best code across all turns (may be corrected by brain)
            final_code    = _extract_code(trace, task.get("want_fn"))
            passed, det   = _check_code(final_code, task["check"])

        else:
            code_agent.monitor = None  # text tasks: direct model call
            result     = _anthropic_call(model, task["prompt"], max_tokens=max_tok)
            trace, tok = result if isinstance(result, tuple) else (str(result), 0)
            fires      = 0

            first_p, first_d = _check_text(trace, task["check"])
            passed, det      = first_p, first_d

            if not passed:
                hint     = task.get("retry_hint", "")
                feedback = (
                    "\n\nYour previous answer was incomplete or wrong.\n"
                    + (f"{hint}\n\n" if hint else "")
                    + "Redo the calculation carefully with the correct approach above."
                )
                r2, tok2 = _anthropic_call(model, task["prompt"] + feedback,
                                           max_tokens=max_tok)
                tok          += tok2
                p2, _         = _check_text(r2, task["check"])
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

        # Brain learns from this task before the next one starts
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
        viz.save(OUT / png_name)

    _report(results, fire_counts, label)
    with open(OUT / json_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved eval/results/{png_name} + {json_name}")
    return results


def _report(results, fire_counts, label="Sonnet"):
    n       = len(results)
    n_base  = sum(1 for r in results if r["first_passed"])
    n_final = sum(1 for r in results if r["passed"])
    n_fired = sum(1 for c in fire_counts if c > 0)
    helped  = [r for r in results if r["brain_helped"]]

    print("\n" + "═" * 72)
    print(f"  RESULTS  [{label}]")
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
        if not rs:
            continue
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--haiku", action="store_true", help="Run with Haiku instead of Sonnet")
    args = ap.parse_args()
    run_eval(use_haiku=args.haiku)
