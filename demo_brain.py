"""demo_brain.py — Brain Agent demo with live graph visualization.

The brain parses each agent trace into a structured reasoning graph, stores
patterns in a semantic graph (neurons = concepts, synapses = co-occurrence
edges), and fires targeted interventions when the current reasoning trajectory
matches past failures.

Two live matplotlib panels:
  Left  — Brain graph: neurons colored by failure rate (green→red), sized by
           evidence count. Edges show co-occurrence links between reasoning steps.
  Right — PCA scatter: trace-level embeddings (green=pass, red=fail, blue=current).

Terminal shows a Rich live table with tasks, P(fail), labels, and any brain warnings.
"""
from __future__ import annotations

import json
import os
import unicodedata
from typing import Optional

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA

import ast, io, contextlib, re as _re
from agents import haiku, opus, tool_agent, _build_openai
from brain import BrainAgent, FailureStore
from pipeline import Forecaster, self_judge


# ── helpers ───────────────────────────────────────────────────────────────────

def _num_check(expected: float, tol: float = 0.02):
    def verify(q: str, trace: str) -> float:
        import re
        tail = trace[-800:]
        nums = re.findall(r"-?\d[\d,]*\.?\d*", tail.replace(",", ""))
        for raw in reversed(nums):
            try:
                v = float(raw)
                if expected == 0:
                    return 1.0 if abs(v) < 0.01 else 0.0
                if abs(v - expected) / abs(expected) <= tol:
                    return 1.0
            except ValueError:
                pass
        return 0.0
    return verify


def _str_check(*accepted, case_sensitive=False):
    import unicodedata
    _SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    _SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    def _norm(s):
        return unicodedata.normalize("NFKC", s).translate(_SUB).translate(_SUP)
    def verify(q, trace):
        tail = _norm(trace[-800:])
        if not case_sensitive:
            tail = tail.lower()
        for a in accepted:
            needle = _norm(a) if case_sensitive else _norm(a).lower()
            if needle in tail:
                return 1.0
        return 0.0
    return verify


_judge = self_judge(opus)


# ── debug task helpers (copied from demo_debug.py to avoid import side-effects) ──

def _last_tool_code(trace: str) -> str | None:
    marker = "[tool:python_exec("
    idx = trace.rfind(marker)
    if idx == -1: return None
    rest = trace[idx + len(marker):]
    end  = rest.find(")] →")
    if end == -1: end = rest.find(")]")
    if end == -1: return None
    try:
        d = ast.literal_eval(rest[:end])
        return d.get("code") if isinstance(d, dict) else None
    except Exception:
        return None


def tool_code_judge(check_fn):
    def verify(question: str, trace: str) -> float:
        code = _last_tool_code(trace)
        if code is None:
            blocks = _re.findall(r"```(?:python|py)?\n(.*?)```", trace, _re.DOTALL)
            code = blocks[-1] if blocks else None
        if not code or not code.strip(): return 0.0
        ns: dict = {}; buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<agent>", "exec"), ns)
            return 1.0 if check_fn(ns, buf.getvalue()) else 0.0
        except Exception:
            return 0.0
    return verify


# ── debug task set (29 tasks, ~35% failure rate) ──────────────────────────────
# These use tool_agent which produces rich traces from actual code execution.
# No CoT prompting needed — the tool calls and execution outputs ARE the trace.

DEBUG_TASKS = [
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef find_max(lst):\n    m = lst[0]\n    for x in lst:\n        if x < m: m = x\n    return m\n\nTests: find_max([3,1,9,2,6]) → 9   find_max([-5,-1,-3]) → -1"), "check": lambda ns, _: ns.get("find_max") and ns["find_max"]([3,1,9,2,6]) == 9 and ns["find_max"]([-5,-1,-3]) == -1},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef factorial(n):\n    result = 0\n    for i in range(1, n+1):\n        result *= i\n    return result\n\nTests: factorial(5) → 120   factorial(0) → 1   factorial(1) → 1"), "check": lambda ns, _: ns.get("factorial") and ns["factorial"](5) == 120 and ns["factorial"](0) == 1 and ns["factorial"](1) == 1},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef sum_evens(lst):\n    return sum(x for x in lst if x % 2 != 0)\n\nTests: sum_evens([1,2,3,4,5,6]) → 12   sum_evens([1,3,5]) → 0"), "check": lambda ns, _: ns.get("sum_evens") and ns["sum_evens"]([1,2,3,4,5,6]) == 12 and ns["sum_evens"]([1,3,5]) == 0},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef is_even(n):\n    return n % 2 == 1\n\nTests: is_even(4) → True   is_even(7) → False   is_even(0) → True"), "check": lambda ns, _: ns.get("is_even") and ns["is_even"](4) == True and ns["is_even"](7) == False and ns["is_even"](0) == True},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef product(lst):\n    result = 1\n    for x in lst:\n        result += x\n    return result\n\nTests: product([2,3,4]) → 24   product([1,1,5]) → 5"), "check": lambda ns, _: ns.get("product") and ns["product"]([2,3,4]) == 24 and ns["product"]([1,1,5]) == 5},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef count_vowels(s):\n    return sum(2 for ch in s.lower() if ch in 'aeiou')\n\nTests: count_vowels('hello') → 2   count_vowels('aeiou') → 5   count_vowels('rhythm') → 0"), "check": lambda ns, _: ns.get("count_vowels") and ns["count_vowels"]("hello") == 2 and ns["count_vowels"]("aeiou") == 5 and ns["count_vowels"]("rhythm") == 0},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef reverse_string(s):\n    return s[::2]\n\nTests: reverse_string('hello') → 'olleh'   reverse_string('ab') → 'ba'\n       reverse_string('a') → 'a'"), "check": lambda ns, _: ns.get("reverse_string") and ns["reverse_string"]("hello") == "olleh" and ns["reverse_string"]("ab") == "ba" and ns["reverse_string"]("a") == "a"},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return b\n\nTests: gcd(48, 18) → 6   gcd(100, 75) → 25   gcd(7, 3) → 1"), "check": lambda ns, _: ns.get("gcd") and ns["gcd"](48, 18) == 6 and ns["gcd"](100, 75) == 25 and ns["gcd"](7, 3) == 1},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = lo + hi // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1\n\nTests: binary_search([1,3,5,7,9], 7) → 3   binary_search([1,3,5,7,9], 4) → -1\n       binary_search([2,4,6,8,10], 2) → 0"), "check": lambda ns, _: ns.get("binary_search") and ns["binary_search"]([1,3,5,7,9], 7) == 3 and ns["binary_search"]([1,3,5,7,9], 4) == -1 and ns["binary_search"]([2,4,6,8,10], 2) == 0},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef is_palindrome(s):\n    return s == s[1:-1]\n\nTests: is_palindrome('racecar') → True   is_palindrome('hello') → False\n       is_palindrome('a') → True   is_palindrome('aa') → True"), "check": lambda ns, _: ns.get("is_palindrome") and ns["is_palindrome"]("racecar") == True and ns["is_palindrome"]("hello") == False and ns["is_palindrome"]("a") == True and ns["is_palindrome"]("aa") == True},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef fib(n):\n    a, b = 0, 0\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nTests: fib(1) → 1   fib(6) → 8   fib(10) → 55   fib(0) → 0"), "check": lambda ns, _: ns.get("fib") and ns["fib"](1) == 1 and ns["fib"](6) == 8 and ns["fib"](10) == 55 and ns["fib"](0) == 0},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef is_prime(n):\n    if n < 2: return True\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True\n\nTests: is_prime(1) → False   is_prime(2) → True   is_prime(9) → False   is_prime(17) → True"), "check": lambda ns, _: ns.get("is_prime") and ns["is_prime"](1) == False and ns["is_prime"](2) == True and ns["is_prime"](9) == False and ns["is_prime"](17) == True},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef bubble_sort(lst):\n    lst = list(lst)\n    n = len(lst)\n    for i in range(n):\n        for j in range(n - i - 1):\n            if lst[j] < lst[j+1]:\n                lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst\n\nTests: bubble_sort([64,34,25,12,22,11,90]) → [11,12,22,25,34,64,90]\n       bubble_sort([3,1,2]) → [1,2,3]"), "check": lambda ns, _: ns.get("bubble_sort") and ns["bubble_sort"]([64,34,25,12,22,11,90]) == [11,12,22,25,34,64,90] and ns["bubble_sort"]([3,1,2]) == [1,2,3]},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef is_anagram(s1, s2):\n    return sorted(s1) != sorted(s2)\n\nTests: is_anagram('listen','silent') → True   is_anagram('hello','world') → False\n       is_anagram('abc','cab') → True"), "check": lambda ns, _: ns.get("is_anagram") and ns["is_anagram"]("listen","silent") == True and ns["is_anagram"]("hello","world") == False and ns["is_anagram"]("abc","cab") == True},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if num in seen:\n            return [seen[num], i]\n        seen[complement] = i\n    return []\n\nTests: two_sum([2,7,11,15],9) → [0,1]   two_sum([3,2,4],6) → [1,2]\n       two_sum([3,3],6) → [0,1]"), "check": lambda ns, _: ns.get("two_sum") and ns["two_sum"]([2,7,11,15],9) == [0,1] and ns["two_sum"]([3,2,4],6) == [1,2] and ns["two_sum"]([3,3],6) == [0,1]},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef rotate(lst, k):\n    n = len(lst)\n    k = k % n\n    return lst[k:] + lst[:k]\n\nTests: rotate([1,2,3,4,5],2) → [4,5,1,2,3]   rotate([1,2,3],1) → [3,1,2]"), "check": lambda ns, _: ns.get("rotate") and ns["rotate"]([1,2,3,4,5],2) == [4,5,1,2,3] and ns["rotate"]([1,2,3],1) == [3,1,2]},
    {"task": ("Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\ndef running_max(lst):\n    result = []\n    curr = 0\n    for x in lst:\n        curr = max(curr, x)\n        result.append(curr)\n    return result\n\nTests: running_max([-3,-1,-4,-1,-5]) → [-3,-1,-1,-1,-1]\n       running_max([1,3,2,5,4]) → [1,3,3,5,5]"), "check": lambda ns, _: ns.get("running_max") and ns["running_max"]([-3,-1,-4,-1,-5]) == [-3,-1,-1,-1,-1] and ns["running_max"]([1,3,2,5,4]) == [1,3,3,5,5]},
    # Group 3 — hidden edge-case bugs
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef max_subarray(nums):\n    max_sum = 0\n    curr_sum = 0\n    for num in nums:\n        curr_sum = max(num, curr_sum + num)\n        max_sum = max(max_sum, curr_sum)\n    return max_sum\n\nSample: max_subarray([-2,1,-3,4,-1,2,1,-5,4]) → 6   max_subarray([1,2,3]) → 6"), "check": lambda ns, _: ns.get("max_subarray") and ns["max_subarray"]([-2,1,-3,4,-1,2,1,-5,4]) == 6 and ns["max_subarray"]([1,2,3]) == 6 and ns["max_subarray"]([-1,-2,-3]) == -1},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef median(nums):\n    s = sorted(nums)\n    n = len(s)\n    m = n // 2\n    return (s[m] + s[m + 1]) / 2 if n % 2 == 0 else float(s[m])\n\nSample: median([1,3,5]) → 3.0   median([2,4,6,8,10]) → 6.0"), "check": lambda ns, _: ns.get("median") and ns["median"]([1,3,5]) == 3.0 and ns["median"]([2,4,6,8,10]) == 6.0 and ns["median"]([1,2,3,4]) == 2.5},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef quicksort(lst):\n    if len(lst) <= 1: return lst\n    pivot = lst[len(lst) // 2]\n    left  = [x for x in lst if x < pivot]\n    right = [x for x in lst if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n\nSample: quicksort([3,1,4,2]) → [1,2,3,4]   quicksort([5,1,3]) → [1,3,5]"), "check": lambda ns, _: ns.get("quicksort") and ns["quicksort"]([3,1,4,2]) == [1,2,3,4] and ns["quicksort"]([5,1,3]) == [1,3,5] and ns["quicksort"]([3,1,3,2]) == [1,2,3,3]},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef running_max(lst):\n    result = []\n    curr = 0\n    for x in lst:\n        curr = max(curr, x)\n        result.append(curr)\n    return result\n\nSample: running_max([3,1,4,1,5]) → [3,3,4,4,5]   running_max([1,2,3]) → [1,2,3]"), "check": lambda ns, _: ns.get("running_max") and ns["running_max"]([3,1,4,1,5]) == [3,3,4,4,5] and ns["running_max"]([1,2,3]) == [1,2,3] and ns["running_max"]([-3,-1,-4,-1,-5]) == [-3,-1,-1,-1,-1]},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(item)\n        else:\n            result.append(item)\n    return result\n\nSample: flatten([[1,2],[3,4]]) → [1,2,3,4]   flatten([1,[2,3],4]) → [1,2,3,4]"), "check": lambda ns, _: ns.get("flatten") and ns["flatten"]([[1,2],[3,4]]) == [1,2,3,4] and ns["flatten"]([1,[2,3],4]) == [1,2,3,4] and ns["flatten"]([1,[2,[3,[4]]]]) == [1,2,3,4]},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef count_vowels(s):\n    return sum(1 for ch in s if ch in 'aeiou')\n\nSample: count_vowels('hello world') → 3   count_vowels('rhythm') → 0"), "check": lambda ns, _: ns.get("count_vowels") and ns["count_vowels"]("hello world") == 3 and ns["count_vowels"]("rhythm") == 0 and ns["count_vowels"]("Apple") == 2},
    # Group 4 — harder hidden edge-case bugs
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef sorted_copy(lst):\n    lst.sort()\n    return lst\n\nSample: sorted_copy([3,1,2]) → [1,2,3]   sorted_copy([5,2,8,1]) → [1,2,5,8]"), "check": lambda ns, _: ns.get("sorted_copy") and ns["sorted_copy"]([3,1,2]) == [1,2,3] and (lambda l: ns["sorted_copy"](l) == [1,2,3] and l == [3,1,2])([3,1,2])},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef title_case(s):\n    return s.title()\n\nSample: title_case('hello world') → 'Hello World'\n        title_case('the quick brown fox') → 'The Quick Brown Fox'"), "check": lambda ns, _: ns.get("title_case") and ns["title_case"]("hello world") == "Hello World" and ns["title_case"]("the quick brown fox") == "The Quick Brown Fox" and ns["title_case"]("don't stop") == "Don't Stop"},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef average(nums):\n    return sum(nums) / len(nums)\n\nSample: average([1,2,3]) → 2.0   average([4,4,4]) → 4.0   average([7]) → 7.0"), "check": lambda ns, _: ns.get("average") and ns["average"]([1,2,3]) == 2.0 and ns["average"]([4,4,4]) == 4.0 and ns["average"]([7]) == 7.0 and ns["average"]([]) == 0.0},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef append_to(item, lst=[]):\n    lst.append(item)\n    return lst\n\nSample: append_to(1, []) → [1]   append_to(2, ['a']) → ['a', 2]"), "check": lambda ns, _: ns.get("append_to") and ns["append_to"](1, []) == [1] and ns["append_to"](2, ["a"]) == ["a", 2] and ns["append_to"](99) == [99] and ns["append_to"](100) == [100]},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1\n\nSample: binary_search([1,5,9,13,17], 9) → 2   binary_search([1,5,9,13,17], 13) → 3"), "check": lambda ns, _: ns.get("binary_search") and ns["binary_search"]([1,5,9,13,17], 9) == 2 and ns["binary_search"]([1,5,9,13,17], 13) == 3 and ns["binary_search"]([1,5,9,13,17], 17) == 4 and ns["binary_search"]([7], 7) == 0},
    {"task": ("Review this function and return the correct implementation. Make sure it passes the sample cases.\n\ndef is_palindrome(s):\n    return s == s[::-1]\n\nSample: is_palindrome('racecar') → True   is_palindrome('hello') → False\n        is_palindrome('madam') → True"), "check": lambda ns, _: ns.get("is_palindrome") and ns["is_palindrome"]("racecar") == True and ns["is_palindrome"]("hello") == False and ns["is_palindrome"]("madam") == True and ns["is_palindrome"]("Race car") == True},
]


# ── task set (same groups as demo_general.py) ─────────────────────────────────

TASKS = [
    # ── Group 1 — Easy ────────────────────────────────────────────────────────
    {"task": "What is the capital city of France? Answer in one word.",
     "verify": _str_check("paris")},
    {"task": "What is 17 × 24? Show your working and give the final number.",
     "verify": _num_check(408)},
    {"task": "Who wrote the novel '1984'? Give the author's full name.",
     "verify": _str_check("george orwell", "orwell")},
    {"task": "How many days are in a leap year? Answer with just the number.",
     "verify": _num_check(366)},
    {"task": "What is the chemical formula for water? Answer in one line.",
     "verify": _str_check("h2o", "H2O")},
    {"task": "What is the square root of 625? Show your reasoning.",
     "verify": _num_check(25)},
    {"task": "What planet is closest to the Sun? Answer in one word.",
     "verify": _str_check("mercury")},
    {"task": "Convert 32 degrees Fahrenheit to Celsius. Give the exact number.",
     "verify": _num_check(0.0, tol=0.05)},
    {"task": "What is the chemical symbol for gold? Answer with just the symbol.",
     "verify": _str_check("Au", case_sensitive=True)},
    {"task": "What is 15% of 200? Show your working.",
     "verify": _num_check(30)},

    # ── Group 2 — Medium ──────────────────────────────────────────────────────
    {"task": ("A store sells a jacket for $120, which is 25% off the original price. "
              "What was the original price? Show your working."),
     "verify": _num_check(160)},
    {"task": "In what year did the Berlin Wall fall?",
     "verify": _num_check(1989, tol=0)},
    {"task": ("What is the sum of interior angles of a regular hexagon? "
              "Derive it from first principles."),
     "verify": _num_check(720)},
    {"task": "What is the atomic number of carbon?",
     "verify": _num_check(6, tol=0)},
    {"task": ("A recipe needs 2.5 cups of flour for 4 servings. "
              "How many cups do you need for 10 servings?"),
     "verify": _num_check(6.25)},
    {"task": ("What is the speed of sound in air at 20°C, in metres per second? "
              "Give an approximate value."),
     "verify": _num_check(343, tol=0.05)},
    {"task": ("A train leaves City A at 09:00 travelling at 90 km/h. "
              "Another train leaves City B (270 km away) at 09:30 travelling toward City A at 60 km/h. "
              "At what time do they meet? Give the answer as HH:MM."),
     "verify": _str_check("11:00", "11:00 am", "1100")},
    {"task": "What is 2^10? Give the exact integer.",
     "verify": _num_check(1024, tol=0)},
    {"task": "Convert 100 degrees Celsius to Fahrenheit. Give the exact number.",
     "verify": _num_check(212, tol=0)},
    {"task": "What is 5! (five factorial)? Give the exact integer.",
     "verify": _num_check(120, tol=0)},

    # ── Group 3 — Hard ────────────────────────────────────────────────────────
    {"task": ("If you invest $1,000 at 6% annual compound interest, "
              "how much will you have after 5 years? Give the answer to the nearest dollar."),
     "verify": _num_check(1338, tol=0.01)},
    {"task": ("How many ways can you arrange the letters in the word MISSISSIPPI? "
              "Show the calculation."),
     "verify": _num_check(34650, tol=0.001)},
    {"task": ("A ball is thrown upward with an initial velocity of 20 m/s. "
              "Using g = 9.8 m/s², how high does it reach in metres? Show your working."),
     "verify": _num_check(20.4, tol=0.05)},
    {"task": "Convert the binary number 11011010 to decimal. Show each step.",
     "verify": _num_check(218, tol=0)},
    {"task": "How many prime numbers are there between 1 and 100? Give the count.",
     "verify": _num_check(25, tol=0)},
    {"task": ("A car drives 150 km from A to B at 60 km/h, then immediately returns "
              "150 km from B to A at 90 km/h. What is the average speed for the whole trip? "
              "Show your working."),
     "verify": _num_check(72, tol=0.01)},
    {"task": "What is 37 × 53? Show your working and give the exact integer.",
     "verify": _num_check(1961, tol=0)},
    {"task": ("A shirt costs $45 after a 40% discount. "
              "What was the original price? Show your working."),
     "verify": _num_check(75, tol=0.01)},
    {"task": ("What is the surface area of a cube with side length 4 cm? "
              "Give the exact value in cm²."),
     "verify": _num_check(96, tol=0)},
    {"task": ("How many ways can you choose 3 items from a set of 6 (order does not matter)? "
              "Show the combination formula and calculation."),
     "verify": _num_check(20, tol=0)},
    {"task": ("Simplify the fraction 2/3 + 3/4 − 1/6. "
              "Give the answer as a single fraction in lowest terms."),
     "verify": _str_check("5/4", "1.25")},
    {"task": "What is the 15th Fibonacci number? (Sequence starts 1, 1, 2, 3, 5, …)",
     "verify": _num_check(610, tol=0)},
    {"task": "How many seconds are in one week? Give the exact integer.",
     "verify": _num_check(604800, tol=0)},
    {"task": ("A water tank is 3/4 full. After adding 12 litres it is 4/5 full. "
              "What is the full capacity of the tank in litres?"),
     "verify": _num_check(240, tol=0.01)},
    {"task": ("What is the sum of interior angles of a regular octagon? "
              "Use the formula for polygons."),
     "verify": _num_check(1080, tol=0)},

    # ── Group 4 — Hardest ─────────────────────────────────────────────────────
    {"task": "What is 18! (18 factorial)? Give the exact integer.",
     "verify": _num_check(6402373705728000, tol=0.001)},
    {"task": ("What is the probability of rolling a sum of 9 with two standard six-sided dice? "
              "Express as a fraction in lowest terms."),
     "verify": _str_check("4/36", "1/9", "one ninth")},
    {"task": ("A jar contains 3 red, 5 blue, and 2 green marbles. You draw 2 without replacement. "
              "What is the probability both are blue? Give as a simplified fraction."),
     "verify": _str_check("2/9", "two ninths")},
    {"task": "What is the LCM (least common multiple) of 12, 18, and 30?",
     "verify": _num_check(180, tol=0)},
    {"task": ("Explain in exactly 3 bullet points why the Monty Hall problem is counter-intuitive "
              "and what the correct answer is. State the exact probability of winning by switching."),
     "verify": _judge},
    {"task": ("Identify the specific logical fallacy and explain why it is a fallacy:\n"
              "'You should take this vitamin supplement — my doctor is a millionaire and takes it every day.'"),
     "verify": _judge},
    {"task": ("Write a Python function `is_prime(n)` that returns True if n is prime, False otherwise. "
              "It must correctly handle n ≤ 1, n = 2, and run in O(√n) time."),
     "verify": _judge},
    {"task": ("Explain the CAP theorem in distributed systems in exactly two sentences. "
              "Name all three guarantees it refers to and state what the theorem says about them."),
     "verify": _judge},
    {"task": ("A car drives 150 km from A to B at 60 km/h, then immediately returns "
              "150 km from B to A at 90 km/h. What is the average speed for the whole trip? "
              "Remember: average speed = total distance / total time."),
     "verify": _num_check(72, tol=0.01)},
    {"task": ("What is the last digit of 17^100? "
              "Find the repeating cycle of last digits of powers of 17 to determine this."),
     "verify": _num_check(1, tol=0)},
    {"task": ("A clock shows exactly 3:15. "
              "What is the angle in degrees between the minute hand and the hour hand? "
              "Remember the hour hand moves continuously."),
     "verify": _num_check(7.5, tol=0.1)},
    {"task": "What is 847 × 293? Show your working step by step and give the exact integer.",
     "verify": _num_check(248171, tol=0)},
    {"task": ("Decode the Caesar cipher 'KHOOR ZRUOG' using a right-shift of 3. "
              "Give the plaintext."),
     "verify": _str_check("hello world")},
    {"task": ("How many digits does 2^100 have? Use logarithms. Show the formula and calculation."),
     "verify": _num_check(31, tol=0)},
    {"task": "Find the GCD of 252 and 105 using the Euclidean algorithm. Show every step.",
     "verify": _num_check(21, tol=0)},
]

CATEGORIES = {
    range(0, 10):  "Group 1 — Easy",
    range(10, 20): "Group 2 — Medium",
    range(20, 35): "Group 3 — Hard",
    range(35, 50): "Group 4 — Hardest",
}


def _group(idx: int) -> str:
    for r, name in CATEGORIES.items():
        if idx in r:
            return name
    return "Unknown"


# ── visualization ─────────────────────────────────────────────────────────────

_CMAP = plt.get_cmap("RdYlGn")   # red=fail, yellow=50%, green=pass


def _pfail_color(p_fail: float) -> tuple:
    return _CMAP(1.0 - p_fail)   # invert: low p_fail → green, high → red


def draw_brain_graph(ax, store: FailureStore, current_vec=None) -> None:
    """Brain store: PCA of all stored trace embeddings, colored by outcome.

    Green = past pass, Red = past fail, Blue star = current partial trace.
    Edges connect current trace to its nearest stored neighbors.
    """
    ax.cla()
    ax.set_facecolor("#0d0d0d")
    ax.set_title(
        f"Brain Store  ({store.n_pass} pass  {store.n_fail} fail)",
        color="white", fontsize=9, pad=4,
    )
    ax.set_axis_off()

    mat, labels = store.all_vecs()
    if mat is None or len(labels) < 2:
        ax.text(0.5, 0.5, f"Accumulating traces… ({store.n} stored)",
                ha="center", va="center", color="#666", transform=ax.transAxes)
        return

    all_vecs = mat
    if current_vec is not None:
        all_vecs = np.vstack([mat, current_vec.reshape(1, -1)])

    from sklearn.decomposition import PCA
    n      = min(len(all_vecs), 60)
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs[:n])

    stored_coords = coords[:len(labels[:n])]
    colors = ["#22cc44" if l == 1 else "#ee3333" for l in labels[:n]]
    sizes  = [18] * len(stored_coords)
    ax.scatter(stored_coords[:, 0], stored_coords[:, 1],
               c=colors, s=sizes, alpha=0.75, edgecolors="none")

    if current_vec is not None and len(all_vecs) <= n:
        cx, cy = coords[-1]
        # Draw lines to 3 nearest stored neighbors
        sims = mat[:len(labels)] @ current_vec
        k    = min(3, len(labels))
        top  = np.argpartition(sims, -k)[-k:]
        for i in top:
            ax.plot([stored_coords[i, 0], cx], [stored_coords[i, 1], cy],
                    color="#4488ff", alpha=0.35, linewidth=0.8, zorder=3)
        ax.scatter([cx], [cy], c="#4488ff", s=70, marker="*",
                   edgecolors="white", linewidths=0.5, zorder=5)

    patches = [
        mpatches.Patch(color="#22cc44", label="past pass"),
        mpatches.Patch(color="#ee3333", label="past fail"),
        mpatches.Patch(color="#4488ff", label="current"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=6,
              facecolor="#111", edgecolor="#333", labelcolor="white")


def draw_scatter(ax, traces: list[str], labels: list[int],
                 embedder, current_trace: Optional[str] = None) -> None:
    """PCA scatter of trace embeddings (green=pass, red=fail, blue=current)."""
    ax.cla()
    ax.set_facecolor("#0d0d0d")
    ax.set_title("Trace Embeddings (PCA)", color="white", fontsize=9, pad=4)
    ax.set_axis_off()

    if len(traces) < 3:
        ax.text(0.5, 0.5, "Accumulating traces…",
                ha="center", va="center", color="#666", transform=ax.transAxes)
        return

    all_traces = traces + ([current_trace] if current_trace else [])
    vecs       = embedder(all_traces)

    n = min(len(vecs), 32)   # cap for speed
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs[:n])

    stored_coords = coords[:len(traces[:n])]
    stored_labels = labels[:n]

    colors = ["#22cc44" if l == 1 else "#ee3333" for l in stored_labels]
    ax.scatter(stored_coords[:, 0], stored_coords[:, 1],
               c=colors, s=25, alpha=0.7, edgecolors="none")

    if current_trace and len(vecs) > len(traces):
        cx, cy = coords[-1]
        ax.scatter([cx], [cy], c="#4488ff", s=60, marker="*",
                   edgecolors="white", linewidths=0.5, zorder=5)

    # Legend
    patches = [
        mpatches.Patch(color="#22cc44", label="pass"),
        mpatches.Patch(color="#ee3333", label="fail"),
        mpatches.Patch(color="#4488ff", label="current"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=6,
              facecolor="#111", edgecolor="#333", labelcolor="white")


# ── Rich terminal table ───────────────────────────────────────────────────────

def _make_table(rows: list[dict], store: FailureStore) -> "rich.table.Table":
    from rich.table import Table
    from rich import box

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan",
              title=f"[cyan]Brain Agent[/cyan] · {store.n_pass}✓ {store.n_fail}✗ stored")
    t.add_column("#",        width=3,  style="dim")
    t.add_column("Task",     width=46, no_wrap=True)
    t.add_column("Group",    width=20)
    t.add_column("P(fail)",  width=8,  justify="right")
    t.add_column("Label",    width=6,  justify="center")
    t.add_column("Brain",    width=6,  justify="center")

    for row in rows:
        pf_str = f"{row['p_fail']:.2f}" if row["p_fail"] is not None else "—"
        lbl    = "[green]PASS[/green]" if row["label"] == 1 else "[red]FAIL[/red]"
        brain  = "[yellow]⚡[/yellow]" if row.get("brain_fired") else ""
        t.add_row(
            str(row["idx"] + 1),
            row["question"][:45],
            row["group"],
            pf_str,
            lbl,
            brain,
        )
    return t


# ── main run loop ─────────────────────────────────────────────────────────────

def main():
    from rich.live import Live
    from rich.panel import Panel
    from rich.console import Console
    from rich.text import Text
    from sklearn.metrics import roc_auc_score

    embedder = _build_openai()

    brain = BrainAgent(
        embedder,
        k              = 5,
        threshold      = 0.45,
        check_interval = 1.5,
        min_chars      = 200,
    )

    # Sonnet thinks natively — brain reads the thinking stream as it arrives.
    # No CoT prompt. No tool execution. The model's internal reasoning IS the trace.
    agent = brain.wrap_thinking("claude-sonnet-4-6")

    # Forecaster for AUC tracking (separate from brain, no conflict)
    fc = Forecaster(embedder, k=5, pca_dim=16)

    console = Console()

    plt.style.use("dark_background")
    fig, (ax_brain, ax_scatter) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("trace_use — Brain Agent  (live mid-generation retry)",
                 color="white", fontsize=11)
    plt.tight_layout(pad=1.5)
    plt.ion()
    plt.show()

    rows:       list[dict] = []
    traces:     list[str]  = []
    labels_all: list[int]  = []

    task_list = [{"task": t["task"], "verify": tool_code_judge(t["check"])}
                 for t in DEBUG_TASKS]
    n_tasks   = len(task_list)

    console.print(Panel(
        "[bold cyan]trace-use — Brain Agent[/bold cyan]\n"
        f"[dim]{n_tasks} Python debugging tasks · tool_agent · live mid-generation retry\n"
        "Brain embeds trace every 1.5s → kNN → stops + retries if P(fail) ≥ 45%[/dim]",
        expand=False,
    ))

    log = open("eval/results/brain_progress.log", "w", buffering=1)

    with Live(console=console, refresh_per_second=4) as live:
        for idx, task_dict in enumerate(task_list):
            task_text = task_dict["task"]
            verifier  = task_dict["verify"]

            live.update(_make_table(rows, brain.failure_store))
            log.write(f"[{idx+1}/{n_tasks}] {task_text[:60]}\n"); log.flush()

            brain_fired_before = brain.last_warning
            result  = agent(task_text)
            trace   = result[0] if isinstance(result, tuple) else str(result)
            fired   = brain.should_bail or (brain.last_warning != brain_fired_before)
            p_fail  = brain.last_p_fail

            label = int(verifier(task_text, trace) >= 0.5)
            brain.store(trace, label)

            fc.add(trace, label)
            fc_p_fail = fc.predict_fail(trace) if len(fc._vecs) >= 4 else p_fail

            log.write(f"  → label={label} p_fail={p_fail} brain={'FIRED' if fired else '-'}\n")
            log.flush()

            if fired:
                console.print(
                    f"  [yellow bold]⚡ BRAIN FIRED[/yellow bold] "
                    f"task #{idx+1}  P(fail)={p_fail:.2f}  "
                    f"→ {'PASS' if label==1 else 'FAIL'} after retry"
                )

            rows.append({
                "idx":         idx,
                "question":    task_text,
                "group":       f"task {idx+1}/{n_tasks}",
                "p_fail":      fc_p_fail,
                "label":       label,
                "brain_fired": fired,
            })
            traces.append(trace)
            labels_all.append(label)

            draw_brain_graph(ax_brain, brain.failure_store)
            draw_scatter(ax_scatter, traces, labels_all, embedder)
            fig.canvas.draw_idle()
            plt.pause(0.05)

            live.update(_make_table(rows, brain.failure_store))

    # ── final metrics ──────────────────────────────────────────────────────
    n_fired = sum(1 for r in rows if r["brain_fired"])
    p_fails = [r["p_fail"] for r in rows if r["p_fail"] is not None]
    lbls    = [r["label"]  for r in rows if r["p_fail"] is not None]

    if len(set(lbls)) == 2 and p_fails:
        auc = roc_auc_score(lbls, [1 - p for p in p_fails])
        console.print(
            f"\n[bold]AUC {auc:.3f}[/bold]   "
            f"brain fired: {n_fired}×   "
            f"stored: {brain.n_stored} ({brain.n_pass} pass / {brain.n_fail} fail)"
        )
    else:
        console.print(
            f"\nbrain fired: {n_fired}×   "
            f"stored: {brain.n_stored} ({brain.n_pass} pass / {brain.n_fail} fail)"
        )

    out = [
        {"task_idx": r["idx"], "question": r["question"], "group": r["group"],
         "label": r["label"], "p_fail": r["p_fail"] or 0.0, "brain_fired": r["brain_fired"]}
        for r in rows
    ]
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/brain_run.json", "w") as f:
        json.dump(out, f, indent=2)

    # Final plot save
    draw_brain_graph(ax_brain, brain.failure_store)
    draw_scatter(ax_scatter, traces, labels_all, embedder)
    plt.ioff()
    fig.savefig("eval/results/brain_viz.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    console.print(f"Plot saved → eval/results/brain_viz.png")

    plt.show(block=True)


if __name__ == "__main__":
    main()
