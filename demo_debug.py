"""
demo_debug.py — trajectory-based failure forecasting on Python debugging.

28 buggy Python functions, ordered easy → hard (same domain, consistent
structure). The kNN forecaster watches HOW the agent reasons: agents that
misidentify a bug write "I think the problem is X..." traces that cluster
near past wrong-diagnosis failures. P(fail) rises before the final answer
is delivered — the forecaster is reading the trajectory, not the result.

By task ~12 the store has enough signal to start flagging. By task ~20
the AUC shows clear discrimination. This is what single-domain deployment
looks like: one type of task, run repeatedly, with the forecaster learning
which reasoning patterns predict failure.

    python3 demo_debug.py
"""
import sys, json, time, re, ast, io, contextlib
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")

from trace_use import tool_agent, haiku
from trace_use.agents import _build_openai, _load_env
from trace_use import run_task, Forecaster
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

_load_env()
console = Console()

# ── agents & forecaster ───────────────────────────────────────────────────
embedder = _build_openai()
agent    = tool_agent(["python_exec"], max_turns=4)   # longer budget → richer trajectory signal
fc       = Forecaster(embedder, k=5, pca_dim=16)

# Bypass decompose: return the full task as the single sub-question.
def _passthrough(prompt):
    return (prompt.split("\n\nTask: ", 1)[-1].strip(), 0)


def _last_tool_code(trace: str) -> str | None:
    """Extract code from the last python_exec tool call in a trace.

    Tool calls appear as: [tool:python_exec({'code': 'def f():\\n  ...'})] → result
    Python's repr() escapes newlines so ast.literal_eval restores them correctly.
    """
    marker = "[tool:python_exec("
    idx = trace.rfind(marker)
    if idx == -1:
        return None
    rest = trace[idx + len(marker):]
    end = rest.find(")] →")         # closing )], start of result arrow
    if end == -1:
        end = rest.find(")]")
    if end == -1:
        return None
    try:
        d = ast.literal_eval(rest[:end])
        return d.get("code") if isinstance(d, dict) else None
    except Exception:
        return None


def tool_code_judge(check_fn):
    """Verifier that works with tool_agent traces.

    Extracts code from the last python_exec call (or falls back to the last
    fenced code block) and runs check_fn(namespace, stdout) on it.
    """
    def verify(question: str, trace: str) -> float:
        code = _last_tool_code(trace)
        if code is None:
            blocks = re.findall(r"```(?:python|py)?\n(.*?)```", trace, re.DOTALL)
            code = blocks[-1] if blocks else None
        if not code or not code.strip():
            return 0.0
        ns: dict = {}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<agent>", "exec"), ns)
            return 1.0 if check_fn(ns, buf.getvalue()) else 0.0
        except Exception:
            return 0.0
    return verify

# ── 28 buggy Python functions, ordered easy → hard ────────────────────────
# Format: task description + buggy code.
# Verifier: haiku reads the agent's reasoning trace and tool execution output,
# then judges whether the fix was correct and produced the expected results.

TASKS = [

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 1 — Obvious single-line bugs (wrong operator / wrong init value)
    # Haiku passes ~95% of these. Builds up clean "pass" examples early.
    # ════════════════════════════════════════════════════════════════════════

    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def find_max(lst):\n"
            "    m = lst[0]\n"
            "    for x in lst:\n"
            "        if x < m: m = x\n"
            "    return m\n\n"
            "Tests: find_max([3,1,9,2,6]) → 9   find_max([-5,-1,-3]) → -1"
        ),
        "check": lambda ns, _: (
            ns.get("find_max") and
            ns["find_max"]([3,1,9,2,6]) == 9 and
            ns["find_max"]([-5,-1,-3]) == -1
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def factorial(n):\n"
            "    result = 0\n"
            "    for i in range(1, n+1):\n"
            "        result *= i\n"
            "    return result\n\n"
            "Tests: factorial(5) → 120   factorial(0) → 1   factorial(1) → 1"
        ),
        "check": lambda ns, _: (
            ns.get("factorial") and
            ns["factorial"](5) == 120 and
            ns["factorial"](0) == 1 and
            ns["factorial"](1) == 1
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def sum_evens(lst):\n"
            "    return sum(x for x in lst if x % 2 != 0)\n\n"
            "Tests: sum_evens([1,2,3,4,5,6]) → 12   sum_evens([1,3,5]) → 0"
        ),
        "check": lambda ns, _: (
            ns.get("sum_evens") and
            ns["sum_evens"]([1,2,3,4,5,6]) == 12 and
            ns["sum_evens"]([1,3,5]) == 0
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def is_even(n):\n"
            "    return n % 2 == 1\n\n"
            "Tests: is_even(4) → True   is_even(7) → False   is_even(0) → True"
        ),
        "check": lambda ns, _: (
            ns.get("is_even") and
            ns["is_even"](4) == True and
            ns["is_even"](7) == False and
            ns["is_even"](0) == True
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def product(lst):\n"
            "    result = 1\n"
            "    for x in lst:\n"
            "        result += x\n"
            "    return result\n\n"
            "Tests: product([2,3,4]) → 24   product([1,1,5]) → 5"
        ),
        "check": lambda ns, _: (
            ns.get("product") and
            ns["product"]([2,3,4]) == 24 and
            ns["product"]([1,1,5]) == 5
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def count_vowels(s):\n"
            "    return sum(2 for ch in s.lower() if ch in 'aeiou')\n\n"
            "Tests: count_vowels('hello') → 2   count_vowels('aeiou') → 5   count_vowels('rhythm') → 0"
        ),
        "check": lambda ns, _: (
            ns.get("count_vowels") and
            ns["count_vowels"]("hello") == 2 and
            ns["count_vowels"]("aeiou") == 5 and
            ns["count_vowels"]("rhythm") == 0
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def reverse_string(s):\n"
            "    return s[::2]\n\n"
            "Tests: reverse_string('hello') → 'olleh'   reverse_string('ab') → 'ba'\n"
            "       reverse_string('a') → 'a'"
        ),
        "check": lambda ns, _: (
            ns.get("reverse_string") and
            ns["reverse_string"]("hello") == "olleh" and
            ns["reverse_string"]("ab") == "ba" and
            ns["reverse_string"]("a") == "a"
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def gcd(a, b):\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return b\n\n"
            "Tests: gcd(48, 18) → 6   gcd(100, 75) → 25   gcd(7, 3) → 1"
        ),
        "check": lambda ns, _: (
            ns.get("gcd") and
            ns["gcd"](48, 18) == 6 and
            ns["gcd"](100, 75) == 25 and
            ns["gcd"](7, 3) == 1
        ),
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 2 — Standard algorithm bugs (~65% pass rate)
    # Bugs are less obvious; Haiku sometimes identifies the wrong cause.
    # ════════════════════════════════════════════════════════════════════════

    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = lo + hi // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: lo = mid + 1\n"
            "        else: hi = mid - 1\n"
            "    return -1\n\n"
            "Tests: binary_search([1,3,5,7,9], 7) → 3   binary_search([1,3,5,7,9], 4) → -1\n"
            "       binary_search([2,4,6,8,10], 2) → 0"
        ),
        "check": lambda ns, _: (
            ns.get("binary_search") and
            ns["binary_search"]([1,3,5,7,9], 7) == 3 and
            ns["binary_search"]([1,3,5,7,9], 4) == -1 and
            ns["binary_search"]([2,4,6,8,10], 2) == 0
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def is_palindrome(s):\n"
            "    return s == s[1:-1]\n\n"
            "Tests: is_palindrome('racecar') → True   is_palindrome('hello') → False\n"
            "       is_palindrome('a') → True   is_palindrome('aa') → True"
        ),
        "check": lambda ns, _: (
            ns.get("is_palindrome") and
            ns["is_palindrome"]("racecar") == True and
            ns["is_palindrome"]("hello") == False and
            ns["is_palindrome"]("a") == True and
            ns["is_palindrome"]("aa") == True
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def fib(n):\n"
            "    a, b = 0, 0\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n\n"
            "Tests: fib(1) → 1   fib(6) → 8   fib(10) → 55   fib(0) → 0"
        ),
        "check": lambda ns, _: (
            ns.get("fib") and
            ns["fib"](1) == 1 and
            ns["fib"](6) == 8 and
            ns["fib"](10) == 55 and
            ns["fib"](0) == 0
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def is_prime(n):\n"
            "    if n < 2: return True\n"
            "    for i in range(2, int(n**0.5) + 1):\n"
            "        if n % i == 0: return False\n"
            "    return True\n\n"
            "Tests: is_prime(1) → False   is_prime(2) → True   is_prime(9) → False   is_prime(17) → True"
        ),
        "check": lambda ns, _: (
            ns.get("is_prime") and
            ns["is_prime"](1) == False and
            ns["is_prime"](2) == True and
            ns["is_prime"](9) == False and
            ns["is_prime"](17) == True
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def bubble_sort(lst):\n"
            "    lst = list(lst)\n"
            "    n = len(lst)\n"
            "    for i in range(n):\n"
            "        for j in range(n - i - 1):\n"
            "            if lst[j] < lst[j+1]:\n"
            "                lst[j], lst[j+1] = lst[j+1], lst[j]\n"
            "    return lst\n\n"
            "Tests: bubble_sort([64,34,25,12,22,11,90]) → [11,12,22,25,34,64,90]\n"
            "       bubble_sort([3,1,2]) → [1,2,3]"
        ),
        "check": lambda ns, _: (
            ns.get("bubble_sort") and
            ns["bubble_sort"]([64,34,25,12,22,11,90]) == [11,12,22,25,34,64,90] and
            ns["bubble_sort"]([3,1,2]) == [1,2,3]
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def is_anagram(s1, s2):\n"
            "    return sorted(s1) != sorted(s2)\n\n"
            "Tests: is_anagram('listen','silent') → True   is_anagram('hello','world') → False\n"
            "       is_anagram('abc','cab') → True"
        ),
        "check": lambda ns, _: (
            ns.get("is_anagram") and
            ns["is_anagram"]("listen","silent") == True and
            ns["is_anagram"]("hello","world") == False and
            ns["is_anagram"]("abc","cab") == True
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if num in seen:\n"
            "            return [seen[num], i]\n"
            "        seen[complement] = i\n"
            "    return []\n\n"
            "Tests: two_sum([2,7,11,15],9) → [0,1]   two_sum([3,2,4],6) → [1,2]\n"
            "       two_sum([3,3],6) → [0,1]"
        ),
        "check": lambda ns, _: (
            ns.get("two_sum") and
            ns["two_sum"]([2,7,11,15],9) == [0,1] and
            ns["two_sum"]([3,2,4],6) == [1,2] and
            ns["two_sum"]([3,3],6) == [0,1]
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def rotate(lst, k):\n"
            "    n = len(lst)\n"
            "    k = k % n\n"
            "    return lst[k:] + lst[:k]\n\n"
            "Tests: rotate([1,2,3,4,5],2) → [4,5,1,2,3]   rotate([1,2,3],1) → [3,1,2]"
        ),
        "check": lambda ns, _: (
            ns.get("rotate") and
            ns["rotate"]([1,2,3,4,5],2) == [4,5,1,2,3] and
            ns["rotate"]([1,2,3],1) == [3,1,2]
        ),
    },
    {
        "task": (
            "Find and fix the one bug. Explain the bug and write the complete corrected function in a Python code block.\n\n"
            "def running_max(lst):\n"
            "    result = []\n"
            "    curr = 0\n"
            "    for x in lst:\n"
            "        curr = max(curr, x)\n"
            "        result.append(curr)\n"
            "    return result\n\n"
            "Tests: running_max([-3,-1,-4,-1,-5]) → [-3,-1,-1,-1,-1]\n"
            "       running_max([1,3,2,5,4]) → [1,3,3,5,5]"
        ),
        "check": lambda ns, _: (
            ns.get("running_max") and
            ns["running_max"]([-3,-1,-4,-1,-5]) == [-3,-1,-1,-1,-1] and
            ns["running_max"]([1,3,2,5,4]) == [1,3,3,5,5]
        ),
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 3 — Hidden-edge-case bugs
    # The shown test cases PASS even with the bug.  The check function tests
    # additional edge cases the agent may not think to run.  Tool agents that
    # only verify the shown examples submit without fixing — a genuine failure
    # the forecaster can learn to predict from the trajectory.
    # ════════════════════════════════════════════════════════════════════════

    # ── G3: visible tests pass with the bug; hidden edge cases in check ──────

    {   # Bug: max_sum=0 means all-negative arrays return 0 instead of the max element
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def max_subarray(nums):\n"
            "    max_sum = 0\n"
            "    curr_sum = 0\n"
            "    for num in nums:\n"
            "        curr_sum = max(num, curr_sum + num)\n"
            "        max_sum = max(max_sum, curr_sum)\n"
            "    return max_sum\n\n"
            "Sample: max_subarray([-2,1,-3,4,-1,2,1,-5,4]) → 6   max_subarray([1,2,3]) → 6"
        ),
        "check": lambda ns, _: (
            ns.get("max_subarray") and
            ns["max_subarray"]([-2,1,-3,4,-1,2,1,-5,4]) == 6 and
            ns["max_subarray"]([1,2,3]) == 6 and
            ns["max_subarray"]([-1,-2,-3]) == -1    # hidden: all-negative
        ),
    },
    {   # Bug: even-length median uses s[m]+s[m+1] instead of s[m-1]+s[m]
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def median(nums):\n"
            "    s = sorted(nums)\n"
            "    n = len(s)\n"
            "    m = n // 2\n"
            "    return (s[m] + s[m + 1]) / 2 if n % 2 == 0 else float(s[m])\n\n"
            "Sample: median([1,3,5]) → 3.0   median([2,4,6,8,10]) → 6.0"
        ),
        "check": lambda ns, _: (
            ns.get("median") and
            ns["median"]([1,3,5]) == 3.0 and
            ns["median"]([2,4,6,8,10]) == 6.0 and
            ns["median"]([1,2,3,4]) == 2.5    # hidden: even-length; bug gives 3.5
        ),
    },
    {   # Bug: quicksort uses strict < and > — identical elements to pivot are silently dropped
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def quicksort(lst):\n"
            "    if len(lst) <= 1: return lst\n"
            "    pivot = lst[len(lst) // 2]\n"
            "    left  = [x for x in lst if x < pivot]\n"
            "    right = [x for x in lst if x > pivot]\n"
            "    return quicksort(left) + [pivot] + quicksort(right)\n\n"
            "Sample: quicksort([3,1,4,2]) → [1,2,3,4]   quicksort([5,1,3]) → [1,3,5]"
        ),
        "check": lambda ns, _: (
            ns.get("quicksort") and
            ns["quicksort"]([3,1,4,2]) == [1,2,3,4] and
            ns["quicksort"]([5,1,3]) == [1,3,5] and
            ns["quicksort"]([3,1,3,2]) == [1,2,3,3]    # hidden: duplicate pivot elements dropped
        ),
    },
    {   # Bug: curr=0 means all-negative arrays track the wrong running max
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def running_max(lst):\n"
            "    result = []\n"
            "    curr = 0\n"
            "    for x in lst:\n"
            "        curr = max(curr, x)\n"
            "        result.append(curr)\n"
            "    return result\n\n"
            "Sample: running_max([3,1,4,1,5]) → [3,3,4,4,5]   running_max([1,2,3]) → [1,2,3]"
        ),
        "check": lambda ns, _: (
            ns.get("running_max") and
            ns["running_max"]([3,1,4,1,5]) == [3,3,4,4,5] and
            ns["running_max"]([1,2,3]) == [1,2,3] and
            ns["running_max"]([-3,-1,-4,-1,-5]) == [-3,-1,-1,-1,-1]    # hidden: all-negative
        ),
    },
    {   # Bug: flatten uses extend (non-recursive) so nested sublists survive
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(item)\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n\n"
            "Sample: flatten([[1,2],[3,4]]) → [1,2,3,4]   flatten([1,[2,3],4]) → [1,2,3,4]"
        ),
        "check": lambda ns, _: (
            ns.get("flatten") and
            ns["flatten"]([[1,2],[3,4]]) == [1,2,3,4] and
            ns["flatten"]([1,[2,3],4]) == [1,2,3,4] and
            ns["flatten"]([1,[2,[3,[4]]]]) == [1,2,3,4]    # hidden: nested >1 level
        ),
    },
    {   # Bug: count_vowels skips uppercase vowels (lowercasing forgotten)
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def count_vowels(s):\n"
            "    return sum(1 for ch in s if ch in 'aeiou')\n\n"
            "Sample: count_vowels('hello world') → 3   count_vowels('rhythm') → 0"
        ),
        "check": lambda ns, _: (
            ns.get("count_vowels") and
            ns["count_vowels"]("hello world") == 3 and
            ns["count_vowels"]("rhythm") == 0 and
            ns["count_vowels"]("Apple") == 2    # hidden: uppercase A missed
        ),
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 4 — Harder hidden-edge-case bugs
    # Even harder to spot: mutation, Python-specific quirks, boundary logic.
    # By the time we reach these the store has ~20 entries and the forecaster
    # should show high P(fail) for trajectories that skip edge-case testing.
    # ════════════════════════════════════════════════════════════════════════

    {   # Bug: sorts in-place (mutates input) instead of returning a sorted copy
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def sorted_copy(lst):\n"
            "    lst.sort()\n"
            "    return lst\n\n"
            "Sample: sorted_copy([3,1,2]) → [1,2,3]   sorted_copy([5,2,8,1]) → [1,2,5,8]"
        ),
        "check": lambda ns, _: (
            ns.get("sorted_copy") and
            ns["sorted_copy"]([3,1,2]) == [1,2,3] and
            # hidden: original must not be mutated
            (lambda l: ns["sorted_copy"](l) == [1,2,3] and l == [3,1,2])([3,1,2])
        ),
    },
    {   # Bug: Python's str.title() capitalises the char after ANY non-alpha (incl apostrophe)
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def title_case(s):\n"
            "    return s.title()\n\n"
            "Sample: title_case('hello world') → 'Hello World'\n"
            "        title_case('the quick brown fox') → 'The Quick Brown Fox'"
        ),
        "check": lambda ns, _: (
            ns.get("title_case") and
            ns["title_case"]("hello world") == "Hello World" and
            ns["title_case"]("the quick brown fox") == "The Quick Brown Fox" and
            ns["title_case"]("don't stop") == "Don't Stop"    # hidden: .title() gives "Don'T Stop"
        ),
    },
    {   # Bug: crashes on empty list (ZeroDivisionError) — not caught by normal tests
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def average(nums):\n"
            "    return sum(nums) / len(nums)\n\n"
            "Sample: average([1,2,3]) → 2.0   average([4,4,4]) → 4.0   average([7]) → 7.0"
        ),
        "check": lambda ns, _: (
            ns.get("average") and
            ns["average"]([1,2,3]) == 2.0 and
            ns["average"]([4,4,4]) == 4.0 and
            ns["average"]([7]) == 7.0 and
            ns["average"]([]) == 0.0    # hidden: empty list → ZeroDivisionError
        ),
    },
    {   # Bug: mutable default argument — all calls without an explicit list share one list
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def append_to(item, lst=[]):\n"
            "    lst.append(item)\n"
            "    return lst\n\n"
            "Sample: append_to(1, []) → [1]   append_to(2, ['a']) → ['a', 2]"
        ),
        "check": lambda ns, _: (
            ns.get("append_to") and
            ns["append_to"](1, []) == [1] and
            ns["append_to"](2, ["a"]) == ["a", 2] and
            # hidden: two calls without explicit list must return independent lists
            ns["append_to"](99) == [99] and
            ns["append_to"](100) == [100]
        ),
    },
    {   # Bug: while lo < hi exits without checking lo==hi; last-element and single-element fail
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: lo = mid + 1\n"
            "        else: hi = mid - 1\n"
            "    return -1\n\n"
            "Sample: binary_search([1,5,9,13,17], 9) → 2   binary_search([1,5,9,13,17], 13) → 3"
        ),
        "check": lambda ns, _: (
            ns.get("binary_search") and
            ns["binary_search"]([1,5,9,13,17], 9) == 2 and
            ns["binary_search"]([1,5,9,13,17], 13) == 3 and
            ns["binary_search"]([1,5,9,13,17], 17) == 4 and    # hidden: last element
            ns["binary_search"]([7], 7) == 0                    # hidden: single element
        ),
    },
    {   # Bug: is_palindrome case-sensitive and space-unaware
        "task": (
            "Review this function and return the correct implementation. "
            "Make sure it passes the sample cases.\n\n"
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n\n"
            "Sample: is_palindrome('racecar') → True   is_palindrome('hello') → False\n"
            "        is_palindrome('madam') → True"
        ),
        "check": lambda ns, _: (
            ns.get("is_palindrome") and
            ns["is_palindrome"]("racecar") == True and
            ns["is_palindrome"]("hello") == False and
            ns["is_palindrome"]("madam") == True and
            ns["is_palindrome"]("Race car") == True    # hidden: spaces and case
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
console.print()
console.print(Panel(
    "[bold cyan]trace_use — trajectory-based failure forecasting[/bold cyan]\n"
    "[dim]29 Python debugging tasks · easy → hard · kNN forecaster learns which "
    "reasoning patterns predict failure · tool_agent + deterministic verifier[/dim]",
    border_style="cyan", padding=(1, 4),
))
console.print()

all_results  = []   # list of dicts for post-run analysis
t0 = time.time()

for idx, t in enumerate(TASKS, 1):
    group = (
        "Group 1 — Obvious"   if idx <= 8  else
        "Group 2 — Standard"  if idx <= 16 else
        "Group 3 — Subtle"    if idx <= 22 else
        "Group 4 — Hard"
    )
    console.rule(f"[bold cyan]{idx}/{len(TASKS)} · {group}[/bold cyan]")
    console.print(f"[dim]{t['task'][:100]}{'...' if len(t['task'])>100 else ''}[/dim]\n")

    result = run_task(
        task=t["task"],
        agent=agent,
        verifier=tool_code_judge(t["check"]),
        forecaster=fc,
        retry=True,
        decompose_agent=_passthrough,   # return full task as single sub-question
        cap=1,                          # each debugging task is ONE component
        display=True,
    )
    all_results.append(result)
    console.print()

elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# Post-session analysis: does the forecaster actually discriminate?
# ─────────────────────────────────────────────────────────────────────────────
console.rule("[bold cyan]Post-Session Forecaster Diagnostic[/bold cyan]")

all_comps = [c for r in all_results for c in r.components]
with_pred  = [c for c in all_comps if c.p_fail is not None]
n_pass     = sum(c.label == 1 for c in all_comps)
n_fail     = sum(c.label == 0 for c in all_comps)
n_retried  = sum(c.retried for c in all_comps)
tp = sum(c.retried and c.label == 0 for c in all_comps)
fp = sum(c.retried and c.label == 1 for c in all_comps)
fn = sum(not c.retried and c.label == 0 for c in all_comps)

# ── session-wide AUC ──────────────────────────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score
    if len(set(c.label for c in with_pred)) == 2:
        session_auc = roc_auc_score(
            [c.label    for c in with_pred],
            [1-c.p_fail for c in with_pred],
        )
    else:
        session_auc = float("nan")
except Exception:
    session_auc = float("nan")

# ── summary table ──────────────────────────────────────────────────────────
tbl = Table(box=box.ROUNDED, border_style="cyan", show_lines=True)
tbl.add_column("Metric",  style="bold white", ratio=3)
tbl.add_column("Value",   justify="right",    ratio=1)
tbl.add_column("Context", justify="left",     ratio=2)

tbl.add_row("Tasks",                    str(len(TASKS)),       "28 debugging problems")
tbl.add_row("Pass (first attempt)",     f"[green]{n_pass}[/green]",  f"{n_pass/len(all_comps)*100:.0f}%")
tbl.add_row("Fail (first attempt)",     f"[red]{n_fail}[/red]",     f"{n_fail/len(all_comps)*100:.0f}%")
tbl.add_row("Retries triggered",        f"[yellow]{n_retried}[/yellow]", f"{n_retried/len(all_comps)*100:.0f}% of tasks")
tbl.add_row("True positives (caught)",  f"[bold green]{tp}[/bold green]", f"{tp/max(1,n_fail)*100:.0f}% of failures flagged")
tbl.add_row("False positives (wasted)", f"[yellow]{fp}[/yellow]",   f"{fp/max(1,n_retried)*100:.0f}% retry precision")
tbl.add_row("Missed failures",          f"[red]{fn}[/red]",         "silent — no retry")
tbl.add_row("Session AUC",              f"[bold cyan]{session_auc:.3f}[/bold cyan]" if session_auc==session_auc else "n/a", "0.5=chance · 1.0=perfect")
tbl.add_row("Store size",               str(len(fc._vecs)),          "traces")
tbl.add_row("Elapsed",                  f"{elapsed/60:.1f} min",     "")
console.print(tbl)
console.print()

# ── AUC learning curve — does the forecaster improve over the run? ─────────
console.print("[bold]Forecaster learning curve — AUC after each task[/bold]")
console.print("[dim](AUC needs ≥2 of each class to be defined)[/dim]\n")

curve_tbl = Table(box=box.SIMPLE, border_style="dim", show_lines=False)
curve_tbl.add_column("After task #", justify="right", style="bold white")
curve_tbl.add_column("Store size",   justify="right")
curve_tbl.add_column("AUC",          justify="right")
curve_tbl.add_column("Signal",       justify="left")

for cutoff in range(1, len(all_results)+1):
    sub = [c for r in all_results[:cutoff] for c in r.components if c.p_fail is not None]
    labels = [c.label for c in sub]
    if len(sub) < 4 or len(set(labels)) < 2:
        continue
    try:
        auc = roc_auc_score(labels, [1-c.p_fail for c in sub])
    except Exception:
        continue
    bar = "█" * int((auc - 0.5) * 20) if auc > 0.5 else ""
    colour = "green" if auc >= 0.65 else "yellow" if auc >= 0.55 else "red"
    curve_tbl.add_row(
        str(cutoff),
        str(len([c for r in all_results[:cutoff] for c in r.components])),
        f"[{colour}]{auc:.3f}[/{colour}]",
        f"[{colour}]{bar}[/{colour}]",
    )

console.print(curve_tbl)
console.print()

# ── P(fail) sorted by outcome — visual separation test ────────────────────
console.print("[bold]P(fail) scores sorted by outcome[/bold]")
console.print("[dim]Failures should cluster toward the top; passes toward the bottom.[/dim]\n")

score_tbl = Table(box=box.MINIMAL, border_style="dim")
score_tbl.add_column("Task",     width=52)
score_tbl.add_column("P(fail)",  justify="right", width=8)
score_tbl.add_column("Actual",   justify="center", width=8)
score_tbl.add_column("Retried",  justify="center", width=8)
score_tbl.add_column("Forecast", justify="center", width=10)

rows = sorted(
    [(c.question[:50], c.p_fail, c.label, c.retried) for c in with_pred],
    key=lambda x: -x[1],
)
for q, pf, label, retried in rows:
    actual  = "[red]FAIL[/red]"    if label == 0 else "[green]pass[/green]"
    retry_s = "[yellow]YES[/yellow]" if retried   else "no"
    pred_fail = pf >= fc.adaptive_threshold if fc._labels else pf >= 0.5
    actually_fail = label == 0
    if pred_fail == actually_fail:  mark = "[green]✓[/green]"
    elif pred_fail and not actually_fail: mark = "[yellow]FP[/yellow]"
    else:                           mark = "[red]FN[/red]"
    score_tbl.add_row(q, f"{pf:.2f}", actual, retry_s, mark)

console.print(score_tbl)

# ── verdict ────────────────────────────────────────────────────────────────
console.print()
if session_auc >= 0.70:
    verdict = (f"[bold green]Strong signal — AUC {session_auc:.2f}. "
               f"The forecaster is reading failure trajectories reliably.[/bold green]")
elif session_auc >= 0.55:
    verdict = (f"[yellow]Moderate signal — AUC {session_auc:.2f}. "
               f"Some discrimination; store needs more examples.[/yellow]")
elif session_auc != session_auc:
    verdict = "[dim]AUC undefined — all pass or all fail.[/dim]"
else:
    verdict = (f"[red]Weak signal — AUC {session_auc:.2f}. "
               f"Failure traces not yet clustering distinctly.[/red]")

console.print(Panel(verdict, border_style="cyan", title="Forecaster verdict"))

# ── save results ───────────────────────────────────────────────────────────
out = []
for i, r in enumerate(all_results):
    for c in r.components:
        out.append({
            "task_idx": i+1,
            "question": c.question[:80],
            "label":    c.label,
            "p_fail":   c.p_fail,
            "retried":  c.retried,
        })
Path("eval/results/debug_run.json").write_text(json.dumps(out, indent=2))
console.print(f"\n[dim]Raw results → eval/results/debug_run.json[/dim]")
