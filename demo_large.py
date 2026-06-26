"""
demo_large.py — large-scale live integration test for trace_use.

20 tasks across 6 domains, genuinely hard, covering coding, math, research,
algorithm implementation, multi-step reasoning, and synthesis. Runs the full
pipeline: tool_agent → decompose → attempt → self_judge(opus) → forecast → retry.

After all tasks, prints a diagnostic: AUC, precision/recall at threshold,
per-domain failure rate, and a breakdown of what the forecaster caught vs missed.

    python3 demo_large.py
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")

from trace_use import haiku, opus, tool_agent
from trace_use.agents import _build_openai, _load_env
from trace_use import run_task, Forecaster, self_judge
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

_load_env()
console = Console()

embedder = _build_openai()
agent    = tool_agent(["wikipedia_search", "calculator", "python_exec"])
verifier = self_judge(judge_agent=opus)
fc       = Forecaster(embedder, k=8)   # pca_dim=16 default

# ─────────────────────────────────────────────────────────────────────────────
# Task bank: 20 tasks, 6 domains, genuinely hard
# ─────────────────────────────────────────────────────────────────────────────
TASKS = [

    # ── DOMAIN 1: Algorithm Implementation ───────────────────────────────────
    {
        "domain": "Algorithm",
        "task": (
            "Implement an LRU (Least Recently Used) cache in Python using only "
            "a dict and a doubly-linked list (no OrderedDict, no deque). "
            "The class should support get(key) and put(key, value) in O(1). "
            "Test it: put 1→A, 2→B, 3→C into a capacity-2 cache and verify "
            "that key 1 is evicted after putting key 3. Print PASS or FAIL."
        ),
    },
    {
        "domain": "Algorithm",
        "task": (
            "Implement Dijkstra's shortest-path algorithm in Python. "
            "Find the shortest path and distance from node A to node F in this "
            "weighted graph: A-B=4, A-C=2, B-D=5, B-C=1, C-D=8, C-E=10, "
            "D-F=2, E-F=3. Print the path and total distance."
        ),
    },
    {
        "domain": "Algorithm",
        "task": (
            "Implement a working merge sort in Python. Sort this list: "
            "[38, 27, 43, 3, 9, 82, 10]. Show each merge step and print the "
            "final sorted result. Also compute and print its time complexity."
        ),
    },
    {
        "domain": "Algorithm",
        "task": (
            "Write a Python function that checks whether a string is valid JSON "
            "without importing the json module. It should handle nested objects, "
            "arrays, strings with escape characters, numbers, booleans, and null. "
            "Test it on: '{\"a\": [1, 2, null], \"b\": true}' (valid) and "
            "'{bad: json}' (invalid). Print PASS/FAIL for each."
        ),
    },

    # ── DOMAIN 2: Debugging ───────────────────────────────────────────────────
    {
        "domain": "Debugging",
        "task": (
            "The following Python binary search has a bug. Find and fix it, "
            "then verify with test cases for found and not-found:\n\n"
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr)\n"
            "    while left < right:\n"
            "        mid = left + right // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: left = mid + 1\n"
            "        else: right = mid\n"
            "    return -1"
        ),
    },
    {
        "domain": "Debugging",
        "task": (
            "This quicksort implementation has two bugs. Identify both, fix them, "
            "and verify the output on [5,3,8,1,9,2,7]:\n\n"
            "def quicksort(arr):\n"
            "    if len(arr) <= 1: return arr\n"
            "    pivot = arr[0]\n"
            "    left  = [x for x in arr if x < pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + quicksort(right)"
        ),
    },
    {
        "domain": "Debugging",
        "task": (
            "Debug this recursive Fibonacci with memoization — it returns wrong "
            "values for n > 10. Find the bug and fix it:\n\n"
            "memo = {}\n"
            "def fib(n):\n"
            "    if n in memo: return memo[n]\n"
            "    if n <= 1: return 1\n"
            "    result = fib(n-1) + fib(n-2)\n"
            "    memo[n+1] = result\n"
            "    return result\n\n"
            "Verify: fib(10) should be 55, fib(15) should be 610."
        ),
    },

    # ── DOMAIN 3: Math & Quantitative ────────────────────────────────────────
    {
        "domain": "Math",
        "task": (
            "A bag contains 4 red, 5 blue, and 3 green balls. "
            "What is the probability of drawing exactly 2 red balls and 1 blue ball "
            "in 3 draws WITHOUT replacement? Compute the exact fraction and decimal."
        ),
    },
    {
        "domain": "Math",
        "task": (
            "Implement Newton's method in Python to compute sqrt(2) to 10 decimal "
            "places. Start from x=1.0, iterate until |x² - 2| < 1e-12. "
            "Print each iteration value and the number of iterations needed. "
            "Compare to math.sqrt(2)."
        ),
    },
    {
        "domain": "Math",
        "task": (
            "Using dynamic programming, compute the number of ways to make change "
            "for exactly $1.00 (100 cents) using coins of 1, 5, 10, 25 cents. "
            "Write the Python DP solution, print the answer, and verify: "
            "ways to make 10 cents should be 4."
        ),
    },
    {
        "domain": "Math",
        "task": (
            "Two trains start toward each other: Train A leaves City X at 8:00am "
            "traveling 75 mph, Train B leaves City Y (390 miles away) at 9:30am "
            "traveling 90 mph. At what time do they meet, and how far from City X? "
            "Show all working."
        ),
    },

    # ── DOMAIN 4: Systems & CS Concepts ──────────────────────────────────────
    {
        "domain": "Systems",
        "task": (
            "Explain the CAP theorem precisely: what are Consistency, Availability, "
            "and Partition Tolerance? Why can a distributed system guarantee at most "
            "two? Give one real-world database example for each of CP, AP, and CA systems."
        ),
    },
    {
        "domain": "Systems",
        "task": (
            "What is the difference between a process and a thread? "
            "Explain when you'd use each, what memory they share, and "
            "describe a race condition with a concrete Python example using threading. "
            "Show how a lock fixes it."
        ),
    },
    {
        "domain": "Systems",
        "task": (
            "Explain TCP's three-way handshake (SYN, SYN-ACK, ACK) step by step. "
            "Why does TCP need a four-way handshake to close a connection? "
            "What is TIME_WAIT and why does it exist?"
        ),
    },

    # ── DOMAIN 5: Research & Synthesis ───────────────────────────────────────
    {
        "domain": "Research",
        "task": (
            "What were the five main causes of the 2008 financial crisis? "
            "For each cause, name a specific institution or instrument involved "
            "and explain its role. What regulatory change did Dodd-Frank make in response?"
        ),
    },
    {
        "domain": "Research",
        "task": (
            "Compare transformer and LSTM architectures for sequence modeling: "
            "attention mechanism vs recurrence, parallelizability, handling of "
            "long-range dependencies, and typical use cases today. "
            "What year did 'Attention Is All You Need' appear and who wrote it?"
        ),
    },
    {
        "domain": "Research",
        "task": (
            "What is the difference between supervised, unsupervised, and "
            "reinforcement learning? Give one concrete algorithm example for each. "
            "For RL specifically: define reward, policy, and value function, "
            "and name a real-world application."
        ),
    },

    # ── DOMAIN 6: Advanced Coding ─────────────────────────────────────────────
    {
        "domain": "Coding",
        "task": (
            "Write a Python decorator @retry(max_attempts=3, delay=0) that retries "
            "a function up to N times on exception, with optional delay between retries. "
            "Test it: a function that fails twice then succeeds should return the "
            "success value. A function that always fails should raise the last exception."
        ),
    },
    {
        "domain": "Coding",
        "task": (
            "Implement a thread-safe bounded queue in Python (without using "
            "queue.Queue) using threading.Lock and threading.Condition. "
            "It should block on put() when full and block on get() when empty. "
            "Test with 2 producer threads and 2 consumer threads, capacity=3."
        ),
    },
    {
        "domain": "Coding",
        "task": (
            "Write a Python context manager @contextmanager that times a block of "
            "code, catches any exception (logging the error), and always prints the "
            "elapsed time on exit. "
            "Test it on: (1) a block that sleeps 0.05s successfully, "
            "(2) a block that raises ValueError. Both should print elapsed time."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
console.print()
console.print(Panel(
    f"[bold cyan]trace_use — large-scale integration test[/bold cyan]\n"
    f"[dim]{len(TASKS)} tasks · 6 domains · Haiku agent + Opus judge · PCA-16 forecaster[/dim]",
    border_style="cyan", padding=(1, 4),
))
console.print()

all_results = []
domain_stats = defaultdict(lambda: {"n": 0, "pass": 0, "fail": 0, "caught": 0, "retried": 0})
t0 = time.time()

for i, t in enumerate(TASKS, 1):
    console.rule(f"[bold cyan]{i}/{len(TASKS)} · {t['domain']}[/bold cyan]")
    console.print(f"[dim]{t['task'][:110]}{'...' if len(t['task'])>110 else ''}[/dim]\n")

    result = run_task(
        task=t["task"],
        agent=agent,
        verifier=verifier,
        forecaster=fc,
        retry=True,
        decompose_agent=haiku,
        display=True,
    )
    result.domain = t["domain"]
    all_results.append(result)

    d = domain_stats[t["domain"]]
    d["n"]       += len(result.components)
    d["pass"]    += result.n_pass
    d["fail"]    += result.n_fail
    d["caught"]  += sum(1 for c in result.components if c.retried and c.p_fail is not None)
    d["retried"] += result.n_intervened
    console.print()

elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# Post-session analysis
# ─────────────────────────────────────────────────────────────────────────────
console.rule("[bold cyan]Post-Session Forecaster Diagnostic[/bold cyan]")

# aggregate
all_components = [c for r in all_results for c in r.components]
total     = len(all_components)
n_pass    = sum(c.label == 1 for c in all_components)
n_fail    = sum(c.label == 0 for c in all_components)
n_retried = sum(c.retried for c in all_components)

# components where we had a real forecast (store was big enough)
with_forecast = [c for c in all_components if c.p_fail is not None]
# true positives: retried AND was a real failure
tp = sum(c.retried and c.label == 0 for c in all_components)
fp = sum(c.retried and c.label == 1 for c in all_components)  # wasted retries
fn = sum(not c.retried and c.label == 0 for c in all_components)  # missed failures

prec   = tp / max(1, tp + fp)
recall = tp / max(1, n_fail)
budget = n_retried / max(1, total)

# AUC on forecasted components
try:
    from sklearn.metrics import roc_auc_score
    pred_labels = [c.label    for c in with_forecast]
    pred_scores = [1 - c.p_fail for c in with_forecast]  # P(pass) for AUC
    if len(set(pred_labels)) == 2:
        session_auc = roc_auc_score(pred_labels, pred_scores)
    else:
        session_auc = float("nan")
except Exception:
    session_auc = float("nan")

# ── summary table ─────────────────────────────────────────────────────────────
tbl = Table(box=box.ROUNDED, border_style="cyan", show_lines=True)
tbl.add_column("Metric",        style="bold white", ratio=3)
tbl.add_column("Value",         justify="right",    ratio=1)
tbl.add_column("Context",       justify="left",     ratio=2)

tbl.add_row("Total components",        str(total),        f"{len(TASKS)} tasks")
tbl.add_row("Passed (first attempt)",  f"[green]{n_pass}[/green]",  f"{n_pass/total*100:.0f}%")
tbl.add_row("Failed (first attempt)",  f"[red]{n_fail}[/red]",     f"{n_fail/total*100:.0f}%")
tbl.add_row("Retries triggered",       f"[yellow]{n_retried}[/yellow]", f"{budget*100:.0f}% budget")
tbl.add_row("True positives (caught)", f"[bold green]{tp}[/bold green]", f"{recall*100:.0f}% of failures")
tbl.add_row("False positives (wasted)",f"[yellow]{fp}[/yellow]",   f"{prec*100:.0f}% retry precision")
tbl.add_row("Missed failures",         f"[red]{fn}[/red]",         "silent — no retry triggered")
tbl.add_row("Session AUC",             f"[bold cyan]{session_auc:.3f}[/bold cyan]" if session_auc==session_auc else "n/a",
            "0.5=chance, 1.0=perfect")
tbl.add_row("Store size (final)",      str(len(fc._vecs)),          "traces in kNN store")
tbl.add_row("Elapsed",                 f"{elapsed/60:.1f} min",     "")

console.print(tbl)
console.print()

# ── per-domain breakdown ──────────────────────────────────────────────────────
dtbl = Table(box=box.SIMPLE, border_style="dim", show_lines=False)
dtbl.add_column("Domain",   style="bold white")
dtbl.add_column("Comps",    justify="right")
dtbl.add_column("Pass",     justify="right")
dtbl.add_column("Fail",     justify="right")
dtbl.add_column("Caught",   justify="right")
dtbl.add_column("Missed",   justify="right")
dtbl.add_column("Fail%",    justify="right")

for domain, d in sorted(domain_stats.items()):
    fail_rate = d["fail"] / max(1, d["n"]) * 100
    missed    = d["fail"] - d["caught"]
    dtbl.add_row(
        domain,
        str(d["n"]),
        f"[green]{d['pass']}[/green]",
        f"[red]{d['fail']}[/red]",
        f"[bold green]{d['caught']}[/bold green]",
        f"[red]{missed}[/red]",
        f"{fail_rate:.0f}%",
    )

console.print(dtbl)

# ── P(fail) score list for all components ────────────────────────────────────
console.print()
console.print("[bold]P(fail) scores — forecaster vs actual outcome[/bold]")
score_tbl = Table(box=box.MINIMAL, border_style="dim")
score_tbl.add_column("Component",  width=52)
score_tbl.add_column("P(fail)",    justify="right", width=8)
score_tbl.add_column("Actual",     justify="center", width=8)
score_tbl.add_column("Retried",    justify="center", width=8)
score_tbl.add_column("Correct?",   justify="center", width=10)

for c in all_components:
    if c.p_fail is None:
        continue
    actual  = "[green]PASS[/green]" if c.label == 1 else "[red]FAIL[/red]"
    retried = "[yellow]YES[/yellow]" if c.retried else "no"
    # was the forecast correct? (0.5 is a display approximation; actual threshold
    # is adaptive per component — see Forecaster.adaptive_threshold)
    predicted_fail = c.p_fail >= 0.5
    actually_fail  = c.label == 0
    if predicted_fail == actually_fail:
        correct = "[green]✓[/green]"
    elif predicted_fail and not actually_fail:
        correct = "[yellow]FP[/yellow]"   # false positive
    else:
        correct = "[red]FN[/red]"         # false negative (missed)
    score_tbl.add_row(
        c.question[:50],
        f"{c.p_fail:.2f}",
        actual,
        retried,
        correct,
    )

console.print(score_tbl)

# ── verdict ───────────────────────────────────────────────────────────────────
console.print()
if session_auc >= 0.70:
    verdict = f"[bold green]Strong — AUC {session_auc:.2f}. Forecaster is learning from this session's traces.[/bold green]"
elif session_auc >= 0.55:
    verdict = f"[yellow]Moderate — AUC {session_auc:.2f}. Some signal; store needs more examples.[/yellow]"
elif session_auc != session_auc:
    verdict = "[dim]AUC undefined — all components passed or all failed (no mixed labels for ROC).[/dim]"
else:
    verdict = f"[red]Weak — AUC {session_auc:.2f}. Forecaster near chance; store likely still too small.[/red]"

console.print(Panel(verdict, border_style="cyan", title="Forecaster verdict"))

# ── save raw results ──────────────────────────────────────────────────────────
out = []
for r in all_results:
    for c in r.components:
        out.append({
            "domain":   r.domain,
            "task":     r.task[:80],
            "question": c.question,
            "label":    c.label,
            "p_fail":   c.p_fail,
            "retried":  c.retried,
        })

Path("eval/results/large_run.json").write_text(json.dumps(out, indent=2))
console.print(f"\n[dim]Raw results saved → eval/results/large_run.json[/dim]")
