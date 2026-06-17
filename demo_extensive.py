"""
Extensive real-world demo — run this to see trace_use working across
coding tasks, research synthesis, and data analysis in real-time.

    python3 demo_extensive.py
"""
import sys
import time
sys.path.insert(0, ".")

from agents import haiku, opus, tool_agent, _build_openai
from pipeline import run_task, Forecaster, self_judge
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

embedder = _build_openai()
agent    = tool_agent(["wikipedia_search", "calculator", "python_exec"])
verifier = self_judge(judge_agent=opus)
fc       = Forecaster(embedder, k=8)

TASKS = [
    {
        "category": "Factual Research",
        "emoji": "🔍",
        "task": (
            "What are the differences between TCP and UDP protocols, "
            "which one does HTTP use, and what port does HTTPS run on?"
        ),
    },
    {
        "category": "Coding",
        "emoji": "💻",
        "task": (
            "Write a Python function implementing the Sieve of Eratosthenes "
            "to find all primes up to N. Run it with N=50, print the primes, "
            "and compute the sum of all primes below 100."
        ),
    },
    {
        "category": "Research Synthesis",
        "emoji": "📚",
        "task": (
            "Compare the founding year, founders, and primary business of "
            "Apple, Google, and Microsoft. Which became a trillion-dollar "
            "company first?"
        ),
    },
    {
        "category": "Data & Math",
        "emoji": "📊",
        "task": (
            "Calculate compound interest on $10,000 at 7% for 30 years. "
            "Then compare: how much more do you earn at 10% vs 5% over the same period?"
        ),
    },
    {
        "category": "Coding (Hard)",
        "emoji": "⚙️",
        "task": (
            "Implement Floyd's cycle detection algorithm in Python to detect "
            "if a linked list has a cycle. Write the Node class and has_cycle "
            "function, then test it: one list with a cycle, one without. "
            "Print PASS or FAIL for each test."
        ),
    },
    {
        "category": "Research Synthesis (Hard)",
        "emoji": "🧬",
        "task": (
            "What is the transformer architecture in machine learning? "
            "Who introduced it, in what year, and what were the two main problems "
            "it solved compared to RNNs? What is its time complexity with respect "
            "to sequence length?"
        ),
    },
]


def print_task_header(i, total, category, emoji, task):
    console.rule(f"[bold cyan]{emoji}  Task {i}/{total} — {category}[/bold cyan]")
    console.print(f"[dim]{task[:120]}{'...' if len(task) > 120 else ''}[/dim]\n")


def print_benefit_summary(all_results):
    console.rule("[bold cyan]Session Benefit Summary[/bold cyan]")

    total_components  = sum(len(r.components) for r in all_results)
    total_pass        = sum(r.n_pass        for r in all_results)
    total_fail        = sum(r.n_fail        for r in all_results)
    total_intervened  = sum(r.n_intervened  for r in all_results)

    # how many retries actually rescued a failure
    rescued = sum(
        1 for r in all_results for c in r.components
        if c.retried and c.label == 1
    )

    # budget used = components where we intervened (spent a retry)
    budget_used_pct  = total_intervened / max(1, total_components) * 100
    # of all failures, how many did we catch (i.e. intervened on)?
    caught = sum(
        1 for r in all_results for c in r.components
        if c.retried and c.p_fail is not None
    )
    catch_rate = caught / max(1, total_fail + rescued) * 100 if total_fail + rescued else 0
    random_catch = budget_used_pct  # random would catch budget_pct% of failures

    tbl = Table(box=box.ROUNDED, border_style="cyan", show_lines=True)
    tbl.add_column("Metric",  style="bold white", ratio=2)
    tbl.add_column("Value",   justify="right",    ratio=1)
    tbl.add_column("vs Naive",justify="right",    ratio=1)

    tbl.add_row("Total components processed", str(total_components), "—")
    tbl.add_row("Passed",  Text(str(total_pass),  style="green"), "—")
    tbl.add_row("Failed",  Text(str(total_fail),  style="red"),   "—")
    tbl.add_row(
        "Interventions triggered",
        Text(str(total_intervened), style="yellow"),
        f"{budget_used_pct:.0f}% budget spent",
    )
    tbl.add_row(
        "Retries that rescued a fail → pass",
        Text(str(rescued), style="bold green"),
        "0 (never retries)",
    )
    tbl.add_row(
        "Failure catch rate at this budget",
        Text(f"{catch_rate:.0f}%", style="bold cyan"),
        Text(f"~{random_catch:.0f}% (random)", style="dim"),
    )
    tbl.add_row(
        "Store size at end",
        str(len(fc._vecs)),
        "grows with use",
    )

    console.print(tbl)
    console.print()

    if rescued > 0:
        console.print(
            Panel(
                f"[bold green]↺ {rescued} component(s) recovered[/bold green] — "
                f"the forecaster predicted failure, triggered a retry, "
                f"and the retry succeeded. Without trace_use these would have been silent failures.",
                border_style="green",
                title="Key win",
            )
        )

    if total_intervened == 0:
        console.print(
            Panel(
                "[dim]No interventions this session — store is still small. "
                "Predictions become sharper after ~50+ traces across both pass and fail outcomes.[/dim]",
                border_style="dim",
            )
        )


# ── run all tasks ─────────────────────────────────────────────────────────────
console.print()
console.print(Panel(
    "[bold cyan]trace_use[/bold cyan] — live failure forecasting across real-world tasks\n"
    "[dim]Coding · Research · Data Analysis · Synthesis[/dim]",
    border_style="cyan",
    padding=(1, 4),
))
console.print()

all_results = []
for i, t in enumerate(TASKS, 1):
    print_task_header(i, len(TASKS), t["category"], t["emoji"], t["task"])
    result = run_task(
        task=t["task"],
        agent=agent,
        verifier=verifier,
        forecaster=fc,
        retry=True,
        display=True,
    )
    all_results.append(result)
    console.print()

print_benefit_summary(all_results)
