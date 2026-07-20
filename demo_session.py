"""demo_session.py — Interactive terminal demo for trace_use failure memory.

Shows the two-layer interception system in action:

  1. TrajectoryDetector: Before you type a prompt, compares your task description
     against stored failure motifs and shows what would be injected into your prompt.

  2. BrainAgent: If you run a task for real (--run mode), intercepts before
     each python_exec call and fires mid-execution when a known logical gap is found.

Usage:
    python demo_session.py                  # inspect mode — no LLM calls to run the task
    python demo_session.py --run            # run mode — actually executes tasks with the agent
    python demo_session.py --clear          # wipe all stored motifs and exit
    python demo_session.py --show           # list all stored motifs and exit
    python demo_session.py --store PATH     # use a custom motifs file

Motifs persist to ~/.trace_use/motifs.json by default.
Set ANTHROPIC_API_KEY to enable LLM calls (trajectory detection + optional --run).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import box
from rich.rule import Rule
from rich.spinner import Spinner
from rich.live import Live
from rich.padding import Padding

console = Console()


# ── Lazy imports (avoid slow sentence-transformers on --help) ─────────────────

def _load_stack(store_path: str):
    from trace_use import build_embedder, PersistentMotifStore, TrajectoryDetector, BrainAgent
    from trace_use.config import DetectorConfig
    embedder = build_embedder()
    store    = PersistentMotifStore(embedder, path=store_path)
    detector = TrajectoryDetector(
        store, embedder,
        config=DetectorConfig(storage_path=store_path),
    )
    brain    = BrainAgent(embedder, motif_store=store)
    return embedder, store, detector, brain


# ── UI helpers ────────────────────────────────────────────────────────────────

def _header(store) -> Panel:
    n = store.count
    motif_str = f"[bold green]{n}[/] motif{'s' if n != 1 else ''} loaded" if n else "[dim]no motifs yet[/]"
    return Panel(
        f"[bold white]trace_use[/] — failure memory for developers\n"
        f"{motif_str}  •  {store.path}",
        style="bold blue",
        box=box.ROUNDED,
    )


def _show_motifs(store) -> None:
    motifs = store.motifs
    if not motifs:
        console.print("[dim]No motifs stored yet.[/]")
        return
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    t.add_column("#",            style="dim",        width=3)
    t.add_column("ID",           style="cyan",       max_width=30)
    t.add_column("Name",         style="bold white", max_width=40)
    t.add_column("Required condition",               max_width=55)
    t.add_column("Recommendation",                   max_width=55)
    for i, m in enumerate(motifs, 1):
        t.add_row(str(i), m.id, m.name, m.required_condition, m.recommendation)
    console.print(t)


def _analyze_task(detector, task: str) -> list:
    """Run trajectory detection with a spinner, return matches."""
    from trace_use.trajectory import MotifMatch
    matches: list[MotifMatch] = []

    if detector._store.count == 0:
        return matches

    with Live(
        "[dim]Checking against stored motifs…[/]",
        console=console, refresh_per_second=8, transient=True,
    ):
        # Check candidates before expensive LLM call
        candidates = detector._store.retrieve(
            task=task, reasoning="", code="",
            top_k=detector._cfg.retrieval_top_k,
            min_sim=detector._cfg.retrieval_min_sim,
        )

    if not candidates:
        console.print("[dim]  No candidates above similarity threshold.[/]")
        return matches

    console.print(f"[dim]  {len(candidates)} candidate motif(s) to judge…[/]")
    for motif in candidates:
        with Live(
            f"  [dim]Judging: {motif.name}…[/]",
            console=console, refresh_per_second=8, transient=True,
        ):
            result = detector._call_judge(motif, task)

        from trace_use.trajectory import _validate_match
        if result and _validate_match(result, task, detector._cfg.judge_threshold):
            console.print(
                f"  [green]✓ MATCH[/]  [bold]{motif.name}[/] "
                f"[dim](conf: {result.get('confidence', 0):.0%})[/]"
            )
            from trace_use.trajectory import MotifMatch as MM
            matches.append(MM(
                motif          = motif,
                confidence     = float(result.get("confidence", 0)),
                evidence_quote = (result.get("evidence_quote") or "").strip(),
                warning        = (result.get("warning")        or "").strip(),
            ))
        else:
            console.print(f"  [dim]  – not relevant: {motif.name}[/]")

    return matches


def _show_matches(matches: list) -> None:
    from trace_use.trajectory import _format_pitfalls
    if not matches:
        console.print(Panel("[green]No known pitfalls found for this task.[/]", box=box.ROUNDED))
        return
    block = _format_pitfalls(matches)
    console.print(Panel(
        block,
        title=f"[bold yellow]⚠️  {len(matches)} pitfall(s) would be injected[/]",
        border_style="yellow",
        box=box.ROUNDED,
    ))


def _show_enriched_prompt(original: str, enriched: str) -> None:
    if original == enriched:
        return
    console.print(Rule("[dim]Enriched prompt preview[/]", style="dim"))
    console.print(Panel(
        enriched,
        title="[dim]What the LLM would receive[/]",
        border_style="dim",
        box=box.SIMPLE,
    ))


def _seed_motif(store, embedder) -> None:
    """Interactively seed a known failure pattern."""
    console.print(Rule("[bold]Add a known failure pattern[/]"))
    mid   = Prompt.ask("  Motif ID (snake_case, short)")
    name  = Prompt.ask("  Name (5–8 words)")
    desc  = Prompt.ask("  Description (the logical principle, no variable names)")
    req   = Prompt.ask("  Required condition (what the task must say for this to apply)")
    viol  = Prompt.ask("  Violation condition (what the code or reasoning must show)")
    rec   = Prompt.ask("  Recommendation (one-line generalizable fix)")

    from trace_use.brain import FailureMotif
    motif = FailureMotif(
        id                  = mid.strip().replace(" ", "_")[:40],
        name                = name.strip()[:80],
        description         = desc.strip()[:250],
        required_condition  = req.strip()[:300],
        violation_condition = viol.strip()[:300],
        recommendation      = rec.strip()[:300],
        source              = "manual",
    )
    store.add(motif)
    console.print(f"[green]✓ Stored motif:[/] {motif.name!r}")


def _teach_failure(brain, store) -> None:
    """After a task fails, record the failure reason so a motif can be extracted."""
    reason = Prompt.ask(
        "  Describe what went wrong (be specific — this becomes the motif explanation)",
        default="",
    )
    if not reason.strip():
        console.print("[dim]Skipped — no metadata provided, nothing stored.[/]")
        return
    # Build a minimal trace stub so store() can extract code if available
    code_snippet = Prompt.ask("  Paste the failed code (or leave blank)", default="")
    trace = ""
    if code_snippet.strip():
        import json
        trace = f'[tool:python_exec({json.dumps({"code": code_snippet})})]'
    brain.store(trace, label=0, metadata=reason.strip())
    n = store.count
    console.print(f"[green]✓ Stored.[/] Motif store now has [bold]{n}[/] motif(s).")


# ── Main loop ─────────────────────────────────────────────────────────────────

def _inspect_loop(store, detector, brain) -> None:
    """Inspect mode: show what would be injected, no actual agent execution."""
    console.print()
    console.print("[dim]Commands: type a task description, or:[/]")
    console.print("[dim]  :seed   — add a known failure pattern manually[/]")
    console.print("[dim]  :show   — list all stored motifs[/]")
    console.print("[dim]  :clear  — wipe all motifs[/]")
    console.print("[dim]  :quit   — exit[/]")
    console.print()

    task_idx = 0
    while True:
        console.print(Rule(style="dim"))
        try:
            task = Prompt.ask("[bold blue]Task[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not task:
            continue
        if task in (":quit", ":q", "quit", "q"):
            console.print("[dim]Goodbye.[/]")
            break
        if task == ":show":
            _show_motifs(store)
            continue
        if task == ":clear":
            if Confirm.ask("Clear all stored motifs?"):
                store.clear()
                console.print("[green]Cleared.[/]")
            continue
        if task == ":seed":
            from trace_use.agents import build_embedder
            _seed_motif(store, None)
            continue

        console.print()

        # ── Trajectory detection ──────────────────────────────────────────────
        matches = _analyze_task(detector, task)
        console.print()
        _show_matches(matches)

        if matches:
            from trace_use.trajectory import _format_pitfalls
            enriched = f"{_format_pitfalls(matches)}\n\n{task}"
            _show_enriched_prompt(task, enriched)

        # ── Teach the system ──────────────────────────────────────────────────
        console.print()
        outcome = Prompt.ask(
            "  Outcome [p=passed / f=failed / s=skip]",
            choices=["p", "f", "s", "passed", "failed", "skip"],
            default="s",
        ).strip().lower()

        if outcome in ("f", "failed"):
            brain.set_task(task_idx, task=task)
            brain.reset()
            _teach_failure(brain, store)
        elif outcome in ("p", "passed"):
            console.print("[green]✓ Marked as passed. Nothing to learn.[/]")
        else:
            console.print("[dim]Skipped.[/]")

        task_idx += 1
        console.print(_header(store))


def _run_loop(store, detector, brain) -> None:
    """Run mode: actually execute tasks with the agent."""
    try:
        from trace_use import tool_agent
    except ImportError:
        console.print("[red]trace_use not importable — install it first.[/]")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY not set — cannot run agent.[/]")
        sys.exit(1)

    agent         = tool_agent(["python_exec"], max_turns=8, model="claude-haiku-4-5-20251001")
    agent.monitor = brain

    console.print()
    console.print("[dim]Run mode: tasks are actually executed with the agent.[/]")
    console.print("[dim]Type a task to run, or :quit to exit.[/]")
    console.print()

    task_idx = 0
    while True:
        console.print(Rule(style="dim"))
        try:
            task = Prompt.ask("[bold blue]Task[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not task or task in (":quit", ":q", "quit", "q"):
            console.print("[dim]Goodbye.[/]")
            break
        if task == ":show":
            _show_motifs(store)
            continue

        # ── Trajectory detection before running ───────────────────────────────
        matches = _analyze_task(detector, task)
        _show_matches(matches)

        if matches:
            from trace_use.trajectory import _format_pitfalls
            enriched_prompt = f"{_format_pitfalls(matches)}\n\n{task}"
        else:
            enriched_prompt = task

        # ── Run agent ─────────────────────────────────────────────────────────
        brain.set_task(task_idx, task=task)
        brain.reset()

        console.print(Rule("[dim]Running agent…[/]", style="dim"))
        t0 = time.time()
        try:
            trace, tokens = agent(enriched_prompt)
        except Exception as e:
            console.print(f"[red]Agent error: {e}[/]")
            task_idx += 1
            continue
        elapsed = time.time() - t0

        console.print(Panel(
            trace[-2000:] if len(trace) > 2000 else trace,
            title=f"[dim]Agent output  ({tokens} tokens, {elapsed:.1f}s)[/]",
            border_style="dim",
            box=box.SIMPLE,
        ))

        if brain.last_fire:
            f = brain.last_fire
            console.print(Panel(
                f"Motif: [bold]{f['motif']}[/]\n"
                f"Confidence: {f.get('confidence', 0):.0%}\n"
                f"Requirement: {f.get('requirement_quote', '')}\n"
                f"Violation:   {f.get('violation_quote', '')}",
                title="[bold yellow]Brain fired mid-execution[/]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        # ── Teach ─────────────────────────────────────────────────────────────
        outcome = Prompt.ask(
            "  Did it pass? [p=passed / f=failed / s=skip]",
            choices=["p", "f", "s"],
            default="s",
        ).strip()

        if outcome == "f":
            _teach_failure(brain, store)
        elif outcome == "p":
            console.print("[green]✓ Marked as passed.[/]")

        task_idx += 1
        console.print(_header(store))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="trace_use interactive demo")
    parser.add_argument("--run",   action="store_true", help="actually run tasks with the agent")
    parser.add_argument("--clear", action="store_true", help="clear all stored motifs and exit")
    parser.add_argument("--show",  action="store_true", help="list stored motifs and exit")
    parser.add_argument("--store", default=None, help="path to motifs JSON file")
    args = parser.parse_args()

    from trace_use.config import _DEFAULT_STORAGE_PATH
    store_path = args.store or _DEFAULT_STORAGE_PATH

    console.print("[dim]Loading embedder…[/]")
    embedder, store, detector, brain = _load_stack(store_path)
    console.print(_header(store))

    if args.clear:
        if Confirm.ask(f"Clear all {store.count} motif(s) in {store_path}?"):
            store.clear()
            console.print("[green]Cleared.[/]")
        return

    if args.show:
        _show_motifs(store)
        return

    if args.run:
        _run_loop(store, detector, brain)
    else:
        _inspect_loop(store, detector, brain)


if __name__ == "__main__":
    main()
