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
import pathlib
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

def _detect_provider() -> tuple[str, str, str]:
    """Returns (provider, agent_model, judge_model)."""
    has_openai    = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_openai:
        return "openai", "gpt-4o-mini", "gpt-4o"
    if has_anthropic:
        return "anthropic", "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"
    return "anthropic", "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"


def _load_stack(store_path: str):
    from trace_use import build_embedder, PersistentMotifStore, TrajectoryDetector, BrainAgent
    from trace_use.config import BrainConfig, DetectorConfig

    provider, agent_model, judge_model = _detect_provider()
    embedder = build_embedder()
    store    = PersistentMotifStore(embedder, path=store_path)
    detector = TrajectoryDetector(
        store, embedder,
        config=DetectorConfig(
            provider     = provider,
            model        = judge_model,
            storage_path = store_path,
        ),
    )
    brain = BrainAgent(
        embedder, motif_store=store,
        config=BrainConfig(
            provider      = provider,
            judge_model   = judge_model,
            extract_model = judge_model,
        ),
    )
    return embedder, store, detector, brain, provider, agent_model


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


def _auto_judge(task: str, trace: str, provider: str, judge_model: str) -> tuple[bool, str]:
    """Ask the judge model whether the agent completed the task correctly.

    Returns (passed, reason). `reason` is non-empty only on failure.
    """
    from trace_use.agents import _llm_call
    # Give the judge the start (tool calls + outputs) and end (agent summary)
    if len(trace) > 3000:
        trace_for_judge = trace[:2000] + "\n...[middle omitted]...\n" + trace[-1000:]
    else:
        trace_for_judge = trace

    prompt = f"""You are evaluating whether an AI agent completed a programming task correctly.

TASK:
{task}

AGENT TRACE (tool calls, outputs, and final summary):
{trace_for_judge}

Answer with a JSON object:
{{
  "passed": true or false,
  "reason": "if failed: name the specific logical mistake. Empty string if passed."
}}

Look for [tool:python_exec(...)] and [tool:write_file(...)] lines — those confirm code ran.
If you see successful tool output (no unhandled errors at the end), it passed.
Reply with ONLY the JSON, no other text."""

    try:
        text, _ = _llm_call(provider, judge_model, prompt, max_tokens=120)
        import json, re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return bool(obj.get("passed")), str(obj.get("reason", ""))
    except Exception:
        pass
    return True, ""   # safe default: don't store a false failure


def _run_loop(store, detector, brain, provider: str, agent_model: str) -> None:
    """Run mode: actually execute tasks with the agent."""
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]No API key set. Export OPENAI_API_KEY or ANTHROPIC_API_KEY.[/]")
        sys.exit(1)

    from trace_use.tools import get_workspace
    workspace = get_workspace()

    if provider == "openai":
        from trace_use.agents import openai_tool_agent
        agent = openai_tool_agent(["python_exec", "write_file"], max_turns=8, model=agent_model)
    else:
        from trace_use import tool_agent
        agent = tool_agent(["python_exec", "write_file"], max_turns=8, model=agent_model)

    agent.monitor = brain
    console.print(f"[dim]provider={provider} | model={agent_model}[/]")
    console.print(f"[dim]workspace={workspace}[/]")

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
            trace[-3000:] if len(trace) > 3000 else trace,
            title=f"[dim]Agent output  ({tokens} tokens, {elapsed:.1f}s)[/]",
            border_style="dim",
            box=box.SIMPLE,
        ))

        # Show files written to workspace this turn
        written = [str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file()]
        if written:
            console.print(f"[dim]Files in workspace: {', '.join(written)}[/]")

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

        # ── Auto-judge ────────────────────────────────────────────────────────
        _code_ran = "[tool:python_exec" in trace or "[tool:write_file" in trace
        if not _code_ran:
            passed, reason = False, "agent did not write or execute any code"
        else:
            with Live("[dim]Judging outcome…[/]", console=console,
                      refresh_per_second=8, transient=True):
                judge_model = "gpt-4o" if provider == "openai" else agent_model
                passed, reason = _auto_judge(task, trace, provider, judge_model)

        # Find intermediate tool errors in the trace even if the task ultimately passed
        import re as _re
        _error_lines = _re.findall(
            r'\[tool:[^\]]+\] → [^\[]*?(?:Traceback \(most recent|[A-Z][a-zA-Z]+Error:|stderr:)[^\[]*',
            trace,
        )
        _had_errors = bool(_error_lines)

        if passed and not _had_errors:
            console.print("[green]✓ PASSED[/] — no errors, nothing to learn.")
            brain.store(trace, label=1)
        elif passed and _had_errors:
            # Agent recovered, but the intermediate mistakes are worth learning from
            error_summary = _error_lines[0][:120].strip()
            console.print(f"[green]✓ PASSED[/] [yellow](recovered from {len(_error_lines)} error(s))[/]")
            console.print(f"[dim]  first error: {error_summary}[/]")
            brain.set_task(task_idx, task=task)
            n_before = store.count
            with Live("[dim]Extracting motif from intermediate errors…[/]", console=console,
                      refresh_per_second=8, transient=True):
                brain.store(trace, label=0,
                            metadata=f"Agent recovered but hit: {error_summary}")
            n_after = store.count
            if n_after > n_before:
                console.print(f"[cyan]→ Motif learned:[/] {store.motifs[-1].name}")
        else:
            console.print(f"[red]✗ FAILED[/] — {reason}")
            brain.set_task(task_idx, task=task)
            n_before = store.count
            with Live("[dim]Extracting motif…[/]", console=console,
                      refresh_per_second=8, transient=True):
                brain.store(trace, label=0, metadata=reason)
            n_after = store.count
            if n_after > n_before:
                console.print(f"[cyan]→ Motif learned:[/] {store.motifs[-1].name}")
            else:
                console.print("[dim]→ No motif extracted (no code in trace or extraction filtered).[/]")

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
    embedder, store, detector, brain, provider, agent_model = _load_stack(store_path)
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
        _run_loop(store, detector, brain, provider, agent_model)
    else:
        _inspect_loop(store, detector, brain)


if __name__ == "__main__":
    main()
