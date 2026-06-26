"""Live terminal display for trace_use sessions using Rich."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── status constants ──────────────────────────────────────────────────────────
_PENDING    = ("⏳", "dim")
_ATTEMPTING = ("⟳ ", "yellow")
_PASS       = ("✓", "green")
_FAIL       = ("✗", "red")
_INTERVENE  = ("↺", "bold yellow")
_SKIP       = ("·", "dim")


@dataclass
class ComponentState:
    question: str
    status:   str = "pending"       # pending | attempting | done
    p_fail:   Optional[float] = None
    label:    Optional[int]   = None
    retried:  bool            = False
    neighbor: Optional[str]   = None   # nearest similar failure, for explain
    elapsed:  float           = 0.0


@dataclass
class SessionState:
    task:       str
    agent_name: str
    store_size: int = 0
    components: List[ComponentState] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def n_done(self):    return sum(1 for c in self.components if c.status == "done")
    @property
    def n_pass(self):    return sum(1 for c in self.components if c.label == 1)
    @property
    def n_fail(self):    return sum(1 for c in self.components if c.label == 0)
    @property
    def n_intervened(self): return sum(1 for c in self.components if c.retried)
    @property
    def elapsed(self):   return time.time() - self.started_at


def _fail_bar(p: float, width: int = 10) -> Text:
    filled = round(p * width)
    bar = "█" * filled + "▁" * (width - filled)
    colour = "red" if p >= 0.6 else "yellow" if p >= 0.35 else "green"
    return Text(bar, style=colour)


def _render(state: SessionState) -> Panel:
    # ── header ────────────────────────────────────────────────────────────────
    task_short = state.task[:72] + ("…" if len(state.task) > 72 else "")
    header = Text.assemble(
        ("Task  ", "bold dim"),
        (task_short, "bold white"),
        "\n",
        ("Agent ", "dim"), (state.agent_name, "cyan"),
        ("  ·  Store: ", "dim"), (str(state.store_size), "cyan"),
        (" traces", "dim"),
    )

    # ── component table ───────────────────────────────────────────────────────
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1),
                expand=True, show_edge=False)
    tbl.add_column("idx",  width=3,  no_wrap=True)
    tbl.add_column("q",    ratio=4,  no_wrap=True)
    tbl.add_column("bar",  width=14, no_wrap=True)
    tbl.add_column("pfail",width=6,  no_wrap=True)
    tbl.add_column("dec",  width=22, no_wrap=True)
    tbl.add_column("out",  width=8,  no_wrap=True)

    for i, comp in enumerate(state.components):
        idx_t = Text(f" {i+1}", style="bold dim")
        q_t   = Text(comp.question[:52], style="white" if comp.status != "pending" else "dim")

        if comp.p_fail is None:
            bar_t  = Text("")
            pf_t   = Text("")
            if comp.status == "attempting":
                dec_t = Text("⟳  attempting…", style="yellow")
            else:
                dec_t = Text("pending", style="dim")
            out_t  = Text("")
        else:
            bar_t = _fail_bar(comp.p_fail)
            pf_t  = Text(f"{comp.p_fail:.2f}",
                         style="red" if comp.p_fail >= 0.6
                         else "yellow" if comp.p_fail >= 0.35 else "green")

            if comp.retried:
                dec_t = Text("↺ intervened + retry", style="bold yellow")
            elif comp.p_fail >= 0.5:
                dec_t = Text("⚑ intervene", style="yellow")
            else:
                dec_t = Text("· skip verify", style="dim")

            if comp.label == 1:
                out_t = Text("✓ pass", style="bold green")
            elif comp.label == 0:
                out_t = Text("✗ fail", style="bold red")
            else:
                out_t = Text("⟳ …", style="yellow")

        tbl.add_row(idx_t, q_t, bar_t, pf_t, dec_t, out_t)

        # show nearest-failure explanation when intervening
        if comp.neighbor and comp.retried:
            nb_t = Text(f"   └ similar past failure: \"{comp.neighbor[:60]}...\"",
                        style="dim italic")
            tbl.add_row(Text(""), nb_t, Text(""), Text(""), Text(""), Text(""))

    # ── footer ────────────────────────────────────────────────────────────────
    done = state.n_done
    total = len(state.components)
    elapsed = f"{state.elapsed:.1f}s"
    footer = Text.assemble(
        (f"{done}/{total} done", "bold white"),
        ("  ·  ", "dim"),
        (f"{state.n_pass} pass", "green"),
        ("  ", "dim"),
        (f"{state.n_fail} fail", "red" if state.n_fail else "dim"),
        ("  ·  ", "dim"),
        (f"{state.n_intervened} interventions", "yellow" if state.n_intervened else "dim"),
        ("  ·  ", "dim"),
        (elapsed, "dim"),
    )

    from rich.console import Group
    body = Group(header, Text(""), tbl, Text(""), footer)
    return Panel(body, title="[bold cyan]trace_use[/bold cyan]",
                 border_style="cyan", padding=(0, 1))


class TraceDisplay:
    """Context manager wrapping a Rich Live display for a trace_use session."""

    def __init__(self, task: str, agent_name: str = "agent", store_size: int = 0):
        self.state = SessionState(task=task, agent_name=agent_name,
                                  store_size=store_size)
        self._live = Live(console=console, refresh_per_second=8,
                          vertical_overflow="crop")

    def __enter__(self):
        self._live.__enter__()
        self._refresh()
        return self

    def __exit__(self, *args):
        self._refresh()
        self._live.__exit__(*args)

    def set_components(self, questions: list[str]):
        self.state.components = [ComponentState(q) for q in questions]
        self._refresh()

    def set_attempting(self, i: int):
        self.state.components[i].status = "attempting"
        self._refresh()

    def set_result(self, i: int, p_fail: float, label: int,
                   retried: bool = False, neighbor: str | None = None):
        c = self.state.components[i]
        c.p_fail   = p_fail
        c.label    = label
        c.retried  = retried
        c.neighbor = neighbor
        c.status   = "done"
        self._refresh()

    def update_store(self, n: int):
        self.state.store_size = n
        self._refresh()

    def _refresh(self):
        self._live.update(_render(self.state))


def print_summary(results: list[dict]):
    """Print a final summary table after the session."""
    tbl = Table(title="Session Summary", box=box.ROUNDED,
                border_style="cyan", show_lines=True)
    tbl.add_column("#",         style="dim",   width=4)
    tbl.add_column("Component", style="white", ratio=3)
    tbl.add_column("P(fail)",   justify="right", width=8)
    tbl.add_column("Action",    width=20)
    tbl.add_column("Outcome",   width=8)

    for i, r in enumerate(results):
        p = r.get("p_fail")
        pf_s = f"{p:.3f}" if p is not None else "—"
        pf_style = ("red" if p and p >= 0.6 else
                    "yellow" if p and p >= 0.35 else "green") if p else "dim"

        action  = "[yellow]↺ retried[/yellow]" if r.get("retried") else (
                  "[yellow]⚑ intervene[/yellow]" if p and p >= 0.5 else
                  "[dim]· skip[/dim]")
        outcome = ("[bold green]✓ pass[/bold green]" if r.get("label") == 1 else
                   "[bold red]✗ fail[/bold red]")

        tbl.add_row(str(i + 1), r["question"][:65],
                    Text(pf_s, style=pf_style), action, outcome)

    console.print()
    console.print(tbl)
