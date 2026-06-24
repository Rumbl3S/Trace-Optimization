"""demo_vibe.py — Brain-monitored vibecoding: expression parser.

Building a recursive-descent expression evaluator incrementally.
Each task adds a new feature. Hidden tests target precedence rules LLMs get wrong:

  - ** is RIGHT-associative: 2**3**2 = 2**(3**2) = 512, NOT (2**3)**2 = 64
  - Unary minus has LOWER precedence than **: -2**2 = -(2**2) = -4, NOT (-2)**2 = 4

Architecture:
  Before each task, the brain queries the failure store.
  If p_fail >= threshold → injects context about past failures into the prompt.
  Haiku always writes a complete response (no mid-stream interruption).

Run:
    python demo_vibe.py
"""
import contextlib
import io
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from use import BrainSession

console = Console()

NO_EVAL = (
    "Do NOT use Python's eval(), compile(), or exec(). "
    "Use standalone functions only — no classes. "
    "The entry point must be a top-level function, not a method."
)
COT = (
    "\n\nWork through your approach step by step, explaining each design decision. "
    "Then provide the complete implementation in a ```python code block."
)


# ── check functions (defined before TASKS) ────────────────────────────────────

def _check_tokenize(ns) -> bool:
    fn = ns.get("tokenize")
    if not fn:
        return False
    def _types(toks):
        return [t[0] for t in toks]
    try:
        # Check token types (flexible on exact value format)
        assert _types(fn("1 + 2"))  == ["NUM", "OP", "NUM"]
        assert _types(fn("2**3"))   == ["NUM", "OP", "NUM"]
        assert _types(fn("2 ** 3")) == ["NUM", "OP", "NUM"]
        assert _types(fn("(2+3)"))  == ["LPAREN", "NUM", "OP", "NUM", "RPAREN"]
        # ** must be a single OP token (not two separate * tokens)
        toks = fn("2**3")
        assert any(v == "**" for _, v in toks), "** not a single token"
        return True
    except Exception:
        return False


def _check_evaluate_basic(ns) -> bool:
    ev = ns.get("evaluate")
    if not ev:
        return False
    try:
        return (
            abs(ev("2 + 3") - 5.0) < 1e-9 and
            abs(ev("10 - 4") - 6.0) < 1e-9 and
            abs(ev("3 * 4") - 12.0) < 1e-9 and
            abs(ev("10 / 2") - 5.0) < 1e-9 and
            abs(ev("2 + 3 * 4") - 14.0) < 1e-9    # basic precedence
        )
    except Exception:
        return False


def _check_power(ns) -> bool:
    ev = ns.get("evaluate")
    if not ev:
        return False
    try:
        return (
            ev("2 ** 3") == 8.0 and
            ev("2 + 3 ** 2") == 11.0 and
            # HIDDEN: right-associativity (LLMs almost always get this wrong)
            ev("2 ** 3 ** 2") == 512.0 and   # 2**(3**2)=512, NOT 64
            ev("2 ** 2 ** 3") == 256.0 and   # 2**(2**3)=256, NOT 64
            ev("(2 ** 3) ** 2") == 64.0      # parens override
        )
    except Exception:
        return False


def _check_unary(ns) -> bool:
    ev = ns.get("evaluate")
    if not ev:
        return False
    try:
        return (
            ev("-3") == -3.0 and
            ev("--3") == 3.0 and
            ev("-(2 + 3)") == -5.0 and
            # HIDDEN: unary minus has LOWER precedence than ** (matches Python)
            ev("-2 ** 2") == -4.0 and       # -(2**2)=-4, NOT (-2)**2=4
            ev("2 ** -1") == 0.5 and        # right-side can be negative
            ev("-2 ** 2 + 1") == -3.0       # (-4)+1=-3
        )
    except Exception:
        return False


def _check_comparison(ns) -> bool:
    ev = ns.get("evaluate")
    if not ev:
        return False
    try:
        return (
            ev("3 > 2") == 1.0 and
            ev("2 > 3") == 0.0 and
            ev("2 + 3 > 4") == 1.0 and
            ev("-2 ** 2 < 0") == 1.0 and     # -4 < 0
            ev("10 - 3 - 2 == 5") == 1.0     # left-assoc preserved
        )
    except Exception:
        return False


def _check_variables(ns) -> bool:
    ev = ns.get("evaluate")
    if not ev:
        return False
    try:
        assert ev("x + 1", {"x": 3}) == 4.0
        assert ev("-x ** 2", {"x": 3}) == -9.0             # -(3**2)=-9
        assert ev("x ** y ** z", {"x": 2, "y": 3, "z": 2}) == 512.0  # right-assoc
        assert ev("10 - a - b == 5", {"a": 3, "b": 2}) == 1.0
        try:
            ev("undef", {})
            return False
        except (NameError, KeyError):
            return True
    except Exception:
        return False


TASKS = [
    {
        "name": "evaluate: +,-,*,/",
        "prompt": (
            "Write a Python function `evaluate(expr: str) -> float` that parses and "
            "evaluates arithmetic expressions with +, -, *, /. "
            "Respect operator precedence (* and / before + and -). Support parentheses. "
            f"{NO_EVAL}"
        ),
        "check": _check_evaluate_basic,
    },
    {
        "name": "evaluate: left-assoc check",
        "prompt": (
            "Write a Python function `evaluate(expr: str) -> float` that handles "
            "+, -, *, / with correct left-to-right associativity and parentheses. "
            "Example: 10-3-2=5 (not 9). "
            f"{NO_EVAL}"
        ),
        "check": _check_evaluate_basic,
    },
    {
        "name": "Add ** (power)",
        "prompt": (
            "Extend your `evaluate` function to also support the ** (power) operator. "
            "** has higher precedence than *, /, +, -. "
            f"{NO_EVAL}"
        ),
        "check": _check_power,
    },
    {
        "name": "Add unary minus",
        "prompt": (
            "Extend your `evaluate` function to support unary minus. "
            "Examples: -3 evaluates to -3, -(2+3) evaluates to -5. "
            f"{NO_EVAL}"
        ),
        "check": _check_unary,
    },
    {
        "name": "Add comparisons",
        "prompt": (
            "Extend your `evaluate` function to support comparison operators: "
            "< > <= >= == != "
            "Return 1.0 for True and 0.0 for False. "
            "Comparisons have lower precedence than arithmetic. "
            f"{NO_EVAL}"
        ),
        "check": _check_comparison,
    },
    {
        "name": "Add variables",
        "prompt": (
            "Extend `evaluate` to accept `evaluate(expr, variables=None) -> float`. "
            "Variable names match [a-zA-Z_][a-zA-Z0-9_]*. "
            "Raise NameError for undefined variables. "
            f"{NO_EVAL}"
        ),
        "check": _check_variables,
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> str | None:
    # Closed blocks (any language tag)
    closed = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    # Unclosed python blocks — haiku sometimes omits the closing ```
    unclosed = re.findall(r"```(?:python|py)[^\n]*\n(.*?)$", text, re.DOTALL)
    blocks = closed + unclosed

    if not blocks:
        return None

    def _try_compile(b: str) -> bool:
        try:
            compile(b, "<check>", "exec")
            return True
        except Exception:
            return False

    # Prefer first block with a definition that compiles
    for b in blocks:
        b = b.strip()
        if ("def " in b or "class " in b) and _try_compile(b):
            return b
    # Fall back: first compilable block
    for b in blocks:
        b = b.strip()
        if _try_compile(b):
            return b
    return None


def _run(code: str, check_fn) -> tuple[bool, str]:
    ns: dict = {}
    try:
        exec(compile(code, "<vibe>", "exec"), ns)
    except Exception as e:
        return False, f"Compile error: {e}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return bool(check_fn(ns)), ""
    except Exception as e:
        return False, f"Test error: {e}"


def _diagnose(code: str) -> str:
    """Run known-failing expressions to name the bug."""
    ns: dict = {}
    try:
        exec(compile(code, "<v>", "exec"), ns)
    except Exception:
        return "exec error"
    ev = ns.get("evaluate")
    if not ev:
        return "evaluate() not defined"
    probes = [
        ("2**3**2",  512.0, "** right-assoc: 2**(3**2)=512"),
        ("-2**2",    -4.0,  "unary minus < **: -(2**2)=-4"),
        ("10-3-2",   5.0,   "subtraction left-assoc"),
        ("2**2**3",  256.0, "** right-assoc: 2**(2**3)=256"),
        ("-2**2+1",  -3.0,  "combined: -4+1=-3"),
    ]
    hits = []
    for expr, expected, label in probes:
        try:
            got = ev(expr)
            if abs(float(got) - expected) > 1e-9:
                hits.append(f"{expr}={got}≠{expected} [{label}]")
        except Exception as exc:
            hits.append(f"{expr}→error({exc})")
    return "; ".join(hits[:2]) if hits else "unknown failure"


# ── pre-seed patterns ─────────────────────────────────────────────────────────

SEEDS = [
    {
        "trace": (
            "def parse_power(tokens, pos):\n"
            "    left, pos = parse_primary(tokens, pos)\n"
            "    while pos < len(tokens) and tokens[pos][1] == '**':\n"
            "        pos += 1\n"
            "        right, pos = parse_primary(tokens, pos)\n"
            "        left = left ** right  # LEFT-ASSOC: (2**3)**2=64, wrong!\n"
            "    return left, pos\n"
            "# 2**3**2 = 64.0 (wrong)"
        ),
        "label": 0,
        "metadata": (
            "** is left-associative (iterative loop). 2**3**2=64 instead of 512. "
            "FIX: right-recursive — `if '**': exp=parse_power(...); return base**exp`"
        ),
    },
    {
        "trace": (
            "def parse_power(tokens, pos):\n"
            "    base, pos = parse_unary(tokens, pos)  # WRONG: power calls unary\n"
            "    if pos < len(tokens) and tokens[pos][1] == '**':\n"
            "        pos += 1\n"
            "        exp, pos = parse_unary(tokens, pos)\n"
            "        return base ** exp, pos\n"
            "    return base, pos\n"
            "# -2**2 = 4.0 (wrong, should be -4)"
        ),
        "label": 0,
        "metadata": (
            "parse_power calls parse_unary → -2**2=(-2)**2=4 (wrong, should be -4). "
            "FIX: parse_power calls parse_primary; parse_unary calls parse_power. "
            "Call chain: mult→power→unary→primary"
        ),
    },
    {
        "trace": (
            "# Correct: right-recursive power, unary wraps power\n"
            "def parse_power(tokens, pos):\n"
            "    base, pos = parse_primary(tokens, pos)\n"
            "    if pos < len(tokens) and tokens[pos][1] == '**':\n"
            "        pos += 1\n"
            "        exp, pos = parse_power(tokens, pos)  # right-recursive\n"
            "        return base ** exp, pos\n"
            "    return base, pos\n"
            "def parse_unary(tokens, pos):\n"
            "    if tokens[pos][1] in ('-', '+'):\n"
            "        op = tokens[pos][1]; pos += 1\n"
            "        val, pos = parse_power(tokens, pos)  # unary wraps power\n"
            "        return (-val if op == '-' else +val), pos\n"
            "    return parse_power(tokens, pos)\n"
            "# 2**3**2=512 ✓  -2**2=-4 ✓"
        ),
        "label": 1,
        "metadata": "Correct: parse_unary calls parse_power (right-recursive). 2**3**2=512, -2**2=-4.",
    },
]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    session  = BrainSession(threshold=0.25, k=5)
    model    = "claude-haiku-4-5-20251001"
    agent    = session.agent(model)

    console.print(Panel(
        "[bold cyan]VibeSession — Expression Parser[/]\n\n"
        "Building a recursive-descent evaluator, one feature per task.\n"
        "Hidden tests: [yellow]**[/] right-assoc  •  unary minus precedence\n\n"
        "Brain checks [bold]before[/] each generation.\n"
        "p_fail ≥ 0.25 → injects past-failure context into the prompt.\n\n"
        "[dim]from use import BrainSession  |  session.agent(model)  |  session.teach(success)[/]",
        box=box.ROUNDED,
    ))

    session.seed(SEEDS)
    console.print(f"  [dim]Brain pre-seeded with {len(SEEDS)} known failure patterns[/]\n")

    rows:    list[tuple] = []
    n_fires: int         = 0

    for i, task in enumerate(TASKS):
        n = i + 1
        console.rule(f"[bold]Task {n}/{len(TASKS)}: {task['name']}[/]")

        # ── Generate (brain predicts + injects context automatically) ─────────
        text = agent(task["prompt"] + COT)
        p_fail = session.p_fail
        fired  = session.fired
        if fired:
            n_fires += 1
            console.print(f"  [yellow]⚡ BRAIN fired  p_fail={p_fail:.2f} — injecting failure context[/]")
        elif p_fail is not None:
            console.print(f"  [dim]p_fail={p_fail:.2f}[/]")

        # ── Verify ────────────────────────────────────────────────────────────
        code = _extract_code(text)
        if code is None:
            passed, detail = False, "No code block found"
        else:
            passed, detail = _run(code, task["check"])
            if not passed and not detail and code and i >= 1:
                detail = _diagnose(code)

        # ── Store outcome ─────────────────────────────────────────────────────
        session.teach(passed, metadata=detail if not passed else "")

        status = "PASS" if passed else "FAIL"
        pf_s   = f"{p_fail:.2f}" if p_fail is not None else "—"
        rows.append((n, task["name"], status, pf_s, fired, detail))

        color = "green" if passed else "red"
        console.print(f"  [{color}]{status}[/{color}]  {detail or ''}")

    # ── Summary ───────────────────────────────────────────────────────────────
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1))
    table.add_column("#",       width=3,  style="dim")
    table.add_column("Task",    width=26)
    table.add_column("Result",  width=6)
    table.add_column("p_fail",  width=7)
    table.add_column("Brain",   width=5)
    table.add_column("Detail",  width=50)

    for n, name, status, pf_s, fired, detail in rows:
        c     = "green" if status == "PASS" else "red"
        fire  = "[yellow]⚡[/]" if fired else "—"
        table.add_row(str(n), name[:26], f"[{c}]{status}[/{c}]", pf_s, fire, detail[:50])

    console.rule("[bold]Session Summary[/]")
    console.print(table)
    console.print(
        f"\n  [bold]{session.n_pass}/{session.n_stored} passed[/]  "
        f"{n_fires} brain fires  "
        f"[dim]{session.n_stored} stored ({session.n_pass}✓ {session.n_fail}✗)[/]"
    )


if __name__ == "__main__":
    main()
