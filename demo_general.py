"""
demo_general.py — trace-based failure forecasting on diverse everyday tasks.

40 tasks a heavy LLM user would actually run: factual lookup, multi-step math,
language/grammar, logic puzzles, and code generation.  The agent is haiku (no
tools) so failures happen from reasoning errors, not execution gaps.  The
forecaster learns that short over-confident wrong-answer traces cluster together
and that careful step-by-step traces cluster differently — task-domain agnostic.

Alongside the Rich terminal output, a live matplotlib window opens and updates
after every task, showing:
  Left  — trace embeddings projected to 2D (PCA).  Green = pass, Red = fail.
           Lines connect each new trace to its 5 nearest stored neighbors.
  Right — forecaster AUC as it builds over time.

    python3 demo_general.py
"""
import sys, json, time, re, math
from pathlib import Path

sys.path.insert(0, ".")

from trace_use import haiku, opus
from trace_use.agents import _build_openai, _load_env
from trace_use import run_task, Forecaster, self_judge
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

_load_env()
console = Console()

# ── live matplotlib visualiser ────────────────────────────────────────────────

class LiveViz:
    """2D PCA scatter of trace embeddings + AUC trend, updating after each task."""

    def __init__(self):
        import matplotlib
        matplotlib.use("MacOSX")          # non-blocking GUI on macOS
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        self._plt     = plt
        self._patches = mpatches
        self.fig, (self.ax_emb, self.ax_auc) = plt.subplots(
            1, 2, figsize=(13, 5.5))
        self.fig.suptitle(
            "trace-use · live forecaster state", fontsize=13, fontweight="bold")
        self.fig.tight_layout(pad=3)
        plt.ion()
        plt.show(block=False)
        self._aucs:   list[float] = []
        self._labels: list[int]   = []

    def update(self, raw_vecs, labels, auc, new_idx, neighbor_idxs):
        import numpy as np
        from sklearn.decomposition import PCA

        self._labels = list(labels)
        if auc is not None:
            self._aucs.append(auc)

        # ── left: PCA scatter ─────────────────────────────────────────────────
        self.ax_emb.clear()
        n = len(raw_vecs)
        if n >= 3:
            arr  = np.array(raw_vecs, dtype="float32")
            dims = min(2, arr.shape[0], arr.shape[1])
            coords = PCA(n_components=dims).fit_transform(arr) if dims == 2 else \
                     np.column_stack([arr[:, 0], np.zeros(n)])

            c_map = {1: "#2ecc71", 0: "#e74c3c"}
            for i, (x, y) in enumerate(coords):
                col  = "gold" if i == new_idx else c_map.get(labels[i], "#aaaaaa")
                size = 140   if i == new_idx else 55
                zord = 4     if i == new_idx else 2
                self.ax_emb.scatter(x, y, c=col, s=size, zorder=zord,
                                    edgecolors="white", linewidths=0.6)

            # kNN neighbour lines from new point
            if new_idx is not None and neighbor_idxs and new_idx < n:
                nx, ny = coords[new_idx]
                for ni in neighbor_idxs:
                    if ni < n:
                        self.ax_emb.plot(
                            [nx, coords[ni, 0]], [ny, coords[ni, 1]],
                            color="black", alpha=0.25, linewidth=0.9, zorder=1)

        passes  = sum(1 for l in labels if l == 1)
        fails   = sum(1 for l in labels if l == 0)
        legend  = [
            self._patches.Patch(fc="#2ecc71", label=f"pass ({passes})"),
            self._patches.Patch(fc="#e74c3c", label=f"fail ({fails})"),
            self._patches.Patch(fc="gold",    label="current"),
        ]
        self.ax_emb.legend(handles=legend, loc="lower right", fontsize=8)
        self.ax_emb.set_title("Trace embeddings (PCA 2D)\nGreen=pass  Red=fail  Gold=current",
                               fontsize=10)
        self.ax_emb.set_xlabel("PC1"); self.ax_emb.set_ylabel("PC2")
        self.ax_emb.tick_params(labelsize=7)

        # ── right: AUC curve ──────────────────────────────────────────────────
        self.ax_auc.clear()
        if self._aucs:
            xs = list(range(1, len(self._aucs) + 1))
            self.ax_auc.plot(xs, self._aucs, "b-o", markersize=4, linewidth=1.5)
            self.ax_auc.axhline(0.5, color="gray", linestyle="--",
                                alpha=0.6, label="chance (0.5)")
            self.ax_auc.fill_between(xs, 0.5, self._aucs,
                                     where=[a > 0.5 for a in self._aucs],
                                     alpha=0.15, color="blue")
            self.ax_auc.set_ylim(0.3, 1.02)
            self.ax_auc.set_title(
                f"Forecaster AUC over time\nCurrent: {self._aucs[-1]:.3f}", fontsize=10)
            self.ax_auc.set_xlabel("Tasks seen"); self.ax_auc.set_ylabel("AUC")
            self.ax_auc.legend(fontsize=8)
            self.ax_auc.tick_params(labelsize=7)

        self.fig.tight_layout(pad=3)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._plt.pause(0.05)

    def save(self, path: str):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        console.print(f"[dim]Plot saved → {path}[/dim]")


# ── verifiers ─────────────────────────────────────────────────────────────────

def _num_check(expected: float, tol: float = 0.02):
    """Exact-ish numeric check — finds any number in the trace within tol of expected."""
    def verify(q: str, trace: str) -> float:
        nums = re.findall(r"-?\d+(?:[,_]\d+)*(?:\.\d+)?(?:[eE][+-]?\d+)?", trace)
        for raw in nums:
            try:
                v = float(raw.replace(",", "").replace("_", ""))
                if abs(v - expected) <= max(abs(expected) * tol, 1e-6):
                    return 1.0
            except ValueError:
                pass
        return 0.0
    return verify


def _str_check(*accepted: str, case_sensitive: bool = False):
    """Returns 1.0 if any accepted string appears in the trace (last 800 chars).
    Normalises Unicode (e.g. H₂O → H2O) before matching so subscript/superscript
    characters don't cause false misses."""
    import unicodedata
    _SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    _SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKC", s).translate(_SUB).translate(_SUP)
    def verify(q: str, trace: str) -> float:
        tail = _norm(trace[-800:])
        if not case_sensitive:
            tail = tail.lower()
        for a in accepted:
            needle = _norm(a) if case_sensitive else _norm(a).lower()
            if needle in tail:
                return 1.0
        return 0.0
    return verify


# ── tasks ─────────────────────────────────────────────────────────────────────
# Each entry: task text shown to haiku, and a verifier callable.
# Groups 1–2 (easy/medium): haiku almost always passes → builds pass-cluster.
# Groups 3–4 (hard): haiku fails ~35–50% → builds fail-cluster.
# The kNN starts recognising "short overconfident answer" traces as fail-signals.

_judge = self_judge(opus)   # opus grades haiku; different model avoids self-grading bias

TASKS = [
    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 1 — Easy  (~95 % pass)
    # Builds the pass-cluster in the store so the kNN has a reference baseline.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "task": "What is the capital city of France? Answer in one word.",
        "verify": _str_check("paris"),
    },
    {
        "task": "What is 17 × 24? Show your working and give the final number.",
        "verify": _num_check(408),
    },
    {
        "task": "Who wrote the novel '1984'? Give the author's full name.",
        "verify": _str_check("george orwell", "orwell"),
    },
    {
        "task": "How many days are in a leap year? Answer with just the number.",
        "verify": _num_check(366),
    },
    {
        "task": "What is the chemical formula for water? Answer in one line.",
        "verify": _str_check("h2o", "H2O"),
    },
    {
        "task": "What is the square root of 625? Show your reasoning.",
        "verify": _num_check(25),
    },
    {
        "task": "What planet is closest to the Sun? Answer in one word.",
        "verify": _str_check("mercury"),
    },
    {
        "task": "Convert 32 degrees Fahrenheit to Celsius. Give the exact number.",
        "verify": _num_check(0.0, tol=0.05),
    },
    {
        "task": "What is the chemical symbol for gold? Answer with just the symbol.",
        "verify": _str_check("Au", case_sensitive=True),
    },
    {
        "task": "What is 15% of 200? Show your working.",
        "verify": _num_check(30),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 2 — Medium  (~75 % pass)
    # Introduces a few failures; both clusters start forming.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "task": (
            "A store sells a jacket for $120, which is 25% off the original price. "
            "What was the original price? Show your working."
        ),
        "verify": _num_check(160),
    },
    {
        "task": "In what year did the Berlin Wall fall?",
        "verify": _num_check(1989, tol=0),
    },
    {
        "task": (
            "What is the sum of interior angles of a regular hexagon? "
            "Derive it from first principles."
        ),
        "verify": _num_check(720),
    },
    {
        "task": "What is the atomic number of carbon?",
        "verify": _num_check(6, tol=0),
    },
    {
        "task": (
            "A recipe needs 2.5 cups of flour for 4 servings. "
            "How many cups do you need for 10 servings?"
        ),
        "verify": _num_check(6.25),
    },
    {
        "task": (
            "What is the speed of sound in air at 20 °C, in metres per second? "
            "Give an approximate value."
        ),
        "verify": _num_check(343, tol=0.05),
    },
    {
        "task": (
            "A train leaves City A at 09:00 travelling at 90 km/h. "
            "Another train leaves City B (270 km away) at 09:30 travelling toward City A at 60 km/h. "
            "At what time do they meet? Give the answer as HH:MM."
        ),
        "verify": _str_check("11:00", "11:00 am", "1100"),
    },
    {
        "task": "What is 2^10? Give the exact integer.",
        "verify": _num_check(1024, tol=0),
    },
    {
        "task": "Convert 100 degrees Celsius to Fahrenheit. Give the exact number.",
        "verify": _num_check(212, tol=0),
    },
    {
        "task": "What is 5! (five factorial)? Give the exact integer.",
        "verify": _num_check(120, tol=0),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 3 — Hard  (~55 % pass)
    # Multi-step problems; arithmetic traps; precise recall.
    # The forecaster begins distinguishing over-confident wrong-answer traces.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "task": (
            "If you invest $1,000 at 6% annual compound interest, "
            "how much will you have after 5 years? Give the answer to the nearest dollar."
        ),
        "verify": _num_check(1338, tol=0.01),
    },
    {
        "task": (
            "How many ways can you arrange the letters in the word MISSISSIPPI? "
            "Show the calculation."
        ),
        "verify": _num_check(34650, tol=0.001),
    },
    {
        "task": (
            "A ball is thrown upward with an initial velocity of 20 m/s. "
            "Using g = 9.8 m/s², how high does it reach in metres? Show your working."
        ),
        "verify": _num_check(20.4, tol=0.05),
    },
    {
        "task": "Convert the binary number 11011010 to decimal. Show each step.",
        "verify": _num_check(218, tol=0),
    },
    {
        "task": "How many prime numbers are there between 1 and 100? Give the count.",
        "verify": _num_check(25, tol=0),
    },
    {
        # Average-speed trap: haiku usually uses the arithmetic mean (75) not harmonic (72).
        "task": (
            "A car drives 150 km from A to B at 60 km/h, then immediately returns "
            "150 km from B to A at 90 km/h. What is the average speed for the whole trip? "
            "Show your working."
        ),
        "verify": _num_check(72, tol=0.01),
    },
    {
        "task": (
            "What is 37 × 53? Show your working and give the exact integer."
        ),
        "verify": _num_check(1961, tol=0),
    },
    {
        "task": (
            "A shirt costs $45 after a 40% discount. "
            "What was the original price? Show your working."
        ),
        "verify": _num_check(75, tol=0.01),
    },
    {
        "task": (
            "What is the surface area of a cube with side length 4 cm? "
            "Give the exact value in cm²."
        ),
        "verify": _num_check(96, tol=0),
    },
    {
        "task": (
            "How many ways can you choose 3 items from a set of 6 (order does not matter)? "
            "Show the combination formula and calculation."
        ),
        "verify": _num_check(20, tol=0),
    },
    {
        "task": (
            "Simplify the fraction 2/3 + 3/4 − 1/6. "
            "Give the answer as a single fraction in lowest terms."
        ),
        "verify": _str_check("5/4", "1.25"),
    },
    {
        "task": "What is the 15th Fibonacci number? (Sequence starts 1, 1, 2, 3, 5, …)",
        "verify": _num_check(610, tol=0),
    },
    {
        "task": "How many seconds are in one week? Give the exact integer.",
        "verify": _num_check(604800, tol=0),
    },
    {
        "task": (
            "A water tank is 3/4 full. After adding 12 litres it is 4/5 full. "
            "What is the full capacity of the tank in litres?"
        ),
        "verify": _num_check(240, tol=0.01),
    },
    {
        "task": (
            "What is the sum of interior angles of a regular octagon? "
            "Use the formula for polygons."
        ),
        "verify": _num_check(1080, tol=0),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 4 — Hardest  (~40 % pass)
    # Complex reasoning, subtle traps, and tasks requiring complete accuracy.
    # Graded by opus — fails when reasoning is incomplete, imprecise, or wrong.
    # By now the kNN has ~35 stored traces; failures here get flagged early.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "task": "What is 18! (18 factorial)? Give the exact integer.",
        "verify": _num_check(6402373705728000, tol=0.001),
    },
    {
        "task": (
            "What is the probability of rolling a sum of 9 with two standard six-sided dice? "
            "Express as a fraction in lowest terms."
        ),
        "verify": _str_check("4/36", "1/9", "one ninth"),
    },
    {
        "task": (
            "A jar contains 3 red, 5 blue, and 2 green marbles. You draw 2 without replacement. "
            "What is the probability both are blue? Give as a simplified fraction."
        ),
        "verify": _str_check("2/9", "two ninths"),
    },
    {
        "task": "What is the LCM (least common multiple) of 12, 18, and 30?",
        "verify": _num_check(180, tol=0),
    },
    {
        # Requires exactly 3 bullet points AND correct switching probability (2/3 not 1/2).
        "task": (
            "Explain in exactly 3 bullet points why the Monty Hall problem is counter-intuitive "
            "and what the correct answer is. State the exact probability of winning by switching."
        ),
        "verify": _judge,
    },
    {
        # Requires naming the fallacy precisely (appeal to authority / ad verecundiam).
        "task": (
            "Identify the specific logical fallacy and explain why it is a fallacy:\n"
            "'You should take this vitamin supplement — my doctor is a millionaire and takes it every day.'"
        ),
        "verify": _judge,
    },
    {
        # Requires working Python satisfying all three conditions: correct, handles n<=1, efficient.
        "task": (
            "Write a Python function `is_prime(n)` that returns True if n is prime, False otherwise. "
            "It must correctly handle n ≤ 1, n = 2, and run in O(√n) time."
        ),
        "verify": _judge,
    },
    {
        # CAP theorem: all three must be named correctly (Consistency, Availability, Partition tolerance).
        "task": (
            "Explain the CAP theorem in distributed systems in exactly two sentences. "
            "Name all three guarantees it refers to and state what the theorem says about them."
        ),
        "verify": _judge,
    },
    {
        # Requires identifying the harmonic-mean trap and computing 72, not 75.
        "task": (
            "A car drives 150 km from A to B at 60 km/h, then immediately returns "
            "150 km from B to A at 90 km/h. What is the average speed for the whole trip? "
            "Show your working."
        ),
        "verify": _num_check(72, tol=0.01),
    },
    {
        # Modular arithmetic via last-digit cycles — requires finding the period of 7^n mod 10.
        "task": (
            "What is the last digit of 17^100? "
            "Find the repeating cycle of last digits of powers of 17 to determine this."
        ),
        "verify": _num_check(1, tol=0),
    },
    {
        # Clock-angle: hour hand at 3:15 is at 97.5°, minute at 90° → difference 7.5°.
        "task": (
            "A clock shows exactly 3:15. "
            "What is the angle in degrees between the minute hand and the hour hand? "
            "Remember the hour hand moves continuously."
        ),
        "verify": _num_check(7.5, tol=0.1),
    },
    {
        # Requires the 'difference of two squares' insight or multi-step multiplication.
        "task": (
            "What is 847 × 293? Show your working step by step and give the exact integer."
        ),
        "verify": _num_check(248171, tol=0),
    },
    {
        # Decode: K→H, H→E, O→L, O→L, R→O / Z→W, R→O, U→R, O→L, G→D → "HELLO WORLD"
        "task": (
            "Decode the Caesar cipher 'KHOOR ZRUOG' using a right-shift of 3 (i.e. each "
            "letter was shifted 3 forward to encrypt). Give the plaintext."
        ),
        "verify": _str_check("hello world"),
    },
    {
        # Requires log calculation: floor(100 × log10(2)) + 1 = floor(30.103) + 1 = 31.
        "task": (
            "How many digits does 2^100 have? Use logarithms. Show the formula and calculation."
        ),
        "verify": _num_check(31, tol=0),
    },
    {
        # Requires complete Euclidean algorithm steps.
        "task": (
            "Find the GCD of 252 and 105 using the Euclidean algorithm. Show every step."
        ),
        "verify": _num_check(21, tol=0),
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _auc(labels, scores):
    """ROC-AUC: P(fail-trace scores higher on P(fail) than pass-trace).
    Uses 1-p_fail as the sklearn score so label=1 (pass) is the positive class
    with high score when the forecaster is working correctly, which matches
    roc_auc_score's convention.  Returns None if fewer than 2 classes present."""
    if len(set(labels)) < 2 or not scores:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, [1 - p for p in scores])
    except Exception:
        return None


CATEGORIES = {
    range(0, 10):  "Group 1 — Easy",
    range(10, 20): "Group 2 — Medium",
    range(20, 35): "Group 3 — Hard",
    range(35, 50): "Group 4 — Hardest",
}

def _group(idx):
    for r, name in CATEGORIES.items():
        if idx in r:
            return name
    return "Unknown"


# ── run ───────────────────────────────────────────────────────────────────────

console.print()
console.print(Panel(
    "[bold cyan]trace-use — general task forecasting[/bold cyan]\n"
    "[dim]50 everyday tasks · easy → hard · diverse domains "
    "(factual, math, logic, traps, long chains)\n"
    "kNN forecaster learns which reasoning patterns predict failure · "
    "live 2D embedding plot[/dim]",
    border_style="cyan", padding=(1, 3),
))

embedder = _build_openai()
fc       = Forecaster(embedder, k=5, pca_dim=16)
viz      = LiveViz()

# Wrap haiku to always show chain-of-thought reasoning.
# This makes traces structurally different between passes and fails:
# correct: step-by-step reasoning arrives at right intermediate values
# wrong:   reasoning goes off-track at a specific step, which embeds differently
# No specific tools needed — the pattern works with any underlying model.
def agent(prompt: str):
    cot = (
        prompt + "\n\n"
        "Think through this step by step, showing every intermediate step explicitly. "
        "Then give your final answer on its own line as 'ANSWER: ...'."
    )
    return haiku(cot)

# Bypass decompose — every task here is already a single atomic question.
def _passthrough(prompt: str):
    return (prompt.split("\n\nTask: ", 1)[-1].strip(), 0)

results: list[dict] = []
all_p_fails: list[float] = []
all_labels:  list[int]   = []

for i, t in enumerate(TASKS):
    group = _group(i)
    console.rule(f"[bold]{i+1}/{len(TASKS)} · {group}[/bold]", style="dim")

    res = run_task(
        task           = t["task"],
        agent          = agent,
        verifier       = t["verify"],
        forecaster     = fc,
        retry          = True,
        decompose_agent = _passthrough,
        cap            = 1,
    )

    comp   = res.components[0] if res.components else type("_", (), {"label":1,"p_fail":None,"retried":False})()
    label  = comp.label
    p_fail = comp.p_fail if comp.p_fail is not None else 0.0

    all_p_fails.append(p_fail)
    all_labels.append(label)
    results.append({
        "task_idx": i,
        "group":    group,
        "question": t["task"][:120],
        "label":    label,
        "p_fail":   p_fail,
        "retried":  comp.retried,
    })

    # ── live visualisation update ─────────────────────────────────────────────
    cur_auc = _auc(all_labels, all_p_fails)
    n_store = len(fc._raw_vecs)
    # find neighbor indices (indices of 5 nearest stored traces)
    neighbor_idxs: list[int] = []
    if n_store > 1:
        import numpy as np
        arr = np.array(fc._vecs, dtype="float32")
        qv  = np.array(fc._vecs[-1], dtype="float32")
        sims = arr[:-1] @ qv
        k    = min(5, len(sims))
        neighbor_idxs = np.argsort(-sims)[:k].tolist()

    viz.update(
        raw_vecs      = fc._raw_vecs,
        labels        = fc._labels,
        auc           = cur_auc,
        new_idx       = n_store - 1,
        neighbor_idxs = neighbor_idxs,
    )

    outcome = "✓ pass" if label == 1 else "✗ fail"
    pf_str  = f"P(fail)={p_fail:.2f}"
    retry_s = "↺ retried" if comp.retried else "· skip"
    console.print(
        f"  [dim]{i+1:02d}[/dim]  "
        f"{'[green]' if label==1 else '[red]'}{outcome}[/]  "
        f"[dim]{pf_str}  {retry_s}[/dim]"
    )

# ── final summary ─────────────────────────────────────────────────────────────
console.print()
passes = sum(r["label"] for r in results)
fails  = len(results) - passes
final_auc = _auc(all_labels, all_p_fails)

console.print(Panel(
    f"[bold]Tasks:[/bold] {len(results)}   "
    f"[bold green]Pass:[/bold green] {passes}   "
    f"[bold red]Fail:[/bold red] {fails}   "
    f"[bold cyan]AUC:[/bold cyan] {final_auc:.3f}",
    title="Final result", border_style="cyan",
))

# sorted P(fail) table
table = Table(box=box.MINIMAL_DOUBLE_HEAD, show_header=True)
table.add_column("Task (truncated)",    style="dim", width=52)
table.add_column("P(fail)", justify="right")
table.add_column("Actual",  justify="center")
table.add_column("Group",   style="dim", width=20)

for r in sorted(results, key=lambda x: -x["p_fail"]):
    table.add_row(
        r["question"][:50],
        f"{r['p_fail']:.2f}",
        "[red]FAIL[/red]" if r["label"] == 0 else "[green]pass[/green]",
        r["group"],
    )
console.print(table)

# save artefacts
out_dir = Path("eval/results")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "general_run.json").write_text(json.dumps(results, indent=2))
viz.save(str(out_dir / "general_viz.png"))

console.print(
    f"\n[bold cyan]AUC {final_auc:.3f}[/bold cyan]   "
    f"results → eval/results/general_run.json   "
    f"plot → eval/results/general_viz.png"
)
