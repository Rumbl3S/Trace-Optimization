# trace_use

**Forecast agent failure from execution traces — spend retries and verification only where they're needed.**

`trace_use` is a self-contained Python toolkit with two complementary layers:

1. **`pipeline.py` — Offline forecaster.** Embeds completed reasoning traces, stores them with pass/fail labels, and predicts P(fail) for new traces via kNN. Used after a task completes to decide whether to verify or retry.

2. **`brain.py` — Online brain.** Intercepts tool calls *during* execution, runs deterministic probe tests on the live code, checks against a kNN store of past code snippets, and injects targeted corrective feedback into the tool result mid-turn — before the model finishes. Requires no training step and no stored history to start.

---

## How it works

### The key insight: the trace carries the failure signal

*How* an agent reasons predicts failure independently of whether the final answer looks wrong. Reasoning-only AUC on structured multi-hop tasks reaches **0.84** — wrong paths diverge from correct ones in embedding space well before the final answer token.

This means:
- Failure can be detected mid-generation, not just after the fact
- The signal transfers to task types never seen before (leave-one-out AUC 0.61–0.73)
- One-liner responses have near-zero signal; multi-step reasoning (tool traces, CoT) is what makes it work

### What trace richness means in practice

| Agent type | Signal quality | Reason |
|---|---|---|
| Tool-use agent (`python_exec`, search) | High (AUC 0.87) | Tool call sequences differ structurally: correct traces show successful execution, failing traces show wrong output or repeated attempts |
| Text agent with explicit chain-of-thought | Good (AUC 0.68) | Wrong reasoning produces wrong intermediate values; correct reasoning forms a coherent chain |
| One-liner text agent | Near chance | "Paris" and "Lyon" have near-identical embeddings |

**Practical rule:** force multi-step output. A CoT wrapper requires no tools and works with any model:

```python
def cot_agent(prompt: str):
    return haiku(
        prompt + "\n\nThink step by step, showing every intermediate step explicitly. "
        "End with 'ANSWER: ...'."
    )
```

---

## The Brain (inference-time, `brain.py`)

`BrainAgent` is a monitor that attaches to any tool-use agent via a hook called after every tool execution. It provides two independent failure signals and combines them into a single P(fail) score:

### Signal 1 — Probe tests (deterministic)

For each task you can register a `probe_fn(ns: dict) -> list[str]`. After the agent's first `python_exec` call, the brain re-runs the code and calls the probe with the resulting namespace. If the probe returns failures, the brain immediately injects specific corrective feedback into the tool result — before the next LLM turn:

```
STOP — your code fails these tests RIGHT NOW:
  ✗ luhn('18')=False, expected True
  FIX: double every SECOND digit from the RIGHTMOST position, not the left;
       '18'→keep 8, double 1→2; sum=10→valid
Fix the specific issue above — do not rewrite the whole function.
```

The agent reads this as the tool result and corrects the bug in the next turn.

### Signal 2 — kNN over stored code snippets (learned)

A `FailureStore` keeps code-snippet embeddings with pass/fail labels. Every `python_exec` input is compared against stored snippets at query time (cosine similarity). When P(fail) exceeds threshold, similar failed snippets surface as context — the agent knows which patterns have caused failures before.

### Signal 3 — Trajectory prefix kNN + Markov chain (learned)

A `TrajectoryStore` stores completed runs as ordered sequences of chunk embeddings:

- **Prefix kNN:** The mean embedding of the live trajectory's prefix is compared to the same-length prefix of each stored run. kNN fraction of failing runs → P(fail). Works from the first stored run.
- **Markov state failure rate:** Once ≥30 chunks are stored, k-means discretizes all chunk embeddings into thought-state clusters. Each state tracks what fraction of runs that visited it eventually failed. At inference: current chunk → nearest cluster → P(fail | state). Captures: *"models reasoning this way tend to get the wrong answer."*

Signals 2 and 3 combine: `p_fail = 0.55 × p_markov + 0.45 × p_prefix`.

### Wiring it up

```python
from agents import build_embedder, tool_agent
from brain import BrainAgent

brain = BrainAgent(build_embedder(), threshold=0.35, k=5)

agent = tool_agent(["python_exec"], max_turns=6, model="claude-haiku-4-5-20251001")
agent.monitor = brain  # single line to attach the brain

def probe_luhn(ns):
    fn = ns.get("luhn_check")
    if not fn: return ["luhn_check not defined"]
    fails = []
    if fn("18") is not True:
        fails.append(
            "luhn('18')=False, expected True. "
            "FIX: double from RIGHTMOST digit, not left"
        )
    return fails

for i, task in enumerate(tasks):
    brain.set_task(i, probe_fn=probe_luhn)  # register probe for this task
    brain.reset()
    trace, tokens = agent(task["prompt"])

    code = extract_code(trace)
    passed = run_checks(code)

    # Store first-attempt trace with first-attempt label
    brain.store(trace, int(passed))
    if code:
        brain.store_code(code, int(passed))
```

The brain fires at most **2 times per task** to avoid warning fatigue. Probes fire on first-attempt bugs; kNN fires later in the run when enough failures are stored.

### `BrainAgent` public API

| Method | Description |
|---|---|
| `brain.set_task(idx, probe_fn=fn)` | Register the current task index and optional deterministic probe |
| `brain.reset()` | Clear buffer, bail flag, and intervention counter before a new task |
| `brain.on_tool_call(name, input_dict, result)` | Hook called by `tool_agent` on every tool execution; returns modified result or `None` |
| `brain.store(trace, label, metadata="")` | Store a completed run's full trajectory |
| `brain.store_code(code, label, metadata="")` | Store a code snippet with its pass/fail label |
| `brain.seed(items)` | Pre-populate stores with known pass/fail examples |
| `brain.wrap_any(agent_or_model)` | Attach brain monitoring to any model string or callable |
| `brain._code_interventions` | How many times the brain fired during the current task |
| `brain.n_stored` | Total runs stored in the trajectory store |

### `BrainAgent` storage invariant

Always store the **first-attempt trace** with the **first-attempt label** — even when a retry fires and recovers a failed component. Storing retry traces conflates recovery patterns with failure patterns and degrades the kNN signal.

---

## The Forecaster (offline, `pipeline.py`)

`Forecaster` operates after task completion. It embeds full traces, stores them with labels, and predicts P(fail) via kNN with optional PCA compression. Integrates with `run_task` for end-to-end orchestration.

### Quickstart

```python
from agents import haiku, opus, build_embedder
from pipeline import run_task, self_judge, Forecaster

embedder   = build_embedder()
forecaster = Forecaster(embedder)
verifier   = self_judge(judge_agent=opus)  # use a different model to avoid self-grading bias

result = run_task(
    task       = "Explain the CAP theorem and name all three properties.",
    agent      = haiku,
    verifier   = verifier,
    forecaster = forecaster,
    retry      = True,
)

print(result.summary())
# Task: Explain the CAP theorem...
# Components: 1  Pass: 1  Fail: 0  Interventions: 0
```

The forecaster accumulates experience across calls. Pass the same instance to every `run_task` call; predictions improve as the store fills.

### With a tool-use agent

Tool traces are naturally rich — 0.87 AUC on Python debugging tasks with no special prompting:

```python
from agents import tool_agent, build_embedder
from pipeline import run_task, code_judge, Forecaster

agent = tool_agent(["python_exec"], max_turns=6)
fc    = Forecaster(build_embedder())

def check(ns, _):
    fn = ns.get("binary_search")
    return fn and fn([1,3,5,7,9], 5) == 2 and fn([1,3,5,7,9], 9) == 4

result = run_task(
    task       = "Fix the off-by-one in this binary search: ...",
    agent      = agent,
    verifier   = code_judge(check),
    forecaster = fc,
    retry      = True,
)
```

### Manual pipeline

For full control without `run_task`:

```python
from agents import haiku, build_embedder
from pipeline import decompose, attempt, self_judge, Forecaster

fc       = Forecaster(build_embedder())
verifier = self_judge(judge_agent=haiku)

sub_questions = decompose(task="Explain photosynthesis and list inputs and outputs.", agent=haiku)

for q in sub_questions:
    trace = attempt(q, context="", agent=haiku)
    label = int(verifier(q, trace) >= 0.5)

    p_fail = fc.predict_fail(trace)
    if p_fail >= fc.adaptive_threshold:
        trace = attempt(q, context="", agent=haiku)  # retry
        label = int(verifier(q, trace) >= 0.5)

    fc.add(trace, label)   # store first-attempt trace
```

### `Forecaster` API

```python
fc = Forecaster(embedder, k=10, pca_dim=16)
```

| Method / property | Description |
|---|---|
| `fc.fit(traces, labels)` | Bulk-load trace strings and int labels (1=pass, 0=fail) |
| `fc.add(trace, label)` | Add one trace online after a task completes |
| `fc.predict_fail(trace)` | `float` in `[0,1]` — P(this trace fails), based on kNN fraction |
| `fc.should_intervene(trace)` | `bool` — uses adaptive threshold |
| `fc.should_intervene(trace, threshold=0.4)` | Override with fixed threshold |
| `fc.explain(trace, k=3)` | Nearest stored traces with similarity, label, and excerpt |
| `fc.nearest_failure(trace)` | Excerpt of the closest stored failure trace |
| `fc.adaptive_threshold` | Auto-computed threshold: `fail_rate + (1 - fail_rate) × 0.20` |

Cold-start: when the store has fewer than `k` examples or only one outcome class, `predict_fail` returns `0.0` and `should_intervene` is `False`. Predictions become reliable at approximately **50 traces** with a mix of passes and failures.

---

## Results

### Multi-hop QA (FanOutQA + MuSiQue)

| Metric | Value |
|---|---|
| Per-component failure AUC | **0.85** (vs 0.45 whole-task, near chance) |
| Reasoning-only AUC (no answer text) | **0.84** |
| Failures caught at 20% verify budget | **31%** (1.56× random baseline) |
| Budget to catch 80% of failures | **58–68%** of components |
| Leave-one-task-type-out AUC | **0.61–0.73** (zero-shot transfer) |

### Python debugging tasks — tool agent (demo_debug.py)

| Metric | Value |
|---|---|
| AUC | **0.87** |
| Tasks | 29 Python debugging tasks with hidden edge-case bugs |
| Agent | `tool_agent(["python_exec"])` |

### Diverse everyday tasks — CoT agent (demo_general.py)

| Metric | Value |
|---|---|
| AUC | **0.68** |
| Tasks | 40 diverse tasks (factual, math, logic, language, code) |
| Agent | CoT-wrapped haiku (no tools) |

### Mixed eval — 30 diverse domains (eval/eval_fires.py)

30 tasks across algorithms, math/logic puzzles, geography, history, and science. Haiku as base model. Brain intercepts tool calls with probe tests.

| Domain | First-attempt | Final (after retry) | Brain fired |
|---|---|---|---|
| Algorithms (11 tasks) | 9/11 (82%) | 9/11 | **5/11 (45%)** |
| Math / logic (6 tasks) | 6/6 (100%) | 6/6 | 0/6 |
| Science (10 tasks) | 9/10 (90%) | 10/10 | 0/10 |
| Geography / History (3 tasks) | 3/3 | 3/3 | 0/3 |
| **Overall** | **27/30 (90%)** | **28/30 (93%)** | **5/30 (17%)** |

All 5 brain fires were on genuine first-attempt bugs (Luhn doubling direction, Soundex H/W rule, day-of-week formula, decode-string stack, look-and-say iteration). The brain's probe gave specific corrective feedback; haiku self-corrected in the next turn. The retry loop handled the remaining failures by feeding back exact failure descriptions.

### Negative results (findings, not failures)

- **GSM8K is too easy.** Grade-school math is solved near-perfectly by Haiku; near-zero failure rate leaves nothing to forecast.
- **Learned embeddings don't help.** Fine-tuned embeddings on this data performed no better than the out-of-the-box sentence-transformers model — the pre-trained semantic space is sufficient.
- **Intervention savings scale with failure rate.** Above ~90% first-attempt accuracy, the marginal gain from interception is small. The brain is most valuable in the 15–40% failure rate band.
- **Text/factual tasks fire rarely.** At temperature 0, haiku is accurate on geography, history, and science recall on first attempt. Brain fires are concentrated on algorithmic tasks where deterministic probe tests can catch bugs.

---

## Live dashboard (`eval/viz_brain.py`)

`BrainViz` renders a 4-panel dark dashboard that updates after every task:

| Panel | What it shows |
|---|---|
| **Neuron Graph** | k-means thought-state nodes sized by visit count, colored green→red by failure rate; transition edges weighted by frequency. Raw scatter shown before Markov activates (≥30 chunks). |
| **Trajectory Map** | PCA-2D of all stored chunk embeddings. Each completed run is a polyline: green=pass, red=fail. Shows clustering of failure trajectories. |
| **Score Timeline** | Per-task pass/fail bars + cumulative accuracy line. |
| **Fire Report** | Brain fires per task + cumulative fire rate vs 30% dashed reference. |

```python
from eval.viz_brain import BrainViz
from pathlib import Path

viz = BrainViz()
# inside your eval loop:
viz.update(brain, results, fire_counts)
viz.save(Path("eval/results/brain_overview.png"))
```

---

## Install

```bash
pip install -r requirements.txt
```

Set API keys (`.env` file at project root, or export directly):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...           # only needed if sentence-transformers unavailable
```

Verify without spending API credits:

```bash
pytest tests/ -q     # ~150 tests, ~2s, fully stubbed
```

---

## Embedder

`build_embedder()` in `agents.py` tries local first, falls back to OpenAI:

1. **`sentence-transformers` (preferred):** `all-MiniLM-L6-v2`, 384-dim, free, ~10ms/chunk on CPU, no API key
2. **OpenAI `text-embedding-3-small`:** 1536-dim, requires `OPENAI_API_KEY`

The brain and the Forecaster accept either — the interface is the same.

---

## Verifiers (`pipeline.py`)

The only task-specific input to the pipeline is a `Verifier`: `(question, answer) -> float` in `[0, 1]`.

### `self_judge` — no gold answer needed (default)

```python
verifier = self_judge(judge_agent=opus, evidence_fn=retriever)
```

A model grades the answer. Use a different model from the one being judged — self-grading is systematically overconfident.

### `tiered_judge` — cheap first, strong on uncertainty

```python
verifier = tiered_judge(fast_agent=haiku, strong_agent=opus, gold=gold,
                        uncertainty_band=(0.35, 0.65))
```

Haiku handles confident cases; opus is called only when haiku is borderline (~0–30% of judgments).

### `code_judge` — execute and check

```python
def check(namespace: dict, stdout: str) -> bool:
    fn = namespace.get("is_palindrome")
    return fn and fn("racecar") and not fn("hello")

verifier = code_judge(check)
```

Extracts the first Python block, execs it into an isolated namespace, calls your checker.

### `gold_judge` — explicit ground truth

```python
verifier = gold_judge(gold="The Treaty of Westphalia was signed in 1648.", agent=haiku)
```

### `self_consistency` — no judge at all

```python
verifier = self_consistency(resample=lambda q: attempt(q, "", agent=haiku), samples=3)
```

Re-runs the question and returns the fraction of runs that agree. Works best for short extractable answers.

---

## `run_task` reference

```python
run_task(
    task            = "...",      # task string
    agent           = haiku,      # callable: prompt -> text or (text, tokens)
    verifier        = verifier,   # callable: (q, trace) -> float
    forecaster      = fc,         # Forecaster instance (optional)
    retriever       = retriever,  # context retriever (optional)
    threshold       = None,       # override adaptive threshold (optional)
    cap             = 8,          # max sub-questions from decompose
    display         = True,       # Rich live terminal output
    retry           = True,       # fire self-critique retry on high P(fail)
    retry_agent     = None,       # different agent for retries
    decompose_agent = None,       # different agent for decomposition
)
```

Returns a `TaskResult` with `.n_pass`, `.n_fail`, `.n_intervened`, `.summary()`, and per-component `.components` (each with `.question`, `.trace`, `.p_fail`, `.label`, `.retried`, `.neighbor`).

---

## Repo layout

| Path | Role |
|---|---|
| `pipeline.py` | Public API: `run_task`, `decompose`, `attempt`, `Forecaster`, `make_retriever`, all verifiers (`gold_judge`, `tiered_judge`, `self_judge`, `self_consistency`, `code_judge`) |
| `brain.py` | `BrainAgent`, `TrajectoryStore`, `FailureStore` — inference-time failure interception |
| `forecast.py` | Primitives: `knn_predict`, `knn_predict_cross`, `auc`, `spearman` |
| `display.py` | Rich live terminal display used by `run_task` |
| `agents.py` | Vendored `haiku`, `opus`, `tool_agent`, `streaming_agent`, `build_embedder` (lazy init, keys from env/`.env`) |
| `demo_general.py` | 40 diverse everyday tasks; CoT haiku agent; live matplotlib plot. AUC ~0.68 |
| `demo_debug.py` | 29 Python debugging tasks with hidden edge-case bugs; tool_agent. AUC ~0.87 |
| `demo_large.py` | Large-scale mixed run; full Rich display |
| `bench/` | Vendored benchmark loaders and scorers (FanOutQA, MuSiQue) |
| `eval/eval_fires.py` | 30-task brain eval across algorithms, math, geography, history, science |
| `eval/viz_brain.py` | Live 4-panel brain dashboard |
| `eval/` | Experiment scripts — each maps to one README finding; results in `eval/results/` |
| `tests/` | Offline test suite: `test_forecast.py`, `test_pipeline.py` (~150 tests, ~2s, fully stubbed) |

---

## Demos

```bash
python demo_general.py   # 40 diverse tasks, CoT, live embedding plot + AUC curve
python demo_debug.py     # 29 debugging tasks, tool agent, AUC ~0.87
python demo_large.py     # 80+ mixed tasks, full Rich display
python eval/eval_fires.py   # 30 diverse-domain tasks, brain interception, live dashboard
```

---

## Limitations

- **Store size matters.** The kNN needs ~50–100 traces with a mix of passes and failures before predictions are reliable. A cold store returns low P(fail) until failures accumulate. The brain's probe tests (deterministic, no history needed) work immediately.
- **Trace richness is a prerequisite.** One-liner responses produce near-identical embeddings regardless of correctness. Force chain-of-thought output or use a tool-use agent.
- **Verifier quality sets the ceiling.** Noisy labels propagate into the store and corrupt the kNN. Programmatic checks (unit tests, exact match) are always preferable to LLM judges when available. Use a different model than the one being judged for self-judge verifiers.
- **Failure rate dependency.** Savings scale with how often the agent fails. Above 90% pass rate the gains are marginal; the tool is most valuable in the 15–40% failure band.
- **Brain fires on code tasks only.** The probe-test mechanism requires `python_exec` tool calls to intercept. Text-only agents use the trajectory kNN signal, which needs accumulated failures to fire.
- **Embedding cost.** Every trace is embedded locally (free, ~10ms/chunk with sentence-transformers) or via the OpenAI API (billed). For high-volume pipelines, cache embeddings by trace hash.
