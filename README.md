# trace_use

**Forecast agent failure from execution traces — spend retries and verification only where they're needed.**

`trace_use` is a self-contained Python toolkit that embeds an agent's reasoning trace into a vector store and predicts, before verification or retry, whether that trace is likely to fail. Instead of verifying every output or none of them, you get a calibrated failure score as the agent reasons.

---

## How it works

### The key insight: forecast at the component level

The signal comes from *how* an agent reasons, not just *what* answer it produces. When broken into atomic sub-questions and forecasted per-component, prediction AUC climbs from ~0.45 (whole-task, near chance) to **0.85** on structured tasks — and the signal transfers to task types the model has never seen.

The mechanism is k-NN over trace embeddings:

1. **Embed** the reasoning trace (OpenAI `text-embedding-3-small` → 1536-dim → PCA-16)
2. **Look up** the k nearest stored traces from prior runs
3. **Score** P(fail) = fraction of those neighbours that failed
4. **Intervene** when P(fail) exceeds a threshold — trigger a self-critique retry, escalate to a stronger model, or route to human review

The forecaster grows online — every completed task adds its trace and outcome to the store. No training step, no fine-tuning, no dataset-specific logic.

### What makes traces discriminative

The forecaster learns from structural differences in how the reasoning unfolds. This matters more than the final answer text:

| Agent / trace type | Why AUC is high or low |
|---|---|
| Tool-use agent (`python_exec`, search, etc.) | Tool call sequences differ structurally: correct traces show successful execution output; failing traces show wrong output, repeated attempts, or no verification step at all |
| Text agent with explicit chain-of-thought | Wrong reasoning produces wrong intermediate values embedded in the trace; correct reasoning produces a coherent chain with consistent intermediate steps |
| One-liner text agent | "Paris" and "Lyon" have near-identical embeddings — minimal structural signal, low AUC |

**Practical rule:** force your agent to show its intermediate steps. A CoT wrapper requires no specific tools and works with any underlying model:

```python
def cot_agent(prompt: str):
    return haiku(
        prompt + "\n\nThink through this step by step, showing every intermediate "
        "step explicitly. Then give your final answer on its own line as 'ANSWER: ...'."
    )
```

---

## Results

### Structured reasoning tasks (FanOutQA + MuSiQue, multi-hop QA)

| Metric | Number |
|---|---|
| Per-component failure AUC | **0.85** (vs 0.45 whole-task, near chance) |
| Failures caught at 20% verify budget | **31%** (1.56× random baseline) |
| Budget to catch 80% of failures | **58–68%** of components (vs 100% naively) |
| Leave-one-task-type-out AUC | **0.61–0.73** (zero-shot transfer) |
| Reasoning-only AUC (no answer) | **0.84** — *how* the agent thinks predicts failure independently of the answer |

### Everyday general tasks (40 diverse tasks: factual, math, logic, language, code)

| Metric | Number |
|---|---|
| Final AUC | **0.684** |
| Failure rate | 20% (8/40 tasks) |
| Retries that recovered | 5 of 8 flagged high-P(fail) tasks |
| Correctly diagnosed irredeemable failures | 3 (retried and still failed) |

The forecaster reaches above-chance AUC within 5 tasks and stabilises above 0.65 for the remainder of the run.

### Negative results (findings, not failures)

- **GSM8K too easy.** Grade-school math is solved near-perfectly by Haiku; with a 98% pass rate there are too few failures to forecast from.
- **Learned representations don't help.** Fine-tuned embeddings on this data performed no better than `text-embedding-3-small` out of the box — the pre-trained semantic space is sufficient.
- **Intervention savings scale with failure rate.** If your agent already succeeds on >90% of components, the marginal gain from gating is small. The tool is most valuable in the 15–40% failure rate band.

---

## Install

```bash
pip install -r requirements.txt
```

Dependencies:

```
anthropic
openai
numpy
scikit-learn
datasets
rich
python-dotenv
pytest
```

Set API keys (or add them to a `.env` file at the project root):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

Verify everything works without spending API credits:

```bash
pytest tests/ -q    # ~150 tests, ~2s, stubbed agent and embedder
```

---

## Quickstart

### Using `run_task` (recommended)

`run_task` is the single-call entry point. It handles decompose → attempt → forecast → retry → verify → store in one shot, with a live Rich terminal display.

```python
from agents import haiku, opus, _build_openai
from pipeline import run_task, self_judge, Forecaster

embedder   = _build_openai()
forecaster = Forecaster(embedder)
verifier   = self_judge(judge_agent=opus)   # opus judges haiku — different model avoids self-grading bias

result = run_task(
    task       = "Explain the CAP theorem and name all three properties.",
    agent      = haiku,
    verifier   = verifier,
    forecaster = forecaster,
    retry      = True,        # fire a self-critique retry when P(fail) is high
)

print(result.summary())
# Task: Explain the CAP theorem...
# Components: 1  Pass: 1  Fail: 0  Interventions: 0
```

The forecaster accumulates experience across calls. Pass the same `Forecaster` instance to every `run_task` call in a session and predictions improve as the store fills.

### With a chain-of-thought agent (general tasks, no tools)

For any task type, wrap the model to emit explicit reasoning steps. This makes pass and fail traces structurally distinguishable in embedding space:

```python
from agents import haiku, opus, _build_openai
from pipeline import run_task, self_judge, Forecaster

def cot_agent(prompt: str):
    return haiku(
        prompt + "\n\nThink through this step by step, showing every intermediate "
        "step explicitly. Then give your final answer on its own line as 'ANSWER: ...'."
    )

embedder   = _build_openai()
forecaster = Forecaster(embedder)

tasks = [
    ("What is 18! (18 factorial)?",             lambda q, t: "6402373705728000" in t),
    ("How many primes are there below 100?",     lambda q, t: "25" in t),
    ("Convert 11011010 binary to decimal.",       lambda q, t: "218" in t),
]

for task, verifier in tasks:
    result = run_task(
        task       = task,
        agent      = cot_agent,
        verifier   = verifier,
        forecaster = forecaster,
        retry      = True,
    )
    c = result.components[0]
    print(f"P(fail)={c.p_fail:.2f}  retried={c.retried}  label={'pass' if c.label else 'fail'}")
```

### With a tool-use agent (coding and execution tasks)

Tool traces are naturally rich and discriminative — the forecaster reaches 0.87 AUC on debugging tasks with no special prompting required.

```python
from agents import tool_agent, opus, _build_openai
from pipeline import run_task, code_judge, Forecaster

agent      = tool_agent(["python_exec"], max_turns=6)
embedder   = _build_openai()
forecaster = Forecaster(embedder)

def check(ns, _):
    fn = ns.get("binary_search")
    return fn and fn([1, 3, 5, 7, 9], 5) == 2 and fn([1, 3, 5, 7, 9], 9) == 4

result = run_task(
    task       = "Fix the bug in this binary search: def binary_search(arr, x): lo, hi = 0, len(arr); while lo < hi: ...",
    agent      = agent,
    verifier   = code_judge(check),
    forecaster = forecaster,
    retry      = True,
)
```

---

## Manual pipeline

For full control, call the primitives directly:

```python
from agents import haiku, _build_openai
from pipeline import decompose, attempt, self_judge, Forecaster

embedder  = _build_openai()
forecaster = Forecaster(embedder)
verifier  = self_judge(judge_agent=haiku)

# 1. Decompose any task into atomic, independently-answerable sub-questions
sub_questions = decompose(task="Explain photosynthesis and list its inputs and outputs.", agent=haiku)
# → ["What is photosynthesis?", "What are the inputs to photosynthesis?", "What are the outputs?"]

# 2. Attempt and verify each sub-question
for q in sub_questions:
    trace = attempt(q, context="", agent=haiku)
    label = int(verifier(q, trace) >= 0.5)

    # 3. Predict failure on the trace before deciding whether to retry
    p_fail = forecaster.predict_fail(trace)
    if p_fail >= forecaster.adaptive_threshold:
        retry_trace = attempt(q, context="", agent=haiku)   # or trigger escalation
        label = int(verifier(q, retry_trace) >= 0.5)
        trace = retry_trace

    # 4. Store the first-attempt trace and its label (not the retry)
    forecaster.add(trace, label)
```

The first-attempt trace is always what you store — it captures the raw reasoning pattern before any correction. Storing retry traces conflates failure patterns with recovery patterns and degrades the kNN signal.

---

## Verifiers

The only task-specific input to the pipeline is a `Verifier`: a `(question, answer) -> float` callable where the float is in `[0, 1]`. Any callable with that signature works.

### `self_judge` — no gold answer needed (recommended default)

```python
from pipeline import self_judge

verifier = self_judge(
    judge_agent=opus,          # use a different model from the one being judged
    evidence_fn=retriever,     # optional: ground the judge in retrieved context
)
```

A model grades whether the answer is correct and well-supported. No ground truth required. Use an independent or stronger model as the judge — a model grading itself is systematically overconfident.

### `tiered_judge` — cheap first, strong on uncertainty

```python
from pipeline import tiered_judge

verifier = tiered_judge(
    fast_agent=haiku,
    strong_agent=opus,
    gold=gold_answer,
    uncertainty_band=(0.35, 0.65),    # escalate to opus only in this range
)
```

Haiku handles confident cases cheaply. Opus is called only when Haiku is borderline — typically 0–30% of judgments. Cost-effective for high-volume pipelines.

### `code_judge` — execute and check

```python
from pipeline import code_judge

def check(namespace: dict, stdout: str) -> bool:
    fn = namespace.get("is_palindrome")
    return (fn is not None
            and fn("racecar") is True
            and fn("Race car") is True    # hidden: case/space normalisation
            and fn("hello") is False)

verifier = code_judge(check)
```

Extracts the first Python block from the agent's answer, `exec()`s it into an isolated namespace, then calls your checker against the namespace and captured stdout. Returns `1.0` on pass, `0.0` on any failure or exception including syntax errors. The forecaster fires a retry *before* execution when the trace resembles past failures — so the agent can self-correct rather than submit broken code.

### `gold_judge` — explicit ground truth

```python
from pipeline import gold_judge

verifier = gold_judge(gold="The Treaty of Westphalia was signed in 1648.", agent=haiku)
```

An LLM compares the agent's answer to a known reference. Use when you have ground truth and want a simple judge without tiering.

### `self_consistency` — no judge at all

```python
from pipeline import self_consistency

verifier = self_consistency(
    resample=lambda q: attempt(q, context="", agent=haiku),
    samples=3,
)
```

Re-runs the question independently and returns the fraction of runs that agree with the given answer. No gold, no judge model. Works best when answers are short and extractable (numbers, named entities). High agreement implies consistency implies likely correctness.

### Custom verifiers

Any callable matching `(question: str, answer: str) -> float` works:

```python
import re

def num_check(expected: float, tol: float = 0.01):
    def verify(q: str, trace: str) -> float:
        for raw in re.findall(r"-?\d[\d,_]*(?:\.\d+)?(?:[eE][+-]?\d+)?", trace[-800:]):
            if abs(float(raw.replace(",", "").replace("_", "")) - expected) <= abs(expected) * tol + 1e-9:
                return 1.0
        return 0.0
    return verify

verifier = num_check(expected=160.0)   # original price of discounted jacket
```

---

## `Forecaster` API

```python
from pipeline import Forecaster
from agents import _build_openai

embedder = _build_openai()
fc = Forecaster(embedder, k=10, pca_dim=16)
```

| Method / property | Description |
|---|---|
| `fc.fit(traces, labels)` | Bulk-load a list of trace strings and int labels (1=pass, 0=fail) |
| `fc.add(trace, label)` | Add one trace online after a task completes |
| `fc.predict_fail(trace)` | `float` in `[0, 1]` — P(this trace fails), based on kNN fraction |
| `fc.should_intervene(trace)` | `bool` — uses `adaptive_threshold` by default |
| `fc.should_intervene(trace, threshold=0.4)` | Override with a fixed threshold |
| `fc.explain(trace, k=3)` | List of k nearest stored traces with similarity, label, and excerpt |
| `fc.nearest_failure(trace)` | Excerpt of the closest stored failure trace |
| `fc.adaptive_threshold` | Current auto-computed threshold (`float`) |

### Cold-start and prediction reliability

When the store has fewer than `k` examples, or only one outcome class has been seen, the forecaster **abstains**: `predict_fail` returns `0.0` and `should_intervene` is `False`. Predictions become reliable at approximately **50 traces** with a meaningful mix of passes and failures.

### PCA compression

Embeddings are reduced from 1536 dimensions to `pca_dim` (default 16) before kNN lookup. PCA is fitted lazily once the store exceeds `pca_dim` examples and refitted on each `add()`. Smaller dimensionality sharpens kNN separation in sparse stores. Set `pca_dim=0` to disable.

### Adaptive threshold

`should_intervene` uses `adaptive_threshold` by default — computed as:

```
threshold = fail_rate + (1 - fail_rate) × 0.20, capped at 0.80
```

This scales the intervention bar with the empirical failure rate of your store, so a domain with 5% failures doesn't trigger on every task and a domain with 40% failures doesn't miss everything. Override per call with an explicit `threshold=` argument.

### Interpretability

```python
neighbors = fc.explain(trace, k=3)
# [{"similarity": 0.91, "label": 0, "outcome": "fail",
#   "excerpt": "I'll review the function... looks correct... ANSWER: ..."},
#  {"similarity": 0.87, "label": 1, "outcome": "pass",
#   "excerpt": "Step 1: compute 120/0.75 = 160... ANSWER: 160"},
#  ...]

failure_ref = fc.nearest_failure(trace)
# "I'll implement flatten. def flatten(lst): result=[]..."
```

`explain` returns what the forecaster is pattern-matching against. `nearest_failure` surfaces the single closest stored failure, shown inline in the live terminal display when `run_task` fires a retry.

---

## Retrieval

```python
from pipeline import make_retriever

retriever = make_retriever(corpus_chunks, embedder)
context   = retriever("photosynthesis inputs", words=1200)   # top chunks up to word budget
```

Embeds the corpus once on construction; subsequent calls are pure dot-product lookup. Pass as `retriever=` to `run_task` to automatically ground each sub-question in relevant retrieved context. Also usable as the `evidence_fn` argument to `self_judge`.

---

## Structured results

`run_task` returns a `TaskResult`:

```python
result = run_task(task, agent=cot_agent, verifier=verifier, forecaster=fc)

result.n_pass          # int — components that passed
result.n_fail          # int — components that failed
result.n_intervened    # int — components where a retry was fired
result.summary()       # formatted one-line string

for c in result.components:
    print(c.question)   # str — the sub-question
    print(c.trace)      # str — full reasoning trace
    print(c.p_fail)     # float | None — forecaster score (None if abstained)
    print(c.label)      # int  — 1=pass, 0=fail
    print(c.retried)    # bool — whether a retry was triggered
    print(c.neighbor)   # str | None — nearest stored failure excerpt shown in display
```

---

## `run_task` parameters

```python
run_task(
    task           = "...",                   # the task string
    agent          = haiku,                   # callable: prompt -> text
    verifier       = verifier,                # callable: (q, trace) -> float
    forecaster     = fc,                      # Forecaster instance (optional)
    retriever      = retriever,               # retriever for context (optional)
    threshold      = None,                    # override adaptive_threshold (optional)
    cap            = 8,                       # max sub-questions from decompose
    display        = True,                    # show Rich live terminal output
    retry          = True,                    # fire self-critique retry on high P(fail)
    retry_agent    = None,                    # different agent for retries (defaults to agent)
    decompose_agent= None,                    # different agent for decomposition
)
```

---

## Self-critique retry

When the forecaster flags a component as high P(fail), `run_task` fires a single self-critique retry — one additional API call beyond the original attempt. The agent receives its previous trace alongside an instruction to identify the exact error:

```
{context}

Question: {sub-question}

Your previous attempt (which may be wrong):
{last 2000 chars of trace}

First, in one sentence quote the exact step above that is wrong or incomplete
and state what NOT to do. Then reattempt from scratch and end with 'ANSWER: ...'
```

The critique is self-generated — no separate diagnosis call, no additional latency beyond the retry itself. The first attempt's trace (not the retry) is stored in the forecaster, so the store captures failure patterns in their raw form.

Use `retry=False` to observe failure predictions without intervening.

---

## trace_use vs answer-only routing

A common alternative is to embed only the final answer and predict failure from that alone. For simple short-answer tasks the gap is small. trace_use has a structural advantage that grows with task complexity:

- **Latency.** The trace is available token-by-token during generation. Failure can be detected and a retry triggered before the model finishes writing. Answer-only routing requires waiting for the full response.
- **Complex reasoning.** On multi-step or tool-use tasks, the trace diverges from correct paths well before the final answer token. Answer-only prediction misses that early signal.
- **Confident-wrong answers.** A model can produce a confident-sounding wrong answer while its reasoning clearly shows the error. The trace exposes this; the answer conceals it.
- **Streaming pipelines.** Mid-generation intervention — rerouting, escalating, early exit — is only possible when prediction runs on the trace.

At a 50% verification budget: trace_use catches **70%** of failures vs **67%** for answer-only. The gap widens on harder tasks.

---

## Demos

| Script | What it shows |
|---|---|
| `python3 demo_general.py` | 40 diverse everyday tasks (factual, math, logic, language, code). CoT haiku agent, no tools. Live matplotlib embedding plot + AUC curve. Final AUC ~0.68. |
| `python3 demo_debug.py` | 29 Python debugging tasks with hidden edge-case bugs. Tool-use agent with `python_exec`. AUC ~0.87. |
| `python3 demo_large.py` | Large-scale run across 80+ mixed tasks. Full Rich display. |

---

## Repo layout

| Path | Role |
|---|---|
| `pipeline.py` | Public API: `run_task`, `decompose`, `attempt`, `Forecaster`, `make_retriever`, `gold_judge`, `tiered_judge`, `self_judge`, `self_consistency`, `code_judge` |
| `forecast.py` | Primitives: kNN (LOO + cross-task), ROC-AUC, Spearman |
| `display.py` | Rich live terminal display used by `run_task` |
| `agents.py` | Vendored `haiku`, `opus`, `tool_agent` + OpenAI embedder (lazy init, keys from env/`.env`) |
| `demo_general.py` | General everyday tasks demo with live visualisation |
| `demo_debug.py` | Python debugging demo with edge-case bug suite |
| `demo_large.py` | Large mixed-task demo |
| `bench/` | Vendored benchmark loaders and scorers (FanOutQA, MuSiQue) |
| `eval/` | Experiment scripts — each maps to a README finding; results in `eval/results/` |
| `tests/` | Offline test suite: `test_forecast.py`, `test_pipeline.py` |

---

## Running the headline experiment

```bash
python eval/component_forecast.py --tasks 18 --max-components 6
```

Runs per-component forecasting on FanOutQA + MuSiQue and reports within-dataset and leave-one-task-type-out AUC. Raw logs and saved trajectories land in `eval/results/`.

---

## Limitations

- **Store size matters.** kNN needs ~50–100 traces with a mix of passes and failures before predictions are reliable. A fresh store returns low P(fail) until failures accumulate.
- **Trace richness is a prerequisite.** One-liner text responses produce near-identical embeddings regardless of correctness. Force chain-of-thought or use a tool-use agent to create discriminative traces.
- **Embedding cost.** Every trace is embedded via the OpenAI API. For high-volume pipelines, batch aggressively or cache embeddings per trace hash.
- **Verifier quality sets the ceiling.** Noisy labels from a weak or self-grading judge propagate into the forecaster store. A programmatic verifier (unit test, exact match, regex) is always preferable to an LLM judge when one is available.
- **Failure rate dependency.** Intervention savings scale with how often your agent fails. Above a 90% pass rate there is little to gate; the tool is most valuable in the 15–40% failure band.
- **Cross-domain cold-start gaps.** When failure patterns in one domain (e.g., algorithm debugging) are geometrically close to pass patterns in another (e.g., math derivations) in the embedding space, P(fail) stays near zero regardless of threshold. This is an embedding geometry issue, not a threshold problem.
