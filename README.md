# trace_use

**Forecast agent failure from execution traces — spend retries and verification only where they're needed.**

`trace_use` is a self-contained Python toolkit that embeds an agent's reasoning trace and predicts, per checkable sub-task, whether it is about to fail. Instead of verifying every output or none of them, you get a failure score before committing to a retry.

---

## Why it works

The key insight is to forecast at the **component level**, not the whole-task level. Breaking a task into atomic, independently-checkable sub-questions and forecasting each one separately raises prediction AUC from ~0.45 (chance) to **0.85** — and the signal transfers to task types the model has never seen.

| What you get | Number |
|---|---|
| Per-component failure AUC | **0.85** (vs 0.45 whole-task) |
| Failures caught at 20% verify budget | **31%** (1.56× random) |
| Budget needed to catch 80% of failures | **58–68%** of components (vs 100% naively) |
| Transfers to unseen task types | AUC **0.61–0.73** (leave-one-out) |

The trace carries the signal. Reasoning-only AUC is **0.84** — *how* the agent thinks predicts failure independently of whether the answer looks wrong.

---

## Install

```bash
pip install -r requirements.txt
```

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

Set your API keys (or put them in a `.env` file at the project root):

```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

Run the offline test suite (no API keys needed):

```bash
pytest tests/ -q    # 150 tests, ~2s
```

---

## Quickstart — `run_task`

The simplest entry point. Handles decompose → attempt → forecast → retry → verify → store in one call:

```python
from agents import haiku, _build_openai
from pipeline import run_task, self_judge, Forecaster

embedder  = _build_openai()
forecaster = Forecaster(embedder)
verifier  = self_judge(haiku)

result = run_task(
    task       = "Implement a binary search tree with insert and search.",
    agent      = haiku,
    verifier   = verifier,
    forecaster = forecaster,
)

print(result.summary())
# Task: Implement a binary search tree...
# Components: 5  Pass: 4  Fail: 1  Interventions: 1
```

`run_task` shows a live Rich terminal display as each component runs — P(fail) bar, intervention status, and a reference to the nearest stored failure that triggered the retry. Pass `display=False` to suppress it.

The forecaster grows with every call. The more tasks it sees, the better its predictions.

---

## Manual pipeline

For full control, call the primitives directly:

```python
from agents import haiku, _build_openai
from pipeline import decompose, attempt, self_judge, Forecaster

embedder = _build_openai()

# 1. Decompose any task into atomic, checkable sub-questions
sub_questions = decompose(task, agent=haiku)

# 2. Attempt each sub-question and verify
verifier = self_judge(haiku)

traces, labels = [], []
for q in sub_questions:
    trace = attempt(q, context=retrieved_context, agent=haiku)
    label = int(verifier(q, trace) >= 0.5)
    traces.append(trace)
    labels.append(label)

# 3. Fit the forecaster on your accumulated trace store
fc = Forecaster(embedder).fit(traces, labels)

# 4. At inference — predict before spending a retry or verification call
new_trace = attempt(new_question, context="", agent=haiku)
if fc.should_intervene(new_trace):
    # P(fail) is high — run your verifier or retry here
    ...

# 5. Grow the store online as more tasks complete
fc.add(new_trace, label=1)    # 1 = success, 0 = failure
```

---

## Self-critique retry

When the forecaster flags a component as likely to fail, `run_task` fires a single self-critique retry — zero extra API calls beyond the retry itself. The agent receives its previous trace alongside an instruction to quote the exact step that went wrong and state what not to do before reattempting:

```
Task: {task description}

Question: {sub-question}

Your previous attempt (which may be wrong):
{last 2000 chars of trace}

First, in one sentence quote the exact step above that is wrong or incomplete
and state what NOT to do. Then reattempt from scratch and end with 'ANSWER: ...'
```

The retry is a single prompt. The critique comes from the model itself — no separate diagnosis call, no additional latency beyond the retry. Use `retry=False` to disable and only observe predictions.

---

## Verifiers

The only task-specific input is a `Verifier`: a `(question, answer) -> float` callable. Four options ship out of the box:

### `self_judge` — no gold answer needed (recommended default)

```python
from pipeline import self_judge

verifier = self_judge(
    judge_agent=haiku,
    evidence_fn=retriever,    # optional: ground the judge in retrieved text
)
```

A model grades whether the answer is correct and well-supported. Works without ground truth. Use an independent or stronger model as the judge to avoid self-grading bias.

### `tiered_judge` — Haiku first, Opus on uncertainty

```python
from pipeline import tiered_judge

verifier = tiered_judge(
    fast_agent=haiku,
    strong_agent=opus,
    gold=gold_answer,
    uncertainty_band=(0.35, 0.65),   # only escalate to Opus in this range
)
```

Haiku handles confident cases for free; Opus is reserved for borderline ones. In practice ~0–30% of judgments escalate.

### `code_judge` — execute and check (for coding tasks)

```python
from pipeline import code_judge

def my_check(namespace: dict, stdout: str) -> bool:
    fn = namespace.get("binary_search")
    return fn is not None and fn([1, 2, 3, 4, 5], 3) == 2

verifier = code_judge(my_check)
```

Extracts Python from the agent's answer, `exec()`s it into a namespace, then calls your checker against the namespace and captured stdout. Returns 1.0 on pass, 0.0 on any failure or exception. The forecaster wraps the verifier — when the agent's trace shows uncertainty, a retry fires *before* execution.

### `gold_judge` — explicit ground truth

```python
from pipeline import gold_judge

verifier = gold_judge(gold=reference_answer, agent=haiku)
```

An LLM compares the agent's answer against a known gold answer. Use when you have ground truth and want a simple yes/no judge without tiering.

### `self_consistency` — no judge at all

```python
from pipeline import self_consistency

verifier = self_consistency(
    resample=lambda q: attempt(q, context="", agent=haiku),
    samples=3,
)
```

Re-runs the question independently and returns the fraction of runs that agree with the given answer. No gold, no judge. Works best when final answers are short and extractable (numbers, entities).

---

## `Forecaster` API

```python
fc = Forecaster(embedder, k=10, pca_dim=16)

fc.fit(traces, labels)           # list[str], list[int] — bulk load
fc.add(trace, label)             # add one trace online after it completes
fc.predict_fail(trace)           # float in [0, 1] — P(this component fails)
fc.should_intervene(trace)       # bool — uses adaptive_threshold by default
fc.should_intervene(trace, threshold=0.4)   # override with a fixed threshold
fc.explain(trace, k=3)           # why was this flagged? nearest stored traces
fc.nearest_failure(trace)        # excerpt of the most similar stored failure
fc.adaptive_threshold            # float — current auto-computed threshold
```

Non-parametric (k-NN over embeddings). No training step; the store grows with `add()` as tasks complete. Predictions become reliable once you have ~50+ traces with a mix of passes and failures.

### Prediction and threshold

`predict_fail` is honest k-NN: it returns the fraction of the *k* nearest stored traces that failed — nothing layered on top. A trace landing among past failures scores high; one among past successes scores low.

When the store has no usable signal yet — empty, or only one outcome class seen — the forecaster **abstains** (`predict_fail` returns `0.0`, so `should_intervene` is `False`). You cannot forecast failure before you have seen both passes and failures, and retrying every component is worse than retrying none. Predictions become meaningful once the store holds a mix of both (~50+ traces in practice).

`should_intervene` uses a fixed default cutoff of **0.35** (`adaptive_threshold`) — a component is flagged when a clear majority of its nearest neighbours failed. Override per call when a domain wants a stricter or looser bar:

```python
fc.should_intervene(trace)                 # default cutoff 0.35
fc.should_intervene(trace, threshold=0.5)  # stricter — only flag strong signals
```

### PCA compression

The forecaster reduces OpenAI's 1536-dimensional embeddings to `pca_dim` dimensions (default 16) before k-NN lookup. PCA is fitted once the store exceeds `pca_dim` examples and refitted on each `add()`. Smaller dimensionality improves k-NN separation in sparse early-stage stores; set `pca_dim=0` to disable.

### Interpretability

```python
neighbors = fc.explain(trace, k=3)
# [{"similarity": 0.91, "label": 0, "outcome": "fail", "excerpt": "..."},
#  {"similarity": 0.87, "label": 1, "outcome": "pass", "excerpt": "..."},
#  ...]

failure_ref = fc.nearest_failure(trace)
# "I'll implement flatten. def flatten(lst): result=[]..."
```

`explain` returns the k nearest stored traces by cosine similarity — what the forecaster is pattern-matching against. `nearest_failure` surfaces the single closest stored failure, which `run_task` shows inline in the live display when a retry fires.

---

## Retrieval

```python
from pipeline import make_retriever

retriever = make_retriever(corpus_chunks, embedder)
context   = retriever(query, words=1200)    # top chunks up to a word budget
```

Embeds the corpus once; subsequent calls are pure dot-product lookup. Pass as `retriever=` to `run_task` to ground each sub-question in retrieved context automatically.

---

## Structured results

`run_task` returns a `TaskResult` with per-component detail:

```python
result = run_task(task, agent=haiku, verifier=verifier, forecaster=fc)

result.n_pass          # int
result.n_fail          # int
result.n_intervened    # int
result.summary()       # formatted string

for c in result.components:
    print(c.question, c.p_fail, c.label, c.retried, c.neighbor)
```

Each `ComponentResult` holds the question, the full reasoning trace, the verifier label, the P(fail) score, whether a retry was triggered, and the nearest stored failure excerpt.

---

## trace_use vs answer-only routing

A common alternative is to embed only the final answer and use that to predict failure. For simple short-answer tasks the numbers are close — but trace_use has a structural advantage that matters in practice: the reasoning trace is available **token-by-token during generation**, so failure can be detected and acted on before the model even finishes writing. Answer-only routing requires waiting for the full response, parsing the final line, and only then deciding whether to intervene. That latency difference is the key edge in any real pipeline.

Beyond latency, trace_use pulls further ahead as tasks grow in complexity:

- **Higher verification budgets.** At a 50% budget, trace_use catches **70%** of failures vs **67%** for answer-only. The trace surfaces failure patterns the answer conceals — a model can produce a confident-sounding wrong answer while its reasoning clearly shows the uncertainty.
- **Complex reasoning tasks.** On multi-step or tool-use tasks, failure diverges from correct paths in the reasoning long before the final answer is written. The advantage of the full trace grows with task depth.
- **Streaming and early-exit pipelines.** Intervening mid-generation — rerouting, escalating, or triggering a retry — is only possible when prediction runs on the trace, not the finished answer.

---

## Repo layout

| Path | Role |
|---|---|
| `pipeline.py` | Public API: `run_task`, `decompose`, `attempt`, `Forecaster`, `make_retriever`, `gold_judge`, `tiered_judge`, `self_judge`, `self_consistency`, `code_judge` |
| `forecast.py` | Primitives: k-NN (LOO + cross-task), ROC-AUC, Spearman |
| `display.py` | Rich live terminal display used by `run_task` |
| `agents.py` | Vendored `haiku` and `opus` agents + OpenAI embedder (lazy init, keys from env/`.env`) |
| `tools.py` | Tool-use agent wrapper for structured function calling |
| `bench/` | Vendored benchmark loaders and scorers (FanOutQA, MuSiQue) |
| `eval/` | Experiment scripts — each maps to a README finding; results in `eval/results/` |
| `tests/` | Offline test suite: `test_forecast.py`, `test_pipeline.py` |

---

## Running the headline experiment

```bash
python eval/component_forecast.py --tasks 18 --max-components 6
```

Runs the per-component forecasting experiment on FanOutQA + MuSiQue and reports within-dataset and leave-one-task-type-out AUC. Raw logs and saved trajectories land in `eval/results/`.

---

## Limitations

- **Store size matters.** The k-NN forecaster needs ~50–100 traces with a mix of passes and failures before predictions are reliable. A fresh store with only successes will always return low P(fail).
- **Cold-start cross-domain gaps.** When failures in one domain (e.g., algorithm implementation) look identical to passes in another (e.g., math derivations) in embedding space, P(fail) stays near zero regardless of threshold. This is an embedding blind spot, not a threshold problem.
- **Failure rate dependency.** The intervention savings scale with how often your agent fails. If the agent already succeeds on >90% of components, there is little to gate.
- **Embedding cost.** Every trace is embedded via the OpenAI API. For high-volume pipelines, batch aggressively or cache embeddings.
- **Verifier quality sets the ceiling.** Noisy labels (from a weak judge) propagate into the forecaster. A programmatic verifier (unit test, exact match) is always preferable to an LLM judge when available.
