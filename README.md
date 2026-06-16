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
datasets
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
pytest tests/ -q    # 48 tests, ~0.2s
```

---

## Quickstart

```python
from agents import haiku, opus, _build_openai
from pipeline import decompose, attempt, tiered_judge, Forecaster

embedder = _build_openai()

# 1. Decompose any task into atomic, checkable sub-questions
sub_questions = decompose(task, agent=haiku)

# 2. Attempt each sub-question and verify with a tiered judge
#    Haiku judges first; Opus only called when Haiku is uncertain (~20-30% of cases)
verifier = tiered_judge(fast_agent=haiku, strong_agent=opus, gold=gold_answer)

traces, labels = [], []
for q in sub_questions:
    trace = attempt(q, context=retrieved_context, agent=haiku)
    label = int(verifier(q, trace) >= 0.5)
    traces.append(trace)
    labels.append(label)

# 3. Fit the forecaster on your accumulated trace store
fc = Forecaster(embedder, k=10).fit(traces, labels)

# 4. At inference — predict before spending a retry or verification call
new_trace = attempt(new_question, context="", agent=haiku)
if fc.should_intervene(new_trace):
    # P(fail) is high — run your verifier or retry here
    ...

# 5. Grow the store online as more tasks complete
fc.add(new_trace, label=1)    # 1 = success, 0 = failure
```

---

## Verifiers

The only task-specific input is a `Verifier`: a `(question, answer) -> float` callable. Three options ship out of the box:

### `tiered_judge` — Haiku first, Opus on uncertainty (recommended)

```python
from pipeline import tiered_judge

verifier = tiered_judge(
    fast_agent=haiku,
    strong_agent=opus,
    gold=gold_answer,
    uncertainty_band=(0.35, 0.65),   # only escalate to Opus in this range
)
```

Haiku handles confident cases for free; Opus is reserved for the borderline ones. In practice ~0–30% of judgments escalate.

### `self_judge` — no gold answer needed

```python
from pipeline import self_judge

verifier = self_judge(
    judge_agent=opus,              # use a different model than the one being judged
    evidence_fn=retriever,         # optional: ground the judge in retrieved text
)
```

A model grades whether the answer is correct and well-supported. Works without ground truth. Use an independent or stronger model as the judge to avoid self-grading bias.

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

## Retrieval

```python
from pipeline import make_retriever

retriever = make_retriever(corpus_chunks, embedder)
context   = retriever(query, words=1200)    # top chunks up to a word budget
```

Embeds the corpus once; subsequent calls are pure dot-product lookup.

---

## `Forecaster` API

```python
fc = Forecaster(embedder, k=10)

fc.fit(traces, labels)           # list[str], list[int] — bulk load
fc.add(trace, label)             # add one trace online after it completes
fc.predict_fail(trace)           # float in [0, 1] — P(this component fails)
fc.should_intervene(trace, threshold=0.5)   # bool — spend a retry/verify here?
```

Non-parametric (k-NN over embeddings). No training step; the store grows with `add()` as tasks complete. Predictions become reliable once you have ~50+ traces with both passes and failures represented.

---

## Repo layout

| Path | Role |
|---|---|
| `pipeline.py` | Public API: `decompose`, `attempt`, `Forecaster`, `make_retriever`, `tiered_judge`, `self_judge`, `self_consistency` |
| `forecast.py` | Primitives: k-NN (LOO + cross-task), ROC-AUC, Spearman |
| `agents.py` | Vendored `haiku` and `opus` agents + OpenAI embedder (lazy init, keys from env/`.env`) |
| `bench/` | Vendored benchmark loaders and scorers (FanOutQA, MuSiQue) |
| `eval/` | Experiment scripts — each maps to a finding; results in `eval/results/` |
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
- **Failure rate dependency.** The intervention savings scale with how often your agent fails. If the agent already succeeds on >90% of components, there is little to gate.
- **Embedding cost.** Every trace is embedded via the OpenAI API. For high-volume pipelines, batch aggressively or cache embeddings.
- **Verifier quality sets the ceiling.** Noisy labels (from a weak judge) propagate into the forecaster. A programmatic verifier (unit test, exact match) is always preferable to an LLM judge when available.
