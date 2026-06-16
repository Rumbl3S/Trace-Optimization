# trace_use — forecasting agent failure from execution traces

> Don't throw away a failed agent attempt. Its **trace** — what it explored, ruled out,
> and couldn't answer — predicts failure on *similar* future work. Use that to **spend
> retries/verification only where they're needed**, instead of everywhere or nowhere.

A **self-contained** toolkit (no external project dependencies). It began as a research
spin-off from a token-efficiency project — see *Why* below — and vendors everything it
needs: a Haiku agent + OpenAI embedder (`agents.py`), the benchmark loaders (`bench/`), and
a couple of helpers (`_util.py`). The product surface is `pipeline.py`; `forecast.py` holds
the primitives.

---

## Why we shifted to this

Our prior work chased **token efficiency** for a single agent solve (single → audit →
gap-fill → compose). It worked, then hit a wall we proved repeatedly: **accuracy ∝ evidence
∝ tokens.** On dispersed tasks you cannot cut tokens without losing accuracy — a full-context
call scored 0.72 at 92k tokens; the retrieval pipeline 0.48 at 34k. Cheaper *or* more
accurate, not both. Every lever (tight caps, dedup, distillation) lived on that one curve.

So we changed the **axis**. Instead of making *one* attempt cheaper, **learn across attempts**:
treat each run's trace as data, predict which work is about to fail, and pay for a fix only
there. That turns "optimize this solve" into "optimize the *workflow*."

**How it differs:** the prior work optimized *within* a single attempt (retrieval,
composition). This is a *meta-layer across* attempts — it doesn't make a solve better, it
decides **where to spend effort** by forecasting failure.

---

## What we tried, and what held up

All numbers are leave-one-out k-NN **AUC** (0.5 = chance), Haiku agent, OpenAI embeddings,
temperature 0. Raw logs in [`eval/results/`](eval/results/).

| question | result | verdict |
|---|---|---|
| Do trace embeddings predict failure? (MuSiQue, controlled) | **AUC 0.85** (task-only 0.74) | ✅ yes |
| Does the *process* predict it, pre-answer? | reasoning-only **0.80**; answer-only 0.90 | ✅ usable before the answer |
| Does it work on fan-out (task-level)? | **0.45 — chance** | ❌ no, at task level |
| …because the binary pass/fail label hides it? | continuous target: fan-out Spearman **+0.19** | ✅ the label was the problem |
| Per *checkable component* instead of per task? | fan-out **0.846** (was 0.45) | ✅ **the fix** |
| Does the signal transfer to unseen task types? | component transfer **0.65–0.73** | ✅ generalizes |
| Does a *learned* representation beat off-the-shelf? | raw k-NN 0.88 > learned 0.79 (n=40) | ❌ data-bound, not method-bound |
| Turn forecasts into saved tokens? | catch 80% of failures at **58% budget** (1.5× random) | ⚠️ real but failure-rate-dependent |
| A non-QA family (GSM8K math)? | Haiku 94% success — too easy to forecast | ⚠️ needs a task the agent fails at |

### The one finding that matters
Predicting per **checkable component** turned fan-out from unpredictable (0.45) to highly
predictable (**0.85**), **generically** — and the signal **transfers across task types**
(0.65–0.73). The trace was never the problem; the coarse, whole-task pass/fail label was.

### Honest limits
- Small scale (≤152 components, single runs); labels are LLM-judge-generated (noisy).
- Learned representations don't beat off-the-shelf embeddings yet — that needs **more data**.
- The intervention payoff scales with the **failure rate**: big when failures are rare,
  small when most things fail. And we have **no non-QA family with enough failures** yet.

---

## The generalizable recipe (and the API)

The product surface is [`pipeline.py`](pipeline.py) — fully task-agnostic. The **only**
task-specific input is a `Verifier` (a unit test, an LLM judge, exact match) — and you can
**auto-generate it with zero manual labeling**: `self_judge(judge_agent)` (reference-free
grading, no gold) or `self_consistency(resample_fn)` (re-run and measure agreement — no
judge at all). Or plug in a real outcome signal you already have (tests pass, tool result,
user accept/reject), which is the most reliable labeler.

```python
from pipeline import decompose, attempt, gold_judge, Forecaster

# 1. break ANY task into checkable components, attempt + verify each
verify = my_test_or_judge                      # (question, answer) -> 0..1  (you provide this)
for q in decompose(task, agent):
    trace = attempt(q, retrieve(q), agent)
    store.append((trace, verify(q, trace)))

# 2. forecast failure on new components; intervene only where it's likely
fc = Forecaster(embedder).fit(*zip(*store))
if fc.should_intervene(new_trace):             # spend a retry/verify only here
    ...
```

Recipe in one line: **forecast per checkable component, bring your own verifier, intervene
only on the risky ones — everything else is task-agnostic.**

---

## Repo map

| path | role |
|---|---|
| `pipeline.py` | the importable API: `decompose / attempt / verify / Forecaster / make_retriever` + auto-verifiers (`self_judge`, `self_consistency`) |
| `forecast.py` | low-level primitives: k-NN (LOO + cross), ROC-AUC, Spearman |
| `agents.py` | vendored Haiku agent + OpenAI embedder (lazy clients, keys from env/`.env`) |
| `bench/` | vendored benchmark loaders + scorers (FanOutQA, MuSiQue) |
| `_util.py` | vendored helper (`select_for_single`) |
| `requirements.txt` | `anthropic openai numpy datasets python-dotenv pytest` |
| `eval/component_forecast.py` | **headline** — per-component forecasting, within + transfer |
| `eval/gen_balanced.py`, `gen_gsm8k.py` | generate labeled trajectories (QA retrieval / math) |
| `eval/analyze.py`, `generalize.py` | within-dataset and leave-one-task-type-out AUC |
| `eval/ablation_reasoning.py` | reasoning-only vs answer-only vs full-trace |
| `eval/learned_repr.py` | learned representation vs off-the-shelf embeddings |
| `eval/intervention_policy.py` | gain curve — failures caught per unit of fix budget |
| `eval/results/` | raw logs + saved trajectory/component datasets |
| `tests/` | offline gates (no API key): `test_forecast.py`, `test_pipeline.py` |

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # offline gates, no API key
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...
python eval/component_forecast.py --tasks 18 --max-components 6   # the headline experiment
```

---

## Status & next

Proven: per-component forecasting works and transfers, with a generic, tested API.
Not yet: scale (hundreds–thousands of trajectories so learned representations can pay off),
a **non-QA family the agent actually fails at** (harder math / coding / tool-use), and a
live intervention loop measuring real tokens/latency saved.
