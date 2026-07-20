# trace_use

[![PyPI](https://img.shields.io/pypi/v/trace-use)](https://pypi.org/project/trace-use/)
[![Python](https://img.shields.io/pypi/pyversions/trace-use)](https://pypi.org/project/trace-use/)

**Learn failure patterns from LLM agent traces and intercept recurrences before they execute.**

`trace_use` attaches to any tool-use LLM agent, learns reusable logical failure patterns from past errors, and fires targeted interventions *before* the next occurrence executes — with zero false positives on held-out near-miss tasks.

---

## The problem

LLM agents fail in systematic, repeatable ways. The same logical mistake — retry all exceptions instead of selective ones, sort on a single key when a tiebreak is required — recurs across different tasks with different surface code. Every recurrence costs tokens on a failure → diagnosis → retry loop. Nothing prevents the mistake from appearing on the next task.

---

## How it works

When a task fails, the brain makes one background LLM call to extract *why* — not what the code looked like, but what logical requirement was violated. This produces a `FailureMotif`:

```
FailureMotif
  required_condition:  "task requires selective retry based on error type"
  violation_condition: "except Exception catches all types without type check"
  recommendation:      "check exception type or .status_code before deciding to retry"
```

On every subsequent task, before `python_exec` runs, the brain retrieves candidate motifs by embedding similarity and asks an LLM judge: *can you find exact text in this task satisfying the required condition, and an exact line in this code matching the violation condition?* If both are concretely grounded in the actual text, the brain fires before execution. If either is absent, it stays silent.

The agent never executes the bad code, never reads wrong output, and never spends a turn diagnosing the error.

---

## Architecture

```
BrainAgent
├── push(text)
│   └── accumulates live reasoning text from the agent's streaming output
│
├── before_tool_call(name, input_dict)  ← fires BEFORE code executes
│   ├── retrieve candidate motifs (embedding cosine similarity ≥ 0.35, top_k=4)
│   ├── for each candidate:
│   │   ├── call applicability judge  (Haiku; structured JSON; max_tokens=320)
│   │   └── validate deterministically (_validate_judge_result, threshold=0.80)
│   └── if any motif passes both → return STOP/FIX message; else return None
│
├── on_tool_call(name, input_dict, result)  ← fires AFTER execution
│   └── stall detection: ≥2 consecutive unproductive calls → redirect message
│
└── store(trace, label, metadata="")
    └── label=0: extract one FailureMotif in a background thread (non-blocking)
        ├── if motif is new → add to MotifStore with embedding
        └── if motif matches existing (via update_id) → update and re-embed
```

### Signal 1 — Stall detector (fires from task 1, no stored history needed)

After 2+ consecutive unproductive `python_exec` calls (empty code or empty output), the brain injects:

```
[BRAIN — STALL after 2 unproductive calls]
Stop repeating the same empty call. Try a completely different approach.
```

### Signal 2 — Learned-motif detection (fires after first failure of a class)

On subsequent tasks, before each `python_exec` call:

1. **Retrieve** candidate motifs by embedding similarity (cosine ≥ 0.35 floor)
2. **Judge** each candidate with a Haiku call returning exact quotes:
   ```json
   {
     "applies": true,
     "confidence": 0.92,
     "requirement_quote": "Only retry exceptions listed in retry_on.",
     "violation_quote": "except Exception as e:",
     "explanation": "code retries all exceptions instead of checking type",
     "recommendation": "check type(e).__name__ in retry_on before retrying"
   }
   ```
3. **Validate deterministically** (`_validate_judge_result`):
   - `applies=true` and `confidence ≥ 0.80`
   - Both quotes non-empty
   - `recommendation` at least 10 characters
   - No vague phrases (`"task implies"`, `"likely"`, `"might"`, `"probably"`, etc.)
   - Both quotes grounded in actual task/code/reasoning text (substring match or ≥70% word overlap)
4. **Fire** only when all checks pass — injects a STOP message before the bad code runs:

```
⚠️ BRAIN:
STOP: The monitor detected a likely logical failure before execution.

Evidence (Learned pattern: Non-Selective Retry Catches All Exception Types):
  - Requirement: Only retry exceptions listed in retry_on.
  - Violation:   except Exception as e:
  - Explanation: code retries all exceptions instead of checking type

Required correction:
  check type(e).__name__ in retry_on before retrying

Revise the code before calling the tool again.
```

### Why not p_fail or trajectory similarity?

Prior iterations used embedding-based trajectory kNN and p_fail scores. These were removed because:

- **False positive problem.** A finance task and a sorting task can have similar reasoning embeddings but completely unrelated failure modes.
- **Vague interventions.** "This trajectory resembles past failures" gives the agent nothing actionable.
- **The structured proof requirement solves both.** A retry motif cannot fire on a sort task because "selective retry" will not appear in a sort task's text. Cross-domain contamination is structurally impossible.

---

## Wiring it up

```python
from trace_use import BrainAgent, build_embedder, tool_agent

brain         = BrainAgent(build_embedder(), k=4, threshold=0.80)
agent         = tool_agent(["python_exec"], max_turns=8, model="claude-haiku-4-5-20251001")
agent.monitor = brain                    # single line to attach

for i, task in enumerate(tasks):
    brain.set_task(i, task=task["prompt"])
    brain.reset()

    trace, tokens = agent(task["prompt"])
    passed        = run_checks(trace)

    # Always store first-attempt traces with first-attempt labels.
    # Never store retry traces — they conflate recovery with failure patterns.
    brain.store(trace, int(passed), metadata=task.get("failure_reason", ""))
```

### `BrainAgent` public API

| Method / property | Description |
|---|---|
| `brain.set_task(idx, task="")` | Register the current task index and task description (passed to the judge for grounding) |
| `brain.reset()` | Clear reasoning buffer and intervention counter before a new task |
| `brain.push(text)` | Accumulate a reasoning chunk; called automatically by `tool_agent` monitor hook |
| `brain.before_tool_call(name, input_dict)` | Pre-execution hook — returns STOP message or `None` |
| `brain.on_tool_call(name, input_dict, result)` | Post-execution hook — stall detection; returns modified result or `None` |
| `brain.store(trace, label, metadata="")` | Store a completed run; on `label=0`, extracts a motif in the background |
| `brain.n_stored` | Number of learned motifs in the store |
| `brain.last_fire` | Dict with task index, motif id, confidence, and both quotes from the most recent fire |

### Storage invariant

Always store the **first-attempt trace** with the **first-attempt label** — even when a retry fires and recovers a failed task. Storing retry traces conflates recovery patterns with failure patterns and produces motifs that fire on legitimate fix attempts.

---

## Results

| Eval | Model | Tasks | Baseline | +Brain | Brain contribution |
|---|---|---|---|---|---|
| 30 diverse domains (`eval_fires`) | Haiku | 30 | 27/30 (90%) | 28/30 (93%) | +1 task, 5 fires |
| Hard one-shot failures (`eval_hard`) | **Sonnet** | 14 | 12/14 (86%) | 13/14 (93%) | +1 task, 1 fire |
| 30-task intensive (`eval_haiku_intensive`) | Haiku | 30 | 26/30 (87%) | 27/30 (90%) | +2 tasks, 2 fires |
| Real-world hard tasks (`eval_real_world`) | Haiku | 30 | 28/30 (93%) | 29/30 (97%) | +1 task, 2 fires |
| Extensive benchmark (`eval_extensive`) | Haiku | 32 | 28/32 (88%) | 28/32 (88%) | 0 tasks, 5 fires |
| Portfolio Risk Analyzer (`eval_project`) | Haiku | 15 | 13/15 (87%) | 14/15 (93%) | +1 task, 4 fires |
| **Cold-start learning (`eval_dev_learning`)** | **Haiku** | **56** | **43/56 (77%)** | **45/56 (80%)** | **+2 tasks; 0% FP on 16 near-miss tasks** |

---

### Cold-start learning benchmark — 56 developer tasks

The most targeted test for the motif system. 56 tasks across 8 programming families, structured so the brain must discover failure patterns from first occurrences and prevent recurrences — starting with zero stored history.

**Structure:** 8 families × 7 tasks = 56 total
- 1 discovery task per family (cold start)
- 4 recurrence tasks per family (brain may fire if motif was learned)
- 2 near-miss tasks per family (same domain, no actual bug — brain must stay silent)

**Families:** `nested_key`, `shared_state`, `off_by_one`, `unit_scale`, `secondary_sort`, `api_key`, `validation_all_errors`, `retry_classification`

| Metric | Value |
|---|---|
| Overall pass rate | **87.5%** (49/56) |
| Motifs extracted | **2/2** (100% of failed discovery tasks produced a learnable motif) |
| Recurrence prevention — retry_classification | **2/4** (50% of recurrences caught and fixed) |
| False positive rate on near-miss tasks | **0%** (0/16) |

**What the brain caught — retry_classification:**

Task 8 failed: agent wrote `except Exception` instead of checking `type(e).__name__ in retry_on`. The extracted motif fired on tasks 16 (`retry_on_type`) and 32 (`safe_request`) — different prompts, different exception types, same underlying logical error — before execution in both cases. Both passed after correction.

**Why api_key recurrences were not caught:**

The `silent_failure_instead_of_exception` motif was correctly learned from `extract_items`. But api_key recurrence failures (tasks 11, 19, 27) had a different root cause — incorrect response key mapping. The brain correctly produced no grounded quotes for these and stayed silent. Firing would have been a false positive.

---

### Portfolio Risk Analyzer — 15 sequential tasks

| # | Task | Baseline | +Brain |
|---|---|---|---|
| 3 | Rolling 20-day statistics (mean, vol, skew) | **✗** | **✓ ⚡×1 FIXED** |
| 14 | Monthly rebalancing with transaction costs | **✗** | **✗** ⚡×2 |
| All others (13) | — | ✓ | ✓ |

Haiku computed `returns.rolling(window).mean().std()` (std of rolling averages) instead of `returns.rolling(window).std()` (rolling std). Wrong volatility at Task 3 would have propagated silently into covariance (Task 4), Sharpe (Task 10), and the final risk report (Task 15). The brain caught it before execution.

---

### Hard one-shot failures — Sonnet + Brain

14 tasks where Sonnet reliably fails in one shot: 7 algorithm tasks and 7 physics/probability problems.

| | Baseline | +Brain |
|---|---|---|
| Code tasks (7) | 6/7 | **7/7** |
| Text tasks (7) | 6/7 | 6/7 |
| **Overall** | **12/14 (86%)** | **13/14 (93%)** |

The brain fixed the histogram (largest rectangle) task — Sonnet's first implementation used naive O(n²) and produced wrong results on edge cases.

---

## Install

```bash
pip install trace-use
```

Or from source:

```bash
git clone https://github.com/Rumbl3S/Trace-Optimization.git
cd Trace-Optimization
pip install -e .
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Run the offline test suite (no API key needed):

```bash
pytest tests/ -q
```

---

## Repo layout

| Path | Role |
|---|---|
| `trace_use/brain.py` | `BrainAgent`, `MotifStore`, `FailureMotif` — the full motif detection system |
| `trace_use/agents.py` | `tool_agent`, `haiku`, `opus`, `build_embedder` (lazy clients, keys from env/`.env`) |
| `trace_use/pipeline.py` | `run_task`, `Forecaster`, verifiers — kNN-based retry orchestration |
| `eval/eval_dev_learning.py` | 56-task cold-start learning benchmark |
| `eval/eval_project.py` | 15-task portfolio risk analyzer session |
| `eval/eval_real_world.py` | 30 hard tasks: competitive programming + GPQA-style science |
| `eval/eval_hard.py` | 14 hard one-shot failures, Sonnet + Haiku |
| `eval/eval_extensive.py` | 32 tasks: LiveCodeBench Pro / ICPC-Eval difficulty |
| `eval/eval_fires.py` | 30-task brain eval, diverse domains |
| `eval/eval_haiku_intensive.py` | 30-task intensive haiku session |
| `eval/results/` | Saved charts and JSON run logs |
| `tests/` | Offline test suite: `test_brain.py`, `test_forecast.py`, `test_pipeline.py` |

---

## Limitations

- **Motifs need a discovery failure to activate.** The brain cannot prevent the first occurrence of a failure class, only recurrences.
- **Retrieval is high-recall, not high-precision.** The 0.35 similarity floor retrieves liberally; the applicability judge narrows. On a 10-motif store, typically 1–3 LLM judge calls fire per task, adding ~300–600ms latency to `before_tool_call`.
- **Motif generalization depends on abstraction quality.** If the extraction call produces a motif with task-specific field names in `required_condition`, it will fail to fire on surface-different recurrences.
- **Trace richness is required.** One-liner responses produce near-identical embeddings regardless of correctness. Use a tool-calling agent, or wrap any text model in a CoT prompt that forces step-by-step output.
- **Brain is most impactful in the 15–40% failure band.** Above ~90% pass rate, fires are rare and gains are marginal. Below ~60%, the model likely needs a fundamentally different approach rather than mid-turn correction.

---

## Negative results

- **GSM8K is too easy.** Haiku solves grade-school math at >95% with no interventions.
- **kNN trajectory scoring produces false positives.** Embedding-based trajectory similarity (p_fail, Markov state tracking) was removed: two tasks with similar reasoning vocabulary spill into each other's motifs regardless of actual logical relationship. The structured proof requirement eliminates this class of error entirely.
- **Intervention is failure-rate-dependent.** When pass rate is above 90%, the store fills slowly with failures and motifs remain sparse. The brain adds value most when there is a recurring failure class.
- **`eval_extensive` fires didn't help.** 5 fires, 0 tasks fixed. When a task fails because the entire algorithm approach is wrong, motif-based feedback cannot recover it.
