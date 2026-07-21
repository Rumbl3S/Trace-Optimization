# trace_use

[![PyPI](https://img.shields.io/pypi/v/trace-use)](https://pypi.org/project/trace-use/)
[![Python](https://img.shields.io/pypi/pyversions/trace-use)](https://pypi.org/project/trace-use/)

**A developer's failure memory — so you never make the same logical mistake twice.**

`trace_use` learns from LLM agent failures, extracts the logical principle behind each one, and warns the agent before the next occurrence — even if it happened a week ago on completely different code.

---

## The story

Agents fail in patterns, not randomly. The same logical mistake — catching all exceptions instead of selective ones, sorting on a single key when a tiebreak is required, returning `None` instead of raising on a missing required field — recurs across different tasks, different code, different sessions. Each recurrence burns tokens on a failure → diagnosis → retry loop that was avoidable.

`trace_use` intercepts at two points:

| Layer | When | What it does |
|---|---|---|
| **`TrajectoryDetector`** | Before the LLM writes any code | Compares the task description against past failures; injects targeted warnings into the prompt |
| **`BrainAgent`** | Before each `write_file` or `python_exec` call | Checks proposed code against stored motifs; fires a STOP message with a concrete fix if the exact logical gap is present |

---

## How the learning works

When a task fails, a single background LLM call extracts *why* — not a description of the code, but a literal snippet showing the bug and what to change. This produces a `FailureMotif`:

```
FailureMotif
  id:                  "dp_obstacle_handling"
  name:                "Incorrect DP obstacle handling logic"
  description:         "DP state transitions fail to correctly account for
                        obstacle constraints in path counting"
  required_condition:  "EXACTLY k obstacle cells"
  violation_condition: "dp[r][c][obs] += dp[r-1][c][obs-1] — adds paths
                        without consuming obstacle slot; should index
                        dp[r-1][c][obs] when grid[r][c]==0"
  recommendation:      "change dp[r][c][obs] += dp[r-1][c][obs-1] to
                        dp[r][c][obs] += dp[r-1][c][obs] when grid[r][c]==0"
```

`required_condition` is a short literal phrase that appears in any task where this bug recurs — not a meta-description. `violation_condition` quotes the exact buggy line and explains why it is wrong. Both are designed to be findable verbatim in future code and task text.

Motifs persist to `~/.trace_use/motifs.json` across sessions. A developer who hits a DP indexing bug on Monday will get a warning injected into their prompt on Friday, before writing a single line of code.

---

## Two-layer detection

### Layer 1 — TrajectoryDetector (pre-prompt, before any code is written)

Before the LLM generates anything, the detector retrieves candidate motifs by embedding similarity and runs a cheap LLM judge call for each. The judge searches the task text for the motif's `required_condition` phrase and quotes it verbatim. If found, a `KNOWN PITFALLS` block is prepended:

```
⚠️  KNOWN PITFALLS — based on recorded failures:

  [1] Incorrect DP obstacle handling logic
      Why this applies: "pass through EXACTLY k obstacle cells (marked 1)"
      Watch out for:    DP update must track obstacle count as a separate state dimension.
      Recommendation:   change dp[r][c][obs] += dp[r-1][c][obs-1] to
                        dp[r][c][obs] += dp[r-1][c][obs] when grid[r][c]==0

Address these before writing your implementation.
```

### Layer 2 — BrainAgent (pre-execution, before each tool call)

Before each `write_file` or `python_exec` call, the brain checks the proposed code against stored motifs. The judge must return a verbatim quote from the task AND the verbatim buggy line from the code. When both are found, a STOP message fires with the specific fix before the bad code runs:

```
⚠️ BRAIN:
STOP: The monitor detected a likely logical failure before execution.

Evidence (Learned pattern: Incorrect DP obstacle handling logic):
  - Requirement: pass through EXACTLY k obstacle cells (marked 1)
  - Bug found:   dp[r][c][obstacles] += (dp[r-1][c][obstacles-1] if r > 0 else 0)
  - Why:         adds paths without decrementing obstacle counter when cell is obstacle

Fix for this specific code:
  Change dp[r][c][obs] += dp[r-1][c][obs-1] to dp[r][c][obs] += dp[r-1][c][obs]
  when grid[r][c]==0; use obs-1 index only when grid[r][c]==1

General pattern fix:
  DP update must index dp[row][col][count-1] when consuming a constrained resource

Rewrite the code to apply the fix above before calling the tool again.
```

The brain intercepts `write_file` (the implementation file) rather than waiting for `python_exec` (the test harness). This means the buggy DP logic is caught before the file is even saved to disk.

Both layers share the same motif store. A motif learned mid-execution is immediately available for pre-prompt injection on the next task.

---

## False positive prevention

Both layers use the same two-stage design: LLM judge for recall, deterministic gate for precision.

The gate requires:
- Confidence ≥ threshold (0.80 for execution-time, 0.70 for pre-prompt)
- Both quotes non-empty and not vague (`"task implies"`, `"likely"`, `"might"`, etc.)
- `requirement_quote` is a verbatim substring of the actual task text (or ≥70% word overlap)
- `violation_quote` is a verbatim substring of the actual proposed code (or ≥40% word overlap for code-like strings where variable names legitimately differ)

**Cross-domain contamination is structurally impossible.** A DP obstacle motif requires quoting the phrase "EXACTLY k obstacle" from the task text — that phrase will not appear in a sort task. No similarity threshold to tune; the predicate either holds or it does not.

Result: **0% false positives on 16 near-miss tasks** across 8 failure families in the cold-start benchmark.

---

## Quick start

```bash
pip install trace-use
```

Works with **Anthropic** or **OpenAI** — set whichever key you have:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # uses claude-haiku-4-5
# or
export OPENAI_API_KEY=sk-proj-...     # uses gpt-4o-mini (agent) + gpt-4o (judge)
```

`build_embedder()` prefers OpenAI embeddings when `OPENAI_API_KEY` is set (avoids loading a local model). Falls back to `sentence-transformers` (free, no key needed) otherwise.

### Minimal setup

```python
from trace_use import (
    BrainAgent, PersistentMotifStore, TrajectoryDetector,
    BrainConfig, build_embedder, openai_tool_agent,
)

embedder = build_embedder()
store    = PersistentMotifStore(embedder)           # loads ~/.trace_use/motifs.json
detector = TrajectoryDetector(store, embedder)      # pre-task injection
brain    = BrainAgent(embedder, motif_store=store)  # mid-execution interception

# OpenAI agent with write_file + python_exec tools, 60s API timeout
agent         = openai_tool_agent(["python_exec", "write_file"], max_turns=8)
agent.monitor = brain

for i, task in enumerate(tasks):
    brain.set_task(i, task=task["prompt"])
    brain.reset()

    # Layer 1: inject known pitfalls before the LLM starts
    enriched_prompt, matches = detector.inject(task["prompt"])
    if matches:
        print(f"⚠️  {len(matches)} pitfall(s) injected")

    # Layer 2: brain fires mid-execution via agent.monitor
    trace, tokens = agent(enriched_prompt)
    passed = run_checks(trace)

    # Always store the first-attempt trace with the first-attempt label
    brain.store(trace, int(passed), metadata=failure_reason_if_failed)
```

### Interactive terminal demo

```bash
python demo_session.py           # inspect mode: type tasks, see what warnings fire
python demo_session.py --run     # run mode: actually executes tasks with the agent
python demo_session.py --show    # list all stored motifs
python demo_session.py --clear   # wipe motif store
python demo_session.py --store ./my_project.json   # project-specific store
```

---

## Tools

The agent has access to two persistent tools:

| Tool | Description |
|---|---|
| `write_file(path, content)` | Writes a file to `~/.trace_use/workspace/`. The workspace persists across turns and sessions and is added to `sys.path` so written modules can be imported. |
| `python_exec(code)` | Runs code in a fresh subprocess (30s timeout). Each call starts a clean Python process — no stale module cache, no risk of hanging on blocking calls. |

The brain intercepts both: `write_file` is checked when the agent writes the implementation, `python_exec` is checked when it runs a test harness.

---

## API reference

### `TrajectoryDetector`

| Method | Description |
|---|---|
| `detector.check(task)` | Returns `list[MotifMatch]` — relevant motifs with grounded evidence |
| `detector.inject(task)` | Returns `(enriched_prompt, matches)` — prepends KNOWN PITFALLS block if matches exist |

### `BrainAgent`

| Method / property | Description |
|---|---|
| `brain.set_task(idx, task="")` | Register task index and description before each task |
| `brain.reset()` | Clear reasoning buffer and counters before each task |
| `brain.push(text)` | Accumulate reasoning chunk — called automatically via `agent.monitor` |
| `brain.before_tool_call(name, input_dict)` | Pre-execution hook — returns STOP message or `None` |
| `brain.on_tool_call(name, input_dict, result)` | Post-execution stall detection |
| `brain.store(trace, label, metadata="")` | Store result; extracts motif on `label=0` (failure) |
| `brain.n_stored` | Number of learned motifs |
| `brain.last_fire` | Dict with motif id, confidence, and verbatim quotes from the most recent fire |

### `PersistentMotifStore`

```python
store = PersistentMotifStore(embedder)                     # default: ~/.trace_use/motifs.json
store = PersistentMotifStore(embedder, path="./proj.json") # project-specific
store.clear()                                              # wipe all motifs
store.count                                                # number of stored motifs
store.motifs                                               # list[FailureMotif]
```

### Configuration

All tunable constants live in `BrainConfig` / `DetectorConfig`. The `provider` field selects the LLM backend:

```python
from trace_use import BrainConfig, DetectorConfig, BrainAgent, TrajectoryDetector

# OpenAI backend — gpt-4o for judge (verbatim-quote compliance), gpt-4o-mini for agent
brain_cfg = BrainConfig(
    provider      = "openai",           # "anthropic" (default) or "openai"
    judge_model   = "gpt-4o",           # stronger model for judge accuracy
    extract_model = "gpt-4o",
    judge_threshold  = 0.80,            # min confidence to fire
    max_interventions = 2,              # max fires per task
    exec_tool_name   = "python_exec",   # tool name to intercept
)
brain = BrainAgent(embedder, config=brain_cfg)

# Anthropic backend
brain_cfg = BrainConfig(
    provider    = "anthropic",
    judge_model = "claude-haiku-4-5-20251001",
)

det_cfg = DetectorConfig(
    provider        = "openai",
    model           = "gpt-4o",
    judge_threshold = 0.70,
    retrieval_top_k = 5,
    storage_path    = "./project_motifs.json",
)
detector = TrajectoryDetector(store, embedder, config=det_cfg)
```

### Storage invariant

Always store the **first-attempt trace** with the **first-attempt label** — even when a retry fires and recovers a failed task. Storing retry traces conflates recovery patterns with failure patterns.

---

## Results

### Cold-start learning — 56 tasks, 8 failure families (`eval_dev_learning`)

The core benchmark. 56 tasks across 8 programming families (nested key access, shared state, API key mapping, validation, off-by-one, unit scaling, secondary sort, retry classification). Each family has 1 discovery task (brain cold-starts with no motifs), 4 recurrence tasks (brain fires if a motif was learned), and 2 near-miss tasks (same domain, correct code — brain must stay silent).

Run: `gpt-4o-mini` agent, `gpt-4o` judge, OpenAI embeddings.

| Metric | Value |
|---|---|
| Overall pass rate | **82.1%** (46/56) |
| Tasks saved by brain fires | **4** (retry_on_type, safe_request, merge_response_lists, rank_leaderboard) |
| False positive rate on 16 near-miss tasks | **0%** |
| Motif generalization | retry_request → retry_on_type + safe_request (different code, same logical error) |

**What a fire looks like in practice:**

```
[BRAIN JUDGE RAW] motif='unconditional_retry_on_failure' applies=True conf=1.00
  req='Only retry exceptions listed in retry_on.'
  viol='except retry_on as e:'
[BRAIN FIRE] motif: unconditional_retry_on_failure, conf: 1.00
[16/56] retry_on_type   retry_classification   | PASS [FIRED] | 13.4s
```

The agent wrote `except retry_on as e:` — iterating the list as a catch-all instead of checking type membership. The brain caught this, injected a STOP with the specific corrected line, the agent fixed the code, and the task passed.

**Where it didn't fire:**

The api_key and nested_key families had recurring failures that the brain missed. Root cause: embedding similarity between abstract motif descriptions and concrete task prompts was below the retrieval threshold (~0.05–0.09 cosine similarity), so those motifs never reached the judge. This is an active limitation — the pre-filter is too coarse for short, concrete task descriptions.

### Portfolio Risk Analyzer — 15 sequential tasks (`eval_project`)

Task 3 (rolling statistics) failed: the agent computed `returns.rolling(window).mean().std()` instead of `returns.rolling(window).std()`. Wrong volatility at Task 3 would have propagated silently into covariance (Task 4), Sharpe ratio (Task 10), and the final risk report (Task 15). Brain caught it before execution.

### Where brain fires don't help

`eval_extensive` — 5 fires, 0 tasks fixed. When a task fails because the entire algorithm is wrong (bitmask TSP that needs DP from scratch), motif-based feedback cannot recover it. The brain's value is highest when the error is localized — a boundary condition, a missing type check, a swallowed exception — not when the approach itself needs replacing.

---

## Repo layout

| Path | Role |
|---|---|
| `trace_use/brain.py` | `BrainAgent`, `MotifStore`, `FailureMotif` — mid-execution motif detection |
| `trace_use/trajectory.py` | `TrajectoryDetector`, `MotifMatch` — pre-task pre-prompt injection |
| `trace_use/motif_store.py` | `PersistentMotifStore` — JSON-backed cross-session persistence |
| `trace_use/config.py` | `BrainConfig`, `DetectorConfig` — all tunable constants |
| `trace_use/agents.py` | `tool_agent`, `openai_tool_agent`, `haiku`, `opus`, `build_embedder`, `_llm_call` |
| `trace_use/tools.py` | `python_exec`, `write_file`, `calculator`, `wikipedia_search` — built-in agent tools |
| `demo_session.py` | Interactive TUI: run tasks with the agent, inspect motif learning live |
| `eval/eval_dev_learning.py` | 56-task cold-start learning benchmark (supports Anthropic + OpenAI) |
| `eval/eval_project.py` | 15-task portfolio risk analyzer session |
| `eval/eval_real_world.py` | 30 hard tasks |
| `eval/results/` | JSON run logs |
| `tests/test_brain.py` | BrainAgent unit tests (offline, stubbed) |
| `tests/test_trajectory.py` | TrajectoryDetector + developer week simulation (offline, stubbed) |
