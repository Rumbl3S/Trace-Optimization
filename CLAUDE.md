# CLAUDE.md — trace_use working context

Read before editing here. Full rationale, results, and API are in [`README.md`](README.md); this is the operational map for contributors.

---

## What this is

Standalone toolkit: forecast agent failure from execution traces, so retries and verification are spent only where needed. Fully self-contained — agent/embedder (`agents.py`), benchmark loaders (`bench/`), and helpers (`_util.py`) are vendored; no external project deps beyond `requirements.txt`.

---

## The core claims (lines to hold)

**1. Forecast per checkable component, not per whole task.**
Decomposing into atomic sub-questions and forecasting each raised AUC from ~0.45 (chance) to 0.85 on structured tasks. Don't regress to whole-task labels.

**2. The trace carries the signal.**
Reasoning-only AUC is 0.84 — *how* the agent thinks predicts failure independently of whether the final answer looks wrong. This is why the approach works across task types.

**3. Trace richness is a prerequisite.**
One-liner text responses produce near-identical embeddings for correct and incorrect answers. To get signal, the agent must show intermediate reasoning steps. Either use a tool-use agent (naturally rich traces from execution) or wrap any text model in a CoT prompt that forces step-by-step output. See `demo_general.py` for the pattern.

**4. Store first-attempt traces with first-attempt labels.**
When a retry fires and recovers a failed component, store the *original* trace with the *original* label — not the retry. The store captures failure patterns in their raw form. Storing retry traces conflates recovery patterns with failure patterns and degrades the kNN signal.

---

## Invariants

- **Generic by construction.** `pipeline.py` has zero dataset logic. The only task-specific input is a `Verifier` callable. Keep it that way — no benchmark-format parsing in `pipeline.py` or `forecast.py`.
- **Compare honestly.** AUC vs 0.5 chance; report within-dataset *and* leave-one-task-type-out (the real generalization test). Temp-0 still wobbles ±1 item — don't over-read small deltas.
- **Negative results stay.** GSM8K-too-easy, learned-repr-doesn't-help, intervention-is-failure-rate-dependent are findings. Keep them in the README.
- **Verifiers must be accurate.** Wrong labels propagate into the store and corrupt the kNN. Mislabeled early tasks are especially damaging — they teach the forecaster wrong associations before the store has enough redundancy to dilute them. Use programmatic checks when possible; when using LLM judges, use a different model than the one being judged.

---

## Layout

| Path | Role |
|---|---|
| `pipeline.py` | Public API: `decompose`, `attempt`, `Forecaster`, `run_task`, `make_retriever`, plus all verifiers (`gold_judge`, `tiered_judge`, `self_judge`, `self_consistency`, `code_judge`) |
| `forecast.py` | Primitives: `knn_predict`, `knn_predict_cross`, `auc`, `spearman` |
| `display.py` | Rich live terminal display used internally by `run_task` |
| `agents.py` | Vendored `haiku`, `opus`, `tool_agent` + OpenAI embedder (lazy clients; keys from env/`.env`) |
| `demo_general.py` | 40 diverse everyday tasks; CoT haiku agent; live matplotlib plot. AUC ~0.68 |
| `demo_debug.py` | 29 Python debugging tasks with hidden edge-case bugs; tool_agent. AUC ~0.87 |
| `demo_large.py` | Large-scale mixed run; full Rich display |
| `bench/` | Vendored benchmark loaders + scorers (`_common`, `run_fanoutqa`, `run_musique`) |
| `_util.py` | Vendored `select_for_single` |
| `eval/` | Experiment ledger — each script = one README finding; data + logs in `eval/results/` |
| `tests/` | Offline gates (stubbed agent/embedder, no API key): `test_forecast.py`, `test_pipeline.py` |

---

## Gotchas

- **`bench/` is NOT `datasets/`.** That name would shadow the HuggingFace `datasets` library the loaders import.
- **Lazy client init.** `agents.py` creates Anthropic and OpenAI clients on first use, so importing never needs keys. Running any eval or demo does need `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`.
- **`run_task` parameter is `task=`, not `q=`.** The decompose agent receives the full task string; sub-questions are internal.
- **Unicode in verifiers.** String verifiers must normalise Unicode before matching — e.g., haiku may write `H₂O` (Unicode subscript `₂`, U+2082) when asked for a chemical formula. `"h2o" in "h₂o"` is `False`. Use `unicodedata.normalize("NFKC", s)` or the `_str_check` helper in `demo_general.py` which handles this.
- **AUC convention.** `forecast.py`'s `auc(labels, scores)` computes `P(score_of_pass > score_of_fail)` (label=1 is pass, label=0 is fail). For P(fail) as a failure predictor, correct prediction means failures score HIGH on P(fail), so this AUC should be LOW if used naively. The display layer uses `roc_auc_score(labels, [1 - p_fail for p in scores])` to get the intuitive AUC (higher = better forecaster). Keep these conventions consistent.
- **"Accuracy ∝ evidence ∝ tokens"** is the prior project's floor and the reason this exists — we predict *where* to spend, not how to beat that floor.
- **Run `pytest tests/ -q` before committing.** Both test files must stay green.

---

## Key results (what the demos show)

| Demo | Agent | AUC | Notes |
|---|---|---|---|
| `demo_debug.py` | `tool_agent(["python_exec"])` | **0.87** | 29 debugging tasks, hidden edge-case bugs. Tool traces naturally discriminative. |
| `demo_general.py` | CoT haiku (no tools) | **0.68** | 40 diverse everyday tasks. CoT forcing required for signal in text-only traces. |
| FanOutQA / MuSiQue | `haiku` | **0.85** | Structured multi-hop QA. Per-component forecasting. |

---

## Adding a new experiment

Each `eval/` script corresponds to one finding in the README. Pattern:

1. Write a self-contained script in `eval/` that imports only from `pipeline.py`, `forecast.py`, and `agents.py`.
2. Save raw results to `eval/results/<experiment_name>.json`.
3. Print the key metric (AUC, Spearman) to stdout.
4. Add the finding to the README under the relevant section with the honest number.
5. If the result is negative, add it to the "Negative results" section — don't omit it.
