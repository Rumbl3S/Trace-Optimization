# CLAUDE.md — trace_use working context

Read before editing here. Full rationale, results, and API are in [`README.md`](README.md);
this is the operational map.

## What this is
Standalone toolkit: forecast agent failure from execution traces, so retries/verification
are spent only where needed. Fully self-contained — agent/embedder (`agents.py`), benchmark
loaders (`bench/`), and helpers (`_util.py`) are vendored; no external project deps.

## The core claim (and the line to hold)
Forecast per **checkable component**, not per whole task — that is what made fan-out
predictable (0.45 → 0.85) and what transfers across task types. The trace was never the
problem; the coarse pass/fail label was. Don't regress to whole-task labels.

## Invariants
- **Generic by construction.** The method (`pipeline.py`) has zero dataset logic. The only
  task-specific input is a `Verifier`. Keep it that way — no benchmark-format parsing in
  `pipeline.py` or `forecast.py`.
- **Compare honestly.** AUC vs 0.5 chance; report within-dataset *and* leave-one-task-type-out
  (the real generalization test). Temp-0 still wobbles ±1 item — don't over-read small deltas.
- **Negative results stay.** GSM8K-too-easy, learned-repr-doesn't-help, intervention-is-
  failure-rate-dependent are findings, not failures. Keep them in the README.

## Layout
- `pipeline.py` — public API: `decompose / attempt / verify / make_retriever / Forecaster`
  + auto-verifiers (`self_judge`, `self_consistency`).
- `forecast.py` — primitives: `knn_predict`, `knn_predict_cross`, `auc`, `spearman`.
- `agents.py` — vendored Haiku agent + OpenAI embedder (lazy clients; keys from env/`.env`).
- `bench/` — vendored benchmark loaders + scorers (`_common`, `run_fanoutqa`, `run_musique`).
- `_util.py` — vendored `select_for_single`.
- `eval/` — experiment ledger (each script = one README finding); data + logs in `eval/results/`.
- `tests/` — offline gates (stubbed agent/embedder, no API key): `test_forecast`, `test_pipeline`.

## Gotchas
- `bench/` is deliberately NOT named `datasets/` — that would shadow the HuggingFace
  `datasets` library the loaders import.
- `agents.py` creates clients lazily, so importing never needs keys; running an eval needs
  `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`.
- "Accuracy ∝ evidence ∝ tokens" is the prior project's floor and the reason this exists —
  we predict *where* to spend, we don't beat that floor.
- Run `pytest tests/ -q` before committing; both test files must stay green.
