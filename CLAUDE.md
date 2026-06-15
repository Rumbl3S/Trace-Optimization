# CLAUDE.md — trace_use working context

Read before editing here. Full rationale, results, and API are in [`README.md`](README.md);
this is the operational map.

## What this is
Branch-scoped research module: forecast agent failure from execution traces, so retries/
verification are spent only where needed. Built on the parent `../` pipeline (imports up;
the parent never imports this).

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
- `pipeline.py` — public API: `decompose / attempt / verify / make_retriever / Forecaster`.
- `forecast.py` — primitives: `knn_predict`, `knn_predict_cross`, `auc`, `spearman`.
- `eval/` — the experiment ledger (each script = one README finding); data + logs in
  `eval/results/`.
- `tests/` — offline gates (stubbed agent/embedder, no API key): `test_forecast`, `test_pipeline`.

## Reused parent infra (import, don't copy)
`../adaptive.py`, `../eval/{run_fanoutqa,run_musique}.py`, `../eval/demo_embed_compare.py`
(`haiku` temp-0 agent, `_build_openai` embedder), `../eval/_common.py` (scorers).

## Gotchas
- Evals call `llm._ensure_api_key()` at import and need `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`.
- "Accuracy ∝ evidence ∝ tokens" is the parent's floor and the reason this project exists —
  trace_use predicts *where* to spend, it does not beat that floor.
- Run `pytest tests/ -q` before committing; both test files must stay green.
