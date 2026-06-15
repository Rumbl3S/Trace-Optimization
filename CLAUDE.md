# CLAUDE.md — trace_use working context

Read this before editing anything in `trace_use/`. Full rationale and plan are in
[`README.md`](README.md); this is the operational map.

## What this branch is
Experiment (`trace_use` branch only): use a **failed agent attempt's trace** — what it
explored, ruled out, found promising, still can't answer — as a **signed retrieval prior**
for the next attempt, instead of discarding the attempt as "wrong."

## The one thing to keep straight
The bet is the **negative signal** (ruled-out regions → prune) plus a **persistent
exploration map** across attempts. The parent pipeline already uses the *positive* /
*unknown* signals (`AREA:` hints → gap retrieval). Do **not** re-pitch those as the
contribution — they exist. New value = pruning dead ends + cross-attempt convergence.

## The hypothesis we must falsify
Trace-guided retry beats **just-retrieve-more at equal token budget** (and beats
blind retry). If it only ties "more evidence," the trace adds nothing — that is the kill
condition. Always evaluate against the add-more-evidence baseline, never only vs naive retry.

## Invariants / guardrails
- **Negative = soft.** Ruled-out units get a similarity *penalty*, never a hard exclusion —
  wrong ≠ dead end (an agent can check the right place and still fail). Hard-excluding can
  prune the answer.
- **Log what it actually saw**, not only the self-report. The ground-truth trace is the set
  of retrieved units; the LLM's verbal "I checked X" is a noisy secondary signal.
- **Equal-budget comparisons only.** Token accounting must include trace-mining cost.
- Keep all new code in this folder. Reuse parent infra by importing upward; never edit
  `../adaptive.py` to serve this experiment (fork behavior here instead).

## Reused parent infra (import, don't copy)
- `../adaptive.py` — `AdaptiveSolver`, retrieval (`_cluster_evidence`, `_embed_top`,
  `_segment`), audit, `memo`.
- `../eval/run_fanoutqa.py`, `../eval/run_musique.py` — dataset loaders + scorers.
- `../eval/demo_embed_compare.py` — `haiku` (temp=0 agent), `_build_openai` (embedder).
- `../tokenmeter.py` — metering / relevance.

## Datasets & why
- **FanOutQA** (dispersed, multi-doc) — primary; this is where there's territory to prune.
- **MuSiQue** (multi-hop) — secondary; tests whether promising-lead chaining helps.

## Build order (see README §5)
0. Harness + **trace-oracle ceiling** (gold-labeled helpful/dead-end units). Go/no-go.
1. `TraceRecord` capture (retrieved units + `CHECKED:`/`OPEN:` self-report).
2. Trace → signed retrieval prior (negative penalty + positive query expansion).
3. Guided-retry loop (trace-directed expansion).
4. Eval vs {blind-retry, add-more-evidence} at equal budget; ablate ±signals.

## Planned files (none yet — scaffolding stage)
- `trace.py` — `TraceRecord`, capture, signed-prior construction.
- `trace_solver.py` — guided-retry loop wrapping the parent solver.
- `eval/oracle_ceiling.py`, `eval/trace_vs_more_evidence.py` — the decisive experiments.
- `tests/` — offline gates (stubbed agents), mirroring the parent's test style.

## Gotchas
- Temp=0 still wobbles ±1 part on the API — average runs before trusting small deltas
  (a lesson from the parent project).
- "Accuracy ∝ evidence ∝ tokens" is the floor every parent experiment hit; trace-use only
  matters if it removes *waste*, not by adding evidence. Hold that line.
