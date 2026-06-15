# trace_use — failure traces as search priors

> **The pivot:** stop throwing away a wrong agent attempt. A botched attempt still
> *explored the space* — the regions it checked and ruled out can be **pruned**, the
> leads it found promising can be **expanded**. Convert "wrong answer" into **a map of
> the search** that makes the next attempt cheaper and more accurate.

This is a branch-scoped experiment that builds on the parent `regime_selector/` pipeline
(measure → split → gap-fill → compose). It reuses that project's agent wrappers, dataset
loaders, embedders, and metering; it changes *what we do with a failed attempt*.

---

## 1. The idea, precisely

A standard pipeline treats an agent attempt as a **binary** outcome: right → keep,
wrong → discard and retry (or escalate). But the attempt produced a **trace** — the
evidence it inspected, the leads it followed, the things it concluded were irrelevant,
and the parts it admits it still can't answer. That trace is **information about the
location of the answer**, even when the answer itself is wrong.

We split the trace into three signals and use each as a **prior for the next retrieval/
reasoning step**:

| signal | from the trace | how we use it |
|---|---|---|
| **Negative** (dead ends) | "I checked X / Y and they don't contain it" | down-weight or exclude X, Y from the next retrieval — don't re-pay for ruled-out evidence |
| **Positive** (promising leads) | "Z looked relevant but I couldn't finish" | boost / expand retrieval around Z; chase the lead deeper |
| **Unknown** (open gaps) | "I still can't answer part P" | the targeted sub-question to fill (this is what the parent pipeline already does) |

The parent pipeline already exploits the **positive** and **unknown** signals (the
`AREA:` hints guide gap retrieval). **The new bet is the *negative* signal and a
*persistent exploration map* across attempts** — systematically ruling out territory so
repeated attempts converge instead of re-treading the same ground.

## 2. Core hypothesis

> On retrieval-heavy, dispersed-evidence tasks, **reusing a failed attempt's trace as a
> retrieval prior** (prune ruled-out regions, expand promising ones) reaches the correct
> answer at **higher accuracy-per-token** than (a) discarding the attempt and retrying
> blind, and (b) spending the same extra tokens on simply retrieving *more* evidence.

The second baseline is the hard one and the whole point: we already proved
**accuracy ∝ evidence ∝ tokens**. Trace-use only earns its keep if *guided* exploration
beats *more* exploration at equal budget — i.e., if the trace removes search waste rather
than just adding compute.

## 3. Honest novelty evaluation

**Verdict: moderate novelty, high practical value, clear experiment.** Worth pursuing —
but the framing has to be sharp, because the *spirit* (use failure as signal) is well
trodden.

Related work and how we differ:

- **Reflexion** (Shinn et al. 2023) — agent writes a *verbal* self-reflection after
  failure and conditions the next attempt on it. Closest relative. **We differ:** our
  signal is **structural** (a map of *which evidence units* were explored / ruled out /
  promising), used as a concrete **retrieval prior**, not a free-text reflection.
- **ReAct / Tree-of-Thoughts / search agents** — explore and backtrack *within* one run;
  dead branches are abandoned implicitly. **We differ:** we persist a cross-attempt
  exploration map and explicitly use *negative* coverage to prune the *next* retrieval.
- **Hindsight Experience Replay** (RL) — relabels failed trajectories as signal for a
  different goal. Conceptually adjacent (failure ≠ waste), different mechanism/domain.
- **Negative caching / "avoid repeated tool calls"** — some agent frameworks log failed
  actions. **We differ:** we turn ruled-out *evidence regions* (not actions) into a
  retrieval down-weight, and combine it with positive lead-expansion.

The defensible novel claim: **"failure-trace as a structured retrieval prior for the next
attempt"** — a persistent, signed (negative + positive) exploration map over the evidence
space for multi-document QA. Not revolutionary; a fresh, testable operationalization.

## 4. Risks & why this could fail (state up front)

1. **Wrong ≠ dead end.** An agent can inspect the *right* region and still fail
   (misread, hallucinate). Pruning "what it checked" may prune the actual answer
   location. → Mitigation: treat negative signal as a *soft* down-weight, never a hard
   exclusion; ablate negative-only vs positive-only.
2. **Trace fidelity.** A single LLM call's "trace" is just its reasoning text — soft and
   possibly confabulated. Faithful traces want ReAct-style explicit retrieve/inspect
   steps. → Mitigation: log the *actual* retrieved units (ground truth of what it saw),
   not only the self-report.
3. **Cost.** Mining the trace + a guided retry spends tokens. It must beat
   "just-add-evidence" at equal budget (hypothesis §2), or it's pointless.
4. **Prior-art overlap** dampens novelty (Reflexion). Position as applied/efficiency.

## 5. The plan (phased, each phase falsifiable)

**Phase 0 — Harness & oracle (no new method).**
Reuse parent loaders (FanOutQA dispersed, MuSiQue multi-hop). Build a *trace-oracle*:
given gold evidence, label each retrieved unit as helpful/dead-end. Measures the
*ceiling* — if a perfect negative/positive prior doesn't help, the idea is dead. **Kill
criterion:** oracle trace-prior ≤ just-add-evidence at equal budget.

**Phase 1 — Trace capture.**
Instrument the solver to record, per attempt: the evidence units actually retrieved/seen,
and a structured self-report (`CHECKED: <unit/topic> → dead-end|partial|promising`,
`OPEN: <sub-question>`). Extends the existing `AREA:` mechanism with a **negative**
channel. Output: a `TraceRecord`.

**Phase 2 — Trace → signed retrieval prior.**
Map the trace onto the embedding pool: negative units get a similarity penalty; promising
leads expand the query (their text becomes additional retrieval anchors); open gaps are
the targets. Persistent across attempts within a task.

**Phase 3 — Guided-retry loop.**
Attempt 1 normal → audit → if incomplete, build trace → Attempt 2+ retrieves under the
signed prior and answers the open gaps → stop on audit-pass or budget. (Mirrors the
parent's expand loop, but the expansion is *trace-directed*, not just band-widening.)

**Phase 4 — Evaluation (the real test).**
Baselines at **equal token budget**: (a) discard-and-retry blind, (b) add-more-evidence
(parent's band expansion), (c) trace-guided. Ablations: negative-only, positive-only,
both. Metrics: accuracy, tokens, **accuracy-per-token**, attempts, exploration coverage.
**Decision rule:** trace-guided must beat (b) on accuracy-per-token to justify the idea;
the negative-only ablation tells us whether dead-end pruning specifically pays.

**Phase 5 — Analysis & positioning.**
Characterize *when* it helps (dispersed search where pruning matters) vs *when* it
doesn't (concentrated tasks where there's nothing to prune). Compare to Reflexion in
cost/accuracy. Write up honestly, including the kill cases.

## 6. Success / kill criteria (decide before running)

- **Pursue further** if trace-guided > add-more-evidence on accuracy-per-token by a
  margin above noise on FanOutQA, *and* the negative-only ablation contributes.
- **Kill** if it only ties "just retrieve more" (then the trace adds nothing over
  evidence), or if negative pruning *hurts* (wrong ≠ dead end dominates).

## 7. Relationship to the parent project

Reuses `../adaptive.py` (solver, retrieval, audit, memo), `../eval/run_fanoutqa.py` /
`run_musique.py`, `../tokenmeter.py`, and the OpenAI/Haiku wrappers from
`../eval/demo_embed_compare.py`. New code lives only in this folder. Nothing here is
imported by the parent.

## 8. Findings so far (MVP + three ablations, Haiku, OpenAI embeddings, temp 0)

All numbers are **leave-one-out k-NN AUC** (0.5 = chance), within-dataset unless noted.
Raw logs in `eval/results/`.

**MVP — do trace embeddings predict failure?** YES, but task-dependent.
- MuSiQue (16/40, balanced control): **trace AUC ~0.85**, beating task-only by +0.08–0.16.
  Robust across two runs. The trajectory genuinely forecasts its own failure beyond task
  difficulty.
- The headline "0.91 over everything" was inflated by a **dataset confound** (FanOutQA
  fails far more than MuSiQue); within-dataset is the honest ~0.85.

**Part 1 — reasoning-only ablation (`eval/ablation_reasoning.py`).** Both signals exist;
the answer is stronger.
- MuSiQue: answer-only ~0.90 > full-trace ~0.85 > **reasoning-only ~0.80** > task-only ~0.74.
- So the *process* predicts failure **before the answer exists** (~0.80, usable for
  mid-trajectory intervention), but "the final answer looks wrong" is the strongest tell.

**Part 2 — balanced FanOutQA (`eval/gen_balanced.py` + `analyze.py`).** A real limitation.
- With richer (embedding-retrieved) context, FanOutQA success rose to 11/50 — testable.
- **FanOutQA trace AUC ≈ chance (0.45–0.56).** Forecasting does **not** work on fan-out,
  even controlled. MuSiQue stays ~0.85. The signal is **task-type dependent**: it works
  where failure is a coherent wrong-chain (multi-hop), not where it's fuzzy partial
  coverage (fan-out aggregation). ⚠️ Fan-out is exactly the *expensive* regime where
  forecasting would be most valuable — and it's where it currently fails.

**Part 3 — learned representation (`eval/learned_repr.py`).** Off-the-shelf wins at this n.
- MuSiQue control: **raw k-NN 0.876 > PCA-kNN 0.831 > logreg 0.789 > LDA 0.760.** Learned
  probes *overfit* at n=40 and underperform the off-the-shelf embedding.
- The apparent learned win on the *confounded* ALL set (logreg 0.805 vs raw 0.733) is just
  the classifier exploiting the dataset split — not a representation gain.
- Verdict: **the bottleneck is DATA, not method.** Representation learning needs far more
  trajectories before it pays; today, OpenAI embeddings + k-NN is the ceiling.

### Net read
The premise has **real but bounded** support: trace embeddings forecast failure well on
**multi-hop** tasks (~0.85, process-only ~0.80), with off-the-shelf embeddings. It
**does not yet generalize to fan-out**, and **learned representations don't help at this
scale**. So before the grand "trajectory graph / failure forecasting product": (a) figure
out *why fan-out failure isn't embedding-separable* (likely the binarized partial-coverage
label) and fix the failure definition, and (b) get to hundreds–thousands of trajectories
so representation learning can be tested honestly. The idea isn't dead — it's **proven on
one task family, unproven on the one that matters most for cost.**

## 8b. Generalizability pass (`eval/generalize.py`) — both fixes show positive signal

Two no-new-data fixes aimed at "works for anything":

**A) Continuous target (per-component proxy)** — predict the coverage *score*, not binary
pass/fail. The 0.5 cutoff was hiding fan-out signal:

| dataset | binary AUC | continuous Spearman |
|---|---|---|
| fanout | 0.449 (chance) | **+0.191** (signal appears!) |
| musique | 0.876 | +0.662 |

→ Fan-out's degree-of-success **is** weakly predictable once we stop binarising — the
cutoff, not the trace, was the problem. Weak (0.19) but no longer chance. Validates the
per-component direction; a true per-*item* label should sharpen it.

**B) Leave-one-task-type-out transfer** — train on one task family, predict an UNSEEN one:

| predict | from | transfer AUC | Spearman |
|---|---|---|---|
| fanout | musique only | **0.605** | +0.034 |
| musique | fanout only | **0.612** | +0.106 |

→ A forecaster trained on one task type predicts a **different, unseen** type at AUC ~0.61
(> 0.5 chance). So the failure signal is **partially transferable** — it isn't memorising
one dataset. Generalises, but weakly (0.61 vs within-task 0.85).

**Read:** both "generalizable" levers work directionally — continuous targets surface
fan-out signal, and cross-task transfer beats chance. Neither is *strong* yet (0.19 / 0.61),
so the path to a general forecaster is: true per-item labels + more task families + more
data, not a new mechanism.

## 8c. Per-component forecasting (`eval/component_forecast.py`) — the fan-out fix lands

GENERIC pipeline (no benchmark logic): `decompose → attempt → verify → forecast`. The unit
of prediction is a single **checkable component** of a task; the only task-specific input is
a **pluggable verifier** (here an LLM gold-judge; in deployment, the user's own check).

| slice | AUC | note |
|---|---|---|
| **FanOutQA within** | **0.846** | was ~0.45 (chance) at the task level |
| MuSiQue within | 0.666 | only ~2 components/task, n=45, judge-noise |
| **FanOutQA ← MuSiQue** (transfer) | **0.732** | up from task-level 0.61 |
| MuSiQue ← FanOutQA (transfer) | 0.650 | |

**This is the payoff.** Decomposing into checkable units turned fan-out from unpredictable
(0.45) to highly predictable (**0.85**) — confirming the diagnosis that the *binary
task-level label*, not the trace, was the blocker. It is **fully generic** (same code on
both datasets; swap the verifier for any task) and **transfers across task types better
than whole-task forecasting** (0.65–0.73). Caveats: MuSiQue-within is weak (few components,
n=45); labels are LLM-judge generated (noisy); single run.

The generalizable recipe is now empirically grounded: **forecast per checkable component,
not per task; bring your own verifier; everything else is task-agnostic.**

## 9. Status

MVP + 3 ablations + generalizability + per-component complete. Code: `forecast.py`,
`eval/{mvp_failure_forecast,ablation_reasoning,gen_balanced,analyze,learned_repr,generalize,
component_forecast}.py`, tests 7/7. Next: add a non-QA task family (coding/tool-use) to test
transfer breadth, and turn forecasts into a token/latency-saving intervention policy.
