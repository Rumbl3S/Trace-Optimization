"""trace_use — learn failure patterns from LLM agent traces.

Install:
    pip install trace-use

Minimal setup (cross-session persistent motifs)::

    from trace_use import BrainAgent, PersistentMotifStore, TrajectoryDetector, build_embedder

    embedder  = build_embedder()
    store     = PersistentMotifStore(embedder)       # loads ~/.trace_use/motifs.json
    detector  = TrajectoryDetector(store, embedder)  # pre-task warning injection
    brain     = BrainAgent(embedder, motif_store=store)  # mid-execution interception

    # Before each task:
    enriched_prompt, matches = detector.inject(task_prompt)
    # ... run agent with enriched_prompt ...
    brain.store(trace, int(passed), metadata="failure reason")
"""

from .brain import BrainAgent

from .motif_store import PersistentMotifStore

from .trajectory import TrajectoryDetector, MotifMatch

from .config import BrainConfig, DetectorConfig

from .agents import (
    haiku,
    opus,
    tool_agent,
    build_embedder,
)

from .pipeline import (
    Forecaster,
    TaskResult,
    ComponentResult,
    run_task,
    decompose,
    attempt,
    gold_judge,
    self_judge,
    self_consistency,
    tiered_judge,
    make_retriever,
    code_judge,
    extract_code,
)

__all__ = [
    # Core
    "BrainAgent",
    "PersistentMotifStore",
    "TrajectoryDetector",
    "MotifMatch",
    # Config
    "BrainConfig",
    "DetectorConfig",
    # Agents & embedder
    "haiku",
    "opus",
    "tool_agent",
    "build_embedder",
    # Forecaster pipeline
    "run_task",
    "Forecaster",
    "TaskResult",
    "ComponentResult",
    # Pipeline primitives
    "decompose",
    "attempt",
    # Verifiers
    "gold_judge",
    "self_judge",
    "self_consistency",
    "tiered_judge",
    "code_judge",
    # Utilities
    "extract_code",
    "make_retriever",
]
