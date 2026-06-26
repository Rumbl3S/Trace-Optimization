"""trace_use — forecast agent failure from execution traces.

Install:
    pip install trace-use

Usage:
    from trace_use import BrainAgent, build_embedder, tool_agent

    brain         = BrainAgent(build_embedder(), threshold=0.30)
    agent         = tool_agent(["python_exec"], model="claude-haiku-4-5-20251001")
    agent.monitor = brain
"""

from .brain import BrainAgent

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
    # Brain
    "BrainAgent",
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
