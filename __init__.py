"""trace_use — forecast agent failure from execution traces.

Quick start:
    from trace_use import run_task, Forecaster
    from trace_use.agents import haiku, opus, tool_agent, _build_openai

    embedder = _build_openai()
    fc = Forecaster(embedder)

    result = run_task(
        task="What is the GDP of France and how does it compare to Germany?",
        agent=haiku,
        forecaster=fc,
    )
    print(result.summary())
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import (
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

from agents import (
    haiku,
    opus,
    tool_agent,
    _build_openai,
)

__all__ = [
    # core pipeline
    "run_task",
    "Forecaster",
    "TaskResult",
    "ComponentResult",
    # primitives
    "decompose",
    "attempt",
    # verifiers
    "gold_judge",
    "self_judge",
    "self_consistency",
    "tiered_judge",
    "code_judge",
    "extract_code",
    # retrieval
    "make_retriever",
    # agents & embedders
    "haiku",
    "opus",
    "tool_agent",
    "_build_openai",
]
