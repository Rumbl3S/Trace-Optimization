"""Centralized configuration for BrainAgent and TrajectoryDetector.

All tunable constants live here — no magic numbers scattered across modules.
Pass a config instance to override any default:

    cfg = BrainConfig(judge_model="claude-sonnet-4-6", judge_threshold=0.75)
    brain = BrainAgent(embedder, config=cfg)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_MODEL        = "claude-haiku-4-5-20251001"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_STORAGE_PATH = os.path.join(os.path.expanduser("~"), ".trace_use", "motifs.json")


@dataclass
class BrainConfig:
    """Configuration for BrainAgent (mid-execution, before_tool_call detection)."""

    # Provider: "anthropic" (default) or "openai"
    provider:           str   = "anthropic"

    # LLM — set to an OpenAI model name when provider="openai"
    judge_model:        str   = _DEFAULT_MODEL
    extract_model:      str   = _DEFAULT_MODEL
    judge_max_tokens:   int   = 320
    extract_max_tokens: int   = 400

    # Firing
    judge_threshold:    float = 0.80   # min confidence to fire
    max_interventions:  int   = 2      # max fires per task

    # Retrieval
    retrieval_top_k:    int   = 4
    retrieval_min_sim:  float = 0.10   # set after observing actual scores via [BRAIN RETRIEVE] diagnostic

    # Agent hooks
    exec_tool_name:         str   = "python_exec"
    stall_streak_threshold: int   = 2
    reasoning_window:       int   = 20   # last N reasoning chunks passed to judge

    # Background extraction — gpt-4o is slower than haiku; give it room
    extract_timeout:    float = 20.0


@dataclass
class DetectorConfig:
    """Configuration for TrajectoryDetector (pre-task, pre-prompt injection)."""

    # Provider: "anthropic" (default) or "openai"
    provider:   str = "anthropic"

    # LLM — set to an OpenAI model name when provider="openai"
    model:      str = _DEFAULT_MODEL
    max_tokens: int = 256

    # Retrieval — wider net than BrainConfig (no code context yet)
    retrieval_top_k:   int   = 5
    retrieval_min_sim: float = 0.10   # OpenAI embeddings score lower than local models

    # Firing — lower bar than execution check; we're advising, not blocking
    judge_threshold: float = 0.70

    # Persistence
    storage_path: str = _DEFAULT_STORAGE_PATH
