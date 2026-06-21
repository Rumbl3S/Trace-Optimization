"""Self-contained agent + embedder for the evals — no external project deps.

`haiku` / `opus` are temperature-0 Anthropic agents (`prompt -> (text, tokens)`);
`_build_openai()` returns an OpenAI `text-embedding-3-small` embedder
(`texts -> L2-normalised vectors`). Clients are created lazily on first use, so importing
this module never needs keys — only running an eval does. Keys load from the environment
or a nearby `.env`.
"""
from __future__ import annotations

import os

_HAIKU_MODEL  = "claude-haiku-4-5-20251001"
_SONNET_MODEL = "claude-sonnet-4-6"
_OPUS_MODEL   = "claude-opus-4-8"
_EMBED_MODEL  = "text-embedding-3-small"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        for p in (".env", os.path.join("..", ".env"), os.path.join("..", "..", ".env")):
            load_dotenv(p)
    except Exception:                                       # noqa: BLE001
        pass


_load_env()
_client = None


def _anthropic_call(model: str, prompt: str, max_tokens: int = 512):
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    # temperature=0 pins sampling where the model allows it; Opus 4 rejects the
    # parameter outright (400), so omit it there. (Haiku 4.5 accepts it.)
    supports_temp = "opus-4" not in model
    kwargs = {"temperature": 0} if supports_temp else {}
    last = None
    for _ in range(2):
        try:
            r = _client.messages.create(model=model, max_tokens=max_tokens,
                                        messages=[{"role": "user", "content": prompt}],
                                        **kwargs)
            return r.content[0].text, r.usage.input_tokens + r.usage.output_tokens
        except Exception as e:                              # noqa: BLE001
            last = e
    raise last


def haiku(prompt: str):
    """Fast, cheap temperature-0 agent. Returns (text, total_tokens)."""
    return _anthropic_call(_HAIKU_MODEL, prompt)


def opus(prompt: str):
    """High-accuracy temperature-0 agent. Returns (text, total_tokens)."""
    return _anthropic_call(_OPUS_MODEL, prompt, max_tokens=1024)


def tool_agent(tools: list | None = None, max_turns: int = 4,
               model: str | None = None):
    """ReAct tool-calling agent. Returns a callable (prompt -> (trace, tokens)).

    `tools` is a subset of ['calculator', 'python_exec', 'wikipedia_search'].
    Defaults to all three. The full trajectory — reasoning, tool calls, and
    tool results — is returned as the trace so the Forecaster can embed it.

    `model` overrides the default (_HAIKU_MODEL). Pass _SONNET_MODEL or any
    Anthropic model ID to trade cost for capability on harder tasks.
    """
    from tools import TOOL_DEFINITIONS, dispatch

    selected = tools or ["calculator", "python_exec", "wikipedia_search"]
    tool_defs = [t for t in TOOL_DEFINITIONS if t["name"] in selected]
    _model = model or _HAIKU_MODEL

    def agent(prompt: str):
        global _client
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic()

        messages = [{"role": "user", "content": prompt}]
        trace_parts: list[str] = []
        total_tokens = 0
        # read once at call-start; run_task sets this before calling attempt()
        bail = getattr(agent, 'bail_fn', None)

        for _ in range(max_turns):
            # 4096 so the model can emit a complete implementation inside a single
            # tool call — at 1024 a large code block is truncated mid-JSON, the
            # tool input arrives empty, nothing runs, and the component fails.
            r = _client.messages.create(
                model=_model, max_tokens=4096, temperature=0,
                tools=tool_defs if tool_defs else [],
                messages=messages,
            )
            total_tokens += r.usage.input_tokens + r.usage.output_tokens

            # collect text and tool calls from this turn
            tool_results = []
            for block in r.content:
                if block.type == "text":
                    trace_parts.append(block.text)
                elif block.type == "tool_use":
                    result = dispatch(block.name, block.input)
                    trace_parts.append(
                        f"[tool:{block.name}({block.input})] → {result[:500]}"
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:4000],
                    })

            if r.stop_reason == "end_turn" or not tool_results:
                break

            # Mid-call checkpoint: if the partial trace already resembles past
            # failures, bail now rather than spending another LLM turn on a
            # trajectory that's likely to end the same way. The outer run_task
            # loop sees the [EARLY_EXIT] trace, scores it high, and fires the
            # informed _RETRY prompt — saving all remaining turns.
            if bail and bail("\n".join(trace_parts)):
                trace_parts.append("[EARLY_EXIT: trajectory matches past failures]")
                break

            messages.append({"role": "assistant", "content": r.content})
            messages.append({"role": "user", "content": tool_results})

        return "\n".join(trace_parts), total_tokens

    agent.bail_fn = None   # wired by run_task once a forecaster is live
    return agent


def _build_openai():
    """Return an embedder: list[str] -> (n, d) float32 L2-normalised array."""
    from openai import OpenAI
    import numpy as np
    client = OpenAI()

    def embed(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), 256):
            batch = [t[:8000] for t in texts[i:i + 256]]
            r = client.embeddings.create(model=_EMBED_MODEL, input=batch)
            out.extend(d.embedding for d in r.data)
        v = np.asarray(out, dtype="float32")
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return embed
