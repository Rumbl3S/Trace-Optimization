"""Self-contained agent + embedder for the evals — no external project deps.

`haiku` / `opus` are temperature-0 Anthropic agents (`prompt -> (text, tokens)`);
`_build_openai()` returns an OpenAI `text-embedding-3-small` embedder
(`texts -> L2-normalised vectors`). Clients are created lazily on first use, so importing
this module never needs keys — only running an eval does. Keys load from the environment
or a nearby `.env`.
"""
from __future__ import annotations

import os

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_OPUS_MODEL  = "claude-opus-4-8"
_EMBED_MODEL = "text-embedding-3-small"


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
    last = None
    for _ in range(2):
        try:
            r = _client.messages.create(model=model, max_tokens=max_tokens, temperature=0,
                                        messages=[{"role": "user", "content": prompt}])
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
