"""Small self-contained helpers (vendored, dependency-free)."""
from __future__ import annotations

import re
from typing import Iterable, Union


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def select_for_single(query: str, context: Union[str, Iterable[str], None], max_words: int) -> str:
    """Coherent, relevance-capped context: rank whole chunks by keyword overlap with the
    query, take the top ones up to `max_words`, and join them in original order."""
    if not context:
        return ""
    chunks = [context] if isinstance(context, str) else list(context)
    qk = _words(query)
    order = {id(c): i for i, c in enumerate(chunks)}
    ranked = sorted(chunks, key=lambda c: len(qk & _words(c)), reverse=True)
    chosen, used = [], 0
    for c in ranked:
        w = len(c.split())
        if chosen and used + w > max_words:
            break
        chosen.append(c)
        used += w
        if used >= max_words:
            break
    chosen.sort(key=lambda c: order[id(c)])
    return "\n".join(chosen)
