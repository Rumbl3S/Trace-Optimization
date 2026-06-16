"""Shared eval plumbing: a dataset record type, answer scoring (exact-match + F1),
and a tiny synthetic fallback so the pipeline can be dry-run with no network.

Each loader returns a list of `Record`s. `context_chunks` is the per-instance
evidence pool (paragraphs/sentences) the orchestrator routes over; `gold` is the
reference answer string.
"""
from __future__ import annotations

import os
import re
import string
import sys
from dataclasses import dataclass, field

# Make the package root importable when a loader is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class Record:
    task: str
    context_chunks: list[str]
    gold: str
    meta: dict = field(default_factory=dict)


# ── Answer normalization + scoring (SQuAD/HotpotQA style) ───────────────────────
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for w in p:
        common[w] = min(p.count(w), g.count(w))
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p)
    recall = n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def contains_gold(pred: str, gold: str) -> float:
    """Looser credit: gold answer string appears in the (often verbose) prediction.
    Reported alongside EM/F1 because budget-limited answers can be wordy."""
    return 1.0 if normalize(gold) and normalize(gold) in normalize(pred) else 0.0


def score(pred: str, gold: str) -> dict:
    return {"em": exact_match(pred, gold),
            "f1": f1_score(pred, gold),
            "contains": contains_gold(pred, gold)}


# ── Synthetic fallback (offline dry-run) ───────────────────────────────────────
def synthetic_records(n: int = 3) -> list[Record]:
    """A few hand-built multi-hop instances so plumbing can be exercised with no
    network and (with stubbed LLM) no API. Deliberately straddles the regime
    boundary: clean single-hop and dispersed multi-hop."""
    base = [
        Record(
            task="What is the capital of France?",
            context_chunks=["Paris is the capital and most populous city of France."],
            gold="Paris", meta={"hops": 1}),
        Record(
            task=("Who directed the highest-grossing film of 1997, and in which "
                  "country was that director born?"),
            context_chunks=[
                "Titanic was the highest-grossing film of 1997.",
                "Bananas are grown in tropical climates.",
                "Titanic was directed by James Cameron.",
                "Asian markets were volatile in 1997.",
                "James Cameron was born in Canada in 1954.",
                "The Great Barrier Reef is off the coast of Australia."],
            gold="Canada", meta={"hops": 2}),
        Record(
            task=("The author who wrote the novel adapted into the film that won "
                  "Best Picture in 1994 was born in which US state?"),
            context_chunks=[
                "Forrest Gump won the Academy Award for Best Picture in 1994.",
                "Forrest Gump is based on the 1986 novel by Winston Groom.",
                "Unrelated: photosynthesis converts light into chemical energy.",
                "Winston Groom was born in Washington, D.C.",
                "Many films are adapted from novels."],
            gold="Washington", meta={"hops": 3}),
    ]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out[:n]


def try_load(loader_fn, n: int, dry_run: bool) -> list[Record]:
    """Run a HuggingFace loader, but fall back to synthetic records on any failure
    (no network, dataset moved, etc.) or when --dry-run is requested."""
    if dry_run:
        return synthetic_records(n)
    try:
        recs = loader_fn(n)
        if recs:
            return recs
    except Exception as e:    # noqa: BLE001 - eval convenience
        print(f"[warn] dataset load failed ({e!r}); using synthetic records.")
    return synthetic_records(n)
