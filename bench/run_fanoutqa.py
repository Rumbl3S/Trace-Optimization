"""FanOutQA loader — fan-out / aggregation QA (Zhu et al. 2024).

Each question's answer must be assembled from facts about MANY entities (fan-out =
4-6), and the provided evidence is HUGE (40k-350k tokens of concatenated Wikipedia
articles). Under a small token budget a single agent can only see a slice of the
evidence, so its context utilization degrades as fan-out / doc size grows — exactly
the DPI regime where decomposing across agents (one sub-question per entity) can
win. The fan-out count is a built-in difficulty knob, so the data spans a real
single-vs-multi boundary instead of being uniformly easy or hard.

Scoring is FanOutQA's "loose" accuracy: the fraction of the gold answer's leaf
strings (the per-entity facts) that appear in the prediction — a continuous,
fan-out-sensitive metric (single should score lower as fan-out rises). `strict` is
all-present. Returns the standard metric keys too so the harness is unchanged.

Source: ragrawal36/fanoutqa (question, answer JSON, pos_doc = provided evidence).
"""
from __future__ import annotations

import ast
import json
import re

from bench._common import Record, try_load


# ── answer / evidence helpers ──────────────────────────────────────────────────
def _parse_answer(ans: str):
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(ans)
        except Exception:
            pass
    return ans


def _leaf_strings(obj) -> list[str]:
    """Flatten an answer (dict/list/scalar) to the fact strings to look for. For a
    {entity: fact} dict we check the FACTS (values); entities are given in the Q."""
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _leaf_strings(v)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out += _leaf_strings(x)
    else:
        out.append(str(obj))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def _chunks(doc: str, max_words: int = 140, cap: int = 1500) -> list[str]:
    """Split provided evidence into retrieval chunks. Strips simple wiki templates,
    splits on blank lines, then windows long paragraphs. Capped to bound TF-IDF
    cost on the largest docs."""
    doc = re.sub(r"\{\{[^{}]*\}\}", " ", doc or "")
    out = []
    for para in re.split(r"\n\s*\n", doc):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < 40:
            continue
        words = para.split()
        for i in range(0, len(words), max_words):
            out.append(" ".join(words[i:i + max_words]))
            if len(out) >= cap:
                return out
    return out


# ── loader ──────────────────────────────────────────────────────────────────────
def _load_hf(n: int) -> list[Record]:
    from datasets import load_dataset
    ds = load_dataset("ragrawal36/fanoutqa", split="train")
    recs = []
    for ex in ds:
        q, ans, doc = ex.get("question"), ex.get("answer"), ex.get("pos_doc") or ""
        if not q or not ans or not doc:
            continue
        chunks = _chunks(doc)
        if not chunks:
            continue
        obj = _parse_answer(ans)
        fanout = len(obj) if isinstance(obj, (dict, list)) else 1
        refs = sorted({_norm(s) for s in _leaf_strings(obj) if _norm(s)})
        if not refs:
            continue
        task = (f"{q}\n\nList EACH item the question asks about and its value, one "
                f"per line as 'Item: value'. Be concise and factual.")
        recs.append(Record(task=task, context_chunks=chunks, gold=str(ans),
                           meta={"fanout": fanout, "ref_strings": refs,
                                 "doc_chunks": len(chunks)}))
        if len(recs) >= n:
            break
    return recs


def load(n: int = 20, dry_run: bool = False) -> list[Record]:
    return try_load(_load_hf, n, dry_run)


# ── scoring (FanOutQA loose / strict) ──────────────────────────────────────────
def score(pred: str, rec: Record) -> dict:
    refs = rec.meta.get("ref_strings", [])
    if not refs:
        return {"loose": 0.0, "strict": 0.0, "contains": 0.0, "f1": 0.0, "em": 0.0}
    p = _norm(pred)
    found = sum(1 for r in refs if r and r in p)
    loose = found / len(refs)
    strict = 1.0 if found == len(refs) else 0.0
    return {"loose": loose, "strict": strict,
            "contains": loose, "f1": loose, "em": strict}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for r in load(args.n, args.dry_run):
        print(f"\nfanout={r.meta['fanout']} chunks={r.meta['doc_chunks']} "
              f"refs={len(r.meta['ref_strings'])}\n  Q: {r.task.splitlines()[0][:90]}\n"
              f"  gold: {r.gold[:90]}")
