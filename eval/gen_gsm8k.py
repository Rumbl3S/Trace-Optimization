"""PART (transfer breadth) — a NON-QA task family: GSM8K math word problems.

Closed-book multi-step arithmetic reasoning — no retrieval, fundamentally unlike the QA
datasets. Its verifier is EXACT NUMERIC MATCH (not an LLM judge), which doubles as proof
that the verifier is genuinely pluggable: swap the check, the rest is unchanged.

Saves trajectories_gsm8k.json (same schema as the QA sets). Merge with the balanced QA set
and run eval/generalize.py to test whether the failure forecaster transfers to a task
family it has never seen.

  python eval/gen_gsm8k.py --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(__file__)
RS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [RS, os.path.join(RS, "eval"), os.path.abspath(os.path.join(HERE, ".."))]

import llm
from datasets import load_dataset
from demo_embed_compare import haiku

llm._ensure_api_key()

PROMPT = ("Answer this quickly using mental math — do NOT write out detailed step-by-step "
          "work. Give a brief one-line justification, then 'ANSWER: <number>'.\n\n"
          "Problem: {q}")


def last_number(s: str):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None


def verify_numeric(trace: str, gold: str) -> int:
    g = last_number(gold)
    m = re.search(r"answer\s*:\s*(.+)", trace, re.I)
    pred = last_number(m.group(1)) if m else last_number(trace)
    if g is None or pred is None:
        return 0
    try:
        return int(abs(float(g) - float(pred)) < 1e-6)
    except ValueError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    ds = load_dataset("gsm8k", "main", split=f"test[:{args.n}]")

    recs = []
    for i, ex in enumerate(ds):
        gold = ex["answer"].split("####")[-1].strip()
        try:
            trace, _ = haiku(PROMPT.format(q=ex["question"]))
        except Exception as e:                                # noqa: BLE001
            print(f"  [{i+1}/{len(ds)}] failed: {e!r}", flush=True); continue
        lab = verify_numeric(trace, gold)
        recs.append({"src": "gsm8k", "task": ex["question"], "score": float(lab),
                     "label": lab, "trace": trace})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ds)}] done", flush=True)

    out = os.path.join(HERE, "results", "trajectories_gsm8k.json")
    json.dump(recs, open(out, "w"))
    succ = sum(r["label"] for r in recs)
    print(f"\ngsm8k: success {succ}/{len(recs)} = {succ/max(1,len(recs)):.2f}  -> {out}")


if __name__ == "__main__":
    main()
